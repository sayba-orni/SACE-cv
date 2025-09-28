# sace_cv/blocks.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def pad_to_shape(this, shp):
    # pad tensor 'this' (N,C,H,W) to match shape tuple 'shp' (N,C,H,W)
    if len(shp) == 4:
        pad = (0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2])
        if pad[1] < 0 or pad[3] < 0:
            # If incoming tensor is larger, do center crop to target (rare)
            return this[..., :shp[2], :shp[3]]
        return F.pad(this, pad)
    return this

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class MRFBlock(nn.Module):
    def __init__(self, in_ch, out_ch, branches=4):
        super().__init__()
        self.out_ch = out_ch
        self.b0 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.b1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=2, dilation=2, bias=False),
                                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.b3 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=5, padding=2, groups=in_ch, bias=False),
                                nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
                                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.attn_logits = nn.Conv2d(branches * out_ch, branches, kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.proj = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)
        self.bn_proj = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        b0 = self.b0(x); b1 = self.b1(x); b2 = self.b2(x); b3 = self.b3(x)
        cat = torch.cat([b0, b1, b2, b3], dim=1)
        logits = self.attn_logits(cat)
        alpha = self.softmax(logits)
        stacked = torch.stack([b0, b1, b2, b3], dim=1)
        alpha_exp = alpha.unsqueeze(2)
        fused = (alpha_exp * stacked).sum(dim=1)
        out = self.proj(fused)
        out = self.bn_proj(out)
        if x.shape[1] == out.shape[1]:
            out = out + x
        out = self.relu(out)
        return out

class TokenModule(nn.Module):
    def __init__(self, stage_channels=(32,64,128,256), K=8, d=64, enabled=True):
        super().__init__()
        self.enabled = enabled
        if not enabled:
            return
        self.stage_channels = stage_channels
        self.K = K
        self.d = d
        self.stage_fc = nn.ModuleList([nn.Linear(c, d) for c in stage_channels])
        self.final = nn.Linear(len(stage_channels) * d, K * d)
    def forward(self, stage_feats):
        if not self.enabled:
            B = stage_feats[0].size(0)
            # return zeros so downstream attention modules can accept
            return torch.zeros(B, 1, 1, device=stage_feats[0].device)
        pooled = []
        for i, f in enumerate(stage_feats):
            v = F.adaptive_avg_pool2d(f, 1).view(f.size(0), -1)
            v = self.stage_fc[i](v)
            pooled.append(v)
        cat = torch.cat(pooled, dim=1)
        tokens = self.final(cat)
        tokens = tokens.view(tokens.size(0), self.K, self.d)
        return tokens

class TokenToPixelAttention(nn.Module):
    def __init__(self, in_ch, d=64, K=8, enabled=True):
        super().__init__()
        self.enabled = enabled
        self.d = d; self.K = K
        if not enabled:
            return
        self.q_proj = nn.Conv2d(in_ch, d, kernel_size=1, bias=False)
        self.out_proj = nn.Sequential(nn.Conv2d(d, in_ch, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(in_ch))
        self.token_k = nn.Linear(d, d, bias=False)
        self.token_v = nn.Linear(d, d, bias=False)
    def forward(self, pixel_feat, tokens):
        if not self.enabled:
            return pixel_feat
        B, C, H, W = pixel_feat.shape
        q = self.q_proj(pixel_feat).view(B, self.d, -1).permute(0,2,1)  # (B,HW,d)
        K_tokens = self.token_k(tokens)  # (B,K,d)
        V_tokens = self.token_v(tokens)
        attn = torch.softmax(torch.matmul(q, K_tokens.permute(0,2,1)) / math.sqrt(self.d), dim=-1)  # (B,HW,K)
        ctx = torch.matmul(attn, V_tokens)  # (B,HW,d)
        ctx = ctx.permute(0,2,1).view(B, self.d, H, W)
        out = self.out_proj(ctx)
        return pixel_feat + out
