# sace_cv/model.py
from .blocks import DoubleConv, MRFBlock, TokenModule, TokenToPixelAttention, pad_to_shape
import torch.nn as nn

class SACE_EncoderDecoder(nn.Module):
    def __init__(self, in_channels=3, base_ch=32, K=8, d=64, use_mrf=True, use_tokens=True):
        super().__init__()
        self.use_mrf = use_mrf
        self.use_tokens = use_tokens

        self.stem = nn.Sequential(nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
                                  nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True))
        c1, c2, c3, c4 = base_ch, base_ch*2, base_ch*4, base_ch*8

        # MRF or simple conv depending on use_mrf
        if use_mrf:
            self.mrf1 = MRFBlock(c1, c1)
            self.mrf2 = MRFBlock(c2, c2)
            self.mrf3 = MRFBlock(c3, c3)
            self.mrf4 = MRFBlock(c4, c4)
        else:
            self.mrf1 = DoubleConv(c1, c1)
            self.mrf2 = DoubleConv(c2, c2)
            self.mrf3 = DoubleConv(c3, c3)
            self.mrf4 = DoubleConv(c4, c4)

        self.down12 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False)
        self.down23 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False)
        self.down34 = nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False)

        self.center = DoubleConv(c4, 512)

        self.token_mod = TokenModule(stage_channels=(c1,c2,c3,c4), K=K, d=d, enabled=use_tokens)
        self.inject1 = TokenToPixelAttention(c1, d=d, K=K, enabled=use_tokens)
        self.inject2 = TokenToPixelAttention(c2, d=d, K=K, enabled=use_tokens)
        self.inject3 = TokenToPixelAttention(c3, d=d, K=K, enabled=use_tokens)
        self.inject4 = TokenToPixelAttention(c4, d=d, K=K, enabled=use_tokens)

        # Decoder
        self.up1 = nn.ConvTranspose2d(512, c4, 2, stride=2)
        self.dec1 = DoubleConv(c4 + c4, c4)
        self.up2 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec2 = DoubleConv(c3 + c3, c3)
        self.up3 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec3 = DoubleConv(c2 + c2, c2)
        self.up4 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec4 = DoubleConv(c1 + c1, c1)
        self.out_head = nn.Sequential(nn.Conv2d(c1, 32, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                      nn.Conv2d(32, 1, kernel_size=1))
    def forward(self, x):
        s = self.stem(x)
        f1 = self.mrf1(s)
        f2_in = self.down12(f1); f2 = self.mrf2(f2_in)
        f3_in = self.down23(f2); f3 = self.mrf3(f3_in)
        f4_in = self.down34(f3); f4 = self.mrf4(f4_in)
        center = self.center(f4)
        tokens = self.token_mod([f1,f2,f3,f4])
        f1c = self.inject1(f1, tokens); f2c = self.inject2(f2, tokens)
        f3c = self.inject3(f3, tokens); f4c = self.inject4(f4, tokens)
        u1 = self.up1(center)
        if u1.shape[2:] != f4c.shape[2:]:
            u1 = pad_to_shape(u1, f4c.shape)
        d1 = self.dec1(torch.cat([u1, f4c], dim=1))
        u2 = self.up2(d1)
        if u2.shape[2:] != f3c.shape[2:]:
            u2 = pad_to_shape(u2, f3c.shape)
        d2 = self.dec2(torch.cat([u2, f3c], dim=1))
        u3 = self.up3(d2)
        if u3.shape[2:] != f2c.shape[2:]:
            u3 = pad_to_shape(u3, f2c.shape)
        d3 = self.dec3(torch.cat([u3, f2c], dim=1))
        u4 = self.up4(d3)
        if u4.shape[2:] != f1c.shape[2:]:
            u4 = pad_to_shape(u4, f1c.shape)
        d4 = self.dec4(torch.cat([u4, f1c], dim=1))
        out = self.out_head(d4)
        return out
