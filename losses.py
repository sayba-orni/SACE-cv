# sace_cv/losses.py
import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import distance_transform_edt

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        tversky = TP / (TP + self.alpha * FP + self.beta * FN + 1e-7)
        return 1 - tversky

class HausdorffLoss(nn.Module):
    def __init__(self, alpha=2.0):
        super().__init__()
        self.alpha = alpha
    def _distance_transform(self, binary_mask: np.ndarray) -> np.ndarray:
        fg_dist = distance_transform_edt(binary_mask == 0)
        bg_dist = distance_transform_edt(binary_mask == 1)
        return fg_dist + bg_dist
    @torch.no_grad()
    def _batch_distance_transform(self, masks: torch.Tensor) -> torch.Tensor:
        b, _, h, w = masks.shape
        dist_maps = []
        for i in range(b):
            mask_np = masks[i, 0].detach().cpu().numpy().astype(np.uint8)
            dist = self._distance_transform(mask_np)
            dist_maps.append(torch.from_numpy(dist).float())
        dist_tensor = torch.stack(dist_maps).unsqueeze(1).to(masks.device)
        return dist_tensor
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert pred.shape == target.shape
        pred_bin = (pred > 0.5).float()
        pred_dist = self._batch_distance_transform(pred_bin)
        target_dist = self._batch_distance_transform(target)
        dist_diff = (pred_dist - target_dist) ** 2
        weighted_loss = dist_diff * (target_dist ** self.alpha)
        return weighted_loss.mean()
