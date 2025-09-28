# sace_cv/metrics.py
import numpy as np
import torch

@torch.no_grad()
def compute_metrics(loader, model, device="cuda", thresh=0.5):
    model.eval()
    eps = 1e-7
    iou_sum = f1_sum = prec_sum = rec_sum = acc_sum = dice_sum = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).unsqueeze(1).float()
        logits = model(x)
        probs = torch.sigmoid(logits)
        preds = (probs > thresh).float()
        tp = (y * preds).sum().float()
        tn = ((1 - y) * (1 - preds)).sum().float()
        fp = ((1 - y) * preds).sum().float()
        fn = (y * (1 - preds)).sum().float()
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        inter = (preds * y).sum()
        union = (preds + y).sum() - inter
        iou = inter / (union + eps)
        dice = 2 * inter / (preds.sum() + y.sum() + eps)
        prec_sum += precision; rec_sum += recall; f1_sum += f1; acc_sum += accuracy
        iou_sum += iou; dice_sum += dice; n += 1
    model.train()
    if n == 0:
        return dict(iou=0,f1=0,precision=0,recall=0,acc=0,dice=0)
    return dict(iou=(iou_sum/n).item(), f1=(f1_sum/n).item(), precision=(prec_sum/n).item(), recall=(rec_sum/n).item(), acc=(acc_sum/n).item(), dice=(dice_sum/n).item())
