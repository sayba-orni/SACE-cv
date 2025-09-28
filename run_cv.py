# sace_cv/run_cv.py
import os
import json
import math
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .config import *
from .utils import find_fold_dirs, find_img_mask_dirs, set_seed
from .dataset import MoNuSeg
from .model import SACE_EncoderDecoder
from .losses import TverskyLoss, HausdorffLoss
from .metrics import compute_metrics

def make_transforms(image_size):
    train_tfms = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize([0,0,0],[1,1,1], max_pixel_value=255.0),
        ToTensorV2()
    ])
    val_tfms = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize([0,0,0],[1,1,1], max_pixel_value=255.0),
        ToTensorV2()
    ])
    return train_tfms, val_tfms

def run_cv(args):
    # reproducibility
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    print("Using device:", device)

    # paths and directories
    root = args.root
    k_fold_dir = args.kfold_dir or os.path.join(root, "k-fold")
    test_dir = args.test_dir or os.path.join(root, "test")
    out_dir = args.out_dir or os.path.join(root, "cv_ckpts")
    os.makedirs(out_dir, exist_ok=True)

    # save the config used
    run_config = vars(args)
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    # transforms
    train_tfms, val_tfms = make_transforms(args.image_size)

    # prepare external test loader
    test_img_dir = os.path.join(test_dir, "img")
    test_mask_dir = os.path.join(test_dir, "labelcol")
    if not os.path.isdir(test_img_dir) or not os.path.isdir(test_mask_dir):
        raise FileNotFoundError(f"External test not found at: {test_img_dir} or {test_mask_dir}")
    test_ds = MoNuSeg(test_img_dir, test_mask_dir, transform=val_tfms)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    print(f"External test set size: {len(test_ds)} images (img: {test_img_dir}, mask: {test_mask_dir})")

    # find folds
    fold_dirs = find_fold_dirs(k_fold_dir)
    print("Found fold directories:", fold_dirs)
    if len(fold_dirs) < args.num_folds:
        print(f"Warning: found {len(fold_dirs)} folds < expected {args.num_folds}. Will iterate over found folds.")

    fold_results = []

    for fold_idx, fold_path in enumerate(fold_dirs[:args.num_folds], start=1):
        print("="*80)
        print(f"FOLD {fold_idx} -> {fold_path}")

        # find train/val
        train_split_dir = os.path.join(fold_path, "train")
        val_split_dir = os.path.join(fold_path, "test")
        if not os.path.isdir(train_split_dir) or not os.path.isdir(val_split_dir):
            print("Could not find exact 'train'/'test' subfolders. Using first two subdirs as fallback.")
            subs = sorted([os.path.join(fold_path, p) for p in os.listdir(fold_path) if os.path.isdir(os.path.join(fold_path,p))])
            if len(subs) >= 2:
                train_split_dir, val_split_dir = subs[0], subs[1]
            else:
                raise FileNotFoundError(f"Fold {fold_path} lacks expected split subfolders.")
        train_img_dir, train_mask_dir = find_img_mask_dirs(train_split_dir)
        val_img_dir, val_mask_dir = find_img_mask_dirs(val_split_dir)
        print(" Using train img/mask:", train_img_dir, train_mask_dir)
        print(" Using val   img/mask:", val_img_dir, val_mask_dir)

        train_ds = MoNuSeg(train_img_dir, train_mask_dir, transform=train_tfms)
        val_ds   = MoNuSeg(val_img_dir, val_mask_dir, transform=val_tfms)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
        val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

        # instantiate model and losses
        model = SACE_EncoderDecoder(in_channels=3, base_ch=args.base_ch, K=args.k_tokens, d=args.token_dim).to(device)
        tversky_loss = TverskyLoss(alpha=args.tv_alpha, beta=args.tv_beta).to(device)
        mse_loss = torch.nn.MSELoss().to(device)
        haus_loss = HausdorffLoss(alpha=2.0).to(device) if args.use_hausdorff else None

        # learnable log vars (uncertainty weighting)
        log_var_t = torch.nn.Parameter(torch.zeros(1, device=device))
        log_var_m = torch.nn.Parameter(torch.zeros(1, device=device))
        log_var_h = torch.nn.Parameter(torch.zeros(1, device=device))
        optimizer = optim.AdamW([{'params': model.parameters()}, {'params': [log_var_t, log_var_m, log_var_h]}], lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
        scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

        best_val_f1 = -1.0
        best_metrics = None
        ckpt_path = os.path.join(out_dir, f"fold{fold_idx}_best.pth.tar")

        # training loop
        for epoch in range(args.num_epochs):
            model.train()
            loop = tqdm(train_loader, desc=f"Fold{fold_idx} Epoch {epoch+1}/{args.num_epochs}", leave=False)
            running_loss = 0.0
            optimizer.zero_grad()
            for batch_idx, (imgs, masks) in enumerate(loop):
                imgs = imgs.to(device)
                masks = masks.to(device).unsqueeze(1).float()
                with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                    logits = model(imgs)
                    l_t = tversky_loss(logits, masks)
                    l_m = mse_loss(torch.sigmoid(logits), masks)
                    if args.use_hausdorff:
                        l_h = haus_loss(torch.sigmoid(logits), masks)
                    else:
                        l_h = torch.tensor(0.0, device=device)
                    precision_t = torch.exp(-log_var_t)
                    precision_m = torch.exp(-log_var_m)
                    precision_h = torch.exp(-log_var_h)
                    loss = (precision_t * l_t + log_var_t +
                            precision_m * l_m + log_var_m +
                            precision_h * l_h + log_var_h)
                    loss = loss / args.accumulation_steps

                scaler.scale(loss).backward()

                if (batch_idx + 1) % args.accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                running_loss += loss.item() * args.accumulation_steps
                loop.set_postfix(loss=running_loss / (batch_idx + 1),
                                 l_t=float(l_t.item()), l_m=float(l_m.item()))
            scheduler.step()

            # validation
            val_metrics = compute_metrics(val_loader, model, device=device, thresh=0.5)
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_metrics = val_metrics
                ckpt = {
                    "fold": fold_idx,
                    "epoch": epoch+1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_f1": best_val_f1,
                    "metrics": val_metrics
                }
                torch.save(ckpt, ckpt_path)

            if (epoch+1) % 10 == 0 or epoch == args.num_epochs - 1:
                print(f"Fold {fold_idx} Epoch {epoch+1}  val_f1={val_metrics['f1']:.4f}  val_iou={val_metrics['iou']:.4f}  loss={running_loss/(len(train_loader)+1):.4f}")

        print(f"=> Fold {fold_idx} finished. Best val F1 = {best_val_f1:.4f}  (checkpoint: {ckpt_path})")

        # evaluate best checkpoint on external test set
        if os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])
            test_metrics = compute_metrics(test_loader, model, device=device, thresh=0.5)
            print(f" Fold {fold_idx} TEST metrics (external test set): F1={test_metrics['f1']:.4f}  IoU={test_metrics['iou']:.4f}  Dice={test_metrics['dice']:.4f}")
        else:
            test_metrics = None
            print(f" No checkpoint found for fold {fold_idx}, skipping external test evaluation.")

        fold_results.append({
            "fold": fold_idx,
            "best_val_f1": best_val_f1,
            "best_val_metrics": best_metrics,
            "test_metrics": test_metrics,
            "ckpt": ckpt_path if os.path.isfile(ckpt_path) else None
        })

    # aggregate
    test_f1s = [r["test_metrics"]["f1"] for r in fold_results if r["test_metrics"] is not None]
    test_ious = [r["test_metrics"]["iou"] for r in fold_results if r["test_metrics"] is not None]
    test_dices = [r["test_metrics"]["dice"] for r in fold_results if r["test_metrics"] is not None]

    summary = {
        "per_fold": fold_results,
        "mean_test_f1": float(np.mean(test_f1s)) if test_f1s else None,
        "std_test_f1": float(np.std(test_f1s)) if test_f1s else None,
        "mean_test_iou": float(np.mean(test_ious)) if test_ious else None,
        "std_test_iou": float(np.std(test_ious)) if test_ious else None,
        "mean_test_dice": float(np.mean(test_dices)) if test_dices else None,
        "std_test_dice": float(np.std(test_dices)) if test_dices else None
    }

    print("\nFinal Cross-Validation -> External test Summary:")
    if summary["mean_test_f1"] is not None:
        print(f"  External Test F1 mean: {summary['mean_test_f1']:.4f}  std: {summary['std_test_f1']:.4f}")
        print(f"  External Test IoU mean: {summary['mean_test_iou']:.4f}  std: {summary['std_test_iou']:.4f}")
        print(f"  External Test Dice mean: {summary['mean_test_dice']:.4f}  std: {summary['std_test_dice']:.4f}")
    else:
        print(" No external test metrics were computed (no ckpts found).")

    # save json
    with open(os.path.join(out_dir, "cv_summary_with_external_test.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved CV summary and best checkpoints to", out_dir)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=ROOT, help="Top-level project folder")
    parser.add_argument("--kfold_dir", type=str, default=None, help="K-fold directory (overrides root/k-fold)")
    parser.add_argument("--test_dir", type=str, default=None, help="External test dir (overrides root/test)")
    parser.add_argument("--out_dir", type=str, default=None, help="Output checkpoint dir")
    parser.add_argument("--num_folds", type=int, default=NUM_FOLDS)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--accumulation_steps", type=int, default=ACCUMULATION_STEPS)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--pin_memory", type=bool, default=PIN_MEMORY)
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--base_ch", type=int, default=BASE_CH)
    parser.add_argument("--k_tokens", type=int, default=K_TOKENS)
    parser.add_argument("--token_dim", type=int, default=TOKEN_DIM)
    parser.add_argument("--tv_alpha", type=float, default=TV_ALPHA)
    parser.add_argument("--tv_beta", type=float, default=TV_BETA)
    parser.add_argument("--use_hausdorff", type=bool, default=USE_HAUSDORFF)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU even if CUDA available")
    parser.add_argument("--quick", action="store_true", help="Quick debug run (2 folds, fewer epochs)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # quick override for fast debugging
    if args.quick:
        args.num_folds = min(2, args.num_folds)
        args.num_epochs = 6
        args.batch_size = max(1, args.batch_size)
    # make args accessible in module
    # keep accumulation steps consistent
    args.accumulation_steps = args.accumulation_steps if hasattr(args, "accumulation_steps") else ACCUMULATION_STEPS
    run_cv(args)
