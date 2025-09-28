# sace_cv/utils.py
import os
import random
import numpy as np
import torch

IMG_CANDIDATES = ["img", "images", "image", "imgs"]
MASK_CANDIDATES = ["labelcol", "masks", "mask", "labels"]

def find_fold_dirs(kfold_root):
    all_entries = sorted(os.listdir(kfold_root))
    folds = []
    for name in all_entries:
        p = os.path.join(kfold_root, name)
        if os.path.isdir(p) and 'fold' in name.lower():
            folds.append(p)
    if len(folds) == 0:
        folds = [os.path.join(kfold_root, p) for p in all_entries if os.path.isdir(os.path.join(kfold_root, p))]
    folds = sorted(folds)
    return folds

def find_img_mask_dirs(split_dir):
    for img_name in IMG_CANDIDATES:
        p = os.path.join(split_dir, img_name)
        if os.path.isdir(p):
            for mask_name in MASK_CANDIDATES:
                q = os.path.join(split_dir, mask_name)
                if os.path.isdir(q):
                    return p, q
    return split_dir, split_dir

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # deterministic cudnn for reproducibility (may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
