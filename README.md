# SACE Cross-Validation & Ablation (Modular)

This repository contains modular code to run 5-fold cross-validation using the SACE (MRF + token attention) segmentation architecture and evaluate each fold's best checkpoint on an external test set.

Structure
- sace_cv/: python package with dataset, losses, model, metrics and runner.
- run with: python -m sace_cv.run_cv or python sace_cv/run_cv.py

Data layout (expected)
- ROOT/k-fold/ : folder containing fold subfolders (e.g. fold1, fold2...). Each fold should include 'train' and 'test' subfolders (or two subfolders).
  - foldX/train/img (or train/) and train/labelcol (or masks/...) — dataset images & masks (shared names).
  - foldX/test/...
- ROOT/test/img and ROOT/test/labelcol : external test dataset (explicitly required).

Reproducibility
- Seeds and deterministic cudnn options are set by default (see sace_cv/utils.set_seed).
- The runner saves run_config.json to the output directory containing used parameters.

Quick start (Colab)
1. Upload repo to Colab.
2. Mount Drive and ensure ROOT points to the folder containing k-fold & test.
3. pip install -r requirements.txt
4. Run:
   python -m sace_cv.run_cv --root /content/drive/MyDrive/MoNuSegnet

For debugging, use --quick to run fewer epochs/folds.
