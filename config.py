# sace_cv/config.py
# Default configuration. You can override these via CLI args when running run_cv.py
import os

ROOT = os.getenv("SACE_ROOT", "/content/Data") #change path to your data root
K_FOLD_DIR = os.path.join(ROOT, "k-fold")
TEST_DIR = os.path.join(ROOT, "test")
OUT_DIR = os.path.join(ROOT, "cv_ckpts")

DEVICE = "cuda"  # auto-detected in runner, this is just default string
NUM_FOLDS = 5
NUM_EPOCHS = 60
BATCH_SIZE = 2
ACCUMULATION_STEPS = 4
NUM_WORKERS = 4
PIN_MEMORY = True
IMAGE_SIZE = 512

# Model hyperparams
BASE_CH = 32
K_TOKENS = 8
TOKEN_DIM = 64

# Loss config
TV_ALPHA = 0.5
TV_BETA = 0.7
USE_HAUSDORFF = True

# Repro
SEED = 42
