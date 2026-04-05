"""
Shared configuration for the thesis project.
All hyperparameters, paths, and constants in one place.
"""
import os
import torch

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
NORMALIZED_DATA_DIR = os.path.join(DATA_DIR, "normalized")
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

# ============================================================
# Dataset
# ============================================================
KAGGLE_DATASET = "forderation/breakhis-400x"

# 8-class mapping (alphabetical for consistency)
CLASS_NAMES = [
    "adenosis",
    "ductal_carcinoma",
    "fibroadenoma",
    "lobular_carcinoma",
    "mucinous_carcinoma",
    "papillary_carcinoma",
    "phyllodes_tumor",
    "tubular_adenoma",
]

# Benign vs Malignant grouping
BENIGN_CLASSES = ["adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma"]
MALIGNANT_CLASSES = ["ductal_carcinoma", "lobular_carcinoma", "mucinous_carcinoma", "papillary_carcinoma"]

NUM_CLASSES = len(CLASS_NAMES)

# ============================================================
# Preprocessing
# ============================================================
IMAGE_SIZE = 224
RANDOM_SEED = 42

# ImageNet normalization (required for pretrained backbones)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Train / Validation / Test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Stain normalization methods to evaluate
STAIN_METHODS = ["none", "macenko", "reinhard"]

# ============================================================
# Training
# ============================================================
BATCH_SIZE = 32
NUM_WORKERS = 4
MAX_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 15
LABEL_SMOOTHING = 0.1

# ============================================================
# Device
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Helpers
# ============================================================
def ensure_dirs():
    """Create all project directories if they don't exist."""
    for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, NORMALIZED_DATA_DIR,
              WEIGHTS_DIR, RESULTS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)
