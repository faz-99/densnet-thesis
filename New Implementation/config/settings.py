"""
Configuration for the Multimodal Histopathology XAI Framework.
All settings: model, data, XAI, LLM, evaluation, logging.
"""
import os
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR.parent / "datasets"
OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
XAI_OUTPUT_DIR = OUTPUT_DIR / "xai_visualizations"
REPORT_OUTPUT_DIR = OUTPUT_DIR / "reports"

for d in [OUTPUT_DIR, CHECKPOINT_DIR, XAI_OUTPUT_DIR, REPORT_OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
DATASET_CONFIG = {
    "name": "BreaKHis",
    "root": str(DATA_DIR / "BreaKHis 400X"),
    "train_dir": str(DATA_DIR / "BreaKHis 400X" / "train"),
    "test_dir": str(DATA_DIR / "BreaKHis 400X" / "test"),
    "magnifications": ["40X", "100X", "200X", "400X"],
    "default_magnification": "400X",
    "img_size": 224,
    "num_workers": 4,
    "batch_size": 16,
    "binary_classes": ["benign", "malignant"],
    "multiclass_classes": [
        "adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma",
        "ductal_carcinoma", "lobular_carcinoma", "mucinous_carcinoma", "papillary_carcinoma",
    ],
    "num_classes_binary": 2,
    "num_classes_multi": 8,
    "mean": (0.5613, 0.5778, 0.6032),
    "std": (0.2114, 0.1957, 0.1590),
    # Stain normalization
    "stain_norm": "macenko",       # "macenko" | "reinhard" | None
    # Patient-level splitting
    "patient_split": True,
    "test_size": 0.2,
    "val_size": 0.1,
    "split_seed": 42,
    # Kaggle download
    "kaggle_dataset": "full",      # "full" (ambarish/breakhis) | "400x"
}

# ──────────────────────────────────────────────
# Model – Hybrid Vision Backbone
# ──────────────────────────────────────────────
MODEL_CONFIG = {
    # Branch A – CNN
    "convnext": {
        "name": "convnext_base",
        "pretrained": True,
        "drop_path_rate": 0.2,
        "feature_dim": 1024,
    },
    # Branch B – Transformer
    "swin": {
        "name": "swin_base_patch4_window7_224",
        "pretrained": True,
        "drop_path_rate": 0.2,
        "feature_dim": 1024,
    },
    # Fusion head
    "fusion": {
        "method": "mlp",          # "weighted_avg" | "mlp"
        "hidden_dim": 512,
        "dropout": 0.3,
    },
    "num_classes": 8,             # 8-class by default; switch to 2 for binary
    "task": "multiclass",         # "binary" | "multiclass"
}

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
TRAIN_CONFIG = {
    "epochs": 100,
    "lr": 3e-4,
    "min_lr": 1e-6,
    "weight_decay": 0.05,
    "scheduler": "cosine",        # "cosine" | "step"
    "warmup_epochs": 5,
    "label_smoothing": 0.1,
    "use_focal_loss": False,
    "focal_alpha": 1.0,
    "focal_gamma": 2.0,
    "use_class_weights": True,
    "early_stopping_patience": 15,
    "grad_clip_norm": 1.0,
    "mixed_precision": True,
    "magnification_aware": True,  # train separate heads per magnification
}

# ──────────────────────────────────────────────
# XAI / Interpretability
# ──────────────────────────────────────────────
XAI_CONFIG = {
    "methods": [
        "grad_cam", "integrated_gradients",
        "shap", "lime",
        "attention_rollout", "counterfactual",
    ],
    "grad_cam": {
        "target_layer": "stages.3",  # ConvNeXt last stage
    },
    "integrated_gradients": {
        "n_steps": 50,
        "internal_batch_size": 8,
    },
    "shap": {
        "n_background": 50,
    },
    "lime": {
        "num_samples": 1000,
        "num_features": 10,
    },
    "attention_rollout": {
        "head_fusion": "mean",       # "mean" | "max" | "min"
        "discard_ratio": 0.1,
    },
    "counterfactual": {
        "max_iter": 200,
        "lr": 0.01,
        "lambda_l1": 0.01,
        "target_confidence": 0.9,
    },
    "tta": {
        "enabled": True,
        "similarity_metric": "ssim",   # "ssim" | "pearson"
        "generate_consensus": True,      # produce noise-reduced consensus heatmaps
    },
}

# ──────────────────────────────────────────────
# Report Decoder (Med-LLM)
# ──────────────────────────────────────────────
LLM_CONFIG = {
    "model_name": "google/medgemma-4b-it",   # or "meta-llama/Meta-Llama-3-8B-Instruct"
    "quantization": "4bit",                   # "4bit" | "8bit" | "none"
    "projection": {
        "type": "linear",                     # "linear" | "qformer"
        "vision_dim": 2048,                   # convnext(1024) + swin(1024)
        "llm_dim": 3072,                      # will be overridden at runtime
        "num_query_tokens": 32,               # for Q-Former only
    },
    "generation": {
        "max_new_tokens": 512,
        "temperature": 0.3,
        "top_p": 0.9,
        "do_sample": True,
    },
}

# ──────────────────────────────────────────────
# Evaluation / Validation Engine
# ──────────────────────────────────────────────
EVAL_CONFIG = {
    "faithfulness": {
        "insertion_steps": 50,
        "deletion_steps": 50,
    },
    "robustness": {
        "epsilon": 0.05,
        "n_perturbations": 10,
    },
    "localization": {
        "use_pseudo_masks": True,
    },
    "textual": {
        "metrics": ["rouge_l", "radgraph_f1"],
    },
}

# ──────────────────────────────────────────────
# Logging (Weights & Biases)
# ──────────────────────────────────────────────
WANDB_CONFIG = {
    "project": "breakhis-xai-framework",
    "entity": None,
    "log_images": True,
    "log_interval": 50,      # steps
}
