"""Configuration for Swin Transformer Experiments on BreakHis"""

# Model Configuration
MODEL_CONFIG = {
    'architecture': 'swin_base_patch4_window7_224',
    'num_classes': 8,
    'pretrained': True,
    'img_size': 224,
    'patch_size': 4,
    'window_size': 7,
    'embed_dim': 128,
    'depths': [2, 2, 18, 2],
    'num_heads': [4, 8, 16, 32]
}

# BreakHis Dataset Configuration
DATASET_CONFIG = {
    'root': 'datasets/BreaKHis 400X',
    'magnification': '400X',
    'num_classes': 8,
    'class_names': [
        'Adenosis',
        'Fibroadenoma', 
        'Phyllodes_tumor',
        'Tubular_adenoma',
        'Ductal_carcinoma',
        'Lobular_carcinoma',
        'Mucinous_carcinoma',
        'Papillary_carcinoma'
    ],
    'train_split': 0.8,
    'val_split': 0.2
}

# Explainability Configuration
EXPLAINABILITY_CONFIG = {
    'methods': ['attention_rollout', 'attention_gradcam', 'multihead'],
    'attention_rollout': {
        'head_fusion': 'mean',  # 'mean', 'max', 'min'
        'discard_ratio': 0.1
    },
    'attention_gradcam': {
        'target_layer': None  # Auto-detect
    },
    'multihead': {
        'num_heads_visualize': 6,
        'target_stage': -1  # -1 for last stage
    },
    'visualization': {
        'dpi': 300,
        'colormap': 'jet',
        'alpha': 0.5
    }
}

# Report Generation Configuration
REPORT_CONFIG = {
    'llm_name': 'template',  # 'template', 'llava-med', 'biomedgpt'
    'swin_dim': 1024,
    'llm_dim': 4096,
    'num_query_tokens': 32,
    'template_based': True,
    'include_attention_description': True
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    'metrics': ['insertion_auc', 'deletion_auc', 'stability'],
    'insertion_steps': 50,
    'deletion_steps': 50,
    'stability_perturbations': 10,
    'noise_level': 0.1,
    'num_samples': 50
}

# Training Configuration (if needed)
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 0.05,
    'optimizer': 'adamw',
    'scheduler': 'cosine',
    'warmup_epochs': 5
}

# Paths
PATHS = {
    'data_root': 'datasets/BreaKHis 400X',
    'model_save': 'weight/swin',
    'results': 'results/swin',
    'explainability': 'results/swin/explainability',
    'reports': 'results/swin/reports',
    'evaluation': 'results/swin/evaluation'
}
