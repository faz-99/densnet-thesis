"""
Multi-class configuration for DenLsNet extension
Supports 8-class BreakHis classification with stain normalization options
"""
import os

# GPU parameters
is_cuda = False
device = 'cpu'
is_parallel = False
gpu_id = "0"
gpu_ids = [0]

# Dataset parameters for multi-class BreakHis
dataset_mean = (0.5613, 0.5778, 0.6032)
dataset_std = (0.2114, 0.1957, 0.1590)

# Model configuration
net_name = "denlsnet_mc"  # DenLsNet Multi-Class
batch_size = 32
num_workers = 0
max_epoch = 100
warmup_epochs = 5
warmup_steps = -1

lr = 0.003
min_lr = 1e-6
weight_decay = 0.05

milestones = [20, 40, 60, 80]
gamma = 0.5

img_s = 224
drop_path = 0.8

# Multi-class configuration (8 subclasses in BreakHis)
class_num = 8
class_names = [
    'Adenosis', 'Fibroadenoma', 'Phyllodes_tumor', 'Tubular_adenoma',  # Benign
    'Ductal_carcinoma', 'Lobular_carcinoma', 'Mucinous_carcinoma', 'Papillary_carcinoma'  # Malignant
]

# Dataset paths for multi-class
train = "datasets/BreaKHis 400X/multiclass/train"
valid = "datasets/BreaKHis 400X/multiclass/test"

# Stain normalization configuration
stain_normalization = {
    'method': 'none',  # Options: 'none', 'macenko', 'reinhard'
    'target_image_path': None,  # Path to target image for normalization
    'save_normalized': True,  # Save normalized images
    'normalized_dir': 'datasets/normalized'
}

# Experiment variants
experiment_variants = {
    'baseline': {
        'stain_method': 'none',
        'model_suffix': 'baseline'
    },
    'macenko': {
        'stain_method': 'macenko',
        'model_suffix': 'macenko'
    },
    'reinhard': {
        'stain_method': 'reinhard',
        'model_suffix': 'reinhard'
    }
}

# Loss function configuration
loss_config = {
    'type': 'categorical_crossentropy',  # Changed from binary
    'class_weights': 'balanced',  # Auto-calculate class weights
    'label_smoothing': 0.1,
    'focal_loss': {
        'use': False,
        'alpha': 1.0,
        'gamma': 2.0
    }
}

# Evaluation metrics for multi-class
evaluation_metrics = [
    'accuracy',
    'precision_macro',
    'recall_macro', 
    'f1_macro',
    'precision_per_class',
    'recall_per_class',
    'f1_per_class',
    'confusion_matrix',
    'classification_report'
]

# Interpretability configuration
interpretability_config = {
    'methods': ['gradcam', 'gradcam_plus', 'shap', 'lime'],
    'target_layers': ['densenet.features.norm5'],
    'num_samples': 50,
    'save_explanations': True,
    'quantitative_eval': True,
    'stability_analysis': True
}

# Academic naming convention
model_names = {
    'binary_baseline': 'DenLsNet',
    'multiclass_baseline': 'DenLsNet-MC',
    'multiclass_macenko': 'DenLsNet-MC-Macenko',
    'multiclass_reinhard': 'DenLsNet-MC-Reinhard',
    'multiclass_none': 'DenLsNet-MC-None',
    'interpretability': 'DenLsNet-XAI'
}

# Output directories
output_dirs = {
    'models': 'weight/multiclass',
    'logs': 'csv/multiclass',
    'results': 'results/multiclass',
    'plots': 'plots/multiclass',
    'normalized_data': 'datasets/normalized',
    'explainability': 'explainability/multiclass'
}

# Ensure directories exist
for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)