# Swin Transformer Explainability & Medical Report Generation for BreakHis

Complete implementation of Swin Transformer with three SOTA explainability methods and automated medical report generation for breast cancer histopathology classification.

## 🎯 Features

### Explainability Methods

1. **Attention Rollout** - Hierarchical attention aggregation across all Swin stages
2. **Attention-based GradCAM** - Class-discriminative visualization with gradient weighting
3. **Multi-Head Visualization** - Feature diversity analysis showing what each attention head learns

### Medical Report Generation

- Template-based pathology reports with clinical terminology
- Structured findings: microscopic observations, key features, diagnosis, differentials
- LLM integration support (LLaVA-Med, BioMedGPT)
- Attention-report alignment

### Quantitative Evaluation

- **Insertion AUC**: Performance when adding important pixels
- **Deletion AUC**: Performance when removing important pixels
- **Stability Score**: Consistency under input perturbations

## 📁 Project Structure

```
swin_explainability/
├── __init__.py
├── attention_rollout.py          # Hierarchical attention aggregation
├── attention_gradcam.py           # Class-discriminative CAM
├── multihead_visualization.py    # Head diversity analysis
└── unified_visualizer.py         # Combined visualization

swin_report_generation/
├── __init__.py
└── report_generator.py           # Medical report generation

scripts/
├── generate_swin_explanations.py # Batch explainability
├── generate_swin_reports.py      # Batch report generation
└── evaluate_swin_explainability.py # Quantitative metrics

config_swin.py                     # Configuration
requirements_swin.txt              # Dependencies
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements_swin.txt

# Optional: Install medical LLMs
pip install llava-med  # If using LLaVA-Med
```

### Usage Examples

#### 1. Generate Explainability Visualizations

```python
from swin_explainability import UnifiedSwinVisualizer
from model.swin_transformer import swin_base_patch4_window7_224
import torch

# Load model
model = swin_base_patch4_window7_224(num_classes=8)
model.load_state_dict(torch.load('path/to/checkpoint.pth'))
model.eval()

# Initialize visualizer
visualizer = UnifiedSwinVisualizer(model)

# Generate comprehensive visualization
image = torch.randn(1, 3, 224, 224)  # Your preprocessed image
original_image = ...  # Original image as numpy array

fig = visualizer.generate_comprehensive_visualization(
    image, 
    original_image,
    save_path='results/explanation.png',
    dpi=300
)
```

#### 2. Generate Medical Reports

```python
from swin_report_generation import SwinMedicalReportGenerator

# Initialize generator
generator = SwinMedicalReportGenerator(model, llm_name='template')

# Generate report
result = generator(image)

# Print formatted report
print(generator.format_report_text(result))

# Access structured data
report = result['report']
print(f"Diagnosis: {report['diagnosis']['primary']}")
print(f"Confidence: {report['diagnosis']['confidence']}")
```

#### 3. Individual Explainability Methods

```python
from swin_explainability import (
    SwinAttentionRollout,
    SwinAttentionGradCAM,
    SwinMultiHeadVisualization
)

# Attention Rollout
rollout = SwinAttentionRollout(model)
heatmap = rollout.generate_rollout(image)

# Attention GradCAM
gradcam = SwinAttentionGradCAM(model)
cam = gradcam.generate_cam(image, target_class=1)

# Multi-Head Visualization
multihead = SwinMultiHeadVisualization(model)
head_maps = multihead.visualize_heads(image, num_heads=6)
diversity = multihead.analyze_head_diversity(image)
```

### Command Line Usage

#### Generate Explanations

```bash
python scripts/generate_swin_explanations.py \
    --model_path checkpoints/swin_breakhis.pth \
    --data_path datasets/BreaKHis\ 400X/test \
    --output_dir results/explainability \
    --num_samples 20
```

#### Generate Reports

```bash
python scripts/generate_swin_reports.py \
    --model_path checkpoints/swin_breakhis.pth \
    --data_path datasets/BreaKHis\ 400X/test \
    --output_dir results/reports \
    --llm_name template \
    --num_samples 50
```

#### Evaluate Explainability

```bash
python scripts/evaluate_swin_explainability.py \
    --model_path checkpoints/swin_breakhis.pth \
    --data_path datasets/BreaKHis\ 400X/test \
    --output_dir results/evaluation \
    --num_samples 50
```

## 📊 Output Examples

### Explainability Visualization

The comprehensive visualization includes:
- Original image
- Attention Rollout overlay
- Attention GradCAM overlay
- Combined attention map
- 6 individual attention heads
- Heatmap-only views

### Medical Report Example

```
HISTOPATHOLOGICAL ANALYSIS REPORT
============================================================

Report ID: BR_20240119_143022
Date: 2024-01-19T14:30:22
Magnification: 400X
Staining: H&E

MICROSCOPIC FINDINGS:
Malignant epithelial proliferation forming irregular ducts. 
Nuclear pleomorphism and increased mitotic activity. Loss of 
myoepithelial layer.

KEY OBSERVATIONS:
  • Nuclear pleomorphism
  • Increased mitotic activity
  • Irregular ductal structures

DIAGNOSIS:
  Primary: Ductal Carcinoma
  Category: Malignant
  Confidence: 94.3%

ATTENTION REGIONS:
Model focused on cellular architecture and nuclear features

DIFFERENTIAL CONSIDERATIONS:
  • Lobular carcinoma
  • Invasive ductal carcinoma

CLINICAL CORRELATION:
Findings consistent with diagnosis
============================================================
```

### Evaluation Metrics

```json
{
  "rollout": {
    "insertion_auc": 0.72,
    "deletion_auc": 0.31,
    "stability": 0.85
  },
  "gradcam": {
    "insertion_auc": 0.68,
    "deletion_auc": 0.28,
    "stability": 0.81
  }
}
```

## 🔧 Configuration

Edit `config_swin.py` to customize:

```python
# Model architecture
MODEL_CONFIG = {
    'architecture': 'swin_base_patch4_window7_224',
    'num_classes': 8,
    'pretrained': True
}

# Explainability settings
EXPLAINABILITY_CONFIG = {
    'methods': ['attention_rollout', 'attention_gradcam', 'multihead'],
    'attention_rollout': {
        'head_fusion': 'mean',
        'discard_ratio': 0.1
    }
}

# Report generation
REPORT_CONFIG = {
    'llm_name': 'template',  # or 'llava-med', 'biomedgpt'
    'include_attention_description': True
}
```

## 📈 Technical Details

### Attention Rollout

- Extracts attention weights from all Swin Transformer blocks
- Handles window-based attention mechanism
- Recursively multiplies attention matrices with residual connections
- Supports stage-specific visualization

### Attention GradCAM

- Combines attention weights with gradient information
- Class-discriminative explanations
- Multi-scale CAM generation across Swin stages
- Gradient weighting for importance

### Multi-Head Visualization

- Analyzes individual attention heads
- Computes head diversity metrics
- Identifies head specialization patterns
- Correlation and entropy analysis

### Report Generation

- Feature extraction from Swin stages
- Learnable projection layer with cross-attention
- Template-based generation with clinical terminology
- LLM integration support for advanced generation

## 🎓 For Thesis

### Key Demonstration Points

1. **Three SOTA Explainability Methods**: Show comprehensive visualization
2. **Quantitative Evaluation**: Present insertion/deletion AUC, stability scores
3. **Medical Report Generation**: Demonstrate automated pathology reports
4. **Clinical Relevance**: Highlight attention on diagnostically relevant regions

### Expected Results

- **Insertion AUC**: 0.65-0.75 (higher is better)
- **Deletion AUC**: 0.25-0.35 (lower is better)
- **Stability**: 0.75-0.90 (higher is better)
- **Report Accuracy**: Consistent with model predictions

## 📚 References

**Explainability:**
1. Abnar & Zuidema, "Quantifying Attention Flow in Transformers" (ACL 2020)
2. Chefer et al., "Transformer Interpretability Beyond Attention Visualization" (CVPR 2021)
3. Chefer et al., "Generic Attention-model Explainability" (ICCV 2021)
4. Caron et al., "Emerging Properties in Self-Supervised Vision Transformers" (ICCV 2021)

**Medical AI:**
1. Li et al., "LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine" (NeurIPS 2023)
2. Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (ICCV 2021)

## 🔍 Troubleshooting

### CUDA Memory Error
```python
# Reduce batch size or use CPU
device = 'cpu'
```

### Attention Extraction Issues
```python
# Verify Swin architecture matches expected structure
# Check hook registration on correct modules
```

### LLM Loading Error
```python
# Fall back to template-based generation
generator = SwinMedicalReportGenerator(model, llm_name='template')
```

## ✅ Success Criteria

- ✅ All three explainability methods produce clear visualizations
- ✅ Attention maps highlight diagnostically relevant regions
- ✅ Generated reports match model predictions
- ✅ Quantitative metrics demonstrate faithfulness
- ✅ Code is well-documented and reproducible

---

**Built for advancing explainable AI in medical image analysis with Swin Transformer architecture**
