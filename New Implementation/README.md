# Multimodal Histopathology XAI Framework (BreaKHis)

**Vision-XAI-Report Pipeline** for breast cancer sub-type classification using the BreaKHis dataset.

## Architecture

```
Input Image (224×224)
    ├── Branch A: ConvNeXt-Base  →  local texture features (1024-d)
    ├── Branch B: Swin-Base      →  global hierarchical features (1024-d)
    └── Fusion (MLP / Weighted Avg)  →  8-class or binary prediction
            │
            ├── XAI Suite (6 methods)
            │     ├── Grad-CAM (ConvNeXt)
            │     ├── Integrated Gradients (axiom-complete)
            │     ├── SHAP (DeepExplainer)
            │     ├── LIME (superpixel)
            │     ├── Attention Rollout (Swin)
            │     └── Counterfactual Explanations
            │
            ├── Validation Engine
            │     ├── Faithfulness: Insertion/Deletion AUC
            │     ├── Robustness: Stability Score
            │     ├── Localization: Localization AUC
            │     └── Textual: ROUGE-L, RadGraph F1
            │
            └── Report Decoder (Med-LLM)
                  ├── Multimodal Projection (Linear / Q-Former)
                  ├── MedGemma / Llama-3-8B-Instruct (4-bit quantized)
                  └── Chain-of-Thought clinical report
```

## Project Structure

```
New Implementation/
├── config/
│   └── settings.py              # All configuration
├── data/
│   ├── breakhis_dataset.py      # Magnification-aware dataset loader
│   └── preprocessing.py         # Transforms & augmentation
├── models/
│   ├── convnext_branch.py       # Branch A: ConvNeXt-Base
│   ├── swin_branch.py           # Branch B: Swin-Transformer-Base
│   ├── fusion.py                # MLP & Weighted Average fusion heads
│   └── ensemble.py              # Full HybridEnsemble
├── xai/
│   ├── interpretability_manager.py  # Centralized XAI orchestrator
│   ├── grad_cam.py              # Grad-CAM
│   ├── integrated_gradients.py  # Integrated Gradients (Captum)
│   ├── shap_explainer.py        # SHAP DeepExplainer
│   ├── lime_explainer.py        # LIME with superpixel segmentation
│   ├── attention_rollout.py     # Attention Rollout for Swin
│   └── counterfactual.py        # Counterfactual explanations
├── report/
│   ├── projection.py            # Linear / Q-Former projection layers
│   └── report_generator.py      # Med-LLM report generation with CoT
├── evaluation/
│   └── validation_engine.py     # Faithfulness, Robustness, Localization, Textual metrics
├── training/
│   └── trainer.py               # Training loop (AMP, WandB, early stopping)
├── run_train.py                 # Training entry point
├── run_inference.py             # Inference + XAI + report pipeline
├── run_evaluate.py              # Comprehensive evaluation
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train

```bash
# 8-class classification on 400X magnification
python run_train.py --task multiclass --magnification 400X

# Binary classification
python run_train.py --task binary --magnification 400X

# Train on all magnifications
python run_train.py --task multiclass --magnification all

# Without WandB
python run_train.py --no-wandb --epochs 50
```

### 3. Inference (XAI + Report)

```bash
# Single image
python run_inference.py --checkpoint outputs/checkpoints/best_model.pth --image path/to/image.png

# Batch inference
python run_inference.py --checkpoint outputs/checkpoints/best_model.pth --batch-dir path/to/images/

# With Med-LLM report generation
python run_inference.py --checkpoint outputs/checkpoints/best_model.pth --image path/to/image.png --load-llm
```

### 4. Evaluate

```bash
python run_evaluate.py --checkpoint outputs/checkpoints/best_model.pth --n-samples 50
```

## Components

### Hybrid Vision Backbone
- **ConvNeXt-Base**: Captures local inductive biases and texture patterns (stroma, nuclei shape)
- **Swin-Transformer-Base**: Captures global dependencies via shifted-window self-attention
- **Fusion**: MLP concatenation (default) or learnable weighted average

### XAI Interpretability Suite
| Method | Type | Target |
|--------|------|--------|
| Grad-CAM | Gradient-based | ConvNeXt last stage |
| Integrated Gradients | Gradient-based (axiom-complete) | Full ensemble |
| SHAP (DeepExplainer) | Model-agnostic | Full ensemble |
| LIME | Model-agnostic (superpixel) | Full ensemble |
| Attention Rollout | Structural | Swin Transformer |
| Counterfactual | Clinical logic | Full ensemble |

### Validation Metrics
| Metric | Category | Interpretation |
|--------|----------|----------------|
| Insertion AUC | Faithfulness | Higher = better (important pixels boost confidence) |
| Deletion AUC | Faithfulness | Lower = better (removing important pixels drops confidence) |
| Stability Score | Robustness | Lower = more robust under noise |
| Localization AUC | Localization | Higher = better overlap with ground truth |
| ROUGE-L | Textual | Higher = better text overlap |
| RadGraph F1 | Textual | Higher = better clinical entity match |

### Report Decoder
- **Projection**: Linear or Q-Former bridge from vision features to LLM embedding space
- **LLM**: MedGemma-4B or Llama-3-8B-Instruct (4-bit quantized)
- **Prompting**: Chain-of-Thought (CoT) requiring XAI evidence citation
- **Fallback**: Template-based reporting when LLM is unavailable

## Technical Stack
- **Framework**: PyTorch, timm, captum, transformers
- **Logging**: Weights & Biases (WandB)
- **Dataset**: BreaKHis (40X, 100X, 200X, 400X)
