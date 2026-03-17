# Breast Cancer Histopathology XAI Pipeline

End-to-end, research-grade pipeline for classification and explainability on the BreaKHis dataset.

---

## Architecture

```
pipeline/
  train_backbones.py        ← Train ConvNeXt / Swin Transformer
  end_to_end_pipeline.py    ← Full evaluation pipeline (all stages)

model/
  convnext_model.py         ← ConvNeXt-Base classifier
  swin_classifier.py        ← Swin-Tiny/Base classifier
  denlsnet_corrected.py     ← DenLsNet (DenseNet-121 + BiLSTM)

data/
  breakhis_dataset.py       ← BreaKHis loader (binary + 8-class)

explainability/
  xai_pipeline.py           ← Unified XAI interface
  grad_cam.py               ← Grad-CAM / Grad-CAM++
  integrated_gradients.py   ← Integrated Gradients
  shap_explainer.py         ← SHAP (DeepExplainer)
  lime_explainer.py         ← LIME
  counterfactual.py         ← Counterfactual explanations (NEW)

swin_explainability/
  attention_rollout.py      ← Attention Rollout (Swin-specific)

xai/metrics/
  fidelity.py               ← Insertion/Deletion AUC per method (NEW)
  cross_method_consistency.py ← Pearson/Spearman/SSIM/Jaccard (NEW)
  sparsity.py               ← Gini / energy concentration (NEW)
  stability.py              ← Robustness under perturbations
  insertion_auc.py          ← Legacy insertion AUC
  deletion_auc.py           ← Legacy deletion AUC

evaluation/
  classification_metrics.py ← Acc/Prec/Rec/Spec/F1/ROC-AUC/PR-AUC/CM
  report_metrics.py         ← BLEU/ROUGE/BERTScore/Hallucination/Grounding

report_generation/
  llm_report_generator.py   ← MedGemma / Llama-3-Medical / rule-based
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements_pipeline.txt
```

### 2. Prepare BreaKHis dataset
```
datasets/
  BreaKHis 400X/
    train/
      benign/   ...
      malignant/ ...
    test/
      benign/   ...
      malignant/ ...
```

### 3. Train backbone models
```bash
# ConvNeXt – binary
python pipeline/train_backbones.py --model convnext --task binary --epochs 50

# Swin Transformer – binary
python pipeline/train_backbones.py --model swin --task binary --epochs 50

# Multiclass (8 subtypes)
python pipeline/train_backbones.py --model convnext --task multiclass --epochs 80
```

### 4. Run the full pipeline
```bash
python pipeline/end_to_end_pipeline.py \
  --data_root "datasets/BreaKHis 400X" \
  --task binary \
  --models convnext swin \
  --convnext_checkpoint checkpoints/convnext/binary/best.pth \
  --swin_checkpoint     checkpoints/swin/binary/best.pth \
  --xai_samples 30 \
  --fidelity_samples 15 \
  --fidelity_steps 50 \
  --llm_model rule_based \
  --results_dir pipeline_results
```

For LLM-based reports (requires GPU + HuggingFace access):
```bash
  --llm_model medgemma   # or llama3
```

---

## Evaluation Metrics

### Classification
| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| Precision / Recall | Per-class and macro-averaged |
| Specificity | TN / (TN + FP) |
| F1-score | Harmonic mean of P/R |
| ROC-AUC | Area under ROC curve |
| PR-AUC | Area under Precision-Recall curve |
| Confusion Matrix | Full per-class breakdown |

### XAI Fidelity (per method)
| Metric | Interpretation |
|--------|----------------|
| Insertion AUC ↑ | Higher = explanation captures truly important pixels |
| Deletion AUC ↓ | Lower = removing important pixels drops confidence fast |
| Fidelity Score | Insertion AUC + (1 − Deletion AUC) |

Methods evaluated: Grad-CAM++, SHAP, LIME, Integrated Gradients, Attention Rollout (Swin), Counterfactual

### XAI Quality
| Metric | Description |
|--------|-------------|
| Stability (SSIM) | Consistency under Gaussian noise / rotation |
| Cross-method Pearson/Spearman | Agreement between methods |
| Cross-method SSIM | Structural similarity of heatmaps |
| Top-k Jaccard | Overlap of most important pixels |
| Gini coefficient | Sparsity / focus of explanation |
| Energy concentration | % energy in top-20% pixels |

### Report Quality
| Metric | Description |
|--------|-------------|
| BLEU-1/2/3/4 | N-gram overlap with reference |
| ROUGE-1/2/L | Recall-oriented overlap |
| BERTScore F1 | Semantic similarity |
| Hallucination rate | Fraction of reports with unsupported claims |
| Grounding score | Text–XAI region alignment |

---

## Output Structure
```
pipeline_results/<timestamp>/
  classification/
    convnext/  confusion_matrix.png  roc_curves.png  pr_curves.png
    swin/      ...
  fidelity/
    convnext/  insertion_curves.png  deletion_curves.png  auc_summary.png  fidelity_summary.csv
    swin/      ...
  xai_quality/
    convnext/  cross_method_consistency.csv
    swin/      ...
  reports/
    convnext/  sample_000_report.txt  ...
    swin/      ...
  comparative_summary.csv
  full_results.json
```
