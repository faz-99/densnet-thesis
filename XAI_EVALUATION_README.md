# 🔬 Comprehensive XAI Evaluation System

## Overview

This system provides **quantitative, human-independent evaluation** of explainability methods (Grad-CAM, SHAP, LIME) for histopathology image classification using DenseNet201 transfer learning.

## 🎯 Key Features

### Quantitative Metrics
- **Insertion AUC** (Faithfulness ↑): Measures how prediction confidence increases when important pixels are progressively added
- **Deletion AUC** (Faithfulness ↓): Measures how prediction confidence decreases when important pixels are removed  
- **Intersection over Union (IoU)** (Localization ↑): Measures spatial overlap between explanations and regions of interest
- **Stability/Robustness** (↑): Measures consistency of explanations under image perturbations

### XAI Methods Evaluated
- **Grad-CAM**: Gradient-based class activation mapping
- **Grad-CAM++**: Enhanced localization with better multi-object handling
- **SHAP**: Shapley value-based feature attribution (when available)
- **LIME**: Local interpretable model-agnostic explanations

## 📁 File Structure

```
xai/
├── __init__.py
├── evaluate_xai.py              # Main evaluation pipeline
├── metrics/
│   ├── __init__.py
│   ├── insertion_auc.py         # Insertion AUC implementation
│   ├── deletion_auc.py          # Deletion AUC implementation
│   ├── iou.py                   # IoU metric implementation
│   └── stability.py             # Stability metric implementation
├── run_xai_evaluation.py        # Standalone evaluation script
├── test_xai_evaluation.py       # Test script
└── XAI_EVALUATION_README.md     # This file
```

## 🚀 Usage

### 1. Interactive Streamlit App (Recommended)

The XAI evaluation is integrated into the main Streamlit app:

```bash
./venv/bin/streamlit run app.py
```

1. Upload a histopathology image
2. Select "Transfer Learning (DenseNet201)" model
3. Enable Grad-CAM and/or LIME
4. Click "🔬 Run Quantitative Analysis" button
5. View quantitative metrics and summary table

### 2. Standalone Evaluation Script

For batch processing and comprehensive analysis:

```bash
./venv/bin/python run_xai_evaluation.py \
    --data_dir data/test \
    --num_samples 20 \
    --num_classes 2 \
    --results_dir results \
    --target_layer features.norm5
```

**Parameters:**
- `--data_dir`: Directory containing test images
- `--num_samples`: Number of images to evaluate
- `--num_classes`: 2 for binary, 8 for multi-class
- `--results_dir`: Output directory for results
- `--target_layer`: Target layer for Grad-CAM

### 3. Python API

```python
from xai.evaluate_xai import XAIEvaluator
import torch

# Initialize evaluator
model, device = load_your_model()
class_names = ['Benign', 'Malignant']
evaluator = XAIEvaluator(model, device, class_names)

# Evaluate images
images = [your_image_tensors]
labels = [your_true_labels]
image_ids = [your_image_ids]

evaluator.evaluate_batch(images, labels, image_ids)
evaluator.compute_aggregated_results()
evaluator.print_summary_table()
evaluator.save_results()
```

## 📊 Output Format

### Summary Table
```
XAI Method | Insertion AUC | Deletion AUC | IoU   | Stability
-----------|---------------|--------------|-------|----------
GRADCAM    |     0.742     |     0.234    | 0.456 |   0.823
LIME       |     0.689     |     0.267    | 0.398 |   0.756
```

### Detailed Results
- **JSON**: Complete results with per-image metrics
- **CSV**: Tabular format for statistical analysis
- **Visualizations**: Bar charts and plots

## 🧮 Metric Interpretation

### Insertion AUC (Higher = Better)
- **Range**: 0.0 - 1.0
- **Good**: > 0.7
- **Interpretation**: How much prediction confidence increases when important pixels are progressively added
- **Baseline**: Random pixels would give ~0.5

### Deletion AUC (Lower = Better)  
- **Range**: 0.0 - 1.0
- **Good**: < 0.3
- **Interpretation**: How much prediction confidence decreases when important pixels are removed
- **Baseline**: Random pixels would give ~0.5

### IoU (Higher = Better)
- **Range**: 0.0 - 1.0
- **Good**: > 0.5
- **Interpretation**: Spatial overlap between explanation and important regions
- **Note**: Uses pseudo-ROI when ground truth not available

### Stability (Higher = Better)
- **Range**: 0.0 - 1.0
- **Good**: > 0.8
- **Interpretation**: Consistency of explanations under perturbations (noise, rotation, flip)
- **Metrics**: SSIM, Pearson correlation, Cosine similarity

## 🔧 Technical Details

### Model Compatibility
- Works with any PyTorch model
- Tested with DenseNet201 transfer learning
- Supports binary and multi-class classification
- GPU acceleration supported

### Perturbation Types
- **Gaussian noise**: Various noise levels
- **Rotation**: ±10 degrees
- **Flipping**: Horizontal and vertical
- **Configurable**: Number and types of perturbations

### Baseline Methods
- **Insertion**: Blurred image baseline
- **Deletion**: Gaussian blur replacement
- **Alternative**: Mean pixel value, zeros

## 📈 Example Results

### Typical Performance Ranges

| Method    | Insertion AUC | Deletion AUC | IoU   | Stability |
|-----------|---------------|--------------|-------|-----------|
| Grad-CAM  | 0.65-0.85     | 0.15-0.35    | 0.3-0.6 | 0.7-0.9   |
| LIME      | 0.60-0.80     | 0.20-0.40    | 0.2-0.5 | 0.6-0.8   |
| SHAP      | 0.70-0.90     | 0.10-0.30    | 0.4-0.7 | 0.8-0.95  |

### Clinical Interpretation

**High Faithfulness (Good Insertion/Deletion AUC):**
- Model focuses on clinically relevant regions
- Explanations reflect actual decision process
- Trustworthy for clinical use

**High Localization (Good IoU):**
- Precise identification of pathological regions
- Focused attention on specific tissue areas
- Useful for pathologist guidance

**High Stability (Good Stability Score):**
- Robust explanations across image variations
- Consistent results under different conditions
- Reliable for clinical deployment

## 🎓 Academic Usage

### For Master's Thesis
- Provides quantitative validation of XAI methods
- Supports claims about explanation quality
- Enables comparison between different approaches
- Suitable for methodology sections

### For Journal Submission
- Human-independent evaluation metrics
- Reproducible experimental setup
- Statistical significance testing
- Comparison with baseline methods

### For Clinical AI Discussion
- Transparency and trustworthiness metrics
- Robustness under real-world conditions
- Localization accuracy for pathologist review
- Faithfulness to model decision process

## 🔍 Troubleshooting

### Common Issues

1. **SHAP Not Available**
   - Expected behavior if cmake not installed
   - App gracefully disables SHAP evaluation
   - Grad-CAM and LIME still work

2. **Layer Not Found**
   - Check target layer name with debug mode
   - Try alternative layer names
   - Use model.named_modules() to list layers

3. **Memory Issues**
   - Reduce batch size
   - Use CPU for evaluation
   - Process images sequentially

4. **Low Metric Scores**
   - Check image preprocessing
   - Verify model is trained properly
   - Consider different target layers

### Performance Optimization
- Use GPU when available
- Batch process multiple images
- Cache model outputs
- Reduce number of perturbations for faster evaluation

## 📚 References

1. **Insertion/Deletion AUC**: Petsiuk et al. "RISE: Randomized Input Sampling for Explanation of Black-box Models"
2. **IoU for XAI**: Mohseni et al. "A Multidisciplinary Survey and Framework for Design and Evaluation of Explainable AI Systems"
3. **Stability Metrics**: Alvarez-Melis & Jaakkola "Towards Robust Interpretability with Self-Explaining Neural Networks"
4. **Grad-CAM**: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"

## 🤝 Contributing

To extend the evaluation system:

1. Add new metrics in `xai/metrics/`
2. Update `XAIEvaluator` class
3. Add tests in `test_xai_evaluation.py`
4. Update documentation

## 📄 License

This XAI evaluation system is part of the DenseNet histopathology classification project and follows the same license terms.