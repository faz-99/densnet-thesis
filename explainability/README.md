# DenseNet Explainability Module

This module provides comprehensive explainability features for the DenseNet-based medical image classification model, implementing multiple state-of-the-art interpretability techniques.

## 🧠 Implemented Techniques

### 1. Grad-CAM & Grad-CAM++
- **Purpose**: Visualize important regions in images that influence model decisions
- **Output**: Heatmaps highlighting discriminative regions
- **Use Case**: Understanding spatial attention patterns

### 2. SHAP (SHapley Additive exPlanations)
- **Purpose**: Explain pixel-level or patch-level contributions to predictions
- **Output**: Attribution maps showing positive/negative contributions
- **Use Case**: Quantitative feature importance analysis

### 3. LIME (Local Interpretable Model-agnostic Explanations)
- **Purpose**: Local interpretability through superpixel perturbation
- **Output**: Superpixel-based explanations with contribution weights
- **Use Case**: Quick local interpretability checks

## 📁 Module Structure

```
explainability/
├── __init__.py                 # Module initialization
├── grad_cam.py                # Grad-CAM implementation
├── shap_explainer.py          # SHAP implementation
├── lime_explainer.py          # LIME implementation
├── explainer.py               # Main comprehensive explainer
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_explainability.txt
```

### 2. Run Comprehensive Analysis
```bash
python run_explainability.py --model_path weight/save/40/iaff40_5.pth
```

### 3. Custom Analysis
```bash
python run_explainability.py \
  --model_path your_model.pth \
  --num_samples 20 \
  --techniques gradcam shap lime \
  --background_size 100
```

## 📊 Output Structure

After running the analysis, you'll get:

```
explainability/
├── performance_analysis.json   # Model performance statistics
├── performance_plots.png      # Performance visualizations
├── explainability_report.json # Comprehensive analysis report
├── README.md                  # Analysis summary
├── gradcam_results/           # Grad-CAM visualizations
│   ├── sample_000_gradcam.png
│   ├── sample_001_gradcam.png
│   └── ...
├── shap_results/              # SHAP explanations
│   ├── sample_000_shap.png
│   ├── sample_001_shap.png
│   ├── shap_summary.png
│   └── ...
└── lime_results/              # LIME explanations
    ├── sample_000_lime.png
    ├── sample_001_lime.png
    ├── lime_comparison.png
    └── ...
```

## 🔧 Usage Examples

### Individual Technique Usage

#### Grad-CAM
```python
from explainability.grad_cam import GradCAM, GradCAMPlusPlus

# Initialize
gradcam = GradCAM(model, target_layer_name='densenet.features.norm5')
gradcam_plus = GradCAMPlusPlus(model, target_layer_name='densenet.features.norm5')

# Generate heatmaps
heatmap = gradcam.generate_cam(input_tensor)
heatmap_plus = gradcam_plus.generate_cam(input_tensor)
```

#### SHAP
```python
from explainability.shap_explainer import SHAPExplainer

# Initialize with background data
shap_explainer = SHAPExplainer(model, background_data, device)

# Generate explanation
shap_values = shap_explainer.explain_image(image_tensor)
```

#### LIME
```python
from explainability.lime_explainer import LIMEExplainer

# Initialize
lime_explainer = LIMEExplainer(model, device, num_samples=1000)

# Generate explanation
explanation, segments = lime_explainer.explain_image(image_numpy)
```

### Comprehensive Analysis
```python
from explainability.explainer import ComprehensiveExplainer

# Initialize
explainer = ComprehensiveExplainer(model, device, class_names)

# Run full analysis
explainer.generate_comprehensive_explanations(
    test_dataloader=test_loader,
    background_dataloader=background_loader,
    num_samples=20,
    techniques=['gradcam', 'shap', 'lime']
)
```

## 📈 Interpretation Guidelines

### Grad-CAM Heatmaps
- **Red/Hot colors**: High importance regions
- **Blue/Cool colors**: Low importance regions
- **Compare**: Grad-CAM vs Grad-CAM++ for localization accuracy

### SHAP Values
- **Positive values**: Features supporting the prediction
- **Negative values**: Features opposing the prediction
- **Magnitude**: Strength of contribution

### LIME Explanations
- **Green superpixels**: Positive contributions
- **Red superpixels**: Negative contributions
- **Transparency**: Contribution strength

## 🎯 Medical Image Analysis Insights

### For Malignant vs Benign Classification:

1. **Texture Analysis**: Look for irregular patterns in malignant cases
2. **Boundary Analysis**: Check if model focuses on tissue boundaries
3. **Artifact Detection**: Ensure model doesn't rely on imaging artifacts
4. **Consistency Check**: Compare explanations across similar cases

### Quality Assurance:

1. **Sanity Checks**: Verify explanations make medical sense
2. **Bias Detection**: Check for dataset-specific biases
3. **Robustness**: Test explanations on augmented images
4. **Expert Validation**: Compare with radiologist annotations

## ⚙️ Configuration Options

### Grad-CAM Parameters
- `target_layer_name`: Layer to analyze (default: 'densenet.features.norm5')
- `class_idx`: Target class (None for predicted class)

### SHAP Parameters
- `background_size`: Background dataset size (default: 50-100)
- `num_samples`: Perturbation samples (automatic)

### LIME Parameters
- `num_samples`: Perturbation samples (default: 1000)
- `num_features`: Superpixels to show (default: 10)
- `segmentation_fn`: Superpixel algorithm (default: quickshift)

## 🔍 Troubleshooting

### Common Issues:

1. **Memory Errors**: Reduce `num_samples` or `background_size`
2. **Slow Performance**: Use fewer samples or GPU acceleration
3. **Layer Not Found**: Check `target_layer_name` for Grad-CAM
4. **Import Errors**: Install all dependencies from requirements file

### Performance Tips:

1. **GPU Usage**: Ensure CUDA is available for faster computation
2. **Batch Processing**: Process multiple samples together when possible
3. **Background Data**: Use representative but small background sets for SHAP
4. **Caching**: Save intermediate results for repeated analysis

## 📚 References

1. **Grad-CAM**: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
2. **Grad-CAM++**: Chattopadhay et al. "Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks"
3. **SHAP**: Lundberg & Lee. "A Unified Approach to Interpreting Model Predictions"
4. **LIME**: Ribeiro et al. "Why Should I Trust You?: Explaining the Predictions of Any Classifier"

## 🤝 Contributing

To extend the explainability module:

1. Add new techniques in separate files
2. Follow the existing interface patterns
3. Include comprehensive documentation
4. Add visualization functions
5. Update the main explainer class

## 📄 License

This explainability module is part of the DenseNet medical image classification project and follows the same license terms.