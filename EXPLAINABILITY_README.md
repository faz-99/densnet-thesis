# Comprehensive Explainability System for Histopathology Analysis

This system provides advanced explainability analysis for histopathology image classification, combining multiple XAI methods with morphological feature extraction and automated clinical report generation.

## Features

### 🔬 Multi-Method Explainability
- **Grad-CAM**: Standard gradient-based class activation mapping
- **Grad-CAM++**: Improved localization with better handling of multiple objects
- **SHAP**: Shapley value-based feature attribution
- **LIME**: Local interpretable model-agnostic explanations

### 📊 Morphological Analysis
Extracts quantitative descriptors from activation maps:
- **Tissue Coverage**: Percentage of tissue area highlighted by the model
- **H&E Stain Analysis**: Hematoxylin vs. Eosin dominance patterns
- **Texture Features**: Entropy, local variance, edge density
- **Color Statistics**: RGB analysis in high-activation regions
- **Morphological Features**: Region count, compactness, eccentricity

### 📝 Automated Report Generation
- **Template-based Descriptions**: Clinical language generation
- **Quantitative Summaries**: Structured feature tables
- **Visual Outputs**: Heatmaps, overlays, and binary masks
- **Multi-format Export**: JSON, HTML, and text reports

## Usage

### 1. Interactive Web Interface

Run the Streamlit app for single-image analysis:

```bash
streamlit run app.py
```

Features:
- Upload histopathology images
- Real-time explainability analysis
- Interactive morphological feature display
- Downloadable HTML reports
- Batch processing capability

### 2. Command Line Batch Processing

Process multiple images programmatically:

```bash
python run_comprehensive_explainability.py \
    --model_path weight/save/40/iaff40_5.pth \
    --data_dir data/BreakHis/test \
    --output_dir explainability_reports \
    --num_images 50 \
    --device cuda
```

### 3. Python API

Use the explainability system in your own code:

```python
from explainability.comprehensive_explainer import ComprehensiveExplainer
import torch

# Initialize explainer
explainer = ComprehensiveExplainer(model, device, class_names)

# Analyze single image
result = explainer.generate_comprehensive_explanation(
    image_tensor, original_image, image_id, save_dir
)

# Access results
print(result['clinical_interpretation'])
print(result['consensus_findings'])
```

## Output Structure

### Directory Layout
```
explainability_reports/
├── visualizations/
│   ├── {image_id}_comprehensive.png
│   ├── {image_id}_gradcam_overlay.png
│   ├── {image_id}_gradcam_raw.npy
│   └── ...
├── {image_id}_comprehensive_report.json
├── {image_id}_comprehensive_report.txt
├── {image_id}_gradcam_detailed.json
└── batch_summary.json
```

### Report Contents

#### Comprehensive Report
- **Metadata**: Image ID, timestamp, prediction details
- **Clinical Interpretation**: Automated clinical summary
- **Consensus Findings**: Aggregated results across methods
- **Method Summaries**: Individual explainer results
- **Visual Outputs**: Paths to generated visualizations

#### Morphological Features
```json
{
  "tissue_area_percent": 42.3,
  "color_features": {
    "mean_rgb": [0.65, 0.45, 0.78],
    "dominant_channel": "blue"
  },
  "texture_features": {
    "entropy": 6.24,
    "edge_density": 0.34
  },
  "stain_analysis": {
    "dominant_stain": "hematoxylin",
    "hematoxylin_intensity": 0.67
  }
}
```

## Clinical Interpretation Examples

### Malignant Case
> "High confidence malignant classification with extensive model attention (42% of tissue) across darkly stained nuclear regions, showing heterogeneous cellular architecture and sharp cellular boundaries. Dense nuclear clustering and irregular tissue structures consistent with malignant pathology (confidence: 93%)."

### Benign Case
> "Moderate confidence benign classification with focal model attention (18% of tissue) to pink cytoplasmic areas, showing uniform cellular pattern and typical tissue organization. Standard morphological features observed consistent with benign pathology (confidence: 78%)."

## Morphological Descriptor Mapping

### Tissue Area Coverage
- **< 20%**: Focal attention → Limited pathological changes
- **20-40%**: Moderate attention → Regional abnormalities  
- **> 40%**: Extensive attention → Widespread pathological features

### Staining Patterns
- **Hematoxylin dominance**: Nuclear focus → Cell proliferation, chromatin changes
- **Eosin dominance**: Cytoplasmic focus → Structural alterations, inflammation
- **Balanced**: Mixed patterns → Complex tissue architecture

### Texture Analysis
- **High entropy (> 6.0)**: Cellular heterogeneity → Malignant transformation
- **High edge density (> 0.3)**: Sharp boundaries → Distinct cellular structures
- **Low variance (< 0.1)**: Uniform texture → Organized tissue architecture

## Configuration

### Model Requirements
- PyTorch model with standard forward() method
- Input shape: (batch_size, 3, 224, 224)
- Output: Class logits or probabilities

### Dependencies
```bash
pip install torch torchvision streamlit
pip install shap lime scikit-image
pip install matplotlib seaborn plotly
pip install opencv-python pillow numpy
```

### Target Layer Configuration
For Grad-CAM methods, specify the target layer:
```python
# Common layer names for DenseNet
target_layers = [
    'densenet.features.norm5',  # Final normalization layer
    'features.norm5',           # Alternative naming
    'classifier'                # Before final classification
]
```

## Advanced Features

### Custom Templates
Modify clinical description templates in `morphological_analyzer.py`:

```python
templates = {
    'tissue_area': {
        'low': "focal areas ({:.1f}%)",
        'high': "extensive regions ({:.1f}%)"
    },
    'stain_dominance': {
        'hematoxylin': "darkly stained nuclear regions",
        'custom_pattern': "your custom description"
    }
}
```

### Batch Processing Options
- **Parallel processing**: Process multiple images simultaneously
- **Memory optimization**: Batch size control for large datasets
- **Progress tracking**: Real-time processing status
- **Error handling**: Graceful failure recovery

### Integration with LLMs
For enhanced report generation, integrate with language models:

```python
# Example with Hugging Face transformers
from transformers import pipeline

generator = pipeline('text-generation', model='flan-t5-base')

prompt = f"""
Generate clinical explanation for histopathology classification.
Class: {predicted_class}
Key findings: {morphological_features}
Model confidence: {confidence}
"""

enhanced_report = generator(prompt, max_length=200)
```

## Validation and Quality Assurance

### Quantitative Metrics
- **Insertion/Deletion AUC**: Faithfulness measurement
- **Stability Score**: Consistency under perturbations
- **Localization Accuracy**: Comparison with expert annotations

### Clinical Validation
- **Expert Review**: Pathologist evaluation of highlighted regions
- **Ground Truth Comparison**: ROI overlap with manual annotations
- **Consistency Analysis**: Inter-method agreement assessment

## Troubleshooting

### Common Issues

1. **SHAP Initialization Fails**
   - Reduce background dataset size
   - Use GradientExplainer instead of DeepExplainer
   - Check model compatibility

2. **Memory Issues**
   - Reduce batch size
   - Use CPU for SHAP analysis
   - Process images sequentially

3. **Layer Not Found**
   - Check model architecture with `model.named_modules()`
   - Use debug mode to list available layers
   - Try alternative layer names

### Performance Optimization
- **GPU Acceleration**: Use CUDA for faster processing
- **Caching**: Enable Streamlit caching for repeated analyses
- **Preprocessing**: Optimize image loading and normalization

## Citation

If you use this explainability system in your research, please cite:

```bibtex
@software{histopath_explainability,
  title={Comprehensive Explainability System for Histopathology Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/explainability}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review example notebooks
- Contact the development team