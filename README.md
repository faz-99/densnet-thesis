# 🔬 DenLsNet: Multi-Class Medical Image Classification with Explainable AI

A comprehensive deep learning system for histopathology image classification featuring multi-class extension, stain normalization ablation, and quantitative interpretability analysis.

## 🎯 Project Overview

This project implements **DenLsNet** (DenseNet with LSTM and SE attention) for medical image classification, extending from binary to 8-class BreakHis histopathology classification with comprehensive explainability analysis and stain normalization studies.

### 🚀 Key Features

#### Core Architecture
- **🧠 DenLsNet**: Enhanced DenseNet-201 with SE layers and iAFF fusion
- **🔄 Multi-Class Extension**: 8-class BreakHis classification (DenLsNet-MC)
- **🎨 Stain Normalization**: Macenko and Reinhard preprocessing ablation
- **📊 LSTM Classification Head**: Temporal feature processing for improved accuracy

#### Explainability Framework (DenLsNet-XAI)
- **🔍 Multiple XAI Methods**: Grad-CAM, Grad-CAM++, SHAP, LIME
- **📈 Quantitative Evaluation**: Insertion/Deletion AUC, Stability Analysis
- **🎯 Clinical Interpretability**: Visual and quantitative explanation assessment
- **⚖️ Method Comparison**: Comprehensive benchmarking of XAI techniques

#### Academic Contributions
- **📚 Systematic Ablation**: Stain normalization impact on performance
- **🔬 Quantitative XAI**: Novel metrics for interpretability evaluation
- **🏥 Clinical Relevance**: Fine-grained histopathology subtype classification
- **📖 Reproducible Research**: Complete pipeline with academic naming conventions

## 🏗️ Architecture

### DenLsNet Model Variants

#### 1. **DenLsNet (Binary)** - Original Implementation
- **Base**: DenseNet-201 with pre-trained ImageNet weights
- **Enhancement**: SE (Squeeze-and-Excitation) layers in dense blocks
- **Fusion**: iAFF (iterative Attentional Feature Fusion) between blocks
- **Classifier**: LSTM-based head with dropout regularization
- **Output**: Binary classification (Benign/Malignant)

#### 2. **DenLsNet-MC (Multi-Class)** - 8-Class Extension
- **Architecture**: Same as DenLsNet with modified classification head
- **Classes**: 8 BreakHis subtypes (4 Benign + 4 Malignant)
- **Loss**: Categorical cross-entropy with class balancing
- **Metrics**: Per-class precision, recall, F1-score, confusion matrix

#### 3. **DenLsNet-MC-Stain** - Stain Normalized Variants
- **DenLsNet-MC-Macenko**: Trained on Macenko normalized images
- **DenLsNet-MC-Reinhard**: Trained on Reinhard normalized images  
- **DenLsNet-MC-None**: Baseline without stain normalization

### Explainability Framework (DenLsNet-XAI)

#### Visual Explanation Methods
1. **Grad-CAM**: Class activation mapping via gradients
2. **Grad-CAM++**: Improved localization with weighted gradients
3. **SHAP**: Shapley value-based pixel attributions
4. **LIME**: Local interpretable model-agnostic explanations

#### Quantitative Evaluation Metrics
- **Insertion AUC**: Performance when adding important pixels
- **Deletion AUC**: Performance when removing important pixels  
- **Stability**: Consistency under input perturbations
- **Localization Accuracy**: IoU with ground truth ROI (when available)

## 📁 Project Structure

```
├── 📁 model/                          # Model architectures
│   ├── model.py                       # Original DenLsNet (binary)
│   ├── multiclass_model.py            # DenLsNet-MC (8-class)
│   ├── SENet.py                       # SE attention layers
│   └── ...
├── 📁 explainability/                 # Explainability framework
│   ├── grad_cam.py                    # Grad-CAM & Grad-CAM++
│   ├── shap_explainer.py              # SHAP analysis
│   ├── lime_explainer.py              # LIME analysis
│   ├── explainer.py                   # Comprehensive explainer
│   └── interpretability_framework.py  # Quantitative XAI evaluation
├── 📁 stain_normalization/            # Stain preprocessing
│   ├── stain_normalizer.py            # Macenko & Reinhard methods
│   └── __init__.py                    # Module initialization
├── 📁 evaluation/                     # Evaluation metrics
│   └── metrics.py                     # Comprehensive evaluation
├── 📁 utils/                          # Utility functions
│   ├── load_dataset2.py               # Binary data loading
│   ├── load_multiclass_dataset.py     # Multi-class data loading
│   └── confusion_matrix.py            # Confusion matrix utils
├── 🎯 Training Scripts
│   ├── train.py                       # Binary training
│   └── train_multiclass.py            # Multi-class training with ablation
├── 🌐 User Interfaces
│   ├── app.py                         # Binary classification UI
│   └── app_multiclass.py              # Multi-class UI with stain options
├── 📊 Pipeline Runners
│   ├── run_evaluation.py              # Model evaluation
│   ├── run_explainability.py          # Interpretability analysis
│   ├── run_complete_pipeline.py       # Binary pipeline
│   └── run_complete_multiclass_pipeline.py  # Complete multi-class pipeline
├── 🔧 Configuration
│   ├── config.py                      # Binary configuration
│   └── config_multiclass.py           # Multi-class configuration
└── 📋 Dependencies
    ├── requirements_explainability.txt
    ├── requirements_ui.txt
    └── requirements_multiclass.txt
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd denlsnet-multiclass

# Install dependencies for multi-class extension
pip install -r requirements_multiclass.txt

# Optional: Install additional packages
pip install -r requirements_explainability.txt
pip install -r requirements_ui.txt
```

### 2. Data Preparation

#### Option A: Binary BreakHis Dataset
```
datasets/BreaKHis 400X/
├── train/
│   ├── benign/
│   └── malignant/
└── test/
    ├── benign/
    └── malignant/
```

#### Option B: Multi-Class Structure (Auto-generated)
```
datasets/BreaKHis 400X/multiclass/
├── train/
│   ├── Adenosis/
│   ├── Fibroadenoma/
│   ├── Phyllodes_tumor/
│   ├── Tubular_adenoma/
│   ├── Ductal_carcinoma/
│   ├── Lobular_carcinoma/
│   ├── Mucinous_carcinoma/
│   └── Papillary_carcinoma/
└── test/
    └── [same structure]
```

### 3. Configuration

#### Binary Classification
Update `config.py`:
```python
class_num = 2
train = "datasets/BreaKHis 400X/train"
valid = "datasets/BreaKHis 400X/test"
```

#### Multi-Class Classification  
Update `config_multiclass.py`:
```python
class_num = 8
class_names = ['Adenosis', 'Fibroadenoma', ...]
train = "datasets/BreaKHis 400X/multiclass/train"
valid = "datasets/BreaKHis 400X/multiclass/test"
```

### 4. Complete Multi-Class Pipeline

#### Run Full Ablation Study
```bash
# Complete pipeline with all stain normalization variants
python run_complete_multiclass_pipeline.py

# Quick test run (reduced epochs)
python run_complete_multiclass_pipeline.py --quick_run
```

#### Individual Components

##### Multi-Class Training
```bash
# Train single variant
python train_multiclass.py --stain_method none

# Run complete ablation study
python train_multiclass.py --ablation
```

##### Model Evaluation
```bash
python run_evaluation.py --model_path weight/multiclass/none/denlsnet_mc_none_best.pth
```

##### Interpretability Analysis
```bash
python run_explainability.py --model_path weight/multiclass/none/denlsnet_mc_none_best.pth
```

##### Interactive Multi-Class UI
```bash
# Multi-class UI with stain normalization options
streamlit run app_multiclass.py

# Original binary UI
streamlit run app.py
```

## 📊 Evaluation Metrics

The system computes comprehensive performance metrics:

### Binary Classification Metrics
- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall (Sensitivity)**: True positive rate
- **Specificity**: True negative rate
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve

### Visualizations Generated
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Class Distribution Analysis
- Confidence Distribution
- Interactive Plotly Charts

## 🧠 Explainability Features

### Grad-CAM Analysis
- Generates visual heatmaps showing important image regions
- Compares standard Grad-CAM with Grad-CAM++
- Creates overlay visualizations on original images

### SHAP Analysis
- Provides quantitative pixel-level attributions
- Shows positive and negative contributions
- Generates summary plots for multiple samples

### LIME Analysis
- Uses superpixel-based local explanations
- Shows contributing and detracting regions
- Provides interpretable local decision boundaries

## 🌐 Interactive UI Features

The Streamlit UI provides:

### Image Upload & Analysis
- Drag-and-drop image upload
- Real-time preprocessing visualization
- Instant classification results

### Explainability Visualization
- Interactive Grad-CAM heatmaps
- SHAP attribution maps
- LIME superpixel explanations
- Side-by-side comparisons

### Performance Metrics
- Confidence scoring
- Probability distributions
- Model performance statistics

### Export Capabilities
- Save analysis results
- Download visualizations
- Export detailed reports

## 📈 Results and Outputs

### Training Outputs
```
weight/save/40/
├── iaff40_5.pth              # Best model checkpoint
└── ...

csv/40/
├── iaff40_5.csv              # Training logs
└── ...
```

### Evaluation Results
```
evaluation_results_YYYYMMDD_HHMMSS/
├── evaluation_results.json   # Complete metrics
├── evaluation_report.md      # Detailed report
├── confusion_matrix.png      # Confusion matrix
├── roc_curve.png            # ROC analysis
├── metrics_summary.png      # Performance overview
└── interactive_*.html       # Interactive plots
```

### Explainability Results
```
explainability/
├── gradcam_results/         # Grad-CAM visualizations
├── shap_results/           # SHAP explanations
├── lime_results/           # LIME analyses
├── performance_analysis.json
└── README.md               # Analysis summary
```

## 🔧 Configuration Options

### Model Configuration (`config.py`)
```python
# Model parameters
net_name = "iaff40"
class_num = 2
img_s = 224

# Training parameters
batch_size = 32
max_epoch = 100
lr = 0.003

# Dataset parameters
dataset_mean = (0.5613, 0.5778, 0.6032)
dataset_std = (0.2114, 0.1957, 0.1590)
```

### Explainability Options
- **Grad-CAM target layer**: `densenet.features.norm5`
- **SHAP background size**: 50-100 samples
- **LIME perturbations**: 100-1000 samples

## 🧪 Testing

Verify your installation:
```bash
# Test explainability modules
python test_explainability.py

# Test UI components
python test_ui.py
```

## 📚 Usage Examples

### Programmatic Usage

```python
# Load trained model
from evaluation.metrics import evaluate_saved_model

results = evaluate_saved_model(
    model_path="weight/save/40/iaff40_5.pth",
    test_dataloader=test_loader,
    class_names=['Benign', 'Malignant']
)

# Run explainability analysis
from explainability.explainer import ComprehensiveExplainer

explainer = ComprehensiveExplainer(model, device, class_names)
explainer.generate_comprehensive_explanations(
    test_dataloader=test_loader,
    background_dataloader=background_loader,
    techniques=['gradcam', 'shap', 'lime']
)
```

### Command Line Usage

```bash
# Custom evaluation
python run_evaluation.py \
  --model_path your_model.pth \
  --save_dir custom_results

# Selective explainability
python run_explainability.py \
  --model_path your_model.pth \
  --techniques gradcam shap \
  --num_samples 10
```

## 🎓 For Thesis Demonstration

### Key Demonstration Points

1. **Model Architecture**: Show DenseNet with attention mechanisms
2. **Training Process**: Display loss curves and accuracy progression
3. **Performance Metrics**: Comprehensive evaluation results
4. **Explainability**: Visual interpretations of model decisions
5. **Interactive Analysis**: Real-time image classification and explanation

### Presentation Flow

1. **Introduction**: Load the Streamlit UI
2. **Model Overview**: Show architecture and training results
3. **Live Demo**: Upload test images and show predictions
4. **Explainability**: Demonstrate Grad-CAM, SHAP, and LIME
5. **Performance**: Review evaluation metrics and visualizations
6. **Medical Insights**: Discuss clinical relevance of explanations

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Memory Error**: Reduce batch size or use CPU
2. **Missing Dependencies**: Install requirements files
3. **Model Loading Error**: Check model path and format
4. **UI Not Loading**: Verify Streamlit installation

### Performance Tips

1. **GPU Usage**: Enable CUDA for faster processing
2. **Background Data**: Use representative samples for SHAP
3. **Sample Size**: Start with fewer samples for testing
4. **Caching**: Streamlit caches model loading automatically

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@misc{densenet-explainable-medical,
  title={DenseNet Medical Image Classification with Explainable AI},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo}}
}
```

## 📞 Support

For questions or issues:
- Check the troubleshooting section
- Review the test scripts output
- Examine the generated log files
- Verify all requirements are installed

## 🔄 Updates and Maintenance

- Regular updates to explainability techniques
- Performance optimizations
- UI enhancements
- Additional evaluation metrics

---

**Built with ❤️ for advancing explainable AI in medical image analysis**
#
# 🔬 Multi-Class Extension Features

### 1. **8-Class BreakHis Classification**

#### Benign Subtypes (Classes 0-3)
- **Adenosis**: Benign proliferative breast disease
- **Fibroadenoma**: Common benign breast tumor
- **Phyllodes Tumor**: Rare benign breast tumor
- **Tubular Adenoma**: Benign epithelial tumor

#### Malignant Subtypes (Classes 4-7)  
- **Ductal Carcinoma**: Most common breast cancer type
- **Lobular Carcinoma**: Second most common type
- **Mucinous Carcinoma**: Rare mucin-producing cancer
- **Papillary Carcinoma**: Rare papillary growth pattern

### 2. **Stain Normalization Ablation Study**

#### Methods Implemented
- **Macenko Normalization**: Optical density-based stain separation
- **Reinhard Normalization**: LAB color space statistics matching
- **Baseline (None)**: No stain normalization

#### Evaluation Protocol
```bash
# Automatic ablation study
python train_multiclass.py --ablation

# Results comparison
python run_complete_multiclass_pipeline.py
```

### 3. **Quantitative Interpretability Framework**

#### Metrics Implemented
- **Insertion AUC**: Measures explanation quality by pixel addition
- **Deletion AUC**: Measures explanation quality by pixel removal
- **Stability**: Consistency under input perturbations
- **Processing Time**: Computational efficiency comparison

#### Usage Example
```python
from explainability.interpretability_framework import InterpretabilityFramework

# Initialize framework
framework = InterpretabilityFramework(model, device, class_names)

# Run comprehensive evaluation
results = framework.comprehensive_evaluation(
    dataloader=test_loader,
    num_samples=50,
    methods=['gradcam', 'gradcam_plus', 'shap', 'lime']
)
```

## 📊 Academic Results Framework

### Model Naming Convention
- **DenLsNet**: Original binary classification model
- **DenLsNet-MC**: Multi-class extension (8 classes)
- **DenLsNet-MC-Macenko**: Multi-class with Macenko normalization
- **DenLsNet-MC-Reinhard**: Multi-class with Reinhard normalization
- **DenLsNet-MC-None**: Multi-class baseline (no normalization)
- **DenLsNet-XAI**: Interpretability framework

### Experimental Design

#### Research Questions Addressed
1. **RQ1**: How does multi-class extension affect model performance?
2. **RQ2**: What is the impact of stain normalization on classification accuracy?
3. **RQ3**: Which interpretability method provides the most reliable explanations?
4. **RQ4**: How stable are explanations across different preprocessing methods?

#### Evaluation Protocol
```bash
# Complete experimental pipeline
python run_complete_multiclass_pipeline.py

# Generates:
# - Training results for all variants
# - Comprehensive evaluation metrics
# - Quantitative interpretability analysis
# - Statistical significance tests
# - Academic-ready visualizations
```

### Output Structure for Research
```
results/complete_pipeline/run_YYYYMMDD_HHMMSS/
├── pipeline_results.json              # Complete numerical results
├── comprehensive_report.md             # Academic report
├── model_comparison.csv                # Performance comparison table
├── interpretability_comparison.csv     # XAI method comparison
├── training_summary.png                # Results visualization
├── evaluations/                        # Detailed evaluation per variant
│   ├── none/                          # Baseline results
│   ├── macenko/                       # Macenko results  
│   └── reinhard/                      # Reinhard results
└── interpretability/                   # XAI analysis per variant
    ├── none/                          # Baseline interpretability
    ├── macenko/                       # Macenko interpretability
    └── reinhard/                      # Reinhard interpretability
```

## 🎓 For Thesis Demonstration

### Key Demonstration Points

#### 1. **Multi-Class Classification Capability**
- Show 8-class BreakHis classification
- Demonstrate per-class performance metrics
- Highlight clinical relevance of fine-grained classification

#### 2. **Stain Normalization Impact**
- Compare performance across normalization methods
- Show visual differences in preprocessing
- Quantify domain adaptation benefits

#### 3. **Comprehensive Interpretability**
- Multiple XAI methods (Grad-CAM, SHAP, LIME)
- Quantitative evaluation metrics
- Stability and reliability analysis

#### 4. **Interactive Analysis Interface**
```bash
# Launch multi-class UI
streamlit run app_multiclass.py

# Features:
# - Real-time 8-class prediction
# - Stain normalization comparison
# - Multiple XAI method visualization
# - Model variant comparison
# - Confidence analysis
```

### Presentation Flow Recommendation

1. **Introduction** (5 min)
   - Problem: Fine-grained histopathology classification
   - Solution: DenLsNet multi-class extension

2. **Architecture Overview** (10 min)
   - DenLsNet components (DenseNet + SE + iAFF + LSTM)
   - Multi-class extension methodology
   - Stain normalization integration

3. **Experimental Design** (10 min)
   - Ablation study protocol
   - Evaluation metrics
   - Interpretability framework

4. **Live Demonstration** (15 min)
   - Interactive UI walkthrough
   - Real-time image classification
   - Explanation generation and comparison
   - Model variant comparison

5. **Results Analysis** (15 min)
   - Performance comparison tables
   - Statistical significance
   - Clinical implications
   - Interpretability insights

6. **Conclusions** (5 min)
   - Key contributions
   - Clinical impact
   - Future work

## 🔧 Advanced Configuration

### Custom Stain Normalization
```python
from stain_normalization import StainNormalizer

# Create custom normalizer
normalizer = StainNormalizer(method='macenko')

# Fit to target image
target_image = load_target_image()
normalizer.fit_target(target_image)

# Normalize dataset
normalized_image = normalizer.normalize(input_image)
```

### Custom Loss Functions
```python
from model.multiclass_model import get_loss_function

# Focal loss for class imbalance
focal_loss = get_loss_function('focal', alpha=1.0, gamma=2.0)

# Label smoothing for regularization
smooth_loss = get_loss_function('label_smoothing', smoothing=0.1)
```

### Interpretability Customization
```python
from explainability.interpretability_framework import InterpretabilityFramework

# Custom evaluation
framework = InterpretabilityFramework(model, device, class_names)

# Evaluate specific methods
results = framework.comprehensive_evaluation(
    dataloader=test_loader,
    methods=['gradcam', 'shap'],  # Select methods
    num_samples=100,              # Increase sample size
    save_dir='custom_analysis'    # Custom output directory
)
```

## 📈 Performance Benchmarks

### Expected Results (BreakHis 400X)

#### Binary Classification (Original DenLsNet)
- **Accuracy**: ~95-97%
- **F1-Score**: ~95-96%
- **AUC**: ~98-99%

#### Multi-Class Classification (DenLsNet-MC)
- **Overall Accuracy**: ~85-90%
- **Macro F1-Score**: ~83-88%
- **Per-Class F1**: 75-95% (varies by class)

#### Stain Normalization Impact
- **Macenko**: +2-5% accuracy improvement
- **Reinhard**: +1-3% accuracy improvement
- **Cross-dataset**: Significant robustness improvement

#### Interpretability Metrics
- **Grad-CAM Insertion AUC**: ~0.65-0.75
- **Grad-CAM Deletion AUC**: ~0.25-0.35
- **Stability Correlation**: ~0.70-0.85

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd denlsnet-multiclass

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements_multiclass.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black .
flake8 .
```

### Adding New Features

#### New Stain Normalization Method
1. Implement in `stain_normalization/stain_normalizer.py`
2. Add to `config_multiclass.py` experiment variants
3. Update training pipeline
4. Add tests

#### New Interpretability Method
1. Implement in `explainability/` directory
2. Add to `InterpretabilityFramework`
3. Update quantitative evaluation
4. Add visualization support

#### New Model Architecture
1. Implement in `model/` directory
2. Update configuration files
3. Modify training scripts
4. Add evaluation support

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{denlsnet-multiclass-2024,
  title={DenLsNet: Multi-Class Histopathology Classification with Stain Normalization and Quantitative Interpretability},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo/denlsnet-multiclass}},
  note={Multi-class extension with comprehensive explainability framework}
}
```

## 📞 Support and Issues

### Common Issues

#### CUDA Memory Error
```bash
# Reduce batch size in config
batch_size = 16  # Instead of 32

# Or use CPU
device = 'cpu'
```

#### Stain Normalization Fails
```bash
# Install staintools (optional)
pip install staintools

# Or use custom implementation (included)
```

#### UI Not Loading
```bash
# Check Streamlit installation
pip install streamlit>=1.10.0

# Run with specific port
streamlit run app_multiclass.py --server.port 8502
```

### Getting Help

1. **Check Documentation**: Review this README and code comments
2. **Run Tests**: Execute `pytest tests/` to verify installation
3. **Check Issues**: Look for similar problems in GitHub issues
4. **Create Issue**: Provide detailed error messages and system info

## 🔄 Updates and Maintenance

### Version History
- **v1.0**: Original binary DenLsNet implementation
- **v2.0**: Multi-class extension with 8-class support
- **v2.1**: Stain normalization ablation study
- **v2.2**: Quantitative interpretability framework
- **v2.3**: Enhanced UI with multi-class support

### Roadmap
- [ ] Additional stain normalization methods
- [ ] Cross-dataset evaluation protocol
- [ ] Real-time inference optimization
- [ ] Clinical validation study
- [ ] Docker containerization
- [ ] Cloud deployment support

---

**Built with ❤️ for advancing explainable AI in medical image analysis**

*This project represents a comprehensive framework for multi-class histopathology classification with state-of-the-art interpretability analysis, designed for academic research and clinical applications.*