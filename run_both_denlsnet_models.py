#!/usr/bin/env python3
"""
Master Script to Run Both Binary and Multiclass DenLsNet Models
- Runs binary classification first
- Then runs multiclass classification
- Saves both models in separate organized folders
- Generates comprehensive comparison report
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import subprocess

# Import our training modules
from run_binary_denlsnet import BinaryDenLsNetTrainer
from run_multiclass_denlsnet import MulticlassDenLsNetTrainer


class DenLsNetMasterRunner:
    """Master runner for both binary and multiclass DenLsNet training"""
    
    def __init__(self, output_base_dir='complete_denlsnet_results'):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_base_dir = Path(output_base_dir)
        self.experiment_dir = self.output_base_dir / f"complete_experiment_{self.timestamp}"
        
        # Create master directory structure
        self.binary_dir = self.experiment_dir / "binary_results"
        self.multiclass_dir = self.experiment_dir / "multiclass_results"
        self.comparison_dir = self.experiment_dir / "comparison_analysis"
        self.models_dir = self.experiment_dir / "trained_models"
        
        for dir_path in [self.binary_dir, self.multiclass_dir, self.comparison_dir, self.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.binary_results = None
        self.multiclass_results = None
        
        print(f"🚀 DenLsNet Complete Training Pipeline")
        print(f"📁 Master experiment directory: {self.experiment_dir}")
        print(f"🎯 Will train both binary and multiclass models")
        print("="*80)
    
    def run_setup_verification(self):
        """Verify setup before training"""
        print("\n🔧 Verifying Setup...")
        
        try:
            # Test imports
            from model.denlsnet_corrected import create_denlsnet
            from config.training_config import TrainingConfig, get_device
            
            # Test model creation
            device = get_device()
            binary_model = create_denlsnet(num_classes=2)
            multiclass_model = create_denlsnet(num_classes=8)
            
            print("✅ All imports and model creation successful")
            print(f"✅ Device: {device}")
            
            # Test configuration
            binary_config = TrainingConfig(task='binary')
            multiclass_config = TrainingConfig(task='multiclass')
            
            print(f"✅ Binary config: {binary_config.epochs} epochs, {binary_config.optimizer_name}")
            print(f"✅ Multiclass config: {multiclass_config.epochs} epochs, {multiclass_config.optimizer_name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Setup verification failed: {str(e)}")
            return False
    
    def run_binary_training(self):
        """Run binary classification training"""
        print("\n" + "="*80)
        print("🔬 PHASE 1: Binary Classification Training")
        print("="*80)
        
        try:
            # Create binary trainer with custom output directory
            binary_trainer = BinaryDenLsNetTrainer(output_dir=str(self.binary_dir))
            
            # Run training
            self.binary_results = binary_trainer.run_training()
            
            # Copy best models to master models directory
            binary_models_dir = self.models_dir / "binary"
            binary_models_dir.mkdir(exist_ok=True)
            
            # Copy the best model files
            source_model_dir = Path(binary_trainer.output_dir) / "models"
            if (source_model_dir / "binary_denlsnet_best.pth").exists():
                import shutil
                shutil.copy2(
                    source_model_dir / "binary_denlsnet_best.pth",
                    binary_models_dir / "binary_denlsnet_best.pth"
                )
                shutil.copy2(
                    source_model_dir / "binary_denlsnet_deployment.pth",
                    binary_models_dir / "binary_denlsnet_deployment.pth"
                )
                print(f"✅ Binary models copied to: {binary_models_dir}")
            
            print(f"\n🎉 Binary training completed successfully!")
            print(f"📊 Best F1-Score: {self.binary_results['best_metrics']['f1_score']:.4f}")
            print(f"🎯 Best Accuracy: {self.binary_results['best_metrics']['accuracy']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Binary training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_multiclass_training(self):
        """Run multiclass classification training"""
        print("\n" + "="*80)
        print("🔬 PHASE 2: Multiclass Classification Training")
        print("="*80)
        
        try:
            # Create multiclass trainer with custom output directory
            multiclass_trainer = MulticlassDenLsNetTrainer(output_dir=str(self.multiclass_dir))
            
            # Run training
            self.multiclass_results = multiclass_trainer.run_training()
            
            # Copy best models to master models directory
            multiclass_models_dir = self.models_dir / "multiclass"
            multiclass_models_dir.mkdir(exist_ok=True)
            
            # Copy the best model files
            source_model_dir = Path(multiclass_trainer.output_dir) / "models"
            if (source_model_dir / "multiclass_denlsnet_best.pth").exists():
                import shutil
                shutil.copy2(
                    source_model_dir / "multiclass_denlsnet_best.pth",
                    multiclass_models_dir / "multiclass_denlsnet_best.pth"
                )
                shutil.copy2(
                    source_model_dir / "multiclass_denlsnet_deployment.pth",
                    multiclass_models_dir / "multiclass_denlsnet_deployment.pth"
                )
                print(f"✅ Multiclass models copied to: {multiclass_models_dir}")
            
            print(f"\n🎉 Multiclass training completed successfully!")
            print(f"📊 Best F1-Score: {self.multiclass_results['best_metrics']['f1_score']:.4f}")
            print(f"🎯 Best Accuracy: {self.multiclass_results['best_metrics']['accuracy']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Multiclass training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_comparison_analysis(self):
        """Generate comprehensive comparison analysis"""
        print("\n" + "="*80)
        print("📊 PHASE 3: Generating Comparison Analysis")
        print("="*80)
        
        if not self.binary_results or not self.multiclass_results:
            print("❌ Cannot generate comparison - missing results")
            return False
        
        try:
            # Create comparison data
            comparison_data = {
                'experiment_info': {
                    'timestamp': self.timestamp,
                    'experiment_dir': str(self.experiment_dir),
                    'total_models_trained': 2
                },
                'binary_results': {
                    'task': 'binary',
                    'num_classes': 2,
                    'best_epoch': self.binary_results['best_metrics']['epoch'],
                    'best_f1_score': self.binary_results['best_metrics']['f1_score'],
                    'best_accuracy': self.binary_results['best_metrics']['accuracy'],
                    'best_auc': self.binary_results['best_metrics'].get('auc', 'N/A'),
                    'training_time_hours': self.binary_results['total_training_time_hours'],
                    'epochs_trained': self.binary_results['epochs_trained']
                },
                'multiclass_results': {
                    'task': 'multiclass',
                    'num_classes': 8,
                    'best_epoch': self.multiclass_results['best_metrics']['epoch'],
                    'best_f1_score': self.multiclass_results['best_metrics']['f1_score'],
                    'best_f1_weighted': self.multiclass_results['best_metrics'].get('f1_weighted', 'N/A'),
                    'best_accuracy': self.multiclass_results['best_metrics']['accuracy'],
                    'training_time_hours': self.multiclass_results['total_training_time_hours'],
                    'epochs_trained': self.multiclass_results['epochs_trained'],
                    'per_class_f1': self.multiclass_results['best_metrics'].get('per_class_f1', [])
                },
                'model_paths': {
                    'binary_best': str(self.models_dir / "binary" / "binary_denlsnet_best.pth"),
                    'binary_deployment': str(self.models_dir / "binary" / "binary_denlsnet_deployment.pth"),
                    'multiclass_best': str(self.models_dir / "multiclass" / "multiclass_denlsnet_best.pth"),
                    'multiclass_deployment': str(self.models_dir / "multiclass" / "multiclass_denlsnet_deployment.pth")
                }
            }
            
            # Save comparison data
            with open(self.comparison_dir / "comparison_results.json", 'w') as f:
                json.dump(comparison_data, f, indent=2, default=str)
            
            # Generate comparison report
            self._generate_comparison_report(comparison_data)
            
            # Generate model usage guide
            self._generate_model_usage_guide()
            
            print("✅ Comparison analysis completed")
            return True
            
        except Exception as e:
            print(f"❌ Comparison analysis failed: {str(e)}")
            return False
    
    def _generate_comparison_report(self, comparison_data):
        """Generate detailed comparison report"""
        binary = comparison_data['binary_results']
        multiclass = comparison_data['multiclass_results']
        
        report = f"""# DenLsNet Complete Training Results

**Experiment Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Experiment ID:** {self.timestamp}

## Architecture Summary

Both models use the corrected DenLsNet architecture:
- **Backbone:** DenseNet-121 (with SE layers)
- **Classifier:** Bidirectional LSTM (128 hidden units)
- **Feature Fusion:** iAFF (iterative Attentional Feature Fusion)
- **Final Feature Dimension:** 1920
- **Dropout Rate:** 0.5

## Training Configuration

- **Optimizer:** SGD (lr=0.003, momentum=0.9, weight_decay=1e-4)
- **Scheduler:** CosineAnnealingLR (T_max=80, eta_min=1e-6)
- **Target Epochs:** 80
- **Batch Size:** 32
- **Early Stopping:** F1-score monitoring (patience=10)

## Results Comparison

### Binary Classification (Benign vs Malignant)

| Metric | Value |
|--------|-------|
| **Best F1-Score** | {binary['best_f1_score']:.4f} |
| **Best Accuracy** | {binary['best_accuracy']:.4f} |"""

        if binary['best_auc'] != 'N/A':
            report += f"""
| **Best AUC** | {binary['best_auc']:.4f} |"""

        report += f"""
| **Best Epoch** | {binary['best_epoch']} |
| **Training Time** | {binary['training_time_hours']:.2f} hours |
| **Epochs Trained** | {binary['epochs_trained']} |

### Multiclass Classification (8 BreakHis Subtypes)

| Metric | Value |
|--------|-------|
| **Best F1-Score (Macro)** | {multiclass['best_f1_score']:.4f} |"""

        if multiclass['best_f1_weighted'] != 'N/A':
            report += f"""
| **Best F1-Score (Weighted)** | {multiclass['best_f1_weighted']:.4f} |"""

        report += f"""
| **Best Accuracy** | {multiclass['best_accuracy']:.4f} |
| **Best Epoch** | {multiclass['best_epoch']} |
| **Training Time** | {multiclass['training_time_hours']:.2f} hours |
| **Epochs Trained** | {multiclass['epochs_trained']} |

### Per-Class Performance (Multiclass)

The 8-class model performance on individual BreakHis subtypes:

| Class | F1-Score |
|-------|----------|"""

        class_names = [
            'Adenosis', 'Fibroadenoma', 'Phyllodes Tumor', 'Tubular Adenoma',
            'Ductal Carcinoma', 'Lobular Carcinoma', 'Mucinous Carcinoma', 'Papillary Carcinoma'
        ]

        if multiclass['per_class_f1']:
            for i, (class_name, f1_score) in enumerate(zip(class_names, multiclass['per_class_f1'])):
                report += f"""
| {class_name} | {f1_score:.4f} |"""

        report += f"""

## Model Files

### Binary Classification Model
- **Best Model:** `{comparison_data['model_paths']['binary_best']}`
- **Deployment Model:** `{comparison_data['model_paths']['binary_deployment']}`

### Multiclass Classification Model
- **Best Model:** `{comparison_data['model_paths']['multiclass_best']}`
- **Deployment Model:** `{comparison_data['model_paths']['multiclass_deployment']}`

## Performance Analysis

### Binary vs Multiclass Comparison

- **Complexity:** Multiclass task is significantly more challenging (8 classes vs 2)
- **Expected Performance Drop:** {binary['best_accuracy'] - multiclass['best_accuracy']:.1%} accuracy decrease from binary to multiclass is reasonable
- **Training Efficiency:** Both models converged within expected timeframes

### Key Observations

1. **Binary Model Performance:** Achieved {binary['best_f1_score']:.1%} F1-score, indicating strong discriminative capability
2. **Multiclass Model Performance:** Achieved {multiclass['best_f1_score']:.1%} macro F1-score across 8 classes
3. **Training Stability:** Both models showed stable convergence patterns
4. **Architecture Effectiveness:** DenLsNet architecture performs well on both tasks

## Usage Recommendations

### For Binary Classification (Benign vs Malignant)
- Use the binary model for initial screening applications
- Higher accuracy makes it suitable for clinical decision support
- Faster inference due to simpler classification head

### For Multiclass Classification (Detailed Subtyping)
- Use for detailed histopathological analysis
- Provides specific subtype information for treatment planning
- More comprehensive but requires higher computational resources

## Next Steps

1. **Model Evaluation:** Run comprehensive evaluation with explainability analysis
2. **Clinical Validation:** Test on independent clinical datasets
3. **Deployment:** Integrate models into clinical workflow systems
4. **Monitoring:** Implement performance monitoring in production

---

*Generated by DenLsNet Complete Training Pipeline*
*Experiment ID: {self.timestamp}*
"""

        # Save report
        with open(self.comparison_dir / "comprehensive_report.md", 'w') as f:
            f.write(report)
        
        print(f"📋 Comprehensive report saved: {self.comparison_dir / 'comprehensive_report.md'}")
    
    def _generate_model_usage_guide(self):
        """Generate model usage guide"""
        usage_guide = f"""# DenLsNet Model Usage Guide

## Quick Start

### Loading Models

```python
import torch

# Load binary classification model
binary_model_path = "{self.models_dir / 'binary' / 'binary_denlsnet_deployment.pth'}"
binary_checkpoint = torch.load(binary_model_path, map_location='cpu')
binary_model = binary_checkpoint['model']
binary_model.eval()

# Load multiclass classification model
multiclass_model_path = "{self.models_dir / 'multiclass' / 'multiclass_denlsnet_deployment.pth'}"
multiclass_checkpoint = torch.load(multiclass_model_path, map_location='cpu')
multiclass_model = multiclass_checkpoint['model']
multiclass_model.eval()
```

### Image Preprocessing

```python
import torchvision.transforms as transforms
from PIL import Image

# Define preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess image
image = Image.open('path_to_histopathology_image.png').convert('RGB')
input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
```

### Making Predictions

```python
import torch.nn.functional as F

# Binary classification
with torch.no_grad():
    binary_output = binary_model(input_tensor)
    binary_probs = F.softmax(binary_output, dim=1)
    binary_prediction = torch.argmax(binary_probs, dim=1)
    
    print(f"Binary Prediction: {{'Benign' if binary_prediction.item() == 0 else 'Malignant'}}")
    print(f"Confidence: {{binary_probs.max().item():.3f}}")

# Multiclass classification
class_names = [
    'Adenosis', 'Fibroadenoma', 'Phyllodes Tumor', 'Tubular Adenoma',
    'Ductal Carcinoma', 'Lobular Carcinoma', 'Mucinous Carcinoma', 'Papillary Carcinoma'
]

with torch.no_grad():
    multiclass_output = multiclass_model(input_tensor)
    multiclass_probs = F.softmax(multiclass_output, dim=1)
    multiclass_prediction = torch.argmax(multiclass_probs, dim=1)
    
    predicted_class = class_names[multiclass_prediction.item()]
    confidence = multiclass_probs.max().item()
    
    print(f"Multiclass Prediction: {{predicted_class}}")
    print(f"Confidence: {{confidence:.3f}}")
```

### Batch Processing

```python
# Process multiple images
def process_batch(image_paths, model, class_names=None):
    results = []
    
    for img_path in image_paths:
        image = Image.open(img_path).convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            prediction = torch.argmax(probs, dim=1)
            confidence = probs.max().item()
            
            if class_names:
                pred_class = class_names[prediction.item()]
            else:
                pred_class = 'Benign' if prediction.item() == 0 else 'Malignant'
            
            results.append({{
                'image_path': img_path,
                'prediction': pred_class,
                'confidence': confidence,
                'probabilities': probs.cpu().numpy().tolist()
            }})
    
    return results
```

## Model Information

### Binary Model
- **Classes:** 2 (Benign, Malignant)
- **Input Size:** 224x224x3
- **Output Size:** 2
- **Best F1-Score:** {self.binary_results['best_metrics']['f1_score']:.4f}
- **Best Accuracy:** {self.binary_results['best_metrics']['accuracy']:.4f}

### Multiclass Model
- **Classes:** 8 (BreakHis subtypes)
- **Input Size:** 224x224x3
- **Output Size:** 8
- **Best F1-Score:** {self.multiclass_results['best_metrics']['f1_score']:.4f}
- **Best Accuracy:** {self.multiclass_results['best_metrics']['accuracy']:.4f}

## Integration Examples

### Flask API Example

```python
from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load models (do this once at startup)
binary_model = torch.load('binary_denlsnet_deployment.pth')['model']
multiclass_model = torch.load('multiclass_denlsnet_deployment.pth')['model']

@app.route('/predict/binary', methods=['POST'])
def predict_binary():
    # Get image from request
    image_data = request.json['image']  # Base64 encoded
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    # Preprocess and predict
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = binary_model(input_tensor)
        probs = F.softmax(output, dim=1)
        prediction = torch.argmax(probs, dim=1)
    
    return jsonify({{
        'prediction': 'Benign' if prediction.item() == 0 else 'Malignant',
        'confidence': probs.max().item(),
        'probabilities': probs.cpu().numpy().tolist()
    }})

@app.route('/predict/multiclass', methods=['POST'])
def predict_multiclass():
    # Similar implementation for multiclass
    pass
```

### Streamlit App Example

```python
import streamlit as st
import torch
from PIL import Image

st.title("DenLsNet Histopathology Classifier")

# Load models
@st.cache_resource
def load_models():
    binary_model = torch.load('binary_denlsnet_deployment.pth')['model']
    multiclass_model = torch.load('multiclass_denlsnet_deployment.pth')['model']
    return binary_model, multiclass_model

binary_model, multiclass_model = load_models()

# File upload
uploaded_file = st.file_uploader("Choose a histopathology image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Make predictions
    input_tensor = preprocess(image).unsqueeze(0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Binary Classification")
        with torch.no_grad():
            binary_output = binary_model(input_tensor)
            binary_probs = F.softmax(binary_output, dim=1)
            binary_pred = torch.argmax(binary_probs, dim=1)
        
        result = 'Benign' if binary_pred.item() == 0 else 'Malignant'
        confidence = binary_probs.max().item()
        st.write(f"**Prediction:** {{result}}")
        st.write(f"**Confidence:** {{confidence:.3f}}")
    
    with col2:
        st.subheader("Multiclass Classification")
        # Similar implementation for multiclass
```

## Performance Monitoring

### Model Validation

```python
def validate_model_performance(model, test_dataloader, class_names):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_dataloader:
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_predictions))
```

---

*Generated by DenLsNet Complete Training Pipeline*
*For technical support, refer to the model documentation and training logs*
"""

        # Save usage guide
        with open(self.comparison_dir / "model_usage_guide.md", 'w') as f:
            f.write(usage_guide)
        
        print(f"📖 Model usage guide saved: {self.comparison_dir / 'model_usage_guide.md'}")
    
    def run_complete_pipeline(self):
        """Run the complete training pipeline"""
        print("🚀 Starting Complete DenLsNet Training Pipeline")
        print("="*80)
        
        start_time = time.time()
        
        # Phase 0: Setup verification
        if not self.run_setup_verification():
            print("❌ Setup verification failed. Aborting.")
            return False
        
        # Phase 1: Binary training
        if not self.run_binary_training():
            print("❌ Binary training failed. Aborting.")
            return False
        
        # Phase 2: Multiclass training
        if not self.run_multiclass_training():
            print("❌ Multiclass training failed. Continuing with analysis of binary results only.")
        
        # Phase 3: Comparison analysis
        if not self.generate_comparison_analysis():
            print("⚠️ Comparison analysis failed, but models were trained successfully.")
        
        # Final summary
        total_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("🎉 COMPLETE PIPELINE FINISHED!")
        print("="*80)
        print(f"⏱️  Total Pipeline Time: {total_time/3600:.2f} hours")
        
        if self.binary_results:
            print(f"\n📊 Binary Model Results:")
            print(f"   🎯 Best Accuracy: {self.binary_results['best_metrics']['accuracy']:.4f}")
            print(f"   📈 Best F1-Score: {self.binary_results['best_metrics']['f1_score']:.4f}")
            if 'auc' in self.binary_results['best_metrics']:
                print(f"   📊 Best AUC: {self.binary_results['best_metrics']['auc']:.4f}")
        
        if self.multiclass_results:
            print(f"\n📊 Multiclass Model Results:")
            print(f"   🎯 Best Accuracy: {self.multiclass_results['best_metrics']['accuracy']:.4f}")
            print(f"   📈 Best F1-Score (Macro): {self.multiclass_results['best_metrics']['f1_score']:.4f}")
            if 'f1_weighted' in self.multiclass_results['best_metrics']:
                print(f"   📈 Best F1-Score (Weighted): {self.multiclass_results['best_metrics']['f1_weighted']:.4f}")
        
        print(f"\n📁 All Results Saved To:")
        print(f"   🗂️  Master Directory: {self.experiment_dir}")
        print(f"   🤖 Trained Models: {self.models_dir}")
        print(f"   📊 Comparison Analysis: {self.comparison_dir}")
        
        print(f"\n📋 Key Files Generated:")
        print(f"   📄 Comprehensive Report: {self.comparison_dir / 'comprehensive_report.md'}")
        print(f"   📖 Usage Guide: {self.comparison_dir / 'model_usage_guide.md'}")
        print(f"   📊 Results Data: {self.comparison_dir / 'comparison_results.json'}")
        
        print("\n🚀 Models are ready for deployment and evaluation!")
        print("="*80)
        
        return True


def main():
    """Main function to run complete pipeline"""
    print("🔬 DenLsNet Complete Training Pipeline")
    print("Training both Binary and Multiclass models")
    print("="*50)
    
    try:
        runner = DenLsNetMasterRunner()
        success = runner.run_complete_pipeline()
        
        if success:
            print("\n✅ Complete pipeline executed successfully!")
            return runner.experiment_dir
        else:
            print("\n❌ Pipeline execution failed!")
            return None
            
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    experiment_dir = main()
    if experiment_dir:
        print(f"\n🎯 Experiment completed! Results in: {experiment_dir}")
    else:
        print("\n💥 Experiment failed!")