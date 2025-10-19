#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for DenLsNet
Runs saved model evaluation with full explainability analysis
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from pathlib import Path
import cv2
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import project modules
import config
from model.model import class_model
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from evaluation.metrics import ModelEvaluator
from explainability.grad_cam import GradCAM, GradCAMPlusPlus, overlay_heatmap
from explainability.shap_explainer import SHAPExplainer
from explainability.lime_explainer import LIMEExplainer

class ComprehensiveEvaluator:
    def __init__(self, model_path, output_dir=None):
        self.model_path = model_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir or f"comprehensive_evaluation_{self.timestamp}"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directories
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{self.output_dir}/explainability").mkdir(exist_ok=True)
        Path(f"{self.output_dir}/gradcam_results").mkdir(exist_ok=True)
        Path(f"{self.output_dir}/shap_results").mkdir(exist_ok=True)
        Path(f"{self.output_dir}/lime_results").mkdir(exist_ok=True)
        Path(f"{self.output_dir}/reports").mkdir(exist_ok=True)
        
        print(f"🔬 DenLsNet Comprehensive Evaluation")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🖥️  Device: {self.device}")
    
    def load_model_and_data(self):
        """Load model and prepare data loaders"""
        print("\n" + "="*60)
        print("📥 Loading Model and Data")
        print("="*60)
        
        try:
            # Load model with PyTorch 2.6 compatibility
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            self.model = checkpoint['model']
            self.model.to(self.device)
            self.model.eval()
            
            # Get model info
            self.best_acc = checkpoint.get('best_acc', 'N/A')
            self.epoch = checkpoint.get('epoch', 'N/A')
            
            print(f"✅ Model loaded successfully")
            print(f"   📊 Best Accuracy: {self.best_acc}")
            print(f"   🔄 Epoch: {self.epoch}")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            sys.exit(1)
        
        # Load data
        try:
            # Create test data loader
            transform_test = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.dataset_mean, std=config.dataset_std)
            ])
            
            test_dataset = datasets.ImageFolder(config.valid, transform=transform_test)
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
            
            print(f"✅ Data loaders created")
            print(f"   📊 Test samples: {len(self.test_loader.dataset)}")
            print(f"   📁 Test path: {config.valid}")
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            sys.exit(1)
    
    def run_model_evaluation(self):
        """Run comprehensive model evaluation"""
        print("\n" + "="*60)
        print("📊 Model Performance Evaluation")
        print("="*60)
        
        evaluator = ModelEvaluator()
        
        # Run evaluation
        results = evaluator.evaluate_model(
            model=self.model,
            dataloader=self.test_loader,
            device=str(self.device),
            save_results=True,
            save_dir=f"{self.output_dir}/reports"
        )
        
        # Save results
        with open(f"{self.output_dir}/evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Model evaluation completed")
        accuracy = results.get('accuracy', 'N/A')
        f1_score = results.get('f1_score', 'N/A')
        auc = results.get('auc', 'N/A')
        
        if accuracy != 'N/A':
            print(f"   🎯 Accuracy: {accuracy:.4f}")
        else:
            print(f"   🎯 Accuracy: {accuracy}")
            
        if f1_score != 'N/A':
            print(f"   📈 F1-Score: {f1_score:.4f}")
        else:
            print(f"   📈 F1-Score: {f1_score}")
            
        if auc != 'N/A':
            print(f"   🔍 AUC: {auc:.4f}")
        else:
            print(f"   🔍 AUC: {auc}")
        
        return results
    
    def initialize_explainers(self):
        """Initialize explainability tools"""
        print("\n" + "="*60)
        print("🧠 Initializing Explainability Tools")
        print("="*60)
        
        try:
            # Initialize Grad-CAM
            self.gradcam = GradCAM(self.model, target_layer_name='densenet.features.norm5')
            self.gradcam_plus = GradCAMPlusPlus(self.model, target_layer_name='densenet.features.norm5')
            print("✅ Grad-CAM and Grad-CAM++ initialized")
            
            # Create background data for SHAP
            background_data = []
            for i, (images, _) in enumerate(self.test_loader):
                background_data.append(images)
                if i >= 2:  # Use first 3 batches as background
                    break
            background_data = torch.cat(background_data, dim=0)[:50].to(self.device)
            
            self.shap_explainer = SHAPExplainer(self.model, background_data, str(self.device))
            print("✅ SHAP explainer initialized")
            
            # Initialize LIME
            self.lime_explainer = LIMEExplainer(self.model, str(self.device), num_samples=100)
            print("✅ LIME explainer initialized")
            
        except Exception as e:
            print(f"❌ Error initializing explainers: {str(e)}")
            return False
        
        return True
    
    def run_explainability_analysis(self, num_samples=20):
        """Run comprehensive explainability analysis"""
        print("\n" + "="*60)
        print(f"🔍 Explainability Analysis ({num_samples} samples)")
        print("="*60)
        
        analysis_results = {
            'gradcam_results': [],
            'shap_results': [],
            'lime_results': [],
            'sample_info': []
        }
        
        sample_count = 0
        
        for batch_idx, (images, labels) in enumerate(tqdm(self.test_loader, desc="Processing samples")):
            if sample_count >= num_samples:
                break
                
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(images)
                probabilities = F.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
            
            for i in range(images.shape[0]):
                if sample_count >= num_samples:
                    break
                
                image = images[i:i+1]
                true_label = labels[i].item()
                pred_label = predicted_classes[i].item()
                confidence = probabilities[i, pred_label].item()
                
                # Store sample info
                sample_info = {
                    'sample_id': sample_count,
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'confidence': confidence,
                    'correct': true_label == pred_label
                }
                analysis_results['sample_info'].append(sample_info)
                
                # Convert image for visualization
                img_np = image[0].cpu().numpy().transpose(1, 2, 0)
                # Denormalize
                mean = np.array(config.dataset_mean)
                std = np.array(config.dataset_std)
                img_np = img_np * std + mean
                img_np = np.clip(img_np, 0, 1)
                
                try:
                    # Grad-CAM analysis
                    gradcam_heatmap = self.gradcam.generate_cam(image, pred_label)
                    gradcam_plus_heatmap = self.gradcam_plus.generate_cam(image, pred_label)
                    
                    # Save Grad-CAM results
                    self.save_gradcam_results(
                        img_np, gradcam_heatmap, gradcam_plus_heatmap,
                        sample_count, sample_info
                    )
                    
                    analysis_results['gradcam_results'].append({
                        'sample_id': sample_count,
                        'gradcam_saved': True,
                        'gradcam_plus_saved': True
                    })
                    
                except Exception as e:
                    print(f"⚠️  Grad-CAM failed for sample {sample_count}: {str(e)}")
                    analysis_results['gradcam_results'].append({
                        'sample_id': sample_count,
                        'gradcam_saved': False,
                        'error': str(e)
                    })
                
                try:
                    # SHAP analysis
                    shap_values = self.shap_explainer.explain_image(image, pred_label)
                    
                    # Save SHAP results
                    self.save_shap_results(img_np, shap_values, sample_count, sample_info)
                    
                    analysis_results['shap_results'].append({
                        'sample_id': sample_count,
                        'shap_saved': True
                    })
                    
                except Exception as e:
                    print(f"⚠️  SHAP failed for sample {sample_count}: {str(e)}")
                    analysis_results['shap_results'].append({
                        'sample_id': sample_count,
                        'shap_saved': False,
                        'error': str(e)
                    })
                
                try:
                    # LIME analysis
                    explanation, segments = self.lime_explainer.explain_image(img_np)
                    
                    # Save LIME results
                    self.save_lime_results(img_np, explanation, pred_label, sample_count, sample_info)
                    
                    analysis_results['lime_results'].append({
                        'sample_id': sample_count,
                        'lime_saved': True
                    })
                    
                except Exception as e:
                    print(f"⚠️  LIME failed for sample {sample_count}: {str(e)}")
                    analysis_results['lime_results'].append({
                        'sample_id': sample_count,
                        'lime_saved': False,
                        'error': str(e)
                    })
                
                sample_count += 1
        
        # Save analysis results
        with open(f"{self.output_dir}/explainability_results.json", 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        print(f"✅ Explainability analysis completed for {sample_count} samples")
        return analysis_results
    
    def save_gradcam_results(self, original_img, gradcam_heatmap, gradcam_plus_heatmap, sample_id, sample_info):
        """Save Grad-CAM visualization results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Grad-CAM heatmap
        axes[0, 1].imshow(gradcam_heatmap, cmap='jet')
        axes[0, 1].set_title('Grad-CAM Heatmap')
        axes[0, 1].axis('off')
        
        # Grad-CAM++ heatmap
        axes[0, 2].imshow(gradcam_plus_heatmap, cmap='jet')
        axes[0, 2].set_title('Grad-CAM++ Heatmap')
        axes[0, 2].axis('off')
        
        # Overlays
        gradcam_overlay = overlay_heatmap(original_img, gradcam_heatmap)
        gradcam_plus_overlay = overlay_heatmap(original_img, gradcam_plus_heatmap)
        
        axes[1, 0].imshow(gradcam_overlay)
        axes[1, 0].set_title('Grad-CAM Overlay')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(gradcam_plus_overlay)
        axes[1, 1].set_title('Grad-CAM++ Overlay')
        axes[1, 1].axis('off')
        
        # Sample info
        info_text = f"Sample {sample_id}\n"
        info_text += f"True: {'Benign' if sample_info['true_label'] == 0 else 'Malignant'}\n"
        info_text += f"Pred: {'Benign' if sample_info['predicted_label'] == 0 else 'Malignant'}\n"
        info_text += f"Conf: {sample_info['confidence']:.3f}\n"
        info_text += f"Correct: {sample_info['correct']}"
        
        axes[1, 2].text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/gradcam_results/sample_{sample_id:03d}_gradcam.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_shap_results(self, original_img, shap_values, sample_id, sample_info):
        """Save SHAP visualization results"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # SHAP positive contributions
        shap_positive = np.maximum(shap_values, 0)
        shap_positive = np.sum(shap_positive, axis=0)
        shap_positive = (shap_positive - shap_positive.min()) / (shap_positive.max() - shap_positive.min() + 1e-8)
        
        axes[1].imshow(shap_positive, cmap='Reds')
        axes[1].set_title('SHAP Positive Contributions')
        axes[1].axis('off')
        
        # SHAP negative contributions
        shap_negative = np.minimum(shap_values, 0)
        shap_negative = np.sum(np.abs(shap_negative), axis=0)
        shap_negative = (shap_negative - shap_negative.min()) / (shap_negative.max() - shap_negative.min() + 1e-8)
        
        axes[2].imshow(shap_negative, cmap='Blues')
        axes[2].set_title('SHAP Negative Contributions')
        axes[2].axis('off')
        
        plt.suptitle(f"Sample {sample_id} - SHAP Analysis")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/shap_results/sample_{sample_id:03d}_shap.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_lime_results(self, original_img, explanation, pred_label, sample_id, sample_info):
        """Save LIME visualization results"""
        try:
            from skimage.segmentation import mark_boundaries
            
            # Get LIME explanation
            temp, mask = explanation.get_image_and_mask(
                pred_label, positive_only=False, num_features=10, hide_rest=True
            )
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(original_img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # LIME explanation
            axes[1].imshow(mark_boundaries(temp, mask))
            axes[1].set_title('LIME Explanation')
            axes[1].axis('off')
            
            # LIME mask only
            axes[2].imshow(mask, cmap='RdYlBu')
            axes[2].set_title('LIME Importance Mask')
            axes[2].axis('off')
            
            plt.suptitle(f"Sample {sample_id} - LIME Analysis")
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/lime_results/sample_{sample_id:03d}_lime.png", 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error saving LIME results for sample {sample_id}: {str(e)}")
    
    def generate_comprehensive_report(self, eval_results, explainability_results):
        """Generate comprehensive evaluation report"""
        print("\n" + "="*60)
        print("📋 Generating Comprehensive Report")
        print("="*60)
        
        report = f"""
# DenLsNet Comprehensive Evaluation Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Model Path:** {self.model_path}
**Device:** {self.device}

## Model Information
- **Best Training Accuracy:** {self.best_acc}
- **Training Epoch:** {self.epoch}
- **Architecture:** DenLsNet (DenseNet-201 + SE + iAFF + LSTM)

## Performance Metrics"""
        
        # Format metrics safely
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'auc']
        for metric in metrics:
            value = eval_results.get(metric, 'N/A')
            if value != 'N/A' and isinstance(value, (int, float)):
                report += f"\n- **{metric.replace('_', ' ').title()}:** {value:.4f}"
            else:
                report += f"\n- **{metric.replace('_', ' ').title()}:** {value}"
        
        report += f"""

## Explainability Analysis Summary
- **Total Samples Analyzed:** {len(explainability_results['sample_info'])}
- **Grad-CAM Success Rate:** {sum(1 for r in explainability_results['gradcam_results'] if r.get('gradcam_saved', False)) / len(explainability_results['gradcam_results']) * 100:.1f}%
- **SHAP Success Rate:** {sum(1 for r in explainability_results['shap_results'] if r.get('shap_saved', False)) / len(explainability_results['shap_results']) * 100:.1f}%
- **LIME Success Rate:** {sum(1 for r in explainability_results['lime_results'] if r.get('lime_saved', False)) / len(explainability_results['lime_results']) * 100:.1f}%

## Sample Analysis
"""
        
        # Add sample-by-sample analysis
        for sample in explainability_results['sample_info'][:10]:  # First 10 samples
            report += f"""
### Sample {sample['sample_id']}
- **True Label:** {'Benign' if sample['true_label'] == 0 else 'Malignant'}
- **Predicted Label:** {'Benign' if sample['predicted_label'] == 0 else 'Malignant'}
- **Confidence:** {sample['confidence']:.3f}
- **Correct:** {sample['correct']}
"""
        
        report += f"""
## Files Generated
- **Evaluation Results:** `evaluation_results.json`
- **Explainability Results:** `explainability_results.json`
- **Grad-CAM Visualizations:** `gradcam_results/` directory
- **SHAP Visualizations:** `shap_results/` directory
- **LIME Visualizations:** `lime_results/` directory
- **Performance Reports:** `reports/` directory

## Usage Instructions
1. Review the performance metrics above
2. Examine individual sample visualizations in the respective directories
3. Use the JSON files for further analysis or integration
4. Check the `reports/` directory for detailed performance analysis

---
*Generated by DenLsNet Comprehensive Evaluation System*
"""
        
        # Save report
        with open(f"{self.output_dir}/comprehensive_report.md", 'w') as f:
            f.write(report)
        
        print(f"✅ Comprehensive report saved")
        print(f"📁 Report location: {self.output_dir}/comprehensive_report.md")
    
    def run_complete_evaluation(self, num_explainability_samples=20):
        """Run the complete evaluation pipeline"""
        print("🚀 Starting Comprehensive DenLsNet Evaluation")
        print("="*80)
        
        # Load model and data
        self.load_model_and_data()
        
        # Run model evaluation
        eval_results = self.run_model_evaluation()
        
        # Initialize explainers
        if self.initialize_explainers():
            # Run explainability analysis
            explainability_results = self.run_explainability_analysis(num_explainability_samples)
        else:
            explainability_results = {'sample_info': [], 'gradcam_results': [], 'shap_results': [], 'lime_results': []}
        
        # Generate comprehensive report
        self.generate_comprehensive_report(eval_results, explainability_results)
        
        print("\n" + "="*80)
        print("🎉 Comprehensive Evaluation Completed!")
        print(f"📁 All results saved to: {self.output_dir}")
        print("="*80)
        
        return eval_results, explainability_results


def main():
    """Main function to run comprehensive evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DenLsNet Comprehensive Evaluation')
    parser.add_argument('--model_path', type=str, default='weight/save/40/iaff40_5.pth',
                       help='Path to the saved model')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of samples for explainability analysis')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        print("Please ensure the model file exists or train a model first.")
        sys.exit(1)
    
    # Run comprehensive evaluation
    evaluator = ComprehensiveEvaluator(args.model_path, args.output_dir)
    eval_results, explainability_results = evaluator.run_complete_evaluation(args.num_samples)
    
    print(f"\n🎯 Final Results Summary:")
    
    accuracy = eval_results.get('accuracy', 'N/A')
    f1_score = eval_results.get('f1_score', 'N/A')
    auc = eval_results.get('auc', 'N/A')
    
    if accuracy != 'N/A':
        print(f"   Accuracy: {accuracy:.4f}")
    else:
        print(f"   Accuracy: {accuracy}")
        
    if f1_score != 'N/A':
        print(f"   F1-Score: {f1_score:.4f}")
    else:
        print(f"   F1-Score: {f1_score}")
        
    if auc != 'N/A':
        print(f"   AUC: {auc:.4f}")
    else:
        print(f"   AUC: {auc}")
        
    print(f"   Explainability samples: {len(explainability_results['sample_info'])}")


if __name__ == "__main__":
    main()