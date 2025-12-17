"""
Comprehensive XAI Evaluation Pipeline
Evaluates Grad-CAM, SHAP, and LIME using quantitative metrics
"""
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import metric classes
from xai.metrics.insertion_auc import InsertionAUC
from xai.metrics.deletion_auc import DeletionAUC
from xai.metrics.iou import IoUMetric
from xai.metrics.stability import StabilityMetric

# Import explainability methods
from explainability.grad_cam import GradCAM, GradCAMPlusPlus
try:
    from explainability.shap_explainer import SHAPExplainer
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
from explainability.lime_explainer import LIMEExplainer


class XAIEvaluator:
    """Comprehensive XAI evaluation pipeline"""
    
    def __init__(self, model, device, class_names: List[str], 
                 target_layer: str = 'features.norm5', results_dir: str = 'results'):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.target_layer = target_layer
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize metrics
        self.insertion_auc = InsertionAUC(model, device)
        self.deletion_auc = DeletionAUC(model, device)
        self.iou_metric = IoUMetric()
        self.stability_metric = StabilityMetric(device)
        
        # Initialize explainers
        self.explainers = self._initialize_explainers()
        
        # Results storage
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_type': str(type(model).__name__),
                'device': str(device),
                'target_layer': target_layer,
                'class_names': class_names
            },
            'per_image_results': [],
            'aggregated_results': {}
        }
    
    def _initialize_explainers(self) -> Dict[str, Any]:
        """Initialize all available explainers"""
        explainers = {}
        
        # Grad-CAM
        try:
            explainers['gradcam'] = GradCAM(self.model, self.target_layer)
            print("✅ Grad-CAM initialized")
        except Exception as e:
            print(f"❌ Grad-CAM initialization failed: {e}")
        
        # Grad-CAM++
        try:
            explainers['gradcam_plus'] = GradCAMPlusPlus(self.model, self.target_layer)
            print("✅ Grad-CAM++ initialized")
        except Exception as e:
            print(f"❌ Grad-CAM++ initialization failed: {e}")
        
        # SHAP
        if SHAP_AVAILABLE:
            try:
                explainers['shap'] = SHAPExplainer(self.model, str(self.device))
                print("✅ SHAP initialized")
            except Exception as e:
                print(f"❌ SHAP initialization failed: {e}")
        else:
            print("⚠️ SHAP not available")
        
        # LIME
        try:
            explainers['lime'] = LIMEExplainer(self.model, str(self.device))
            print("✅ LIME initialized")
        except Exception as e:
            print(f"❌ LIME initialization failed: {e}")
        
        return explainers
    
    def generate_explanation(self, explainer_name: str, image: torch.Tensor, 
                           target_class: int) -> Optional[np.ndarray]:
        """Generate explanation using specified explainer"""
        if explainer_name not in self.explainers:
            return None
        
        explainer = self.explainers[explainer_name]
        
        try:
            if explainer_name in ['gradcam', 'gradcam_plus']:
                explanation = explainer.generate_cam(image, target_class)
            elif explainer_name == 'shap':
                # SHAP returns different format, need to adapt
                explanation = explainer.explain_image(image.cpu().numpy(), target_class)
                if isinstance(explanation, list):
                    explanation = explanation[0] if explanation else np.zeros((224, 224))
                if len(explanation.shape) == 3:
                    explanation = np.sum(np.abs(explanation), axis=0)
            elif explainer_name == 'lime':
                # Convert tensor to numpy for LIME
                img_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                # Denormalize for LIME (assuming ImageNet normalization)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = img_np * std + mean
                img_np = np.clip(img_np, 0, 1)
                
                lime_explanation, _ = explainer.explain_image(img_np)
                temp, mask = lime_explanation.get_image_and_mask(
                    target_class, positive_only=False, num_features=10, hide_rest=False
                )
                explanation = mask.astype(np.float32)
            else:
                return None
            
            # Ensure explanation is 2D numpy array
            if isinstance(explanation, torch.Tensor):
                explanation = explanation.cpu().numpy()
            
            if len(explanation.shape) > 2:
                explanation = np.sum(np.abs(explanation), axis=0)
            
            return explanation
            
        except Exception as e:
            print(f"Error generating {explainer_name} explanation: {e}")
            return None
    
    def evaluate_single_image(self, image: torch.Tensor, true_class: int, 
                            image_id: str = None) -> Dict[str, Any]:
        """Evaluate all XAI methods on a single image"""
        if image_id is None:
            image_id = f"image_{len(self.results['per_image_results'])}"
        
        # Get model prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
            probs = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, predicted_class].item()
        
        image_results = {
            'image_id': image_id,
            'true_class': true_class,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'correct_prediction': true_class == predicted_class,
            'explainer_results': {}
        }
        
        # Evaluate each explainer
        for explainer_name in self.explainers.keys():
            print(f"Evaluating {explainer_name} for {image_id}...")
            
            # Generate explanation
            explanation = self.generate_explanation(explainer_name, image, predicted_class)
            
            if explanation is None:
                print(f"Failed to generate {explainer_name} explanation")
                continue
            
            # Evaluate metrics
            explainer_results = {}
            
            # Insertion AUC
            try:
                insertion_auc, _ = self.insertion_auc.compute_insertion_auc(
                    image, explanation, predicted_class)
                explainer_results['insertion_auc'] = insertion_auc
            except Exception as e:
                print(f"Insertion AUC failed for {explainer_name}: {e}")
                explainer_results['insertion_auc'] = 0.0
            
            # Deletion AUC
            try:
                deletion_auc, _ = self.deletion_auc.compute_deletion_auc(
                    image, explanation, predicted_class)
                explainer_results['deletion_auc'] = deletion_auc
            except Exception as e:
                print(f"Deletion AUC failed for {explainer_name}: {e}")
                explainer_results['deletion_auc'] = 1.0  # Worst case for deletion
            
            # IoU (using pseudo-ROI approach since we don't have ground truth)
            try:
                # Create pseudo-ROI from the explanation itself (top 20% pixels)
                pseudo_roi = self.iou_metric.create_pseudo_roi(explanation, top_k_percent=0.2)
                iou_score = self.iou_metric.compute_iou_with_roi(explanation, pseudo_roi)
                explainer_results['iou'] = iou_score
            except Exception as e:
                print(f"IoU failed for {explainer_name}: {e}")
                explainer_results['iou'] = 0.0
            
            # Stability
            try:
                def explanation_generator(img, target):
                    return self.generate_explanation(explainer_name, img, target)
                
                stability, _ = self.stability_metric.evaluate_stability(
                    image, explanation_generator, predicted_class, similarity_metric='ssim')
                explainer_results['stability'] = stability
            except Exception as e:
                print(f"Stability failed for {explainer_name}: {e}")
                explainer_results['stability'] = 0.0
            
            image_results['explainer_results'][explainer_name] = explainer_results
        
        return image_results
    
    def evaluate_batch(self, images: List[torch.Tensor], true_classes: List[int],
                      image_ids: List[str] = None) -> None:
        """Evaluate XAI methods on a batch of images"""
        if image_ids is None:
            image_ids = [f"image_{i}" for i in range(len(images))]
        
        print(f"🔍 Evaluating XAI methods on {len(images)} images...")
        
        for i, (image, true_class, image_id) in enumerate(zip(images, true_classes, image_ids)):
            print(f"\nProcessing {image_id} ({i+1}/{len(images)})...")
            
            # Ensure image has batch dimension
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            
            image_results = self.evaluate_single_image(image, true_class, image_id)
            self.results['per_image_results'].append(image_results)
    
    def compute_aggregated_results(self) -> Dict[str, Any]:
        """Compute aggregated statistics across all evaluated images"""
        if not self.results['per_image_results']:
            return {}
        
        # Initialize aggregated results structure
        aggregated = {
            'overall': {},
            'per_class': {},
            'summary_table': {}
        }
        
        # Get all explainer names
        explainer_names = set()
        for result in self.results['per_image_results']:
            explainer_names.update(result['explainer_results'].keys())
        
        explainer_names = list(explainer_names)
        
        # Compute overall statistics
        for explainer_name in explainer_names:
            metrics = ['insertion_auc', 'deletion_auc', 'iou', 'stability']
            explainer_stats = {}
            
            for metric in metrics:
                values = []
                for result in self.results['per_image_results']:
                    if explainer_name in result['explainer_results']:
                        value = result['explainer_results'][explainer_name].get(metric, 0.0)
                        values.append(value)
                
                if values:
                    explainer_stats[metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'count': len(values)
                    }
                else:
                    explainer_stats[metric] = {
                        'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0
                    }
            
            aggregated['overall'][explainer_name] = explainer_stats
        
        # Compute per-class statistics
        for class_idx, class_name in enumerate(self.class_names):
            class_results = [r for r in self.results['per_image_results'] 
                           if r['true_class'] == class_idx]
            
            if not class_results:
                continue
            
            aggregated['per_class'][class_name] = {}
            
            for explainer_name in explainer_names:
                metrics = ['insertion_auc', 'deletion_auc', 'iou', 'stability']
                explainer_stats = {}
                
                for metric in metrics:
                    values = []
                    for result in class_results:
                        if explainer_name in result['explainer_results']:
                            value = result['explainer_results'][explainer_name].get(metric, 0.0)
                            values.append(value)
                    
                    if values:
                        explainer_stats[metric] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'count': len(values)
                        }
                    else:
                        explainer_stats[metric] = {
                            'mean': 0.0, 'std': 0.0, 'count': 0
                        }
                
                aggregated['per_class'][class_name][explainer_name] = explainer_stats
        
        # Create summary table
        summary_table = []
        for explainer_name in explainer_names:
            if explainer_name in aggregated['overall']:
                stats = aggregated['overall'][explainer_name]
                row = {
                    'XAI Method': explainer_name.upper(),
                    'Insertion AUC': f"{stats['insertion_auc']['mean']:.3f}",
                    'Deletion AUC': f"{stats['deletion_auc']['mean']:.3f}",
                    'IoU': f"{stats['iou']['mean']:.3f}",
                    'Stability': f"{stats['stability']['mean']:.3f}"
                }
                summary_table.append(row)
        
        aggregated['summary_table'] = summary_table
        
        self.results['aggregated_results'] = aggregated
        return aggregated
    
    def save_results(self, filename: str = None) -> str:
        """Save evaluation results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"xai_evaluation_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"✅ Results saved to {filepath}")
        return str(filepath)
    
    def save_csv_results(self, filename: str = None) -> str:
        """Save per-image results to CSV file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"xai_evaluation_{timestamp}.csv"
        
        filepath = self.results_dir / filename
        
        # Flatten results for CSV
        csv_data = []
        for result in self.results['per_image_results']:
            base_row = {
                'image_id': result['image_id'],
                'true_class': result['true_class'],
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence'],
                'correct_prediction': result['correct_prediction']
            }
            
            for explainer_name, explainer_results in result['explainer_results'].items():
                row = base_row.copy()
                row['explainer'] = explainer_name
                row.update(explainer_results)
                csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(filepath, index=False)
        
        print(f"✅ CSV results saved to {filepath}")
        return str(filepath)
    
    def print_summary_table(self) -> None:
        """Print formatted summary table"""
        if 'aggregated_results' not in self.results:
            self.compute_aggregated_results()
        
        summary_table = self.results['aggregated_results']['summary_table']
        
        if not summary_table:
            print("No results to display")
            return
        
        print("\n" + "="*80)
        print("📊 XAI EVALUATION SUMMARY")
        print("="*80)
        
        # Create DataFrame for nice formatting
        df = pd.DataFrame(summary_table)
        print(df.to_string(index=False))
        
        print("\n📝 Interpretation:")
        print("• Insertion AUC: Higher is better (faithfulness)")
        print("• Deletion AUC: Lower is better (faithfulness)")  
        print("• IoU: Higher is better (localization)")
        print("• Stability: Higher is better (robustness)")
        print("="*80)
    
    def create_visualization(self, save_path: str = None) -> None:
        """Create visualization of evaluation results"""
        if 'aggregated_results' not in self.results:
            self.compute_aggregated_results()
        
        aggregated = self.results['aggregated_results']['overall']
        
        if not aggregated:
            print("No results to visualize")
            return
        
        # Prepare data for plotting
        explainers = list(aggregated.keys())
        metrics = ['insertion_auc', 'deletion_auc', 'iou', 'stability']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            means = [aggregated[exp][metric]['mean'] for exp in explainers]
            stds = [aggregated[exp][metric]['std'] for exp in explainers]
            
            bars = axes[i].bar(explainers, means, yerr=stds, capsize=5, alpha=0.7)
            axes[i].set_title(f'{metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{mean:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.results_dir / f"xai_evaluation_plot_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualization saved to {save_path}")


def main():
    """Example usage of XAI evaluation pipeline"""
    print("🔬 XAI Evaluation Pipeline")
    print("This is an example - integrate with your actual model and data")
    
    # Example setup (replace with your actual model and data)
    # model = your_trained_model
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # class_names = ['Benign', 'Malignant']  # or your 8 subclasses
    
    # evaluator = XAIEvaluator(model, device, class_names)
    
    # Example evaluation
    # images = [your_test_images]
    # true_classes = [your_true_labels]
    # image_ids = [your_image_ids]
    
    # evaluator.evaluate_batch(images, true_classes, image_ids)
    # evaluator.compute_aggregated_results()
    # evaluator.print_summary_table()
    # evaluator.save_results()
    # evaluator.save_csv_results()
    # evaluator.create_visualization()


if __name__ == "__main__":
    main()