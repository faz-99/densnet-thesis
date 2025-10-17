"""
Comprehensive interpretability framework for DenLsNet-XAI
Implements quantitative evaluation of explainability methods
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Dict, List, Tuple, Optional, Union
import os
import json
from datetime import datetime
from sklearn.metrics import auc
import seaborn as sns
from scipy import ndimage
from skimage.segmentation import slic
from skimage.measure import regionprops

from .grad_cam import GradCAM, GradCAMPlusPlus
from .shap_explainer import SHAPExplainer
from .lime_explainer import LIMEExplainer


class InterpretabilityFramework:
    """
    Comprehensive framework for quantitative interpretability evaluation
    """
    
    def __init__(self, 
                 model, 
                 device: str = 'cpu',
                 class_names: List[str] = None):
        """
        Initialize interpretability framework
        
        Args:
            model: Trained model
            device: Computing device
            class_names: List of class names
        """
        self.model = model
        self.device = device
        self.class_names = class_names or [f'Class_{i}' for i in range(8)]
        
        # Initialize explainers
        self.explainers = self._initialize_explainers()
        
        # Metrics storage
        self.quantitative_results = {}
        self.stability_results = {}
        
    def _initialize_explainers(self) -> Dict:
        """Initialize all explainability methods"""
        explainers = {}
        
        try:
            # Grad-CAM variants
            explainers['gradcam'] = GradCAM(
                self.model, 
                target_layer_name='densenet.features.norm5'
            )
            explainers['gradcam_plus'] = GradCAMPlusPlus(
                self.model,
                target_layer_name='densenet.features.norm5'
            )
            
            # SHAP (with dummy background)
            background_data = torch.randn(50, 3, 224, 224).to(self.device)
            explainers['shap'] = SHAPExplainer(
                self.model, 
                background_data, 
                str(self.device)
            )
            
            # LIME
            explainers['lime'] = LIMEExplainer(
                self.model,
                str(self.device),
                num_samples=500
            )
            
        except Exception as e:
            print(f"Warning: Failed to initialize some explainers: {e}")
        
        return explainers
    
    def generate_explanations(self, 
                            image: torch.Tensor, 
                            target_class: int,
                            methods: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Generate explanations using specified methods
        
        Args:
            image: Input image tensor (1, 3, H, W)
            target_class: Target class for explanation
            methods: List of methods to use
            
        Returns:
            Dictionary of explanation maps
        """
        if methods is None:
            methods = list(self.explainers.keys())
        
        explanations = {}
        
        for method in methods:
            if method not in self.explainers:
                print(f"Warning: Method {method} not available")
                continue
            
            try:
                if method in ['gradcam', 'gradcam_plus']:
                    explanation = self.explainers[method].generate_cam(image, target_class)
                elif method == 'shap':
                    explanation = self.explainers[method].explain_image(image, target_class)
                    # Convert SHAP values to heatmap
                    if explanation is not None:
                        explanation = np.sum(np.abs(explanation), axis=0)
                        explanation = (explanation - explanation.min()) / (explanation.max() - explanation.min() + 1e-8)
                elif method == 'lime':
                    # Convert tensor to numpy for LIME
                    image_np = image[0].cpu().numpy().transpose(1, 2, 0)
                    # Denormalize for LIME
                    mean = np.array([0.5613, 0.5778, 0.6032])
                    std = np.array([0.2114, 0.1957, 0.1590])
                    image_np = image_np * std + mean
                    image_np = np.clip(image_np, 0, 1)
                    
                    lime_explanation, segments = self.explainers[method].explain_image(image_np)
                    temp, mask = lime_explanation.get_image_and_mask(
                        target_class, positive_only=False, num_features=10, hide_rest=False
                    )
                    explanation = mask.astype(np.float32)
                    explanation = (explanation - explanation.min()) / (explanation.max() - explanation.min() + 1e-8)
                
                explanations[method] = explanation
                
            except Exception as e:
                print(f"Warning: Failed to generate {method} explanation: {e}")
        
        return explanations
    
    def evaluate_insertion_deletion(self, 
                                  image: torch.Tensor,
                                  explanation: np.ndarray,
                                  target_class: int,
                                  num_steps: int = 50) -> Tuple[float, float]:
        """
        Evaluate explanation using insertion/deletion curves
        
        Args:
            image: Input image tensor
            explanation: Explanation heatmap
            target_class: Target class
            num_steps: Number of steps for curve
            
        Returns:
            Tuple of (insertion_auc, deletion_auc)
        """
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(image)
            baseline_prob = F.softmax(baseline_output, dim=1)[0, target_class].item()
        
        # Resize explanation to match image
        if explanation.shape != (224, 224):
            explanation = cv2.resize(explanation, (224, 224))
        
        # Get pixel importance order
        flat_explanation = explanation.flatten()
        sorted_indices = np.argsort(flat_explanation)
        
        # Insertion curve (start with black image, add important pixels)
        insertion_probs = []
        masked_image = torch.zeros_like(image)
        
        for step in range(num_steps + 1):
            if step == 0:
                prob = 0.0  # Black image
            else:
                # Add most important pixels
                num_pixels = int((step / num_steps) * len(sorted_indices))
                pixels_to_add = sorted_indices[-num_pixels:]
                
                # Create mask
                mask = np.zeros_like(flat_explanation)
                mask[pixels_to_add] = 1
                mask = mask.reshape(224, 224)
                
                # Apply mask
                for c in range(3):
                    masked_image[0, c] = image[0, c] * torch.from_numpy(mask).to(self.device)
                
                with torch.no_grad():
                    output = self.model(masked_image)
                    prob = F.softmax(output, dim=1)[0, target_class].item()
            
            insertion_probs.append(prob)
        
        # Deletion curve (start with full image, remove important pixels)
        deletion_probs = []
        masked_image = image.clone()
        
        for step in range(num_steps + 1):
            if step == 0:
                prob = baseline_prob  # Full image
            else:
                # Remove most important pixels
                num_pixels = int((step / num_steps) * len(sorted_indices))
                pixels_to_remove = sorted_indices[-num_pixels:]
                
                # Create mask
                mask = np.ones_like(flat_explanation)
                mask[pixels_to_remove] = 0
                mask = mask.reshape(224, 224)
                
                # Apply mask
                for c in range(3):
                    masked_image[0, c] = image[0, c] * torch.from_numpy(mask).to(self.device)
                
                with torch.no_grad():
                    output = self.model(masked_image)
                    prob = F.softmax(output, dim=1)[0, target_class].item()
            
            deletion_probs.append(prob)
        
        # Calculate AUC
        x = np.linspace(0, 1, num_steps + 1)
        insertion_auc = auc(x, insertion_probs)
        deletion_auc = auc(x, deletion_probs)
        
        return insertion_auc, deletion_auc
    
    def evaluate_localization_accuracy(self, 
                                     explanation: np.ndarray,
                                     ground_truth_mask: np.ndarray = None) -> Dict[str, float]:
        """
        Evaluate localization accuracy if ground truth masks are available
        
        Args:
            explanation: Explanation heatmap
            ground_truth_mask: Ground truth ROI mask
            
        Returns:
            Dictionary of localization metrics
        """
        if ground_truth_mask is None:
            return {'iou': None, 'dice': None, 'precision': None, 'recall': None}
        
        # Threshold explanation to create binary mask
        threshold = np.percentile(explanation, 80)  # Top 20% of pixels
        explanation_mask = (explanation >= threshold).astype(np.uint8)
        
        # Resize to match ground truth
        if explanation_mask.shape != ground_truth_mask.shape:
            explanation_mask = cv2.resize(
                explanation_mask, 
                (ground_truth_mask.shape[1], ground_truth_mask.shape[0])
            )
        
        # Calculate metrics
        intersection = np.logical_and(explanation_mask, ground_truth_mask)
        union = np.logical_or(explanation_mask, ground_truth_mask)
        
        iou = np.sum(intersection) / (np.sum(union) + 1e-8)
        dice = 2 * np.sum(intersection) / (np.sum(explanation_mask) + np.sum(ground_truth_mask) + 1e-8)
        
        precision = np.sum(intersection) / (np.sum(explanation_mask) + 1e-8)
        recall = np.sum(intersection) / (np.sum(ground_truth_mask) + 1e-8)
        
        return {
            'iou': float(iou),
            'dice': float(dice),
            'precision': float(precision),
            'recall': float(recall)
        }
    
    def evaluate_stability(self, 
                          image: torch.Tensor,
                          target_class: int,
                          method: str,
                          num_perturbations: int = 10,
                          noise_level: float = 0.1) -> Dict[str, float]:
        """
        Evaluate explanation stability under input perturbations
        
        Args:
            image: Input image tensor
            target_class: Target class
            method: Explanation method
            num_perturbations: Number of perturbations to test
            noise_level: Noise level for perturbations
            
        Returns:
            Stability metrics
        """
        if method not in self.explainers:
            return {'mean_correlation': None, 'std_correlation': None}
        
        # Generate baseline explanation
        baseline_explanation = self.generate_explanations(
            image, target_class, [method]
        ).get(method)
        
        if baseline_explanation is None:
            return {'mean_correlation': None, 'std_correlation': None}
        
        correlations = []
        
        for _ in range(num_perturbations):
            # Add noise to image
            noise = torch.randn_like(image) * noise_level
            perturbed_image = image + noise
            perturbed_image = torch.clamp(perturbed_image, -3, 3)  # Reasonable range for normalized images
            
            # Generate explanation for perturbed image
            perturbed_explanation = self.generate_explanations(
                perturbed_image, target_class, [method]
            ).get(method)
            
            if perturbed_explanation is not None:
                # Calculate correlation
                corr = np.corrcoef(
                    baseline_explanation.flatten(),
                    perturbed_explanation.flatten()
                )[0, 1]
                
                if not np.isnan(corr):
                    correlations.append(corr)
        
        if correlations:
            return {
                'mean_correlation': float(np.mean(correlations)),
                'std_correlation': float(np.std(correlations)),
                'num_valid': len(correlations)
            }
        else:
            return {'mean_correlation': None, 'std_correlation': None, 'num_valid': 0}
    
    def comprehensive_evaluation(self, 
                               dataloader,
                               num_samples: int = 50,
                               methods: List[str] = None,
                               save_dir: str = 'interpretability_evaluation') -> Dict:
        """
        Run comprehensive quantitative evaluation
        
        Args:
            dataloader: Data loader for evaluation
            num_samples: Number of samples to evaluate
            methods: Methods to evaluate
            save_dir: Directory to save results
            
        Returns:
            Comprehensive evaluation results
        """
        if methods is None:
            methods = list(self.explainers.keys())
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting comprehensive interpretability evaluation...")
        print(f"Methods: {methods}")
        print(f"Samples: {num_samples}")
        
        results = {method: {
            'insertion_aucs': [],
            'deletion_aucs': [],
            'stability_correlations': [],
            'localization_ious': [],
            'processing_times': []
        } for method in methods}
        
        sample_count = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            if sample_count >= num_samples:
                break
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            for i in range(images.size(0)):
                if sample_count >= num_samples:
                    break
                
                image = images[i:i+1]
                label = labels[i].item()
                
                print(f"Evaluating sample {sample_count + 1}/{num_samples}")
                
                # Generate explanations for all methods
                explanations = self.generate_explanations(image, label, methods)
                
                for method in methods:
                    if method not in explanations:
                        continue
                    
                    explanation = explanations[method]
                    
                    try:
                        # Insertion/Deletion evaluation
                        start_time = time.time()
                        ins_auc, del_auc = self.evaluate_insertion_deletion(
                            image, explanation, label
                        )
                        processing_time = time.time() - start_time
                        
                        results[method]['insertion_aucs'].append(ins_auc)
                        results[method]['deletion_aucs'].append(del_auc)
                        results[method]['processing_times'].append(processing_time)
                        
                        # Stability evaluation
                        stability = self.evaluate_stability(image, label, method)
                        if stability['mean_correlation'] is not None:
                            results[method]['stability_correlations'].append(
                                stability['mean_correlation']
                            )
                        
                    except Exception as e:
                        print(f"Warning: Evaluation failed for {method}: {e}")
                
                sample_count += 1
        
        # Calculate summary statistics
        summary_results = {}
        
        for method in methods:
            method_results = results[method]
            
            summary_results[method] = {
                'insertion_auc': {
                    'mean': float(np.mean(method_results['insertion_aucs'])) if method_results['insertion_aucs'] else None,
                    'std': float(np.std(method_results['insertion_aucs'])) if method_results['insertion_aucs'] else None,
                    'values': method_results['insertion_aucs']
                },
                'deletion_auc': {
                    'mean': float(np.mean(method_results['deletion_aucs'])) if method_results['deletion_aucs'] else None,
                    'std': float(np.std(method_results['deletion_aucs'])) if method_results['deletion_aucs'] else None,
                    'values': method_results['deletion_aucs']
                },
                'stability': {
                    'mean': float(np.mean(method_results['stability_correlations'])) if method_results['stability_correlations'] else None,
                    'std': float(np.std(method_results['stability_correlations'])) if method_results['stability_correlations'] else None,
                    'values': method_results['stability_correlations']
                },
                'processing_time': {
                    'mean': float(np.mean(method_results['processing_times'])) if method_results['processing_times'] else None,
                    'std': float(np.std(method_results['processing_times'])) if method_results['processing_times'] else None,
                    'values': method_results['processing_times']
                },
                'num_samples': len(method_results['insertion_aucs'])
            }
        
        # Save results
        results_path = os.path.join(save_dir, 'quantitative_evaluation.json')
        with open(results_path, 'w') as f:
            json.dump(summary_results, f, indent=2)
        
        # Create visualizations
        self._create_evaluation_plots(summary_results, save_dir)
        
        # Create summary report
        self._create_evaluation_report(summary_results, save_dir)
        
        print(f"Evaluation completed! Results saved to: {save_dir}")
        
        return summary_results
    
    def _create_evaluation_plots(self, results: Dict, save_dir: str):
        """Create evaluation visualization plots"""
        methods = list(results.keys())
        
        # Prepare data for plotting
        metrics = ['insertion_auc', 'deletion_auc', 'stability']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Insertion AUC comparison
        insertion_means = [results[m]['insertion_auc']['mean'] for m in methods if results[m]['insertion_auc']['mean'] is not None]
        insertion_stds = [results[m]['insertion_auc']['std'] for m in methods if results[m]['insertion_auc']['mean'] is not None]
        valid_methods = [m for m in methods if results[m]['insertion_auc']['mean'] is not None]
        
        if insertion_means:
            axes[0].bar(valid_methods, insertion_means, yerr=insertion_stds, capsize=5, alpha=0.7)
            axes[0].set_title('Insertion AUC Comparison')
            axes[0].set_ylabel('AUC')
            axes[0].tick_params(axis='x', rotation=45)
        
        # Deletion AUC comparison
        deletion_means = [results[m]['deletion_auc']['mean'] for m in methods if results[m]['deletion_auc']['mean'] is not None]
        deletion_stds = [results[m]['deletion_auc']['std'] for m in methods if results[m]['deletion_auc']['mean'] is not None]
        valid_methods_del = [m for m in methods if results[m]['deletion_auc']['mean'] is not None]
        
        if deletion_means:
            axes[1].bar(valid_methods_del, deletion_means, yerr=deletion_stds, capsize=5, alpha=0.7, color='orange')
            axes[1].set_title('Deletion AUC Comparison')
            axes[1].set_ylabel('AUC')
            axes[1].tick_params(axis='x', rotation=45)
        
        # Stability comparison
        stability_means = [results[m]['stability']['mean'] for m in methods if results[m]['stability']['mean'] is not None]
        stability_stds = [results[m]['stability']['std'] for m in methods if results[m]['stability']['mean'] is not None]
        valid_methods_stab = [m for m in methods if results[m]['stability']['mean'] is not None]
        
        if stability_means:
            axes[2].bar(valid_methods_stab, stability_means, yerr=stability_stds, capsize=5, alpha=0.7, color='green')
            axes[2].set_title('Stability (Correlation) Comparison')
            axes[2].set_ylabel('Correlation')
            axes[2].tick_params(axis='x', rotation=45)
        
        # Processing time comparison
        time_means = [results[m]['processing_time']['mean'] for m in methods if results[m]['processing_time']['mean'] is not None]
        valid_methods_time = [m for m in methods if results[m]['processing_time']['mean'] is not None]
        
        if time_means:
            axes[3].bar(valid_methods_time, time_means, alpha=0.7, color='red')
            axes[3].set_title('Processing Time Comparison')
            axes[3].set_ylabel('Time (seconds)')
            axes[3].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'quantitative_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed box plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Box plot for insertion AUC
        insertion_data = [results[m]['insertion_auc']['values'] for m in methods if results[m]['insertion_auc']['values']]
        insertion_labels = [m for m in methods if results[m]['insertion_auc']['values']]
        
        if insertion_data:
            axes[0].boxplot(insertion_data, labels=insertion_labels)
            axes[0].set_title('Insertion AUC Distribution')
            axes[0].set_ylabel('AUC')
            axes[0].tick_params(axis='x', rotation=45)
        
        # Box plot for deletion AUC
        deletion_data = [results[m]['deletion_auc']['values'] for m in methods if results[m]['deletion_auc']['values']]
        deletion_labels = [m for m in methods if results[m]['deletion_auc']['values']]
        
        if deletion_data:
            axes[1].boxplot(deletion_data, labels=deletion_labels)
            axes[1].set_title('Deletion AUC Distribution')
            axes[1].set_ylabel('AUC')
            axes[1].tick_params(axis='x', rotation=45)
        
        # Box plot for stability
        stability_data = [results[m]['stability']['values'] for m in methods if results[m]['stability']['values']]
        stability_labels = [m for m in methods if results[m]['stability']['values']]
        
        if stability_data:
            axes[2].boxplot(stability_data, labels=stability_labels)
            axes[2].set_title('Stability Distribution')
            axes[2].set_ylabel('Correlation')
            axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'distribution_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_evaluation_report(self, results: Dict, save_dir: str):
        """Create comprehensive evaluation report"""
        report = f"""# Interpretability Evaluation Report

## Analysis Overview
- **Date**: {datetime.now().isoformat()}
- **Methods Evaluated**: {', '.join(results.keys())}
- **Evaluation Metrics**: Insertion AUC, Deletion AUC, Stability

## Quantitative Results

### Summary Table
| Method | Insertion AUC | Deletion AUC | Stability | Processing Time (s) | Samples |
|--------|---------------|--------------|-----------|-------------------|---------|
"""
        
        for method, metrics in results.items():
            ins_auc = f"{metrics['insertion_auc']['mean']:.3f} ± {metrics['insertion_auc']['std']:.3f}" if metrics['insertion_auc']['mean'] else "N/A"
            del_auc = f"{metrics['deletion_auc']['mean']:.3f} ± {metrics['deletion_auc']['std']:.3f}" if metrics['deletion_auc']['mean'] else "N/A"
            stability = f"{metrics['stability']['mean']:.3f} ± {metrics['stability']['std']:.3f}" if metrics['stability']['mean'] else "N/A"
            proc_time = f"{metrics['processing_time']['mean']:.3f}" if metrics['processing_time']['mean'] else "N/A"
            samples = metrics['num_samples']
            
            report += f"| {method} | {ins_auc} | {del_auc} | {stability} | {proc_time} | {samples} |\n"
        
        report += f"""
### Metric Interpretations

#### Insertion AUC
- **Higher is better** (closer to 1.0)
- Measures how quickly prediction confidence increases when adding important pixels
- Good explanations should achieve high confidence with fewer pixels

#### Deletion AUC  
- **Lower is better** (closer to 0.0)
- Measures how quickly prediction confidence decreases when removing important pixels
- Good explanations should cause rapid confidence drop when important regions are removed

#### Stability
- **Higher is better** (closer to 1.0)
- Measures consistency of explanations under input perturbations
- Stable explanations should be robust to small input changes

### Method Rankings

"""
        
        # Rank methods by each metric
        valid_methods = [m for m in results.keys() if results[m]['insertion_auc']['mean'] is not None]
        
        if valid_methods:
            # Insertion AUC ranking (higher is better)
            ins_ranking = sorted(valid_methods, key=lambda m: results[m]['insertion_auc']['mean'], reverse=True)
            report += "#### Insertion AUC Ranking\n"
            for i, method in enumerate(ins_ranking, 1):
                score = results[method]['insertion_auc']['mean']
                report += f"{i}. {method}: {score:.3f}\n"
            
            # Deletion AUC ranking (lower is better)
            del_ranking = sorted(valid_methods, key=lambda m: results[m]['deletion_auc']['mean'])
            report += "\n#### Deletion AUC Ranking\n"
            for i, method in enumerate(del_ranking, 1):
                score = results[method]['deletion_auc']['mean']
                report += f"{i}. {method}: {score:.3f}\n"
            
            # Stability ranking (higher is better)
            stab_methods = [m for m in valid_methods if results[m]['stability']['mean'] is not None]
            if stab_methods:
                stab_ranking = sorted(stab_methods, key=lambda m: results[m]['stability']['mean'], reverse=True)
                report += "\n#### Stability Ranking\n"
                for i, method in enumerate(stab_ranking, 1):
                    score = results[method]['stability']['mean']
                    report += f"{i}. {method}: {score:.3f}\n"
        
        report += f"""
## Recommendations

Based on the quantitative evaluation:

"""
        
        # Generate recommendations
        if valid_methods:
            best_insertion = max(valid_methods, key=lambda m: results[m]['insertion_auc']['mean'])
            best_deletion = min(valid_methods, key=lambda m: results[m]['deletion_auc']['mean'])
            
            report += f"- **Best for Insertion**: {best_insertion} (AUC: {results[best_insertion]['insertion_auc']['mean']:.3f})\n"
            report += f"- **Best for Deletion**: {best_deletion} (AUC: {results[best_deletion]['deletion_auc']['mean']:.3f})\n"
            
            stab_methods = [m for m in valid_methods if results[m]['stability']['mean'] is not None]
            if stab_methods:
                best_stability = max(stab_methods, key=lambda m: results[m]['stability']['mean'])
                report += f"- **Most Stable**: {best_stability} (Correlation: {results[best_stability]['stability']['mean']:.3f})\n"
        
        report += """
## Files Generated
- `quantitative_evaluation.json`: Complete numerical results
- `quantitative_comparison.png`: Bar chart comparisons
- `distribution_plots.png`: Box plot distributions
- `evaluation_report.md`: This report

## Usage Notes
- Results are averaged across multiple samples
- Error bars represent standard deviation
- Processing times are method-dependent and hardware-specific
- Stability evaluation uses Gaussian noise perturbations
"""
        
        # Save report
        report_path = os.path.join(save_dir, 'evaluation_report.md')
        with open(report_path, 'w') as f:
            f.write(report)


# Import time for processing time measurement
import time


def run_interpretability_evaluation(
    model_path: str,
    test_dataloader,
    class_names: List[str],
    device: str = 'cpu',
    num_samples: int = 50,
    methods: List[str] = None,
    save_dir: str = 'interpretability_evaluation'
) -> Dict:
    """
    Convenience function to run complete interpretability evaluation
    
    Args:
        model_path: Path to trained model
        test_dataloader: Test data loader
        class_names: List of class names
        device: Computing device
        num_samples: Number of samples to evaluate
        methods: Methods to evaluate
        save_dir: Directory to save results
        
    Returns:
        Evaluation results
    """
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = checkpoint['model']
    model.to(device)
    model.eval()
    
    print(f"Loaded model for interpretability evaluation")
    
    # Initialize framework
    framework = InterpretabilityFramework(model, device, class_names)
    
    # Run evaluation
    results = framework.comprehensive_evaluation(
        dataloader=test_dataloader,
        num_samples=num_samples,
        methods=methods,
        save_dir=save_dir
    )
    
    return results


if __name__ == "__main__":
    print("Interpretability Framework for DenLsNet-XAI")
    print("Quantitative evaluation of explainability methods")