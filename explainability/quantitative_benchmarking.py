"""
Quantitative Explainability Benchmarking for DenLsNet
Implements IoU, Insertion/Deletion AUC, and Stability metrics for XAI method comparison
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import cv2
from sklearn.metrics import jaccard_score, auc
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

from .grad_cam import GradCAM, GradCAMPlusPlus
from .shap_explainer import SHAPExplainer
from .lime_explainer import LIMEExplainer


class QuantitativeExplainabilityBenchmark:
    """
    Comprehensive quantitative evaluation of explainability methods
    """
    
    def __init__(self, 
                 model, 
                 device: str = 'cpu',
                 class_names: List[str] = None):
        """
        Initialize benchmark
        
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
        
        # Results storage
        self.benchmark_results = {}
    
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
            
            # SHAP (with dummy background for initialization)
            background_data = torch.randn(10, 3, 224, 224).to(self.device)
            explainers['shap'] = SHAPExplainer(
                self.model, 
                background_data, 
                str(self.device)
            )
            
            # LIME
            explainers['lime'] = LIMEExplainer(
                self.model,
                str(self.device),
                num_samples=100  # Reduced for faster benchmarking
            )
            
        except Exception as e:
            print(f"Warning: Failed to initialize some explainers: {e}")
        
        return explainers
    
    def compute_iou_with_ground_truth(self, 
                                    explanation: np.ndarray,
                                    ground_truth_mask: np.ndarray,
                                    threshold_percentile: float = 80) -> float:
        """
        Compute IoU between explanation and ground truth mask
        
        Args:
            explanation: Explanation heatmap (H, W)
            ground_truth_mask: Ground truth ROI mask (H, W)
            threshold_percentile: Percentile threshold for binarizing explanation
            
        Returns:
            IoU score
        """
        # Resize explanation to match ground truth if needed
        if explanation.shape != ground_truth_mask.shape:
            explanation = cv2.resize(explanation, 
                                   (ground_truth_mask.shape[1], ground_truth_mask.shape[0]))
        
        # Threshold explanation to create binary mask
        threshold = np.percentile(explanation, threshold_percentile)
        explanation_binary = (explanation >= threshold).astype(np.uint8)
        
        # Ensure ground truth is binary
        ground_truth_binary = (ground_truth_mask > 0).astype(np.uint8)
        
        # Calculate IoU
        intersection = np.logical_and(explanation_binary, ground_truth_binary)
        union = np.logical_or(explanation_binary, ground_truth_binary)
        
        iou = np.sum(intersection) / (np.sum(union) + 1e-8)
        
        return float(iou)
    
    def compute_insertion_deletion_auc(self, 
                                     image: torch.Tensor,
                                     explanation: np.ndarray,
                                     target_class: int,
                                     num_steps: int = 50) -> Tuple[float, float]:
        """
        Compute insertion and deletion AUC for faithfulness evaluation
        
        Args:
            image: Input image tensor (1, C, H, W)
            explanation: Explanation heatmap (H, W)
            target_class: Target class index
            num_steps: Number of steps for curve computation
            
        Returns:
            Tuple of (insertion_auc, deletion_auc)
        """
        self.model.eval()
        
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
    
    def compute_stability_score(self, 
                              image: torch.Tensor,
                              method: str,
                              target_class: int,
                              num_perturbations: int = 5,
                              noise_level: float = 0.1) -> float:
        """
        Compute stability score under random perturbations
        
        Args:
            image: Input image tensor (1, C, H, W)
            method: Explanation method name
            target_class: Target class index
            num_perturbations: Number of perturbations to test
            noise_level: Noise level for perturbations
            
        Returns:
            Average correlation coefficient
        """
        if method not in self.explainers:
            return 0.0
        
        # Generate baseline explanation
        baseline_explanation = self._generate_explanation(image, method, target_class)
        
        if baseline_explanation is None:
            return 0.0
        
        correlations = []
        
        for _ in range(num_perturbations):
            # Add noise to image
            noise = torch.randn_like(image) * noise_level
            perturbed_image = image + noise
            perturbed_image = torch.clamp(perturbed_image, -3, 3)
            
            # Generate explanation for perturbed image
            perturbed_explanation = self._generate_explanation(perturbed_image, method, target_class)
            
            if perturbed_explanation is not None:
                # Calculate correlation
                try:
                    corr, _ = pearsonr(baseline_explanation.flatten(), perturbed_explanation.flatten())
                    if not np.isnan(corr):
                        correlations.append(corr)
                except:
                    continue
        
        return float(np.mean(correlations)) if correlations else 0.0
    
    def _generate_explanation(self, 
                            image: torch.Tensor, 
                            method: str, 
                            target_class: int) -> Optional[np.ndarray]:
        """
        Generate explanation using specified method
        
        Args:
            image: Input image tensor (1, C, H, W)
            method: Method name
            target_class: Target class index
            
        Returns:
            Explanation heatmap or None if failed
        """
        try:
            if method == 'gradcam':
                return self.explainers['gradcam'].generate_cam(image, target_class)
            elif method == 'gradcam_plus':
                return self.explainers['gradcam_plus'].generate_cam(image, target_class)
            elif method == 'shap':
                shap_values = self.explainers['shap'].explain_image(image, target_class)
                if shap_values is not None:
                    # Convert SHAP values to heatmap
                    explanation = np.sum(np.abs(shap_values), axis=0)
                    explanation = (explanation - explanation.min()) / (explanation.max() - explanation.min() + 1e-8)
                    return explanation
            elif method == 'lime':
                # Convert tensor to numpy for LIME
                image_np = image[0].cpu().numpy().transpose(1, 2, 0)
                # Denormalize for LIME
                mean = np.array([0.5613, 0.5778, 0.6032])
                std = np.array([0.2114, 0.1957, 0.1590])
                image_np = image_np * std + mean
                image_np = np.clip(image_np, 0, 1)
                
                lime_explanation, segments = self.explainers['lime'].explain_image(image_np)
                temp, mask = lime_explanation.get_image_and_mask(
                    target_class, positive_only=False, num_features=10, hide_rest=False
                )
                explanation = mask.astype(np.float32)
                explanation = (explanation - explanation.min()) / (explanation.max() - explanation.min() + 1e-8)
                return explanation
        except Exception as e:
            print(f"Error generating {method} explanation: {e}")
            return None
        
        return None
    
    def run_comprehensive_benchmark(self, 
                                  dataloader,
                                  num_samples: int = 50,
                                  methods: List[str] = None,
                                  ground_truth_masks: Optional[Dict] = None,
                                  save_dir: str = 'results/explainability_metrics') -> Dict:
        """
        Run comprehensive quantitative benchmark
        
        Args:
            dataloader: Data loader for evaluation
            num_samples: Number of samples to evaluate
            methods: Methods to evaluate (default: all available)
            ground_truth_masks: Optional ground truth masks for IoU computation
            save_dir: Directory to save results
            
        Returns:
            Comprehensive benchmark results
        """
        if methods is None:
            methods = list(self.explainers.keys())
        
        print(f"Starting quantitative explainability benchmark...")
        print(f"Methods: {methods}")
        print(f"Samples: {num_samples}")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize results storage
        results = {method: {
            'insertion_aucs': [],
            'deletion_aucs': [],
            'stability_scores': [],
            'iou_scores': [],
            'processing_times': [],
            'success_count': 0
        } for method in methods}
        
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                if sample_count >= num_samples:
                    break
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Get predictions
                outputs = self.model(images)
                probabilities = F.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
                
                for i in range(images.size(0)):
                    if sample_count >= num_samples:
                        break
                    
                    image = images[i:i+1]
                    true_label = labels[i].item()
                    pred_label = predicted_classes[i].item()
                    
                    print(f"Evaluating sample {sample_count + 1}/{num_samples}")
                    
                    # Evaluate each method
                    for method in methods:
                        try:
                            import time
                            start_time = time.time()
                            
                            # Generate explanation
                            explanation = self._generate_explanation(image, method, pred_label)
                            
                            processing_time = time.time() - start_time
                            results[method]['processing_times'].append(processing_time)
                            
                            if explanation is not None:
                                # Insertion/Deletion AUC
                                ins_auc, del_auc = self.compute_insertion_deletion_auc(
                                    image, explanation, pred_label
                                )
                                results[method]['insertion_aucs'].append(ins_auc)
                                results[method]['deletion_aucs'].append(del_auc)
                                
                                # Stability score
                                stability = self.compute_stability_score(
                                    image, method, pred_label
                                )
                                results[method]['stability_scores'].append(stability)
                                
                                # IoU with ground truth (if available)
                                if ground_truth_masks and sample_count in ground_truth_masks:
                                    gt_mask = ground_truth_masks[sample_count]
                                    iou = self.compute_iou_with_ground_truth(explanation, gt_mask)
                                    results[method]['iou_scores'].append(iou)
                                
                                results[method]['success_count'] += 1
                            
                        except Exception as e:
                            print(f"Error evaluating {method}: {e}")
                            continue
                    
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
                    'mean': float(np.mean(method_results['stability_scores'])) if method_results['stability_scores'] else None,
                    'std': float(np.std(method_results['stability_scores'])) if method_results['stability_scores'] else None,
                    'values': method_results['stability_scores']
                },
                'iou': {
                    'mean': float(np.mean(method_results['iou_scores'])) if method_results['iou_scores'] else None,
                    'std': float(np.std(method_results['iou_scores'])) if method_results['iou_scores'] else None,
                    'values': method_results['iou_scores']
                },
                'processing_time': {
                    'mean': float(np.mean(method_results['processing_times'])) if method_results['processing_times'] else None,
                    'std': float(np.std(method_results['processing_times'])) if method_results['processing_times'] else None,
                    'values': method_results['processing_times']
                },
                'success_rate': method_results['success_count'] / sample_count if sample_count > 0 else 0,
                'total_samples': method_results['success_count']
            }
        
        # Save results
        results_path = os.path.join(save_dir, 'explainability_summary.csv')
        self._save_results_csv(summary_results, results_path)
        
        json_path = os.path.join(save_dir, 'quantitative_benchmark_results.json')
        with open(json_path, 'w') as f:
            json.dump(summary_results, f, indent=2)
        
        # Create visualizations
        self._create_benchmark_visualizations(summary_results, save_dir)
        
        # Create detailed report
        self._create_benchmark_report(summary_results, save_dir)
        
        print(f"Quantitative benchmark completed!")
        print(f"Results saved to: {save_dir}")
        
        self.benchmark_results = summary_results
        return summary_results
    
    def _save_results_csv(self, results: Dict, save_path: str):
        """Save results to CSV format"""
        data = []
        
        for method, metrics in results.items():
            row = {'Method': method}
            
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and 'mean' in metric_data:
                    row[f'{metric_name}_mean'] = metric_data['mean']
                    row[f'{metric_name}_std'] = metric_data['std']
                elif metric_name in ['success_rate', 'total_samples']:
                    row[metric_name] = metric_data
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
        print(f"Results saved to CSV: {save_path}")
    
    def _create_benchmark_visualizations(self, results: Dict, save_dir: str):
        """Create comprehensive benchmark visualizations"""
        methods = list(results.keys())
        
        # Create comparison bar chart
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        metrics_to_plot = [
            ('insertion_auc', 'Insertion AUC', 'Higher is Better'),
            ('deletion_auc', 'Deletion AUC', 'Lower is Better'),
            ('stability', 'Stability Score', 'Higher is Better'),
            ('iou', 'IoU Score', 'Higher is Better'),
            ('processing_time', 'Processing Time (s)', 'Lower is Better'),
            ('success_rate', 'Success Rate', 'Higher is Better')
        ]
        
        for idx, (metric, title, note) in enumerate(metrics_to_plot):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            values = []
            errors = []
            method_names = []
            
            for method in methods:
                if metric in results[method]:
                    if isinstance(results[method][metric], dict):
                        mean_val = results[method][metric]['mean']
                        std_val = results[method][metric]['std']
                    else:
                        mean_val = results[method][metric]
                        std_val = 0
                    
                    if mean_val is not None:
                        values.append(mean_val)
                        errors.append(std_val if std_val is not None else 0)
                        method_names.append(method)
            
            if values:
                bars = ax.bar(method_names, values, yerr=errors, capsize=5, alpha=0.7)
                ax.set_title(f'{title}\n({note})')
                ax.set_ylabel(title)
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'quantitative_comparison_chart.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed distribution plots
        self._create_distribution_plots(results, save_dir)
    
    def _create_distribution_plots(self, results: Dict, save_dir: str):
        """Create distribution plots for each metric"""
        methods = list(results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics_to_plot = ['insertion_auc', 'deletion_auc', 'stability', 'processing_time']
        titles = ['Insertion AUC Distribution', 'Deletion AUC Distribution', 
                 'Stability Distribution', 'Processing Time Distribution']
        
        for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            data_to_plot = []
            labels = []
            
            for method in methods:
                if metric in results[method] and results[method][metric]['values']:
                    data_to_plot.append(results[method][metric]['values'])
                    labels.append(method)
            
            if data_to_plot:
                ax.boxplot(data_to_plot, labels=labels)
                ax.set_title(title)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'metric_distributions.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_benchmark_report(self, results: Dict, save_dir: str):
        """Create comprehensive benchmark report"""
        report_content = f"""# Quantitative Explainability Benchmark Report

## Analysis Overview
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Methods Evaluated**: {', '.join(results.keys())}
- **Metrics Computed**: Insertion AUC, Deletion AUC, Stability, IoU, Processing Time

## Summary Results

### Performance Ranking

#### Insertion AUC (Higher is Better)
"""
        
        # Rank methods by insertion AUC
        valid_methods = [(method, data['insertion_auc']['mean']) 
                        for method, data in results.items() 
                        if data['insertion_auc']['mean'] is not None]
        valid_methods.sort(key=lambda x: x[1], reverse=True)
        
        for i, (method, score) in enumerate(valid_methods, 1):
            report_content += f"{i}. **{method}**: {score:.3f}\n"
        
        report_content += "\n#### Deletion AUC (Lower is Better)\n"
        
        # Rank methods by deletion AUC (lower is better)
        valid_methods = [(method, data['deletion_auc']['mean']) 
                        for method, data in results.items() 
                        if data['deletion_auc']['mean'] is not None]
        valid_methods.sort(key=lambda x: x[1])
        
        for i, (method, score) in enumerate(valid_methods, 1):
            report_content += f"{i}. **{method}**: {score:.3f}\n"
        
        report_content += "\n#### Stability (Higher is Better)\n"
        
        # Rank methods by stability
        valid_methods = [(method, data['stability']['mean']) 
                        for method, data in results.items() 
                        if data['stability']['mean'] is not None]
        valid_methods.sort(key=lambda x: x[1], reverse=True)
        
        for i, (method, score) in enumerate(valid_methods, 1):
            report_content += f"{i}. **{method}**: {score:.3f}\n"
        
        report_content += f"""
## Detailed Results

| Method | Insertion AUC | Deletion AUC | Stability | Processing Time (s) | Success Rate |
|--------|---------------|--------------|-----------|-------------------|--------------|
"""
        
        for method, data in results.items():
            ins_auc = f"{data['insertion_auc']['mean']:.3f} ± {data['insertion_auc']['std']:.3f}" if data['insertion_auc']['mean'] else "N/A"
            del_auc = f"{data['deletion_auc']['mean']:.3f} ± {data['deletion_auc']['std']:.3f}" if data['deletion_auc']['mean'] else "N/A"
            stability = f"{data['stability']['mean']:.3f} ± {data['stability']['std']:.3f}" if data['stability']['mean'] else "N/A"
            proc_time = f"{data['processing_time']['mean']:.3f}" if data['processing_time']['mean'] else "N/A"
            success_rate = f"{data['success_rate']:.1%}"
            
            report_content += f"| {method} | {ins_auc} | {del_auc} | {stability} | {proc_time} | {success_rate} |\n"
        
        report_content += f"""
## Metric Interpretations

### Insertion AUC
- **Range**: 0.0 - 1.0
- **Interpretation**: Measures how quickly model confidence increases when adding important pixels
- **Good Score**: > 0.6 indicates faithful explanations

### Deletion AUC
- **Range**: 0.0 - 1.0  
- **Interpretation**: Measures how quickly model confidence decreases when removing important pixels
- **Good Score**: < 0.4 indicates explanations identify truly important regions

### Stability Score
- **Range**: -1.0 - 1.0
- **Interpretation**: Correlation between explanations under input perturbations
- **Good Score**: > 0.7 indicates robust explanations

### Processing Time
- **Interpretation**: Computational efficiency of explanation generation
- **Consideration**: Balance between quality and speed for practical applications

## Recommendations

Based on the quantitative evaluation:
"""
        
        # Generate recommendations
        if valid_methods:
            # Best overall method (considering multiple metrics)
            method_scores = {}
            for method, data in results.items():
                score = 0
                count = 0
                
                if data['insertion_auc']['mean'] is not None:
                    score += data['insertion_auc']['mean']
                    count += 1
                
                if data['deletion_auc']['mean'] is not None:
                    score += (1 - data['deletion_auc']['mean'])  # Invert since lower is better
                    count += 1
                
                if data['stability']['mean'] is not None:
                    score += data['stability']['mean']
                    count += 1
                
                if count > 0:
                    method_scores[method] = score / count
            
            if method_scores:
                best_method = max(method_scores.items(), key=lambda x: x[1])
                report_content += f"- **Best Overall Method**: {best_method[0]} (composite score: {best_method[1]:.3f})\n"
        
        report_content += """
- Use methods with high insertion AUC and low deletion AUC for faithful explanations
- Prioritize methods with high stability for consistent clinical interpretation
- Consider processing time constraints for real-time applications
- Validate explanations with domain experts regardless of quantitative scores

## Files Generated
- `quantitative_benchmark_results.json`: Complete numerical results
- `explainability_summary.csv`: Summary table in CSV format
- `quantitative_comparison_chart.png`: Performance comparison visualization
- `metric_distributions.png`: Distribution analysis plots

## Usage Notes
- Results are averaged across multiple samples for statistical reliability
- Error bars represent standard deviation across samples
- Success rate indicates percentage of samples successfully processed
- Consider both quantitative metrics and qualitative clinical relevance
"""
        
        # Save report
        report_path = os.path.join(save_dir, 'quantitative_benchmark_report.md')
        with open(report_path, 'w') as f:
            f.write(report_content)


def run_quantitative_benchmark(
    model_path: str,
    test_dataloader,
    class_names: List[str],
    device: str = 'cpu',
    num_samples: int = 50,
    methods: List[str] = None,
    save_dir: str = 'results/explainability_metrics'
) -> Dict:
    """
    Convenience function to run quantitative explainability benchmark
    
    Args:
        model_path: Path to trained model
        test_dataloader: Test data loader
        class_names: List of class names
        device: Computing device
        num_samples: Number of samples to evaluate
        methods: Methods to evaluate
        save_dir: Directory to save results
        
    Returns:
        Benchmark results
    """
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = checkpoint['model']
    model.to(device)
    model.eval()
    
    print(f"Loaded model for quantitative benchmarking")
    
    # Initialize benchmark
    benchmark = QuantitativeExplainabilityBenchmark(model, device, class_names)
    
    # Run benchmark
    results = benchmark.run_comprehensive_benchmark(
        dataloader=test_dataloader,
        num_samples=num_samples,
        methods=methods,
        save_dir=save_dir
    )
    
    return results


if __name__ == "__main__":
    print("Quantitative Explainability Benchmarking Module")
    print("Evaluates XAI methods using IoU, Insertion/Deletion AUC, and Stability metrics")