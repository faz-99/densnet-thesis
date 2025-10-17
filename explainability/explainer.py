"""
Main explainability module that integrates Grad-CAM, SHAP, and LIME
for comprehensive model interpretability analysis
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime

from .grad_cam import generate_gradcam_explanations
from .shap_explainer import generate_shap_explanations
from .lime_explainer import generate_lime_explanations


class ComprehensiveExplainer:
    """
    Comprehensive explainer that combines multiple interpretability techniques
    """
    
    def __init__(
        self, 
        model, 
        device: str = 'cpu',
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize comprehensive explainer
        
        Args:
            model: Trained model
            device: Computing device
            class_names: List of class names
        """
        self.model = model
        self.device = device
        self.class_names = class_names or [f'Class_{i}' for i in range(2)]  # Default for binary classification
        
        # Create main results directory
        self.base_dir = 'explainability'
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'total_samples': 0,
            'correct_predictions': 0,
            'incorrect_predictions': 0,
            'class_distribution': {name: 0 for name in self.class_names},
            'prediction_distribution': {name: 0 for name in self.class_names},
            'confidence_stats': {'mean': 0, 'std': 0, 'min': 1, 'max': 0}
        }
    
    def analyze_model_performance(
        self, 
        dataloader,
        save_analysis: bool = True
    ) -> Dict:
        """
        Analyze model performance and collect statistics
        
        Args:
            dataloader: Data loader for analysis
            save_analysis: Whether to save analysis results
            
        Returns:
            Dictionary with performance statistics
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        # Calculate statistics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_confidences = np.array(all_confidences)
        
        accuracy = np.mean(all_predictions == all_labels)
        
        # Update stats
        self.stats['total_samples'] = len(all_predictions)
        self.stats['correct_predictions'] = np.sum(all_predictions == all_labels)
        self.stats['incorrect_predictions'] = np.sum(all_predictions != all_labels)
        
        # Class distributions
        for i, class_name in enumerate(self.class_names):
            self.stats['class_distribution'][class_name] = np.sum(all_labels == i)
            self.stats['prediction_distribution'][class_name] = np.sum(all_predictions == i)
        
        # Confidence statistics
        self.stats['confidence_stats'] = {
            'mean': float(np.mean(all_confidences)),
            'std': float(np.std(all_confidences)),
            'min': float(np.min(all_confidences)),
            'max': float(np.max(all_confidences))
        }
        
        performance_stats = {
            'accuracy': accuracy,
            'total_samples': len(all_predictions),
            'correct_predictions': int(np.sum(all_predictions == all_labels)),
            'incorrect_predictions': int(np.sum(all_predictions != all_labels)),
            'class_distribution': {self.class_names[i]: int(np.sum(all_labels == i)) for i in range(len(self.class_names))},
            'prediction_distribution': {self.class_names[i]: int(np.sum(all_predictions == i)) for i in range(len(self.class_names))},
            'confidence_stats': self.stats['confidence_stats']
        }
        
        if save_analysis:
            analysis_path = os.path.join(self.base_dir, 'performance_analysis.json')
            with open(analysis_path, 'w') as f:
                json.dump(performance_stats, f, indent=2)
            
            # Create performance visualization
            self._create_performance_plots(all_predictions, all_labels, all_confidences)
        
        return performance_stats
    
    def _create_performance_plots(
        self, 
        predictions: np.ndarray, 
        labels: np.ndarray, 
        confidences: np.ndarray
    ):
        """Create performance visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels, predictions)
        
        im = axes[0, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted Label')
        axes[0, 0].set_ylabel('True Label')
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[0, 0].text(j, i, str(cm[i, j]), ha='center', va='center')
        
        # Class distribution
        class_counts = [np.sum(labels == i) for i in range(len(self.class_names))]
        axes[0, 1].bar(self.class_names, class_counts)
        axes[0, 1].set_title('True Class Distribution')
        axes[0, 1].set_ylabel('Count')
        
        # Confidence distribution
        axes[1, 0].hist(confidences, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Confidence Distribution')
        axes[1, 0].set_xlabel('Confidence')
        axes[1, 0].set_ylabel('Frequency')
        
        # Confidence by correctness
        correct_mask = predictions == labels
        correct_conf = confidences[correct_mask]
        incorrect_conf = confidences[~correct_mask]
        
        axes[1, 1].hist([correct_conf, incorrect_conf], bins=15, alpha=0.7, 
                       label=['Correct', 'Incorrect'], color=['green', 'red'])
        axes[1, 1].set_title('Confidence by Prediction Correctness')
        axes[1, 1].set_xlabel('Confidence')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_dir, 'performance_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_explanations(
        self,
        test_dataloader,
        background_dataloader,
        num_samples: int = 20,
        techniques: List[str] = ['gradcam', 'shap', 'lime']
    ):
        """
        Generate comprehensive explanations using multiple techniques
        
        Args:
            test_dataloader: Test data loader
            background_dataloader: Background data for SHAP
            num_samples: Number of samples to explain
            techniques: List of techniques to use ['gradcam', 'shap', 'lime']
        """
        print("Starting comprehensive explainability analysis...")
        
        # Analyze model performance first
        print("Analyzing model performance...")
        performance_stats = self.analyze_model_performance(test_dataloader)
        print(f"Model accuracy: {performance_stats['accuracy']:.3f}")
        
        # Generate explanations for each technique
        if 'gradcam' in techniques:
            print("\nGenerating Grad-CAM explanations...")
            generate_gradcam_explanations(
                model=self.model,
                dataloader=test_dataloader,
                device=self.device,
                class_names=self.class_names,
                save_dir=os.path.join(self.base_dir, 'gradcam_results'),
                num_samples=num_samples
            )
        
        if 'shap' in techniques:
            print("\nGenerating SHAP explanations...")
            generate_shap_explanations(
                model=self.model,
                dataloader=test_dataloader,
                background_loader=background_dataloader,
                device=self.device,
                class_names=self.class_names,
                save_dir=os.path.join(self.base_dir, 'shap_results'),
                num_samples=min(num_samples, 10),  # SHAP is computationally expensive
                background_size=50
            )
        
        if 'lime' in techniques:
            print("\nGenerating LIME explanations...")
            generate_lime_explanations(
                model=self.model,
                dataloader=test_dataloader,
                device=self.device,
                class_names=self.class_names,
                save_dir=os.path.join(self.base_dir, 'lime_results'),
                num_samples=min(num_samples, 10),  # LIME is also computationally expensive
                lime_samples=500  # Reduced for faster computation
            )
        
        # Create summary report
        self._create_summary_report(techniques, performance_stats)
        
        print(f"\nExplainability analysis complete! Results saved to: {self.base_dir}")
    
    def _create_summary_report(self, techniques: List[str], performance_stats: Dict):
        """Create a comprehensive summary report"""
        report = {
            'analysis_date': datetime.now().isoformat(),
            'model_performance': performance_stats,
            'techniques_used': techniques,
            'class_names': self.class_names,
            'summary': {
                'total_samples_analyzed': performance_stats['total_samples'],
                'model_accuracy': performance_stats['accuracy'],
                'techniques_applied': len(techniques),
                'output_directories': {
                    technique: f"{self.base_dir}/{technique}_results" 
                    for technique in techniques
                }
            },
            'recommendations': self._generate_recommendations(performance_stats)
        }
        
        # Save JSON report
        report_path = os.path.join(self.base_dir, 'explainability_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create markdown report
        self._create_markdown_report(report)
    
    def _generate_recommendations(self, performance_stats: Dict) -> List[str]:
        """Generate recommendations based on performance analysis"""
        recommendations = []
        
        accuracy = performance_stats['accuracy']
        confidence_stats = performance_stats['confidence_stats']
        
        if accuracy < 0.8:
            recommendations.append("Model accuracy is below 80%. Consider retraining with more data or different architecture.")
        
        if confidence_stats['mean'] < 0.7:
            recommendations.append("Average confidence is low. Model may be uncertain about predictions.")
        
        if confidence_stats['std'] > 0.3:
            recommendations.append("High confidence variance detected. Review model calibration.")
        
        # Check class imbalance
        class_dist = performance_stats['class_distribution']
        total_samples = sum(class_dist.values())
        class_ratios = [count/total_samples for count in class_dist.values()]
        
        if max(class_ratios) > 0.8:
            recommendations.append("Significant class imbalance detected. Consider data balancing techniques.")
        
        if not recommendations:
            recommendations.append("Model performance looks good. Use explainability results to understand decision patterns.")
        
        return recommendations
    
    def _create_markdown_report(self, report: Dict):
        """Create a markdown summary report"""
        md_content = f"""# Model Explainability Analysis Report

## Analysis Overview
- **Date**: {report['analysis_date']}
- **Model Accuracy**: {report['model_performance']['accuracy']:.3f}
- **Total Samples**: {report['model_performance']['total_samples']}
- **Techniques Used**: {', '.join(report['techniques_used'])}

## Performance Summary
- **Correct Predictions**: {report['model_performance']['correct_predictions']}
- **Incorrect Predictions**: {report['model_performance']['incorrect_predictions']}

### Class Distribution
"""
        
        for class_name, count in report['model_performance']['class_distribution'].items():
            md_content += f"- **{class_name}**: {count} samples\n"
        
        md_content += f"""
### Confidence Statistics
- **Mean Confidence**: {report['model_performance']['confidence_stats']['mean']:.3f}
- **Std Confidence**: {report['model_performance']['confidence_stats']['std']:.3f}
- **Min Confidence**: {report['model_performance']['confidence_stats']['min']:.3f}
- **Max Confidence**: {report['model_performance']['confidence_stats']['max']:.3f}

## Explainability Techniques Applied

"""
        
        for technique in report['techniques_used']:
            md_content += f"### {technique.upper()}\n"
            md_content += f"Results saved to: `{report['summary']['output_directories'][technique]}`\n\n"
        
        md_content += "## Recommendations\n\n"
        for i, rec in enumerate(report['recommendations'], 1):
            md_content += f"{i}. {rec}\n"
        
        md_content += f"""
## Output Structure
```
{self.base_dir}/
├── performance_analysis.json
├── performance_plots.png
├── explainability_report.json
├── README.md
"""
        
        for technique in report['techniques_used']:
            md_content += f"├── {technique}_results/\n"
        
        md_content += "```\n"
        
        # Save markdown report
        md_path = os.path.join(self.base_dir, 'README.md')
        with open(md_path, 'w') as f:
            f.write(md_content)


def run_explainability_analysis(
    model_path: str,
    test_dataloader,
    background_dataloader,
    class_names: List[str],
    device: str = 'cpu',
    num_samples: int = 20,
    techniques: List[str] = ['gradcam', 'shap', 'lime']
):
    """
    Convenience function to run complete explainability analysis
    
    Args:
        model_path: Path to trained model
        test_dataloader: Test data loader
        background_dataloader: Background data for SHAP
        class_names: List of class names
        device: Computing device
        num_samples: Number of samples to explain
        techniques: List of techniques to use
    """
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = checkpoint['model']
    model.to(device)
    model.eval()
    
    print(f"Loaded model with best accuracy: {checkpoint.get('best_acc', 'N/A')}")
    
    # Initialize explainer
    explainer = ComprehensiveExplainer(model, device, class_names)
    
    # Run analysis
    explainer.generate_comprehensive_explanations(
        test_dataloader=test_dataloader,
        background_dataloader=background_dataloader,
        num_samples=num_samples,
        techniques=techniques
    )
    
    return explainer