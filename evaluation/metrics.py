"""
Comprehensive evaluation metrics for DenseNet model performance assessment
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
import pandas as pd
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ModelEvaluator:
    """Comprehensive model evaluation with multiple metrics and visualizations"""
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize evaluator
        
        Args:
            class_names: List of class names (default: ['Benign', 'Malignant'])
        """
        self.class_names = class_names or ['Benign', 'Malignant']
        self.num_classes = len(self.class_names)
        self.results = {}
        
    def evaluate_model(
        self, 
        model, 
        dataloader, 
        device: str = 'cpu',
        save_results: bool = True,
        save_dir: str = 'evaluation_results'
    ) -> Dict:
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            dataloader: Test data loader
            device: Computing device
            save_results: Whether to save results
            save_dir: Directory to save results
            
        Returns:
            Dictionary with all evaluation metrics
        """
        print("Starting comprehensive model evaluation...")
        
        # Collect predictions
        all_predictions, all_labels, all_probabilities = self._collect_predictions(
            model, dataloader, device
        )
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_labels, all_probabilities)
        
        # Store results
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'predictions': all_predictions.tolist(),
            'labels': all_labels.tolist(),
            'probabilities': all_probabilities.tolist(),
            'class_names': self.class_names
        }
        
        # Create visualizations
        if save_results:
            os.makedirs(save_dir, exist_ok=True)
            self._create_visualizations(
                all_predictions, all_labels, all_probabilities, save_dir
            )
            self._save_results(save_dir)
        
        return self.results
    
    def _collect_predictions(self, model, dataloader, device):
        """Collect model predictions and ground truth labels"""
        model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return (
            np.array(all_predictions),
            np.array(all_labels),
            np.array(all_probabilities)
        )
    
    def _calculate_metrics(self, predictions, labels, probabilities):
        """Calculate comprehensive evaluation metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = float(accuracy_score(labels, predictions))
        metrics['precision'] = float(precision_score(labels, predictions, average='weighted'))
        metrics['recall'] = float(recall_score(labels, predictions, average='weighted'))
        metrics['f1_score'] = float(f1_score(labels, predictions, average='weighted'))
        
        # Per-class metrics
        precision_per_class = precision_score(labels, predictions, average=None)
        recall_per_class = recall_score(labels, predictions, average=None)
        f1_per_class = f1_score(labels, predictions, average=None)
        
        metrics['per_class'] = {}
        for i, class_name in enumerate(self.class_names):
            metrics['per_class'][class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i])
            }
        
        # Binary classification specific metrics (for 2-class problems)
        if self.num_classes == 2:
            # Sensitivity (True Positive Rate) = Recall
            metrics['sensitivity'] = float(recall_score(labels, predictions, pos_label=1))
            
            # Specificity (True Negative Rate)
            tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
            metrics['specificity'] = float(tn / (tn + fp))
            
            # ROC-AUC
            metrics['roc_auc'] = float(roc_auc_score(labels, probabilities[:, 1]))
            
            # Additional binary metrics
            metrics['positive_predictive_value'] = float(precision_score(labels, predictions, pos_label=1))
            metrics['negative_predictive_value'] = float(tn / (tn + fn))
            
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        report = classification_report(labels, predictions, target_names=self.class_names, output_dict=True)
        metrics['classification_report'] = report
        
        # Sample distribution
        metrics['sample_distribution'] = {
            self.class_names[i]: int(np.sum(labels == i)) 
            for i in range(self.num_classes)
        }
        
        # Prediction distribution
        metrics['prediction_distribution'] = {
            self.class_names[i]: int(np.sum(predictions == i)) 
            for i in range(self.num_classes)
        }
        
        # Confidence statistics
        max_probs = np.max(probabilities, axis=1)
        metrics['confidence_stats'] = {
            'mean': float(np.mean(max_probs)),
            'std': float(np.std(max_probs)),
            'min': float(np.min(max_probs)),
            'max': float(np.max(max_probs)),
            'median': float(np.median(max_probs))
        }
        
        return metrics
    
    def _create_visualizations(self, predictions, labels, probabilities, save_dir):
        """Create comprehensive visualizations"""
        
        # 1. Confusion Matrix
        self._plot_confusion_matrix(predictions, labels, save_dir)
        
        # 2. ROC Curve (for binary classification)
        if self.num_classes == 2:
            self._plot_roc_curve(labels, probabilities, save_dir)
            self._plot_precision_recall_curve(labels, probabilities, save_dir)
        
        # 3. Class distribution
        self._plot_class_distribution(labels, predictions, save_dir)
        
        # 4. Confidence distribution
        self._plot_confidence_distribution(probabilities, labels, predictions, save_dir)
        
        # 5. Performance metrics summary
        self._plot_metrics_summary(save_dir)
        
        # 6. Interactive plots with Plotly
        self._create_interactive_plots(predictions, labels, probabilities, save_dir)
    
    def _plot_confusion_matrix(self, predictions, labels, save_dir):
        """Plot confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Add percentage annotations
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1f}%)', 
                        ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curve(self, labels, probabilities, save_dir):
        """Plot ROC curve for binary classification"""
        fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
        auc = roc_auc_score(labels, probabilities[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curve(self, labels, probabilities, save_dir):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(labels, probabilities[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_class_distribution(self, labels, predictions, save_dir):
        """Plot class distribution comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # True distribution
        true_counts = [np.sum(labels == i) for i in range(self.num_classes)]
        ax1.bar(self.class_names, true_counts, color='skyblue', alpha=0.7)
        ax1.set_title('True Class Distribution', fontweight='bold')
        ax1.set_ylabel('Count')
        
        # Add count labels
        for i, count in enumerate(true_counts):
            ax1.text(i, count + max(true_counts) * 0.01, str(count), 
                    ha='center', va='bottom', fontweight='bold')
        
        # Predicted distribution
        pred_counts = [np.sum(predictions == i) for i in range(self.num_classes)]
        ax2.bar(self.class_names, pred_counts, color='lightcoral', alpha=0.7)
        ax2.set_title('Predicted Class Distribution', fontweight='bold')
        ax2.set_ylabel('Count')
        
        # Add count labels
        for i, count in enumerate(pred_counts):
            ax2.text(i, count + max(pred_counts) * 0.01, str(count), 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_distribution(self, probabilities, labels, predictions, save_dir):
        """Plot confidence distribution analysis"""
        max_probs = np.max(probabilities, axis=1)
        correct_mask = predictions == labels
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Overall confidence distribution
        ax1.hist(max_probs, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Overall Confidence Distribution', fontweight='bold')
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Frequency')
        ax1.axvline(np.mean(max_probs), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(max_probs):.3f}')
        ax1.legend()
        
        # Confidence by correctness
        correct_conf = max_probs[correct_mask]
        incorrect_conf = max_probs[~correct_mask]
        
        ax2.hist([correct_conf, incorrect_conf], bins=15, alpha=0.7, 
                label=['Correct', 'Incorrect'], color=['green', 'red'])
        ax2.set_title('Confidence by Prediction Correctness', fontweight='bold')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        # Confidence by class
        for i, class_name in enumerate(self.class_names):
            class_mask = labels == i
            class_conf = max_probs[class_mask]
            ax3.hist(class_conf, bins=15, alpha=0.6, label=class_name)
        
        ax3.set_title('Confidence Distribution by True Class', fontweight='bold')
        ax3.set_xlabel('Confidence')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        # Confidence vs Accuracy
        confidence_bins = np.linspace(0, 1, 11)
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        bin_accuracies = []
        
        for i in range(len(confidence_bins) - 1):
            mask = (max_probs >= confidence_bins[i]) & (max_probs < confidence_bins[i + 1])
            if np.sum(mask) > 0:
                bin_accuracy = np.mean(correct_mask[mask])
                bin_accuracies.append(bin_accuracy)
            else:
                bin_accuracies.append(0)
        
        ax4.plot(bin_centers, bin_accuracies, 'o-', color='blue', linewidth=2, markersize=6)
        ax4.plot([0, 1], [0, 1], '--', color='gray', alpha=0.7, label='Perfect Calibration')
        ax4.set_title('Confidence vs Accuracy (Calibration)', fontweight='bold')
        ax4.set_xlabel('Confidence')
        ax4.set_ylabel('Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confidence_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_summary(self, save_dir):
        """Plot metrics summary"""
        metrics = self.results['metrics']
        
        # Prepare data for plotting
        if self.num_classes == 2:
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Sensitivity', 'Specificity', 'ROC-AUC']
            metric_values = [
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1_score'],
                metrics['sensitivity'],
                metrics['specificity'],
                metrics['roc_auc']
            ]
        else:
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            metric_values = [
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1_score']
            ]
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metric_names, metric_values, color='skyblue', alpha=0.8, edgecolor='navy')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Model Performance Metrics Summary', fontsize=16, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'metrics_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_interactive_plots(self, predictions, labels, probabilities, save_dir):
        """Create interactive Plotly visualizations"""
        
        # Interactive confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=self.class_names,
            y=self.class_names,
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False
        ))
        
        fig_cm.update_layout(
            title='Interactive Confusion Matrix',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            width=600,
            height=500
        )
        
        fig_cm.write_html(os.path.join(save_dir, 'interactive_confusion_matrix.html'))
        
        # Interactive ROC curve (for binary classification)
        if self.num_classes == 2:
            fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
            auc = roc_auc_score(labels, probabilities[:, 1])
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {auc:.3f})',
                line=dict(color='darkorange', width=3)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(color='navy', width=2, dash='dash')
            ))
            
            fig_roc.update_layout(
                title='Interactive ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                width=700,
                height=500
            )
            
            fig_roc.write_html(os.path.join(save_dir, 'interactive_roc_curve.html'))
    
    def _save_results(self, save_dir):
        """Save evaluation results to files"""
        
        # Save JSON results
        json_path = os.path.join(save_dir, 'evaluation_results.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save CSV summary
        metrics = self.results['metrics']
        summary_data = {
            'Metric': [],
            'Value': []
        }
        
        # Add main metrics
        main_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        if self.num_classes == 2:
            main_metrics.extend(['sensitivity', 'specificity', 'roc_auc'])
        
        for metric in main_metrics:
            summary_data['Metric'].append(metric.replace('_', ' ').title())
            summary_data['Value'].append(metrics[metric])
        
        # Add per-class metrics
        for class_name in self.class_names:
            class_metrics = metrics['per_class'][class_name]
            for metric_name, value in class_metrics.items():
                summary_data['Metric'].append(f'{class_name} {metric_name.replace("_", " ").title()}')
                summary_data['Value'].append(value)
        
        df_summary = pd.DataFrame(summary_data)
        csv_path = os.path.join(save_dir, 'metrics_summary.csv')
        df_summary.to_csv(csv_path, index=False)
        
        # Create detailed report
        self._create_detailed_report(save_dir)
        
        print(f"Evaluation results saved to: {save_dir}")
    
    def _create_detailed_report(self, save_dir):
        """Create detailed markdown report"""
        metrics = self.results['metrics']
        timestamp = self.results['timestamp']
        
        report = f"""# Model Evaluation Report

## Analysis Overview
- **Timestamp**: {timestamp}
- **Classes**: {', '.join(self.class_names)}
- **Total Samples**: {sum(metrics['sample_distribution'].values())}

## Performance Metrics

### Overall Performance
- **Accuracy**: {metrics['accuracy']:.4f}
- **Precision**: {metrics['precision']:.4f}
- **Recall**: {metrics['recall']:.4f}
- **F1-Score**: {metrics['f1_score']:.4f}

"""
        
        if self.num_classes == 2:
            report += f"""### Binary Classification Metrics
- **Sensitivity (True Positive Rate)**: {metrics['sensitivity']:.4f}
- **Specificity (True Negative Rate)**: {metrics['specificity']:.4f}
- **ROC-AUC**: {metrics['roc_auc']:.4f}
- **Positive Predictive Value**: {metrics['positive_predictive_value']:.4f}
- **Negative Predictive Value**: {metrics['negative_predictive_value']:.4f}

"""
        
        report += """### Per-Class Performance
"""
        
        for class_name in self.class_names:
            class_metrics = metrics['per_class'][class_name]
            report += f"""
#### {class_name}
- **Precision**: {class_metrics['precision']:.4f}
- **Recall**: {class_metrics['recall']:.4f}
- **F1-Score**: {class_metrics['f1_score']:.4f}
"""
        
        report += f"""
## Data Distribution

### True Class Distribution
"""
        for class_name, count in metrics['sample_distribution'].items():
            percentage = count / sum(metrics['sample_distribution'].values()) * 100
            report += f"- **{class_name}**: {count} samples ({percentage:.1f}%)\n"
        
        report += f"""
### Predicted Class Distribution
"""
        for class_name, count in metrics['prediction_distribution'].items():
            percentage = count / sum(metrics['prediction_distribution'].values()) * 100
            report += f"- **{class_name}**: {count} predictions ({percentage:.1f}%)\n"
        
        confidence_stats = metrics['confidence_stats']
        report += f"""
## Confidence Analysis
- **Mean Confidence**: {confidence_stats['mean']:.4f}
- **Std Confidence**: {confidence_stats['std']:.4f}
- **Min Confidence**: {confidence_stats['min']:.4f}
- **Max Confidence**: {confidence_stats['max']:.4f}
- **Median Confidence**: {confidence_stats['median']:.4f}

## Files Generated
- `evaluation_results.json`: Complete results in JSON format
- `metrics_summary.csv`: Summary metrics in CSV format
- `confusion_matrix.png`: Confusion matrix visualization
- `class_distribution.png`: Class distribution comparison
- `confidence_analysis.png`: Confidence distribution analysis
- `metrics_summary.png`: Performance metrics summary
"""
        
        if self.num_classes == 2:
            report += """- `roc_curve.png`: ROC curve
- `precision_recall_curve.png`: Precision-Recall curve
- `interactive_roc_curve.html`: Interactive ROC curve
"""
        
        report += """- `interactive_confusion_matrix.html`: Interactive confusion matrix

## Recommendations
"""
        
        # Add recommendations based on performance
        accuracy = metrics['accuracy']
        if accuracy >= 0.95:
            report += "- Excellent model performance! Consider deployment.\n"
        elif accuracy >= 0.90:
            report += "- Very good model performance. Minor improvements possible.\n"
        elif accuracy >= 0.80:
            report += "- Good model performance. Consider additional training or data augmentation.\n"
        else:
            report += "- Model performance needs improvement. Consider architecture changes or more data.\n"
        
        if self.num_classes == 2:
            sensitivity = metrics['sensitivity']
            specificity = metrics['specificity']
            
            if sensitivity < 0.8:
                report += "- Low sensitivity detected. Model may miss positive cases.\n"
            if specificity < 0.8:
                report += "- Low specificity detected. Model may have false positives.\n"
        
        # Check class imbalance
        sample_counts = list(metrics['sample_distribution'].values())
        if max(sample_counts) / min(sample_counts) > 3:
            report += "- Significant class imbalance detected. Consider balancing techniques.\n"
        
        confidence_mean = confidence_stats['mean']
        if confidence_mean < 0.7:
            report += "- Low average confidence. Model may be uncertain about predictions.\n"
        
        # Save report
        report_path = os.path.join(save_dir, 'evaluation_report.md')
        with open(report_path, 'w') as f:
            f.write(report)


def evaluate_saved_model(
    model_path: str,
    test_dataloader,
    class_names: List[str] = None,
    device: str = 'cpu',
    save_dir: str = 'evaluation_results'
) -> Dict:
    """
    Convenience function to evaluate a saved model
    
    Args:
        model_path: Path to saved model
        test_dataloader: Test data loader
        class_names: List of class names
        device: Computing device
        save_dir: Directory to save results
        
    Returns:
        Evaluation results dictionary
    """
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = checkpoint['model']
    model.to(device)
    model.eval()
    
    print(f"Loaded model with best accuracy: {checkpoint.get('best_acc', 'N/A')}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(class_names)
    
    # Run evaluation
    results = evaluator.evaluate_model(
        model=model,
        dataloader=test_dataloader,
        device=device,
        save_results=True,
        save_dir=save_dir
    )
    
    return results