"""
Evaluation and Comparative Analysis for DenseNet vs Swin Transformer V2
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime

from preprocessing_pipeline import ModelFactory
from simple_dataset import MyDataset
import config_clean as config

class ModelEvaluator:
    """Evaluate trained models"""
    
    def __init__(self, model, model_name, device):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.model.eval()
    
    def evaluate(self, test_loader):
        """Comprehensive evaluation"""
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                probabilities = F.softmax(output, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted')
        cm = confusion_matrix(all_targets, all_predictions)
        
        return {
            'model_name': self.model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }

class ComprehensiveComparison:
    """Detailed comparison between models"""
    
    def __init__(self):
        self.evaluation_results = {}
    
    def add_evaluation(self, model_name, results):
        """Add evaluation results"""
        self.evaluation_results[model_name] = results
    
    def generate_comparison_report(self):
        """Generate detailed comparison report"""
        if len(self.evaluation_results) < 2:
            return "Need at least 2 models for comparison"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_evaluated': list(self.evaluation_results.keys()),
            'detailed_comparison': {},
            'summary': {}
        }
        
        # Detailed metrics comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for metric in metrics:
            report['detailed_comparison'][metric] = {}
            for model_name, results in self.evaluation_results.items():
                report['detailed_comparison'][metric][model_name] = results[metric]
        
        # Find best performing model for each metric
        best_models = {}
        for metric in metrics:
            best_model = max(self.evaluation_results.keys(), 
                           key=lambda x: self.evaluation_results[x][metric])
            best_models[metric] = {
                'model': best_model,
                'value': self.evaluation_results[best_model][metric]
            }
        
        report['summary'] = {
            'best_models_by_metric': best_models,
            'overall_best': self._determine_overall_best(),
            'performance_gaps': self._calculate_performance_gaps()
        }
        
        return report
    
    def _determine_overall_best(self):
        """Determine overall best model based on weighted metrics"""
        weights = {'accuracy': 0.4, 'precision': 0.2, 'recall': 0.2, 'f1_score': 0.2}
        
        scores = {}
        for model_name, results in self.evaluation_results.items():
            weighted_score = sum(weights[metric] * results[metric] for metric in weights.keys())
            scores[model_name] = weighted_score
        
        best_model = max(scores, key=scores.get)
        return {
            'model': best_model,
            'weighted_score': scores[best_model],
            'all_scores': scores
        }
    
    def _calculate_performance_gaps(self):
        """Calculate performance gaps between models"""
        if len(self.evaluation_results) != 2:
            return "Performance gap calculation requires exactly 2 models"
        
        models = list(self.evaluation_results.keys())
        model1, model2 = models[0], models[1]
        
        gaps = {}
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for metric in metrics:
            gap = self.evaluation_results[model1][metric] - self.evaluation_results[model2][metric]
            gaps[metric] = {
                'absolute_gap': gap,
                'percentage_gap': (gap / self.evaluation_results[model2][metric]) * 100,
                'better_model': model1 if gap > 0 else model2
            }
        
        return gaps
    
    def create_visualizations(self):
        """Create comparison visualizations"""
        if len(self.evaluation_results) < 2:
            return
        
        # Create results directory
        os.makedirs('results/visualizations', exist_ok=True)
        
        # 1. Metrics comparison bar chart
        self._plot_metrics_comparison()
        
        # 2. Confusion matrices
        self._plot_confusion_matrices()
        
        print("Visualizations saved to results/visualizations/")
    
    def _plot_metrics_comparison(self):
        """Plot metrics comparison"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        models = list(self.evaluation_results.keys())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, model in enumerate(models):
            values = [self.evaluation_results[model][metric] for metric in metrics]
            ax.bar(x + i*width, values, width, label=model)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/visualizations/metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrices(self):
        """Plot confusion matrices for both models"""
        fig, axes = plt.subplots(1, len(self.evaluation_results), figsize=(12, 5))
        
        if len(self.evaluation_results) == 1:
            axes = [axes]
        
        for i, (model_name, results) in enumerate(self.evaluation_results.items()):
            cm = np.array(results['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{model_name} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('results/visualizations/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_report(self, filename='detailed_comparison_report.json'):
        """Save detailed comparison report"""
        report = self.generate_comparison_report()
        
        os.makedirs('results', exist_ok=True)
        with open(f'results/{filename}', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Detailed comparison report saved to results/{filename}")
        return report

def evaluate_models():
    """Main evaluation function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create preprocessing
    preprocessing = ModelFactory.get_preprocessing()
    
    # Create test dataset
    test_dataset = MyDataset(config.valid, preprocessing.get_densenet_transform())
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    test_dataset_swin = MyDataset(config.valid, preprocessing.get_swin_transform())
    test_loader_swin = DataLoader(test_dataset_swin, batch_size=config.batch_size, shuffle=False)
    
    # Initialize comparison
    comparison = ComprehensiveComparison()
    
    # Evaluate DenseNet
    print("Evaluating DenseNet...")
    densenet = ModelFactory.create_densenet()
    
    # Load trained weights if available
    densenet_path = 'weight/DenseNet_best.pth'
    if os.path.exists(densenet_path):
        densenet.load_state_dict(torch.load(densenet_path, map_location=device))
        print("Loaded DenseNet weights")
    else:
        print("No trained DenseNet weights found, using random initialization")
    
    densenet_evaluator = ModelEvaluator(densenet, "DenseNet", device)
    densenet_results = densenet_evaluator.evaluate(test_loader)
    comparison.add_evaluation("DenseNet", densenet_results)
    
    # Evaluate Swin Transformer
    print("Evaluating Swin Transformer V2...")
    swin = ModelFactory.create_swin()
    
    # Load trained weights if available
    swin_path = 'weight/SwinTransformerV2_best.pth'
    if os.path.exists(swin_path):
        swin.load_state_dict(torch.load(swin_path, map_location=device))
        print("Loaded Swin Transformer weights")
    else:
        print("No trained Swin Transformer weights found, using random initialization")
    
    swin_evaluator = ModelEvaluator(swin, "SwinTransformerV2", device)
    swin_results = swin_evaluator.evaluate(test_loader_swin)
    comparison.add_evaluation("SwinTransformerV2", swin_results)
    
    # Generate comprehensive report
    print("="*50)
    print("GENERATING COMPREHENSIVE COMPARISON REPORT")
    print("="*50)
    
    report = comparison.save_report()
    comparison.create_visualizations()
    
    # Print summary
    print("\nEVALUATION RESULTS:")
    print("-" * 30)
    
    for model_name, results in comparison.evaluation_results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1-Score: {results['f1_score']:.4f}")
    
    print(f"\nOverall Best Model: {report['summary']['overall_best']['model']}")
    print(f"Weighted Score: {report['summary']['overall_best']['weighted_score']:.4f}")
    
    return report

if __name__ == "__main__":
    evaluate_models()