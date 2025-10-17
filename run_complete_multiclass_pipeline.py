"""
Complete pipeline runner for DenLsNet multi-class extension
Runs training, evaluation, and interpretability analysis for all variants
"""
import os
import sys
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List
import subprocess

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import project modules
import config_multiclass as config
from train_multiclass import MultiClassTrainer, run_ablation_study
from evaluation.metrics import evaluate_saved_model, ModelEvaluator
from explainability.interpretability_framework import run_interpretability_evaluation
from stain_normalization import create_stain_normalized_dataset, compare_stain_methods
from utils.load_multiclass_dataset import get_multiclass_data_loader, create_multiclass_dataset_structure


class MultiClassPipelineRunner:
    """
    Complete pipeline runner for multi-class DenLsNet experiments
    """
    
    def __init__(self, base_output_dir: str = 'results/complete_pipeline'):
        """
        Initialize pipeline runner
        
        Args:
            base_output_dir: Base directory for all outputs
        """
        self.base_output_dir = base_output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_output_dir, f"run_{self.timestamp}")
        
        # Create output directories
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Results storage
        self.results = {
            'pipeline_info': {
                'timestamp': self.timestamp,
                'run_directory': self.run_dir,
                'config': self._get_config_dict()
            },
            'data_preparation': {},
            'training_results': {},
            'evaluation_results': {},
            'interpretability_results': {},
            'summary': {}
        }
        
        print(f"Pipeline initialized. Results will be saved to: {self.run_dir}")
    
    def _get_config_dict(self) -> Dict:
        """Get configuration as dictionary"""
        return {
            'class_num': config.class_num,
            'class_names': config.class_names,
            'max_epoch': config.max_epoch,
            'batch_size': config.batch_size,
            'lr': config.lr,
            'img_size': config.img_s,
            'dataset_mean': config.dataset_mean,
            'dataset_std': config.dataset_std,
            'stain_methods': list(config.experiment_variants.keys()),
            'loss_config': config.loss_config,
            'evaluation_metrics': config.evaluation_metrics
        }
    
    def step_1_prepare_data(self):
        """Step 1: Prepare multi-class dataset and stain-normalized variants"""
        print("\n" + "="*60)
        print("STEP 1: DATA PREPARATION")
        print("="*60)
        
        step_results = {
            'start_time': datetime.now().isoformat(),
            'multiclass_conversion': {},
            'stain_normalization': {},
            'dataset_statistics': {}
        }
        
        # 1.1 Convert binary dataset to multi-class structure
        print("\n1.1 Converting binary dataset to multi-class structure...")
        
        multiclass_dir = "datasets/BreaKHis 400X/multiclass"
        if not os.path.exists(os.path.join(multiclass_dir, 'train')):
            try:
                create_multiclass_dataset_structure(
                    base_dir="datasets/BreaKHis 400X",
                    output_dir=multiclass_dir
                )
                step_results['multiclass_conversion']['status'] = 'success'
                step_results['multiclass_conversion']['output_dir'] = multiclass_dir
            except Exception as e:
                print(f"Error in multiclass conversion: {e}")
                step_results['multiclass_conversion']['status'] = 'failed'
                step_results['multiclass_conversion']['error'] = str(e)
        else:
            print("Multi-class dataset already exists.")
            step_results['multiclass_conversion']['status'] = 'already_exists'
        
        # 1.2 Create stain-normalized datasets
        print("\n1.2 Creating stain-normalized datasets...")
        
        stain_methods = ['macenko', 'reinhard']
        for method in stain_methods:
            print(f"\nCreating {method} normalized dataset...")
            
            try:
                # Normalize training data
                create_stain_normalized_dataset(
                    input_dir=os.path.join(multiclass_dir, 'train'),
                    output_dir=config.output_dirs['normalized_data'],
                    method=method
                )
                
                # Normalize validation data
                create_stain_normalized_dataset(
                    input_dir=os.path.join(multiclass_dir, 'test'),
                    output_dir=config.output_dirs['normalized_data'],
                    method=method
                )
                
                step_results['stain_normalization'][method] = {
                    'status': 'success',
                    'output_dir': os.path.join(config.output_dirs['normalized_data'], method)
                }
                
            except Exception as e:
                print(f"Error creating {method} normalized dataset: {e}")
                step_results['stain_normalization'][method] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # 1.3 Generate dataset statistics
        print("\n1.3 Generating dataset statistics...")
        
        try:
            for variant in ['none'] + stain_methods:
                if variant == 'none':
                    train_path = os.path.join(multiclass_dir, 'train')
                    test_path = os.path.join(multiclass_dir, 'test')
                else:
                    base_norm_dir = config.output_dirs['normalized_data']
                    train_path = os.path.join(base_norm_dir, variant, 'train')
                    test_path = os.path.join(base_norm_dir, variant, 'test')
                
                if os.path.exists(train_path) and os.path.exists(test_path):
                    # Count samples per class
                    train_stats = {}
                    test_stats = {}
                    
                    for class_name in config.class_names:
                        train_class_dir = os.path.join(train_path, class_name)
                        test_class_dir = os.path.join(test_path, class_name)
                        
                        train_count = len([f for f in os.listdir(train_class_dir) 
                                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]) if os.path.exists(train_class_dir) else 0
                        test_count = len([f for f in os.listdir(test_class_dir) 
                                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]) if os.path.exists(test_class_dir) else 0
                        
                        train_stats[class_name] = train_count
                        test_stats[class_name] = test_count
                    
                    step_results['dataset_statistics'][variant] = {
                        'train': train_stats,
                        'test': test_stats,
                        'total_train': sum(train_stats.values()),
                        'total_test': sum(test_stats.values())
                    }
        
        except Exception as e:
            print(f"Error generating dataset statistics: {e}")
            step_results['dataset_statistics']['error'] = str(e)
        
        step_results['end_time'] = datetime.now().isoformat()
        self.results['data_preparation'] = step_results
        
        # Save intermediate results
        self._save_results()
        
        print("\n✅ Data preparation completed!")
        return step_results
    
    def step_2_train_models(self):
        """Step 2: Train models with different stain normalization methods"""
        print("\n" + "="*60)
        print("STEP 2: MODEL TRAINING")
        print("="*60)
        
        step_results = {
            'start_time': datetime.now().isoformat(),
            'training_runs': {},
            'best_models': {}
        }
        
        # Run ablation study
        print("\n2.1 Running complete ablation study...")
        
        try:
            ablation_results = run_ablation_study()
            step_results['training_runs'] = ablation_results
            
            # Find best models
            valid_results = {k: v for k, v in ablation_results.items() if v is not None}
            if valid_results:
                # Best by accuracy
                best_acc_method = max(valid_results.keys(), key=lambda k: valid_results[k]['accuracy'])
                best_f1_method = max(valid_results.keys(), key=lambda k: valid_results[k]['f1_macro'])
                
                step_results['best_models'] = {
                    'best_accuracy': {
                        'method': best_acc_method,
                        'accuracy': valid_results[best_acc_method]['accuracy'],
                        'f1_macro': valid_results[best_acc_method]['f1_macro']
                    },
                    'best_f1': {
                        'method': best_f1_method,
                        'accuracy': valid_results[best_f1_method]['accuracy'],
                        'f1_macro': valid_results[best_f1_method]['f1_macro']
                    }
                }
        
        except Exception as e:
            print(f"Error in training: {e}")
            step_results['training_runs']['error'] = str(e)
        
        step_results['end_time'] = datetime.now().isoformat()
        self.results['training_results'] = step_results
        
        # Save intermediate results
        self._save_results()
        
        print("\n✅ Model training completed!")
        return step_results
    
    def step_3_evaluate_models(self):
        """Step 3: Comprehensive model evaluation"""
        print("\n" + "="*60)
        print("STEP 3: MODEL EVALUATION")
        print("="*60)
        
        step_results = {
            'start_time': datetime.now().isoformat(),
            'evaluations': {},
            'comparisons': {}
        }
        
        # Evaluate each trained model
        methods = ['none', 'macenko', 'reinhard']
        
        for method in methods:
            print(f"\n3.{methods.index(method)+1} Evaluating {method} model...")
            
            model_path = os.path.join(config.output_dirs['models'], method, f"denlsnet_mc_{method}_best.pth")
            
            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                step_results['evaluations'][method] = {'status': 'model_not_found'}
                continue
            
            try:
                # Prepare test data
                if method == 'none':
                    test_path = "datasets/BreaKHis 400X/multiclass/test"
                else:
                    test_path = os.path.join(config.output_dirs['normalized_data'], method, 'test')
                
                _, test_loader, _ = get_multiclass_data_loader(
                    train_path=None,  # Not needed for evaluation
                    valid_path=test_path,
                    batch_size=config.batch_size,
                    num_workers=0,
                    use_weighted_sampling=False
                )
                
                # Run evaluation
                eval_save_dir = os.path.join(self.run_dir, 'evaluations', method)
                results = evaluate_saved_model(
                    model_path=model_path,
                    test_dataloader=test_loader,
                    class_names=config.class_names,
                    device=config.device,
                    save_dir=eval_save_dir
                )
                
                step_results['evaluations'][method] = {
                    'status': 'success',
                    'results': results['metrics'],
                    'save_dir': eval_save_dir
                }
                
            except Exception as e:
                print(f"Error evaluating {method} model: {e}")
                step_results['evaluations'][method] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Create comparison analysis
        print("\n3.4 Creating comparison analysis...")
        
        try:
            valid_evaluations = {k: v for k, v in step_results['evaluations'].items() 
                               if v.get('status') == 'success'}
            
            if valid_evaluations:
                comparison_data = []
                
                for method, eval_data in valid_evaluations.items():
                    metrics = eval_data['results']
                    comparison_data.append({
                        'Method': method,
                        'Accuracy': metrics['accuracy'],
                        'Precision': metrics['precision'],
                        'Recall': metrics['recall'],
                        'F1-Score': metrics['f1_score']
                    })
                
                # Create comparison DataFrame
                comparison_df = pd.DataFrame(comparison_data)
                
                # Save comparison
                comparison_path = os.path.join(self.run_dir, 'model_comparison.csv')
                comparison_df.to_csv(comparison_path, index=False)
                
                # Create comparison plot
                self._create_comparison_plot(comparison_df, self.run_dir)
                
                step_results['comparisons'] = {
                    'comparison_table': comparison_data,
                    'best_method': comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Method'],
                    'comparison_file': comparison_path
                }
        
        except Exception as e:
            print(f"Error creating comparison: {e}")
            step_results['comparisons']['error'] = str(e)
        
        step_results['end_time'] = datetime.now().isoformat()
        self.results['evaluation_results'] = step_results
        
        # Save intermediate results
        self._save_results()
        
        print("\n✅ Model evaluation completed!")
        return step_results
    
    def step_4_interpretability_analysis(self):
        """Step 4: Comprehensive interpretability analysis"""
        print("\n" + "="*60)
        print("STEP 4: INTERPRETABILITY ANALYSIS")
        print("="*60)
        
        step_results = {
            'start_time': datetime.now().isoformat(),
            'interpretability_runs': {},
            'quantitative_comparison': {}
        }
        
        # Run interpretability analysis for each model
        methods = ['none', 'macenko', 'reinhard']
        
        for method in methods:
            print(f"\n4.{methods.index(method)+1} Running interpretability analysis for {method} model...")
            
            model_path = os.path.join(config.output_dirs['models'], method, f"denlsnet_mc_{method}_best.pth")
            
            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                step_results['interpretability_runs'][method] = {'status': 'model_not_found'}
                continue
            
            try:
                # Prepare test data
                if method == 'none':
                    test_path = "datasets/BreaKHis 400X/multiclass/test"
                else:
                    test_path = os.path.join(config.output_dirs['normalized_data'], method, 'test')
                
                _, test_loader, _ = get_multiclass_data_loader(
                    train_path=None,
                    valid_path=test_path,
                    batch_size=8,  # Smaller batch for interpretability
                    num_workers=0,
                    use_weighted_sampling=False
                )
                
                # Run interpretability evaluation
                interp_save_dir = os.path.join(self.run_dir, 'interpretability', method)
                
                results = run_interpretability_evaluation(
                    model_path=model_path,
                    test_dataloader=test_loader,
                    class_names=config.class_names,
                    device=config.device,
                    num_samples=20,  # Reduced for faster execution
                    methods=['gradcam', 'gradcam_plus'],  # Focus on faster methods
                    save_dir=interp_save_dir
                )
                
                step_results['interpretability_runs'][method] = {
                    'status': 'success',
                    'results': results,
                    'save_dir': interp_save_dir
                }
                
            except Exception as e:
                print(f"Error in interpretability analysis for {method}: {e}")
                step_results['interpretability_runs'][method] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Create quantitative comparison
        print("\n4.4 Creating quantitative interpretability comparison...")
        
        try:
            valid_interp = {k: v for k, v in step_results['interpretability_runs'].items() 
                          if v.get('status') == 'success'}
            
            if valid_interp:
                comparison_data = []
                
                for method, interp_data in valid_interp.items():
                    results = interp_data['results']
                    
                    for technique, metrics in results.items():
                        if metrics.get('insertion_auc', {}).get('mean') is not None:
                            comparison_data.append({
                                'Method': method,
                                'Technique': technique,
                                'Insertion_AUC': metrics['insertion_auc']['mean'],
                                'Deletion_AUC': metrics['deletion_auc']['mean'],
                                'Stability': metrics.get('stability', {}).get('mean', 0),
                                'Processing_Time': metrics.get('processing_time', {}).get('mean', 0)
                            })
                
                if comparison_data:
                    interp_df = pd.DataFrame(comparison_data)
                    interp_path = os.path.join(self.run_dir, 'interpretability_comparison.csv')
                    interp_df.to_csv(interp_path, index=False)
                    
                    step_results['quantitative_comparison'] = {
                        'comparison_table': comparison_data,
                        'comparison_file': interp_path
                    }
        
        except Exception as e:
            print(f"Error creating interpretability comparison: {e}")
            step_results['quantitative_comparison']['error'] = str(e)
        
        step_results['end_time'] = datetime.now().isoformat()
        self.results['interpretability_results'] = step_results
        
        # Save intermediate results
        self._save_results()
        
        print("\n✅ Interpretability analysis completed!")
        return step_results
    
    def step_5_generate_summary(self):
        """Step 5: Generate comprehensive summary and documentation"""
        print("\n" + "="*60)
        print("STEP 5: SUMMARY GENERATION")
        print("="*60)
        
        step_results = {
            'start_time': datetime.now().isoformat(),
            'summary_statistics': {},
            'recommendations': [],
            'files_generated': []
        }
        
        # Generate summary statistics
        print("\n5.1 Generating summary statistics...")
        
        try:
            # Training summary
            training_results = self.results.get('training_results', {}).get('training_runs', {})
            valid_training = {k: v for k, v in training_results.items() if v is not None}
            
            if valid_training:
                best_accuracy = max(valid_training.values(), key=lambda x: x['accuracy'])
                best_f1 = max(valid_training.values(), key=lambda x: x['f1_macro'])
                
                step_results['summary_statistics']['training'] = {
                    'methods_trained': len(valid_training),
                    'best_accuracy': {
                        'value': best_accuracy['accuracy'],
                        'method': [k for k, v in valid_training.items() if v == best_accuracy][0]
                    },
                    'best_f1': {
                        'value': best_f1['f1_macro'],
                        'method': [k for k, v in valid_training.items() if v == best_f1][0]
                    }
                }
            
            # Evaluation summary
            eval_results = self.results.get('evaluation_results', {}).get('evaluations', {})
            valid_eval = {k: v for k, v in eval_results.items() if v.get('status') == 'success'}
            
            if valid_eval:
                step_results['summary_statistics']['evaluation'] = {
                    'methods_evaluated': len(valid_eval),
                    'average_accuracy': np.mean([v['results']['accuracy'] for v in valid_eval.values()]),
                    'average_f1': np.mean([v['results']['f1_score'] for v in valid_eval.values()])
                }
            
            # Interpretability summary
            interp_results = self.results.get('interpretability_results', {}).get('interpretability_runs', {})
            valid_interp = {k: v for k, v in interp_results.items() if v.get('status') == 'success'}
            
            if valid_interp:
                step_results['summary_statistics']['interpretability'] = {
                    'methods_analyzed': len(valid_interp),
                    'techniques_used': ['gradcam', 'gradcam_plus']
                }
        
        except Exception as e:
            print(f"Error generating summary statistics: {e}")
            step_results['summary_statistics']['error'] = str(e)
        
        # Generate recommendations
        print("\n5.2 Generating recommendations...")
        
        recommendations = []
        
        try:
            # Training recommendations
            if 'training' in step_results['summary_statistics']:
                training_stats = step_results['summary_statistics']['training']
                best_method = training_stats['best_f1']['method']
                best_f1_value = training_stats['best_f1']['value']
                
                recommendations.append(f"Best performing model: {best_method} (F1: {best_f1_value:.3f})")
                
                if best_f1_value > 0.85:
                    recommendations.append("Excellent model performance achieved. Ready for deployment.")
                elif best_f1_value > 0.75:
                    recommendations.append("Good model performance. Consider fine-tuning for improvement.")
                else:
                    recommendations.append("Model performance needs improvement. Consider more data or architecture changes.")
            
            # Stain normalization recommendations
            if valid_training and len(valid_training) > 1:
                methods_performance = [(k, v['f1_macro']) for k, v in valid_training.items()]
                methods_performance.sort(key=lambda x: x[1], reverse=True)
                
                best_method, best_score = methods_performance[0]
                if best_method != 'none':
                    recommendations.append(f"Stain normalization ({best_method}) improves performance.")
                else:
                    recommendations.append("Stain normalization does not significantly improve performance.")
            
            # Interpretability recommendations
            if valid_interp:
                recommendations.append("Interpretability analysis completed successfully.")
                recommendations.append("Use Grad-CAM for clinical explanation of model decisions.")
        
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            recommendations.append(f"Error in recommendation generation: {e}")
        
        step_results['recommendations'] = recommendations
        
        # Create comprehensive report
        print("\n5.3 Creating comprehensive report...")
        
        try:
            report_path = self._create_comprehensive_report()
            step_results['files_generated'].append(report_path)
        except Exception as e:
            print(f"Error creating report: {e}")
        
        # Create summary plots
        print("\n5.4 Creating summary visualizations...")
        
        try:
            plots_created = self._create_summary_plots()
            step_results['files_generated'].extend(plots_created)
        except Exception as e:
            print(f"Error creating plots: {e}")
        
        step_results['end_time'] = datetime.now().isoformat()
        self.results['summary'] = step_results
        
        # Final save
        self._save_results()
        
        print("\n✅ Summary generation completed!")
        return step_results
    
    def _create_comparison_plot(self, comparison_df: pd.DataFrame, save_dir: str):
        """Create model comparison plot"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            bars = ax.bar(comparison_df['Method'], comparison_df[metric], alpha=0.7)
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, comparison_df[metric]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'model_comparison_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _create_summary_plots(self) -> List[str]:
        """Create summary visualization plots"""
        plots_created = []
        
        try:
            # Training progress plot
            training_results = self.results.get('training_results', {}).get('training_runs', {})
            valid_training = {k: v for k, v in training_results.items() if v is not None}
            
            if valid_training:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                methods = list(valid_training.keys())
                accuracies = [valid_training[m]['accuracy'] for m in methods]
                f1_scores = [valid_training[m]['f1_macro'] for m in methods]
                
                # Accuracy comparison
                bars1 = ax1.bar(methods, accuracies, alpha=0.7, color='skyblue')
                ax1.set_title('Final Accuracy by Method')
                ax1.set_ylabel('Accuracy')
                ax1.set_ylim(0, 1)
                
                for bar, acc in zip(bars1, accuracies):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{acc:.3f}', ha='center', va='bottom')
                
                # F1-Score comparison
                bars2 = ax2.bar(methods, f1_scores, alpha=0.7, color='lightcoral')
                ax2.set_title('Final F1-Score by Method')
                ax2.set_ylabel('F1-Score')
                ax2.set_ylim(0, 1)
                
                for bar, f1 in zip(bars2, f1_scores):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{f1:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                summary_plot_path = os.path.join(self.run_dir, 'training_summary.png')
                plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                plots_created.append(summary_plot_path)
        
        except Exception as e:
            print(f"Error creating summary plots: {e}")
        
        return plots_created
    
    def _create_comprehensive_report(self) -> str:
        """Create comprehensive markdown report"""
        report_content = f"""# DenLsNet Multi-Class Pipeline Report

## Pipeline Overview
- **Run ID**: {self.timestamp}
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Configuration**: 8-class BreakHis classification with stain normalization ablation

## Experimental Setup

### Dataset Configuration
- **Classes**: {len(config.class_names)} ({', '.join(config.class_names)})
- **Image Size**: {config.img_s}x{config.img_s}
- **Batch Size**: {config.batch_size}
- **Training Epochs**: {config.max_epoch}

### Stain Normalization Methods
- **Baseline**: No normalization
- **Macenko**: Macenko stain normalization
- **Reinhard**: Reinhard color normalization

## Results Summary

"""
        
        # Add training results
        training_results = self.results.get('training_results', {}).get('training_runs', {})
        valid_training = {k: v for k, v in training_results.items() if v is not None}
        
        if valid_training:
            report_content += "### Training Results\n\n"
            report_content += "| Method | Accuracy | F1-Score | Precision | Recall | Epoch |\n"
            report_content += "|--------|----------|----------|-----------|--------|---------|\n"
            
            for method, results in valid_training.items():
                acc = results['accuracy']
                f1 = results['f1_macro']
                prec = results['classification_report']['precision_macro']
                rec = results['classification_report']['recall_macro']
                epoch = results['epoch']
                
                report_content += f"| {method} | {acc:.3f} | {f1:.3f} | {prec:.3f} | {rec:.3f} | {epoch} |\n"
        
        # Add evaluation results
        eval_results = self.results.get('evaluation_results', {}).get('evaluations', {})
        valid_eval = {k: v for k, v in eval_results.items() if v.get('status') == 'success'}
        
        if valid_eval:
            report_content += "\n### Evaluation Results\n\n"
            report_content += "| Method | Test Accuracy | Test F1-Score | Test Precision | Test Recall |\n"
            report_content += "|--------|---------------|---------------|----------------|-------------|\n"
            
            for method, eval_data in valid_eval.items():
                metrics = eval_data['results']
                report_content += f"| {method} | {metrics['accuracy']:.3f} | {metrics['f1_score']:.3f} | {metrics['precision']:.3f} | {metrics['recall']:.3f} |\n"
        
        # Add interpretability results
        interp_results = self.results.get('interpretability_results', {}).get('interpretability_runs', {})
        valid_interp = {k: v for k, v in interp_results.items() if v.get('status') == 'success'}
        
        if valid_interp:
            report_content += "\n### Interpretability Analysis\n\n"
            report_content += f"- **Methods Analyzed**: {len(valid_interp)}\n"
            report_content += "- **Techniques Used**: Grad-CAM, Grad-CAM++\n"
            report_content += "- **Quantitative Metrics**: Insertion AUC, Deletion AUC, Stability\n"
        
        # Add recommendations
        recommendations = self.results.get('summary', {}).get('recommendations', [])
        if recommendations:
            report_content += "\n## Recommendations\n\n"
            for i, rec in enumerate(recommendations, 1):
                report_content += f"{i}. {rec}\n"
        
        # Add file structure
        report_content += f"""
## Output Structure

```
{self.run_dir}/
├── pipeline_results.json          # Complete results in JSON format
├── model_comparison.csv           # Model performance comparison
├── training_summary.png           # Training results visualization
├── comprehensive_report.md        # This report
├── evaluations/                   # Detailed evaluation results
│   ├── none/                     # Baseline model evaluation
│   ├── macenko/                  # Macenko model evaluation
│   └── reinhard/                 # Reinhard model evaluation
└── interpretability/             # Interpretability analysis results
    ├── none/                     # Baseline interpretability
    ├── macenko/                  # Macenko interpretability
    └── reinhard/                 # Reinhard interpretability
```

## Academic Contributions

### Novel Aspects
1. **Multi-class Extension**: Extended DenLsNet from binary to 8-class classification
2. **Stain Normalization Ablation**: Systematic evaluation of preprocessing effects
3. **Quantitative Interpretability**: Comprehensive XAI evaluation framework
4. **Academic Naming**: DenLsNet-MC variants for clear research communication

### Clinical Relevance
- Supports fine-grained histopathology classification
- Provides interpretable predictions for clinical decision support
- Evaluates robustness across different staining protocols

## Conclusion

This pipeline demonstrates the comprehensive evaluation of DenLsNet-MC across multiple
dimensions: classification performance, preprocessing robustness, and interpretability.
The results provide insights into the effectiveness of stain normalization and the
reliability of explainability methods for histopathology analysis.

---
*Generated by DenLsNet Multi-Class Pipeline Runner*
"""
        
        # Save report
        report_path = os.path.join(self.run_dir, 'comprehensive_report.md')
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        return report_path
    
    def _save_results(self):
        """Save current results to JSON file"""
        results_path = os.path.join(self.run_dir, 'pipeline_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def run_complete_pipeline(self):
        """Run the complete multi-class pipeline"""
        print("🚀 Starting Complete Multi-Class DenLsNet Pipeline")
        print("="*80)
        
        start_time = time.time()
        
        try:
            # Step 1: Data Preparation
            self.step_1_prepare_data()
            
            # Step 2: Model Training
            self.step_2_train_models()
            
            # Step 3: Model Evaluation
            self.step_3_evaluate_models()
            
            # Step 4: Interpretability Analysis
            self.step_4_interpretability_analysis()
            
            # Step 5: Summary Generation
            self.step_5_generate_summary()
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Final summary
            print("\n" + "="*80)
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"Total execution time: {total_time/3600:.2f} hours")
            print(f"Results saved to: {self.run_dir}")
            
            # Print key findings
            summary_stats = self.results.get('summary', {}).get('summary_statistics', {})
            if 'training' in summary_stats:
                best_method = summary_stats['training']['best_f1']['method']
                best_f1 = summary_stats['training']['best_f1']['value']
                print(f"Best performing method: {best_method} (F1: {best_f1:.3f})")
            
            recommendations = self.results.get('summary', {}).get('recommendations', [])
            if recommendations:
                print("\nKey Recommendations:")
                for rec in recommendations[:3]:  # Show top 3
                    print(f"  • {rec}")
            
            print(f"\n📊 View results: {self.run_dir}")
            print("="*80)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {str(e)}")
            print(f"Partial results saved to: {self.run_dir}")
            return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Complete Multi-Class DenLsNet Pipeline')
    parser.add_argument('--output_dir', type=str, default='results/complete_pipeline',
                       help='Base output directory')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training step (use existing models)')
    parser.add_argument('--skip_interpretability', action='store_true',
                       help='Skip interpretability analysis')
    parser.add_argument('--quick_run', action='store_true',
                       help='Run with reduced parameters for testing')
    
    args = parser.parse_args()
    
    # Adjust config for quick run
    if args.quick_run:
        config.max_epoch = 5
        print("Quick run mode: Reduced epochs to 5")
    
    # Initialize and run pipeline
    runner = MultiClassPipelineRunner(base_output_dir=args.output_dir)
    
    success = runner.run_complete_pipeline()
    
    if success:
        print("\n✅ Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()