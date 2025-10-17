"""
Comprehensive Explainability Pipeline Runner for DenLsNet
Integrates all explainability methods with quantitative benchmarking and report generation
"""
import os
import sys
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

import torch
import numpy as np

# Import project modules
import config_multiclass as config
from utils.load_multiclass_dataset import get_multiclass_data_loader
from explainability.grad_cam import generate_comprehensive_gradcam_analysis
from explainability.shap_explainer import generate_comprehensive_shap_analysis
from explainability.lime_explainer import generate_comprehensive_lime_analysis
from explainability.quantitative_benchmarking import run_quantitative_benchmark
from explainability.report_generator import generate_explainability_report
from evaluation.metrics import evaluate_saved_model


class ComprehensiveExplainabilityPipeline:
    """
    Complete pipeline for explainability analysis with all requested features
    """
    
    def __init__(self, 
                 model_path: str,
                 test_data_path: str,
                 device: str = 'cpu',
                 class_names: List[str] = None,
                 output_dir: str = 'explainability_analysis'):
        """
        Initialize comprehensive explainability pipeline
        
        Args:
            model_path: Path to trained model
            test_data_path: Path to test dataset
            device: Computing device
            class_names: List of class names
            output_dir: Base output directory
        """
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.device = device
        self.class_names = class_names or config.class_names
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, f"comprehensive_analysis_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Results storage
        self.results = {
            'pipeline_info': {
                'timestamp': timestamp,
                'model_path': model_path,
                'test_data_path': test_data_path,
                'device': device,
                'class_names': self.class_names,
                'output_dir': self.output_dir
            },
            'model_performance': {},
            'gradcam_results': {},
            'shap_results': {},
            'lime_results': {},
            'benchmark_results': {},
            'execution_times': {}
        }
        
        print(f"Comprehensive Explainability Pipeline initialized")
        print(f"Output directory: {self.output_dir}")
    
    def load_model_and_data(self):
        """Load model and prepare data loaders"""
        print("\n" + "="*60)
        print("LOADING MODEL AND DATA")
        print("="*60)
        
        start_time = time.time()
        
        # Load model
        print(f"Loading model from: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model = checkpoint['model']
        self.model.to(self.device)
        self.model.eval()
        
        # Get model info
        model_info = {
            'best_accuracy': checkpoint.get('best_metrics', {}).get('accuracy', 'N/A'),
            'best_f1': checkpoint.get('best_metrics', {}).get('f1_macro', 'N/A'),
            'epoch': checkpoint.get('best_metrics', {}).get('epoch', 'N/A'),
            'stain_method': checkpoint.get('config', {}).get('stain_method', 'unknown')
        }
        
        print(f"Model loaded successfully:")
        print(f"  - Best Accuracy: {model_info['best_accuracy']}")
        print(f"  - Best F1-Score: {model_info['best_f1']}")
        print(f"  - Training Epoch: {model_info['epoch']}")
        print(f"  - Stain Method: {model_info['stain_method']}")
        
        # Load data
        print(f"\\nLoading test data from: {self.test_data_path}")
        _, self.test_loader, _ = get_multiclass_data_loader(
            train_path=None,
            valid_path=self.test_data_path,
            batch_size=16,  # Smaller batch for explainability
            num_workers=0,
            use_weighted_sampling=False
        )
        
        # Create background loader for SHAP (subset of test data)
        _, self.background_loader, _ = get_multiclass_data_loader(
            train_path=None,
            valid_path=self.test_data_path,
            batch_size=8,
            num_workers=0,
            use_weighted_sampling=False
        )
        
        print(f"Data loaded successfully:")
        print(f"  - Test samples: {len(self.test_loader.dataset)}")
        print(f"  - Test batches: {len(self.test_loader)}")
        
        load_time = time.time() - start_time
        self.results['execution_times']['data_loading'] = load_time
        print(f"  - Loading time: {load_time:.2f} seconds")
    
    def evaluate_model_performance(self):
        """Evaluate model performance for context"""
        print("\\n" + "="*60)
        print("MODEL PERFORMANCE EVALUATION")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Run comprehensive evaluation
            eval_results = evaluate_saved_model(
                model_path=self.model_path,
                test_dataloader=self.test_loader,
                class_names=self.class_names,
                device=self.device,
                save_dir=os.path.join(self.output_dir, 'model_evaluation')
            )
            
            # Store key metrics
            self.results['model_performance'] = eval_results['metrics']
            
            print("Model evaluation completed:")
            print(f"  - Accuracy: {eval_results['metrics']['accuracy']:.3f}")
            print(f"  - F1-Score: {eval_results['metrics']['f1_score']:.3f}")
            print(f"  - Precision: {eval_results['metrics']['precision']:.3f}")
            print(f"  - Recall: {eval_results['metrics']['recall']:.3f}")
            
        except Exception as e:
            print(f"Error in model evaluation: {e}")
            self.results['model_performance'] = {'error': str(e)}
        
        eval_time = time.time() - start_time
        self.results['execution_times']['model_evaluation'] = eval_time
        print(f"  - Evaluation time: {eval_time:.2f} seconds")
    
    def run_gradcam_analysis(self, 
                           correct_samples: int = 5, 
                           incorrect_samples: int = 5):
        """Run comprehensive Grad-CAM analysis"""
        print("\\n" + "="*60)
        print("GRAD-CAM ANALYSIS")
        print("="*60)
        
        start_time = time.time()
        
        try:
            gradcam_results = generate_comprehensive_gradcam_analysis(
                model=self.model,
                dataloader=self.test_loader,
                device=self.device,
                class_names=self.class_names,
                save_dir=os.path.join(self.output_dir, 'gradcam'),
                correct_samples=correct_samples,
                incorrect_samples=incorrect_samples
            )
            
            self.results['gradcam_results'] = gradcam_results
            
            print("Grad-CAM analysis completed:")
            print(f"  - Samples analyzed: {gradcam_results['samples_analyzed']}")
            print(f"  - Insertion AUC: {gradcam_results['insertion_auc']['mean']:.3f}")
            print(f"  - Deletion AUC: {gradcam_results['deletion_auc']['mean']:.3f}")
            print(f"  - Stability: {gradcam_results['stability']['mean']:.3f}")
            
        except Exception as e:
            print(f"Error in Grad-CAM analysis: {e}")
            self.results['gradcam_results'] = {'error': str(e)}
        
        gradcam_time = time.time() - start_time
        self.results['execution_times']['gradcam_analysis'] = gradcam_time
        print(f"  - Analysis time: {gradcam_time:.2f} seconds")
    
    def run_shap_analysis(self, samples_per_class: int = 5):
        """Run comprehensive SHAP analysis"""
        print("\\n" + "="*60)
        print("SHAP ANALYSIS")
        print("="*60)
        
        start_time = time.time()
        
        try:
            shap_results = generate_comprehensive_shap_analysis(
                model=self.model,
                dataloader=self.test_loader,
                background_loader=self.background_loader,
                device=self.device,
                class_names=self.class_names,
                save_dir=os.path.join(self.output_dir, 'shap'),
                samples_per_class=samples_per_class,
                background_size=50
            )
            
            self.results['shap_results'] = shap_results
            
            print("SHAP analysis completed:")
            print(f"  - Samples analyzed: {shap_results['total_samples']}")
            print(f"  - Background size: {shap_results['background_size']}")
            print(f"  - Mean importance: {shap_results['overall_statistics']['mean_importance']:.3f}")
            
        except Exception as e:
            print(f"Error in SHAP analysis: {e}")
            self.results['shap_results'] = {'error': str(e)}
        
        shap_time = time.time() - start_time
        self.results['execution_times']['shap_analysis'] = shap_time
        print(f"  - Analysis time: {shap_time:.2f} seconds")
    
    def run_lime_analysis(self, samples_per_class: int = 2):
        """Run comprehensive LIME analysis"""
        print("\\n" + "="*60)
        print("LIME ANALYSIS")
        print("="*60)
        
        start_time = time.time()
        
        try:
            lime_results = generate_comprehensive_lime_analysis(
                model=self.model,
                dataloader=self.test_loader,
                device=self.device,
                class_names=self.class_names,
                save_dir=os.path.join(self.output_dir, 'lime'),
                samples_per_class=samples_per_class,
                lime_samples=500
            )
            
            self.results['lime_results'] = lime_results
            
            print("LIME analysis completed:")
            print(f"  - Samples analyzed: {lime_results['total_samples']}")
            print(f"  - Avg superpixels per sample: {lime_results['overall_statistics']['avg_superpixels_per_sample']:.1f}")
            print(f"  - Avg positive features: {lime_results['overall_statistics']['avg_positive_features']:.1f}")
            
        except Exception as e:
            print(f"Error in LIME analysis: {e}")
            self.results['lime_results'] = {'error': str(e)}
        
        lime_time = time.time() - start_time
        self.results['execution_times']['lime_analysis'] = lime_time
        print(f"  - Analysis time: {lime_time:.2f} seconds")
    
    def run_quantitative_benchmark(self, num_samples: int = 30):
        """Run quantitative benchmarking of all methods"""
        print("\\n" + "="*60)
        print("QUANTITATIVE BENCHMARKING")
        print("="*60)
        
        start_time = time.time()
        
        try:
            benchmark_results = run_quantitative_benchmark(
                model_path=self.model_path,
                test_dataloader=self.test_loader,
                class_names=self.class_names,
                device=self.device,
                num_samples=num_samples,
                methods=['gradcam', 'gradcam_plus', 'shap', 'lime'],
                save_dir=os.path.join(self.output_dir, 'quantitative_benchmark')
            )
            
            self.results['benchmark_results'] = benchmark_results
            
            print("Quantitative benchmarking completed:")
            for method, results in benchmark_results.items():
                if results.get('insertion_auc', {}).get('mean'):
                    ins_auc = results['insertion_auc']['mean']
                    del_auc = results['deletion_auc']['mean']
                    stability = results['stability']['mean']
                    print(f"  - {method}: Ins={ins_auc:.3f}, Del={del_auc:.3f}, Stab={stability:.3f}")
            
        except Exception as e:
            print(f"Error in quantitative benchmarking: {e}")
            self.results['benchmark_results'] = {'error': str(e)}
        
        benchmark_time = time.time() - start_time
        self.results['execution_times']['quantitative_benchmark'] = benchmark_time
        print(f"  - Benchmarking time: {benchmark_time:.2f} seconds")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive visual report"""
        print("\\n" + "="*60)
        print("REPORT GENERATION")
        print("="*60)
        
        start_time = time.time()
        
        try:
            report_files = generate_explainability_report(
                results_dir=self.output_dir,
                model_performance=self.results['model_performance'],
                class_names=self.class_names,
                output_dir=os.path.join(self.output_dir, 'reports')
            )
            
            self.results['report_files'] = report_files
            
            print("Report generation completed:")
            for format_type, path in report_files.items():
                if path and os.path.exists(path):
                    print(f"  - {format_type.upper()}: {os.path.basename(path)}")
            
        except Exception as e:
            print(f"Error in report generation: {e}")
            self.results['report_files'] = {'error': str(e)}
        
        report_time = time.time() - start_time
        self.results['execution_times']['report_generation'] = report_time
        print(f"  - Report generation time: {report_time:.2f} seconds")
    
    def save_pipeline_results(self):
        """Save comprehensive pipeline results"""
        results_path = os.path.join(self.output_dir, 'comprehensive_results.json')
        
        # Calculate total execution time
        total_time = sum(self.results['execution_times'].values())
        self.results['execution_times']['total_pipeline'] = total_time
        
        # Save results
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\\nPipeline results saved: {results_path}")
    
    def create_summary_report(self):
        """Create executive summary of the analysis"""
        summary_content = f"""# Comprehensive Explainability Analysis Summary

## Analysis Overview
- **Date**: {self.results['pipeline_info']['timestamp']}
- **Model**: {os.path.basename(self.model_path)}
- **Classes**: {len(self.class_names)} ({', '.join(self.class_names)})
- **Total Execution Time**: {self.results['execution_times']['total_pipeline']:.2f} seconds

## Model Performance
"""
        
        if 'accuracy' in self.results['model_performance']:
            perf = self.results['model_performance']
            summary_content += f"""- **Accuracy**: {perf['accuracy']:.3f}
- **F1-Score**: {perf['f1_score']:.3f}
- **Precision**: {perf['precision']:.3f}
- **Recall**: {perf['recall']:.3f}

"""
        
        summary_content += "## Explainability Analysis Results\\n\\n"
        
        # Grad-CAM results
        if 'samples_analyzed' in self.results['gradcam_results']:
            gradcam = self.results['gradcam_results']
            summary_content += f"""### Grad-CAM Analysis
- **Samples Analyzed**: {gradcam['samples_analyzed']}
- **Insertion AUC**: {gradcam['insertion_auc']['mean']:.3f} ± {gradcam['insertion_auc']['std']:.3f}
- **Deletion AUC**: {gradcam['deletion_auc']['mean']:.3f} ± {gradcam['deletion_auc']['std']:.3f}
- **Stability**: {gradcam['stability']['mean']:.3f} ± {gradcam['stability']['std']:.3f}

"""
        
        # SHAP results
        if 'total_samples' in self.results['shap_results']:
            shap = self.results['shap_results']
            summary_content += f"""### SHAP Analysis
- **Samples Analyzed**: {shap['total_samples']}
- **Background Size**: {shap['background_size']}
- **Mean Importance**: {shap['overall_statistics']['mean_importance']:.3f}

"""
        
        # LIME results
        if 'total_samples' in self.results['lime_results']:
            lime = self.results['lime_results']
            summary_content += f"""### LIME Analysis
- **Samples Analyzed**: {lime['total_samples']}
- **Avg Superpixels per Sample**: {lime['overall_statistics']['avg_superpixels_per_sample']:.1f}
- **Avg Positive Features**: {lime['overall_statistics']['avg_positive_features']:.1f}
- **Avg Negative Features**: {lime['overall_statistics']['avg_negative_features']:.1f}

"""
        
        # Benchmark results
        if self.results['benchmark_results']:
            summary_content += "### Quantitative Benchmarking\\n\\n"
            summary_content += "| Method | Insertion AUC | Deletion AUC | Stability |\\n"
            summary_content += "|--------|---------------|--------------|-----------|\\n"
            
            for method, results in self.results['benchmark_results'].items():
                if 'error' not in results:
                    ins_auc = results.get('insertion_auc', {}).get('mean', 0)
                    del_auc = results.get('deletion_auc', {}).get('mean', 0)
                    stability = results.get('stability', {}).get('mean', 0)
                    summary_content += f"| {method} | {ins_auc:.3f} | {del_auc:.3f} | {stability:.3f} |\\n"
        
        # Execution times
        summary_content += f"""
## Execution Times
- **Data Loading**: {self.results['execution_times'].get('data_loading', 0):.2f}s
- **Model Evaluation**: {self.results['execution_times'].get('model_evaluation', 0):.2f}s
- **Grad-CAM Analysis**: {self.results['execution_times'].get('gradcam_analysis', 0):.2f}s
- **SHAP Analysis**: {self.results['execution_times'].get('shap_analysis', 0):.2f}s
- **LIME Analysis**: {self.results['execution_times'].get('lime_analysis', 0):.2f}s
- **Quantitative Benchmark**: {self.results['execution_times'].get('quantitative_benchmark', 0):.2f}s
- **Report Generation**: {self.results['execution_times'].get('report_generation', 0):.2f}s
- **Total Pipeline**: {self.results['execution_times'].get('total_pipeline', 0):.2f}s

## Output Files
All analysis results are saved in: `{self.output_dir}`

### Key Directories:
- `gradcam/`: Grad-CAM visualizations and metrics
- `shap/`: SHAP explanations and per-class analysis
- `lime/`: LIME explanations and JSON exports
- `quantitative_benchmark/`: Comparative metrics and benchmarking
- `reports/`: Comprehensive visual reports (Jupyter notebook, HTML, PDF)
- `model_evaluation/`: Model performance evaluation

### Generated Reports:
"""
        
        if 'report_files' in self.results:
            for format_type, path in self.results['report_files'].items():
                if path and os.path.exists(path):
                    summary_content += f"- **{format_type.upper()}**: {os.path.basename(path)}\\n"
        
        summary_content += f"""
## Recommendations

Based on the comprehensive analysis:

1. **Best Explanation Method**: Review quantitative benchmarking results to identify the most reliable method
2. **Clinical Integration**: Use Grad-CAM for quick visual explanations and SHAP for detailed analysis
3. **Quality Assurance**: Monitor explanation stability and faithfulness metrics
4. **Validation**: Cross-reference explanations with expert pathologist knowledge

## Next Steps

1. Review the comprehensive Jupyter notebook report for detailed analysis
2. Validate explanations with domain experts
3. Integrate best-performing method into clinical workflow
4. Monitor explanation quality in production environment

---

*This analysis was generated by the DenLsNet Comprehensive Explainability Pipeline*
"""
        
        # Save summary
        summary_path = os.path.join(self.output_dir, 'EXECUTIVE_SUMMARY.md')
        with open(summary_path, 'w') as f:
            f.write(summary_content)
        
        print(f"Executive summary saved: {summary_path}")
    
    def run_complete_pipeline(self, 
                            gradcam_samples: Tuple[int, int] = (5, 5),
                            shap_samples_per_class: int = 5,
                            lime_samples_per_class: int = 2,
                            benchmark_samples: int = 30):
        """
        Run the complete explainability pipeline
        
        Args:
            gradcam_samples: Tuple of (correct_samples, incorrect_samples) for Grad-CAM
            shap_samples_per_class: Number of samples per class for SHAP
            lime_samples_per_class: Number of samples per class for LIME
            benchmark_samples: Number of samples for quantitative benchmarking
        """
        print("🚀 Starting Comprehensive Explainability Analysis Pipeline")
        print("="*80)
        
        pipeline_start_time = time.time()
        
        try:
            # Step 1: Load model and data
            self.load_model_and_data()
            
            # Step 2: Evaluate model performance
            self.evaluate_model_performance()
            
            # Step 3: Run Grad-CAM analysis
            self.run_gradcam_analysis(
                correct_samples=gradcam_samples[0],
                incorrect_samples=gradcam_samples[1]
            )
            
            # Step 4: Run SHAP analysis
            self.run_shap_analysis(samples_per_class=shap_samples_per_class)
            
            # Step 5: Run LIME analysis
            self.run_lime_analysis(samples_per_class=lime_samples_per_class)
            
            # Step 6: Run quantitative benchmarking
            self.run_quantitative_benchmark(num_samples=benchmark_samples)
            
            # Step 7: Generate comprehensive report
            self.generate_comprehensive_report()
            
            # Step 8: Save results and create summary
            self.save_pipeline_results()
            self.create_summary_report()
            
            # Calculate total time
            total_time = time.time() - pipeline_start_time
            
            # Final summary
            print("\\n" + "="*80)
            print("🎉 COMPREHENSIVE EXPLAINABILITY ANALYSIS COMPLETED!")
            print("="*80)
            print(f"Total execution time: {total_time/60:.2f} minutes")
            print(f"Results directory: {self.output_dir}")
            
            # Print key findings
            if 'accuracy' in self.results['model_performance']:
                accuracy = self.results['model_performance']['accuracy']
                print(f"Model accuracy: {accuracy:.1%}")
            
            if self.results['benchmark_results']:
                print("\\nExplainability method performance:")
                for method, results in self.results['benchmark_results'].items():
                    if 'error' not in results and results.get('insertion_auc', {}).get('mean'):
                        ins_auc = results['insertion_auc']['mean']
                        print(f"  - {method}: Insertion AUC = {ins_auc:.3f}")
            
            print(f"\\n📊 View comprehensive report: {self.output_dir}/reports/")
            print(f"📋 Executive summary: {self.output_dir}/EXECUTIVE_SUMMARY.md")
            print("="*80)
            
            return True
            
        except Exception as e:
            print(f"\\n❌ Pipeline failed: {str(e)}")
            print(f"Partial results saved to: {self.output_dir}")
            return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Comprehensive Explainability Analysis Pipeline')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test dataset')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Computing device')
    parser.add_argument('--output_dir', type=str, default='explainability_analysis',
                       help='Base output directory')
    parser.add_argument('--gradcam_correct', type=int, default=5,
                       help='Number of correct samples for Grad-CAM')
    parser.add_argument('--gradcam_incorrect', type=int, default=5,
                       help='Number of incorrect samples for Grad-CAM')
    parser.add_argument('--shap_samples', type=int, default=5,
                       help='Number of samples per class for SHAP')
    parser.add_argument('--lime_samples', type=int, default=2,
                       help='Number of samples per class for LIME')
    parser.add_argument('--benchmark_samples', type=int, default=30,
                       help='Number of samples for quantitative benchmarking')
    parser.add_argument('--quick_run', action='store_true',
                       help='Run with reduced parameters for testing')
    
    args = parser.parse_args()
    
    # Adjust parameters for quick run
    if args.quick_run:
        args.gradcam_correct = 2
        args.gradcam_incorrect = 2
        args.shap_samples = 2
        args.lime_samples = 1
        args.benchmark_samples = 10
        print("Quick run mode: Using reduced sample sizes")
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.test_data):
        print(f"Error: Test data directory not found: {args.test_data}")
        sys.exit(1)
    
    # Initialize and run pipeline
    pipeline = ComprehensiveExplainabilityPipeline(
        model_path=args.model_path,
        test_data_path=args.test_data,
        device=args.device,
        class_names=config.class_names,
        output_dir=args.output_dir
    )
    
    success = pipeline.run_complete_pipeline(
        gradcam_samples=(args.gradcam_correct, args.gradcam_incorrect),
        shap_samples_per_class=args.shap_samples,
        lime_samples_per_class=args.lime_samples,
        benchmark_samples=args.benchmark_samples
    )
    
    if success:
        print("\\n✅ Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\\n❌ Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()