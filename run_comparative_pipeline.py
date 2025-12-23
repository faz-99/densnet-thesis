"""
Complete Pipeline Runner for DenseNet vs Swin Transformer V2 Comparison
Executes training, evaluation, and comparative analysis
"""
import os
import sys
import argparse
from datetime import datetime

def run_complete_pipeline(quick_run=False):
    """Run the complete comparative analysis pipeline"""
    
    print("="*60)
    print("DENSENET vs SWIN TRANSFORMER V2 COMPARATIVE ANALYSIS")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if quick_run:
        print("Running in QUICK MODE (reduced epochs)")
        # Temporarily modify config for quick run
        import config_clean as config
        original_epochs = config.max_epoch
        config.max_epoch = 2
    
    try:
        # Step 1: Training both models
        print("\n" + "="*50)
        print("STEP 1: TRAINING MODELS")
        print("="*50)
        
        from train_comparative import main as train_main
        train_main()
        
        # Step 2: Detailed evaluation
        print("\n" + "="*50)
        print("STEP 2: DETAILED EVALUATION")
        print("="*50)
        
        from evaluate_comparative import evaluate_models
        evaluation_report = evaluate_models()
        
        # Step 3: Generate final summary
        print("\n" + "="*50)
        print("STEP 3: FINAL SUMMARY")
        print("="*50)
        
        generate_final_summary(evaluation_report)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Results saved in:")
        print("- results/comparative_analysis.json")
        print("- results/detailed_comparison_report.json")
        print("- results/visualizations/")
        print("- weight/ (trained models)")
        
    except Exception as e:
        print(f"Pipeline failed with error: {str(e)}")
        sys.exit(1)
    
    finally:
        if quick_run:
            # Restore original config
            config.max_epoch = original_epochs

def generate_final_summary(evaluation_report):
    """Generate final summary of the comparison"""
    
    summary = {
        'pipeline_completion_time': datetime.now().isoformat(),
        'models_compared': ['DenseNet (DenLsNet)', 'Swin Transformer V2'],
        'key_findings': {},
        'recommendations': {}
    }
    
    if 'summary' in evaluation_report:
        best_model = evaluation_report['summary']['overall_best']['model']
        best_score = evaluation_report['summary']['overall_best']['weighted_score']
        
        summary['key_findings'] = {
            'best_performing_model': best_model,
            'performance_score': best_score,
            'performance_metrics': evaluation_report['detailed_comparison']
        }
        
        # Generate recommendations
        if best_model == 'DenseNet':
            summary['recommendations'] = {
                'primary_choice': 'DenseNet (DenLsNet)',
                'reason': 'Better overall performance on medical image classification',
                'use_cases': ['Medical image classification', 'Feature-rich analysis', 'Interpretability focus']
            }
        else:
            summary['recommendations'] = {
                'primary_choice': 'Swin Transformer V2',
                'reason': 'Superior performance with attention mechanisms',
                'use_cases': ['Large-scale image analysis', 'Attention-based interpretability', 'Scalable deployment']
            }
    
    # Save final summary
    os.makedirs('results', exist_ok=True)
    import json
    with open('results/final_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("FINAL COMPARISON SUMMARY:")
    print("-" * 40)
    if 'key_findings' in summary and summary['key_findings']:
        print(f"Best Model: {summary['key_findings']['best_performing_model']}")
        print(f"Performance Score: {summary['key_findings']['performance_score']:.4f}")
        
        if 'recommendations' in summary:
            print(f"Recommendation: {summary['recommendations']['primary_choice']}")
            print(f"Reason: {summary['recommendations']['reason']}")
    
    print("\nDetailed results available in results/ directory")

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='DenseNet vs Swin Transformer V2 Comparison')
    parser.add_argument('--quick_run', action='store_true', 
                       help='Run with reduced epochs for quick testing')
    parser.add_argument('--eval_only', action='store_true',
                       help='Run evaluation only (skip training)')
    
    args = parser.parse_args()
    
    if args.eval_only:
        print("Running evaluation only...")
        from evaluate_comparative import evaluate_models
        evaluation_report = evaluate_models()
        generate_final_summary(evaluation_report)
    else:
        run_complete_pipeline(quick_run=args.quick_run)

if __name__ == "__main__":
    main()