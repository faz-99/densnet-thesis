"""
Script to run comprehensive model evaluation
"""
import torch
import argparse
import os
import sys
from datetime import datetime

# Import project modules
import config
from utils.load_dataset2 import get_data_loader
from evaluation.metrics import evaluate_saved_model


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive model evaluation')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model (.pth file)')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu). If None, uses config setting')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)
    
    # Set save directory
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = f"evaluation_results_{timestamp}"
    
    # Load test data
    print("Loading test dataset...")
    try:
        _, test_loader, _ = get_data_loader()
        print(f"Test dataset loaded with {len(test_loader.dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        sys.exit(1)
    
    # Define class names (adjust based on your dataset)
    class_names = ['Benign', 'Malignant']  # For BreaKHis dataset
    
    print(f"Starting comprehensive evaluation...")
    print(f"Model: {args.model_path}")
    print(f"Classes: {class_names}")
    print(f"Results will be saved to: {args.save_dir}")
    
    try:
        # Run evaluation
        results = evaluate_saved_model(
            model_path=args.model_path,
            test_dataloader=test_loader,
            class_names=class_names,
            device=str(device),
            save_dir=args.save_dir
        )
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        metrics = results['metrics']
        print(f"📊 PERFORMANCE SUMMARY:")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        
        if len(class_names) == 2:
            print(f"   Sensitivity: {metrics['sensitivity']:.4f}")
            print(f"   Specificity: {metrics['specificity']:.4f}")
            print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
        
        print(f"\n📁 GENERATED FILES:")
        print(f"   📈 evaluation_results.json - Complete results")
        print(f"   📊 metrics_summary.csv - Summary table")
        print(f"   📋 evaluation_report.md - Detailed report")
        print(f"   🎯 confusion_matrix.png - Confusion matrix")
        print(f"   📊 metrics_summary.png - Performance overview")
        print(f"   📈 class_distribution.png - Data distribution")
        print(f"   📊 confidence_analysis.png - Confidence analysis")
        
        if len(class_names) == 2:
            print(f"   📈 roc_curve.png - ROC curve")
            print(f"   📊 precision_recall_curve.png - PR curve")
            print(f"   🌐 interactive_roc_curve.html - Interactive ROC")
        
        print(f"   🌐 interactive_confusion_matrix.html - Interactive CM")
        
        print(f"\n📍 All results saved to: {args.save_dir}")
        
        # Per-class performance
        print(f"\n📋 PER-CLASS PERFORMANCE:")
        for class_name in class_names:
            class_metrics = metrics['per_class'][class_name]
            print(f"   {class_name}:")
            print(f"     Precision: {class_metrics['precision']:.4f}")
            print(f"     Recall: {class_metrics['recall']:.4f}")
            print(f"     F1-Score: {class_metrics['f1_score']:.4f}")
        
        # Sample distribution
        print(f"\n📊 SAMPLE DISTRIBUTION:")
        total_samples = sum(metrics['sample_distribution'].values())
        for class_name, count in metrics['sample_distribution'].items():
            percentage = count / total_samples * 100
            print(f"   {class_name}: {count} samples ({percentage:.1f}%)")
        
        # Confidence statistics
        conf_stats = metrics['confidence_stats']
        print(f"\n🎯 CONFIDENCE STATISTICS:")
        print(f"   Mean: {conf_stats['mean']:.4f}")
        print(f"   Std: {conf_stats['std']:.4f}")
        print(f"   Range: [{conf_stats['min']:.4f}, {conf_stats['max']:.4f}]")
        
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()