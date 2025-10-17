"""
Script to run comprehensive explainability analysis on the trained DenseNet model
"""
import torch
import numpy as np
import os
import sys
from torch.utils.data import DataLoader, Subset
import argparse

# Import existing modules
import config
from utils.load_dataset2 import get_data_loader
from explainability.explainer import ComprehensiveExplainer, run_explainability_analysis


def create_background_dataloader(train_loader, background_size=100):
    """
    Create a background dataset for SHAP from the training data
    
    Args:
        train_loader: Training data loader
        background_size: Size of background dataset
        
    Returns:
        Background data loader
    """
    # Get a subset of training data for background
    dataset = train_loader.dataset
    indices = np.random.choice(len(dataset), min(background_size, len(dataset)), replace=False)
    background_dataset = Subset(dataset, indices)
    
    background_loader = DataLoader(
        background_dataset,
        batch_size=min(32, background_size),
        shuffle=False,
        num_workers=0
    )
    
    return background_loader


def main():
    parser = argparse.ArgumentParser(description='Run explainability analysis on DenseNet model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model (.pth file)')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of samples to analyze (default: 20)')
    parser.add_argument('--techniques', nargs='+', default=['gradcam', 'shap', 'lime'],
                       choices=['gradcam', 'shap', 'lime'],
                       help='Explainability techniques to use')
    parser.add_argument('--background_size', type=int, default=100,
                       help='Size of background dataset for SHAP (default: 100)')
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
    
    # Load data
    print("Loading datasets...")
    train_loader, test_loader, _ = get_data_loader()
    
    # Create background dataset for SHAP
    print("Creating background dataset for SHAP...")
    background_loader = create_background_dataloader(train_loader, args.background_size)
    
    # Define class names (adjust based on your dataset)
    class_names = ['Benign', 'Malignant']  # For BreaKHis dataset
    
    print(f"Starting explainability analysis with techniques: {args.techniques}")
    print(f"Analyzing {args.num_samples} samples...")
    
    # Run explainability analysis
    try:
        explainer = run_explainability_analysis(
            model_path=args.model_path,
            test_dataloader=test_loader,
            background_dataloader=background_loader,
            class_names=class_names,
            device=str(device),
            num_samples=args.num_samples,
            techniques=args.techniques
        )
        
        print("\n" + "="*50)
        print("EXPLAINABILITY ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Results saved to: explainability/")
        print("\nGenerated files:")
        print("- performance_analysis.json: Model performance statistics")
        print("- performance_plots.png: Performance visualization")
        print("- explainability_report.json: Comprehensive analysis report")
        print("- README.md: Human-readable summary report")
        
        for technique in args.techniques:
            print(f"- {technique}_results/: {technique.upper()} explanations and visualizations")
        
        print("\nRecommendations:")
        # Load and display recommendations
        import json
        with open('explainability/explainability_report.json', 'r') as f:
            report = json.load(f)
        
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")
            
    except Exception as e:
        print(f"Error during explainability analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()