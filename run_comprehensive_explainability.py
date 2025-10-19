"""
Comprehensive Explainability Analysis for BreakHis Dataset
Generates morphological descriptors and template-based reports
"""
import torch
import os
import glob
from pathlib import Path
import argparse

from model.model import class_model
from explainability.comprehensive_explainer import process_dataset_batch


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive explainability analysis')
    parser.add_argument('--model_path', type=str, default='weight/save/40/iaff40_5.pth',
                       help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='data/BreakHis/test',
                       help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, default='explainability_reports',
                       help='Output directory for reports')
    parser.add_argument('--num_images', type=int, default=20,
                       help='Number of images to process (0 for all)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}")
    try:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        model = checkpoint['model']
        model.to(device)
        model.float()
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Class names for BreakHis dataset
    class_names = ['Benign', 'Malignant']
    
    # Find test images
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.data_dir, '**', ext), recursive=True))
    
    if not image_paths:
        print(f"❌ No images found in {args.data_dir}")
        return
    
    # Limit number of images if specified
    if args.num_images > 0:
        image_paths = image_paths[:args.num_images]
    
    print(f"Found {len(image_paths)} images to process")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process images
    print("Starting comprehensive explainability analysis...")
    results = process_dataset_batch(
        model=model,
        device=str(device),
        class_names=class_names,
        image_paths=image_paths,
        save_dir=args.output_dir
    )
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Results saved to: {args.output_dir}")
    print(f"📊 Processed: {len(results)} images")
    
    # Print summary statistics
    if results:
        predictions = [r['metadata']['predicted_class'] for r in results]
        confidences = [r['metadata']['confidence'] for r in results]
        
        print(f"\n📈 Summary Statistics:")
        print(f"   • Mean confidence: {sum(confidences)/len(confidences):.1%}")
        print(f"   • Benign predictions: {predictions.count('Benign')}")
        print(f"   • Malignant predictions: {predictions.count('Malignant')}")
        
        # Show sample findings
        print(f"\n🔍 Sample Clinical Interpretations:")
        for i, result in enumerate(results[:3]):
            print(f"   {i+1}. {result['metadata']['image_id']}: {result['clinical_interpretation'][:100]}...")


if __name__ == "__main__":
    main()