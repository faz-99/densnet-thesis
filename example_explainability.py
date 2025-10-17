"""
Example script demonstrating how to use the explainability features
with the DenseNet model for medical image classification
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# Import project modules
import config
from utils.load_dataset2 import get_data_loader
from explainability.explainer import ComprehensiveExplainer
from explainability.grad_cam import GradCAM, GradCAMPlusPlus, visualize_gradcam_results
from explainability.shap_explainer import SHAPExplainer
from explainability.lime_explainer import LIMEExplainer


def example_gradcam_analysis():
    """Example of using Grad-CAM for single image analysis"""
    print("=== Grad-CAM Example ===")
    
    # Load model (replace with your actual model path)
    model_path = "weight/save/40/iaff40_5.pth"  # Update this path
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please update the path.")
        return
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model = checkpoint['model']
    model.to(device)
    model.eval()
    
    # Load data
    _, test_loader, _ = get_data_loader()
    
    # Get a single sample
    for images, labels in test_loader:
        single_image = images[0:1].to(device)
        true_label = labels[0].item()
        break
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer_name='densenet.features.norm5')
    gradcam_plus = GradCAMPlusPlus(model, target_layer_name='densenet.features.norm5')
    
    # Generate explanations
    with torch.no_grad():
        output = model(single_image)
        predicted_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, predicted_class].item()
    
    gradcam_heatmap = gradcam.generate_cam(single_image)
    gradcam_plus_heatmap = gradcam_plus.generate_cam(single_image)
    
    # Prepare image for visualization
    original_img = single_image[0].cpu().numpy().transpose(1, 2, 0)
    original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
    
    # Create visualization
    class_names = ['Benign', 'Malignant']
    visualize_gradcam_results(
        original_img,
        gradcam_heatmap,
        gradcam_plus_heatmap,
        class_names[predicted_class],
        class_names[true_label],
        confidence,
        'example_gradcam_result.png'
    )
    
    print(f"Grad-CAM visualization saved to: example_gradcam_result.png")
    print(f"Predicted: {class_names[predicted_class]}, True: {class_names[true_label]}")
    print(f"Confidence: {confidence:.3f}")


def example_comprehensive_analysis():
    """Example of running comprehensive explainability analysis"""
    print("\n=== Comprehensive Analysis Example ===")
    
    # Check if model exists
    model_path = "weight/save/40/iaff40_5.pth"  # Update this path
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please update the path.")
        print("To run this example, you need a trained model file.")
        return
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = checkpoint['model']
    model.to(device)
    
    # Load data
    train_loader, test_loader, _ = get_data_loader()
    
    # Create background dataset (subset of training data for SHAP)
    background_images = []
    background_count = 0
    for images, _ in train_loader:
        for i in range(images.size(0)):
            if background_count >= 50:  # Small background set for example
                break
            background_images.append(images[i:i+1])
            background_count += 1
        if background_count >= 50:
            break
    
    background_data = torch.cat(background_images, dim=0)
    
    # Initialize comprehensive explainer
    class_names = ['Benign', 'Malignant']
    explainer = ComprehensiveExplainer(model, str(device), class_names)
    
    # Analyze model performance
    print("Analyzing model performance...")
    performance_stats = explainer.analyze_model_performance(test_loader)
    print(f"Model accuracy: {performance_stats['accuracy']:.3f}")
    
    # Generate explanations (using fewer samples for example)
    print("Generating explanations...")
    
    # Create a simple dataloader for background
    from torch.utils.data import TensorDataset, DataLoader
    background_dataset = TensorDataset(background_data, torch.zeros(len(background_data)))
    background_loader = DataLoader(background_dataset, batch_size=16, shuffle=False)
    
    explainer.generate_comprehensive_explanations(
        test_dataloader=test_loader,
        background_dataloader=background_loader,
        num_samples=5,  # Small number for example
        techniques=['gradcam']  # Start with just Grad-CAM for faster execution
    )
    
    print("Comprehensive analysis completed!")
    print("Check the 'explainability/' folder for results.")


def example_individual_techniques():
    """Example of using individual explainability techniques"""
    print("\n=== Individual Techniques Example ===")
    
    # This example shows how to use each technique individually
    # without running the full comprehensive analysis
    
    model_path = "weight/save/40/iaff40_5.pth"  # Update this path
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}.")
        print("This example requires a trained model file.")
        return
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = checkpoint['model']
    model.to(device)
    model.eval()
    
    # Load data
    _, test_loader, _ = get_data_loader()
    
    # Get a sample
    for images, labels in test_loader:
        sample_image = images[0:1].to(device)
        sample_label = labels[0].item()
        break
    
    print("1. Grad-CAM Analysis:")
    gradcam = GradCAM(model, target_layer_name='densenet.features.norm5')
    heatmap = gradcam.generate_cam(sample_image)
    print(f"   Generated heatmap shape: {heatmap.shape}")
    
    print("2. SHAP Analysis (requires background data):")
    print("   SHAP requires background dataset - see comprehensive example")
    
    print("3. LIME Analysis:")
    lime_explainer = LIMEExplainer(model, str(device), num_samples=100)
    
    # Convert image for LIME
    image_np = sample_image[0].cpu().numpy().transpose(1, 2, 0)
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    
    explanation, segments = lime_explainer.explain_image(image_np)
    print(f"   Generated LIME explanation with {len(segments)} superpixels")


def print_usage_instructions():
    """Print instructions for using the explainability features"""
    print("\n" + "="*60)
    print("EXPLAINABILITY USAGE INSTRUCTIONS")
    print("="*60)
    
    print("\n1. QUICK START:")
    print("   Run comprehensive analysis with:")
    print("   python run_explainability.py --model_path weight/save/40/iaff40_5.pth")
    
    print("\n2. CUSTOM ANALYSIS:")
    print("   python run_explainability.py \\")
    print("     --model_path your_model.pth \\")
    print("     --num_samples 10 \\")
    print("     --techniques gradcam shap \\")
    print("     --background_size 50")
    
    print("\n3. INDIVIDUAL TECHNIQUES:")
    print("   Use the classes in explainability/ folder:")
    print("   - GradCAM, GradCAMPlusPlus from grad_cam.py")
    print("   - SHAPExplainer from shap_explainer.py")
    print("   - LIMEExplainer from lime_explainer.py")
    
    print("\n4. OUTPUT STRUCTURE:")
    print("   explainability/")
    print("   ├── performance_analysis.json")
    print("   ├── performance_plots.png")
    print("   ├── README.md")
    print("   ├── gradcam_results/")
    print("   ├── shap_results/")
    print("   └── lime_results/")
    
    print("\n5. REQUIREMENTS:")
    print("   Install dependencies with:")
    print("   pip install -r requirements_explainability.txt")


def main():
    """Main function to run examples"""
    print("DenseNet Explainability Examples")
    print("="*40)
    
    # Print usage instructions
    print_usage_instructions()
    
    # Check if we have a model to work with
    model_path = "weight/save/40/iaff40_5.pth"
    
    if os.path.exists(model_path):
        print(f"\nFound model at: {model_path}")
        print("Running examples...")
        
        try:
            # Run individual technique examples
            example_individual_techniques()
            
            # Run Grad-CAM example
            example_gradcam_analysis()
            
            # Optionally run comprehensive analysis (commented out for speed)
            # example_comprehensive_analysis()
            
        except Exception as e:
            print(f"Error running examples: {str(e)}")
            print("Make sure all dependencies are installed:")
            print("pip install -r requirements_explainability.txt")
    
    else:
        print(f"\nModel not found at: {model_path}")
        print("Please train a model first or update the model path in the examples.")
        print("\nTo train a model, run:")
        print("python train.py")
        
    print("\n" + "="*60)
    print("For full explainability analysis, run:")
    print("python run_explainability.py --model_path your_model.pth")
    print("="*60)


if __name__ == "__main__":
    main()