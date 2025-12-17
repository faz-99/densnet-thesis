#!/usr/bin/env python3
"""
Test script for Integrated Gradients implementation
"""
import torch
import torchvision.models as models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def test_integrated_gradients():
    """Test the Integrated Gradients implementation"""
    print("🧪 Testing Integrated Gradients Implementation")
    print("="*50)
    
    # Create a simple model
    print("1. Loading DenseNet201 model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.densenet201(weights='IMAGENET1K_V1')
    
    # Modify for binary classification
    num_features = model.classifier.in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(num_features, 2)
    )
    
    model.to(device)
    model.eval()
    print(f"✅ Model loaded on {device}")
    
    # Test Integrated Gradients
    print("\n2. Testing Integrated Gradients...")
    try:
        from explainability.integrated_gradients import IntegratedGradients
        
        # Create synthetic image
        synthetic_image = torch.randn(1, 3, 224, 224).to(device)
        
        # Initialize IG
        ig = IntegratedGradients(model, device)
        print("✅ Integrated Gradients initialized")
        
        # Generate attribution
        attribution, metadata = ig.generate_integrated_gradients(
            synthetic_image, target_class=0, num_steps=20
        )
        print(f"✅ Attribution generated: shape {attribution.shape}")
        print(f"   - Range: [{metadata['attribution_range'][0]:.3f}, {metadata['attribution_range'][1]:.3f}]")
        
        # Test histopathology analysis
        original_img = np.random.rand(224, 224, 3)
        analysis = ig.analyze_histopathology_features(attribution, original_img)
        print(f"✅ Histopathology analysis completed")
        print(f"   - Tissue proportion: {analysis['tissue_proportion']:.1%}")
        print(f"   - Dominant stain: {analysis['dominant_stain']}")
        
        # Test textual explanation
        prediction_info = {
            'predicted_class': 0,
            'confidence': 0.85,
            'class_names': ['Benign', 'Malignant']
        }
        
        textual_explanation = ig.generate_textual_explanation(analysis, prediction_info)
        print(f"✅ Textual explanation generated ({len(textual_explanation)} characters)")
        
    except Exception as e:
        print(f"❌ Integrated Gradients test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Textual Explainer
    print("\n3. Testing Textual Explainer...")
    try:
        from explainability.textual_explainer import HistopathologyTextualExplainer
        
        class_names = ['Benign', 'Malignant']
        textual_explainer = HistopathologyTextualExplainer(class_names)
        print("✅ Textual Explainer initialized")
        
        # Generate comprehensive report
        attribution_map = np.random.rand(224, 224) * 2 - 1  # Random attribution
        original_image = np.random.rand(224, 224, 3)
        prediction_info = {
            'predicted_class': 1,
            'confidence': 0.75,
            'class_names': class_names
        }
        
        report = textual_explainer.generate_comprehensive_report(
            attribution_map, original_image, prediction_info, "Integrated Gradients"
        )
        print(f"✅ Comprehensive report generated ({len(report)} characters)")
        
        # Show first few lines
        print("\n📄 Report Preview:")
        print("-" * 40)
        print(report[:500] + "..." if len(report) > 500 else report)
        
    except Exception as e:
        print(f"❌ Textual Explainer test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test LRP (if available)
    print("\n4. Testing LRP...")
    try:
        from explainability.lrp import LRPDenseNet
        
        lrp = LRPDenseNet(model, device)
        print("✅ LRP initialized")
        
        # Generate LRP explanation
        synthetic_image = torch.randn(1, 3, 224, 224).to(device)
        relevance, metadata = lrp.generate_lrp_explanation(synthetic_image, target_class=0)
        print(f"✅ LRP relevance generated: shape {relevance.shape}")
        print(f"   - Range: [{metadata['relevance_range'][0]:.3f}, {metadata['relevance_range'][1]:.3f}]")
        
    except Exception as e:
        print(f"❌ LRP test failed: {e}")
    
    print("\n🎉 Integrated Gradients Implementation Test Complete!")
    print("="*50)


if __name__ == "__main__":
    test_integrated_gradients()