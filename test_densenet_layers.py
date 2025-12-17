#!/usr/bin/env python3
"""
Quick test to see DenseNet201 layer structure
"""
import torch
import torchvision.models as models

def test_densenet_layers():
    """Test DenseNet201 layer structure"""
    print("🔍 Testing DenseNet201 Layer Structure")
    print("="*50)
    
    # Load DenseNet201
    model = models.densenet201(pretrained=True)
    
    # Modify classifier for binary classification
    num_features = model.classifier.in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(num_features, 2)
    )
    
    print("📋 Available Layers:")
    print("-" * 30)
    
    layer_count = 0
    for name, module in model.named_modules():
        if any(keyword in name.lower() for keyword in ['norm', 'conv', 'dense', 'features']):
            print(f"{layer_count:2d}. {name}")
            layer_count += 1
            if layer_count > 20:  # Limit output
                print("    ... (showing first 20 relevant layers)")
                break
    
    print("\n🎯 Recommended Target Layers for Grad-CAM:")
    print("-" * 40)
    
    recommended_layers = [
        'features.norm5',
        'features.denseblock4.denselayer16.norm2',
        'features.denseblock4',
        'features.transition3.norm',
        'features.denseblock3'
    ]
    
    for i, layer in enumerate(recommended_layers, 1):
        # Check if layer exists
        layer_exists = any(name == layer for name, _ in model.named_modules())
        status = "✅" if layer_exists else "❌"
        print(f"{i}. {layer} {status}")
    
    print("\n🧪 Testing with sample input:")
    print("-" * 30)
    
    # Test with sample input
    sample_input = torch.randn(1, 3, 224, 224)
    model.eval()
    
    with torch.no_grad():
        output = model(sample_input)
        print(f"Input shape: {sample_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output classes: {output.argmax(dim=1).item()}")
    
    print("\n✅ DenseNet201 structure analysis complete!")

if __name__ == "__main__":
    test_densenet_layers()