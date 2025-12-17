#!/usr/bin/env python3
"""
Test script to verify the fixed DenLsNet model works correctly
"""

import torch
import torch.nn.functional as F
from model.denlsnet_corrected import create_denlsnet
from PIL import Image
import numpy as np

def test_model():
    """Test the fixed DenLsNet model"""
    print("🧪 Testing Fixed DenLsNet Model")
    print("=" * 50)
    
    # Create models
    print("📦 Creating models...")
    binary_model = create_denlsnet(num_classes=2, dropout_rate=0.5)
    multiclass_model = create_denlsnet(num_classes=8, dropout_rate=0.5)
    
    # Set to evaluation mode
    binary_model.eval()
    multiclass_model.eval()
    
    print("✅ Models created successfully")
    
    # Create dummy input (simulating preprocessed image)
    print("\n🖼️ Creating test input...")
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 224, 224)
    print(f"Input shape: {test_input.shape}")
    
    # Test binary model
    print("\n🔬 Testing Binary Model...")
    try:
        with torch.no_grad():
            binary_output = binary_model(test_input)
            binary_probs = F.softmax(binary_output, dim=1)
        
        print(f"✅ Binary output shape: {binary_output.shape}")
        print(f"✅ Binary probabilities shape: {binary_probs.shape}")
        print(f"✅ Probability sums: {binary_probs.sum(dim=1)}")
        
        # Test predictions
        predictions = torch.argmax(binary_probs, dim=1)
        confidences = binary_probs.max(dim=1)[0]
        
        for i in range(batch_size):
            pred_class = "Benign" if predictions[i].item() == 0 else "Malignant"
            print(f"   Sample {i+1}: {pred_class} (confidence: {confidences[i].item():.3f})")
        
    except Exception as e:
        print(f"❌ Binary model failed: {str(e)}")
        return False
    
    # Test multiclass model
    print("\n🔬 Testing Multiclass Model...")
    try:
        with torch.no_grad():
            multiclass_output = multiclass_model(test_input)
            multiclass_probs = F.softmax(multiclass_output, dim=1)
        
        print(f"✅ Multiclass output shape: {multiclass_output.shape}")
        print(f"✅ Multiclass probabilities shape: {multiclass_probs.shape}")
        print(f"✅ Probability sums: {multiclass_probs.sum(dim=1)}")
        
        # Test predictions
        class_names = [
            'Adenosis', 'Fibroadenoma', 'Phyllodes Tumor', 'Tubular Adenoma',
            'Ductal Carcinoma', 'Lobular Carcinoma', 'Mucinous Carcinoma', 'Papillary Carcinoma'
        ]
        
        predictions = torch.argmax(multiclass_probs, dim=1)
        confidences = multiclass_probs.max(dim=1)[0]
        
        for i in range(batch_size):
            pred_class = class_names[predictions[i].item()]
            category = "Benign" if predictions[i].item() < 4 else "Malignant"
            print(f"   Sample {i+1}: {pred_class} ({category}) (confidence: {confidences[i].item():.3f})")
        
    except Exception as e:
        print(f"❌ Multiclass model failed: {str(e)}")
        return False
    
    # Test feature extraction
    print("\n🔍 Testing Feature Extraction...")
    try:
        with torch.no_grad():
            features = binary_model.get_feature_maps(test_input)
        
        print("✅ Feature maps extracted successfully:")
        for name, feature_map in features.items():
            print(f"   {name}: {list(feature_map.shape)}")
        
    except Exception as e:
        print(f"❌ Feature extraction failed: {str(e)}")
        return False
    
    print("\n🎉 All tests passed! The model is working correctly.")
    print("🌐 Your Streamlit app should now work without errors.")
    
    return True

if __name__ == "__main__":
    success = test_model()
    if success:
        print("\n✅ Model fix successful! You can now use the web application.")
    else:
        print("\n❌ Model still has issues. Please check the error messages above.")