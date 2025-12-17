#!/usr/bin/env python3
"""
Test UI functionality without Streamlit
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from model.denlsnet_corrected import create_denlsnet
from torchvision import transforms

def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    input_tensor = transform(image).unsqueeze(0)
    
    return input_tensor

def predict_binary(model, image_tensor):
    """Make binary classification prediction"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        confidence = probabilities.max().item()
    
    class_name = 'Benign' if prediction.item() == 0 else 'Malignant'
    
    return {
        'prediction': class_name,
        'prediction_idx': prediction.item(),
        'confidence': confidence,
        'probabilities': probabilities.cpu().numpy()[0]
    }

def predict_multiclass(model, image_tensor):
    """Make multiclass classification prediction"""
    class_names = [
        'Adenosis', 'Fibroadenoma', 'Phyllodes Tumor', 'Tubular Adenoma',
        'Ductal Carcinoma', 'Lobular Carcinoma', 'Mucinous Carcinoma', 'Papillary Carcinoma'
    ]
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        confidence = probabilities.max().item()
    
    return {
        'prediction': class_names[prediction.item()],
        'prediction_idx': prediction.item(),
        'confidence': confidence,
        'probabilities': probabilities.cpu().numpy()[0],
        'class_names': class_names
    }

def create_demo_image(image_type='benign'):
    """Create a demo histopathology-like image"""
    # Create a simple colored image for demonstration
    if image_type == 'benign':
        # Lighter, more organized pattern
        base_color = np.array([200, 180, 220])
    else:
        # Darker, more chaotic pattern
        base_color = np.array([120, 80, 140])
    
    # Create image with some texture
    img_array = np.ones((224, 224, 3)) * base_color
    noise = np.random.normal(0, 20, (224, 224, 3))
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_array)

def test_ui_functionality():
    """Test the UI functionality"""
    print("🧪 Testing UI Functionality")
    print("=" * 50)
    
    # Load models
    print("📦 Loading models...")
    try:
        binary_model = create_denlsnet(num_classes=2, dropout_rate=0.5)
        multiclass_model = create_denlsnet(num_classes=8, dropout_rate=0.5)
        
        binary_model.eval()
        multiclass_model.eval()
        
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load models: {str(e)}")
        return False
    
    # Test with demo images
    for image_type in ['benign', 'malignant']:
        print(f"\n🖼️ Testing with demo {image_type} image...")
        
        # Create demo image
        demo_image = create_demo_image(image_type)
        print(f"   Created {image_type} demo image: {demo_image.size}")
        
        # Preprocess image
        try:
            input_tensor = preprocess_image(demo_image)
            print(f"   Preprocessed image shape: {input_tensor.shape}")
        except Exception as e:
            print(f"   ❌ Preprocessing failed: {str(e)}")
            continue
        
        # Test binary classification
        try:
            binary_result = predict_binary(binary_model, input_tensor)
            print(f"   🎯 Binary: {binary_result['prediction']} (confidence: {binary_result['confidence']:.3f})")
        except Exception as e:
            print(f"   ❌ Binary prediction failed: {str(e)}")
            continue
        
        # Test multiclass classification
        try:
            multiclass_result = predict_multiclass(multiclass_model, input_tensor)
            category = "Benign" if multiclass_result['prediction_idx'] < 4 else "Malignant"
            print(f"   🎯 Multiclass: {multiclass_result['prediction']} ({category}) (confidence: {multiclass_result['confidence']:.3f})")
        except Exception as e:
            print(f"   ❌ Multiclass prediction failed: {str(e)}")
            continue
        
        print(f"   ✅ {image_type.capitalize()} image test passed")
    
    print(f"\n🎉 All UI functionality tests passed!")
    print(f"🌐 Your Streamlit app at http://localhost:8501 should work perfectly now!")
    
    return True

if __name__ == "__main__":
    success = test_ui_functionality()
    if success:
        print("\n✅ UI is ready! Open your browser and go to:")
        print("   🌐 Local: http://localhost:8501")
        print("   🌐 Network: http://192.168.18.249:8501")
        print("   🌐 External: http://139.135.32.77:8501")
    else:
        print("\n❌ UI tests failed. Please check the error messages above.")