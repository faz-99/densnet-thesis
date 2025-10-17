"""
Test script to verify UI components and functionality
"""
import sys
import os
import torch
import numpy as np
from PIL import Image
import tempfile

def test_imports():
    """Test if all UI-related imports work"""
    print("Testing UI imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit")
    except ImportError:
        print("❌ Streamlit - install with: pip install streamlit")
        return False
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ Plotly")
    except ImportError:
        print("❌ Plotly - install with: pip install plotly")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib")
    except ImportError:
        print("❌ Matplotlib")
        return False
    
    try:
        import cv2
        print("✅ OpenCV")
    except ImportError:
        print("❌ OpenCV - install with: pip install opencv-python")
        return False
    
    try:
        from sklearn.metrics import confusion_matrix
        print("✅ Scikit-learn")
    except ImportError:
        print("❌ Scikit-learn")
        return False
    
    return True


def test_model_loading():
    """Test model loading functionality"""
    print("\nTesting model loading...")
    
    # Check if model file exists
    model_path = "weight/save/40/iaff40_5.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Train a model first with: python train.py")
        return False
    
    try:
        device = torch.device('cpu')  # Use CPU for testing
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model' not in checkpoint:
            print("❌ Model checkpoint missing 'model' key")
            return False
        
        model = checkpoint['model']
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully")
        print(f"   Best accuracy: {checkpoint.get('best_acc', 'N/A')}")
        print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        return False


def test_image_processing():
    """Test image preprocessing functionality"""
    print("\nTesting image processing...")
    
    try:
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_image = Image.fromarray(dummy_image)
        
        # Test preprocessing (simplified version)
        image_array = np.array(pil_image)
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            print("✅ RGB image handling")
        
        # Test resize
        import cv2
        resized = cv2.resize(image_array, (224, 224))
        if resized.shape == (224, 224, 3):
            print("✅ Image resizing")
        
        # Test normalization
        normalized = resized.astype(np.float32) / 255.0
        if 0 <= normalized.min() and normalized.max() <= 1:
            print("✅ Image normalization")
        
        # Test tensor conversion
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
        if tensor.shape == (1, 3, 224, 224):
            print("✅ Tensor conversion")
        
        return True
        
    except Exception as e:
        print(f"❌ Image processing failed: {str(e)}")
        return False


def test_explainability_imports():
    """Test explainability module imports"""
    print("\nTesting explainability imports...")
    
    try:
        from explainability.grad_cam import GradCAM, GradCAMPlusPlus
        print("✅ Grad-CAM modules")
    except ImportError as e:
        print(f"❌ Grad-CAM modules: {e}")
        return False
    
    try:
        from explainability.shap_explainer import SHAPExplainer
        print("✅ SHAP module")
    except ImportError as e:
        print(f"❌ SHAP module: {e}")
        return False
    
    try:
        from explainability.lime_explainer import LIMEExplainer
        print("✅ LIME module")
    except ImportError as e:
        print(f"❌ LIME module: {e}")
        return False
    
    try:
        from evaluation.metrics import ModelEvaluator
        print("✅ Evaluation module")
    except ImportError as e:
        print(f"❌ Evaluation module: {e}")
        return False
    
    return True


def test_config():
    """Test configuration import"""
    print("\nTesting configuration...")
    
    try:
        import config
        
        required_attrs = ['dataset_mean', 'dataset_std', 'class_num', 'img_s']
        for attr in required_attrs:
            if hasattr(config, attr):
                print(f"✅ config.{attr}: {getattr(config, attr)}")
            else:
                print(f"❌ Missing config.{attr}")
                return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False


def test_file_structure():
    """Test if all required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        'app.py',
        'config.py',
        'train.py',
        'run_evaluation.py',
        'run_explainability.py',
        'explainability/__init__.py',
        'explainability/grad_cam.py',
        'explainability/shap_explainer.py',
        'explainability/lime_explainer.py',
        'explainability/explainer.py',
        'evaluation/__init__.py',
        'evaluation/metrics.py',
        'requirements_ui.txt',
        'requirements_explainability.txt'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\nMissing files: {missing_files}")
        return False
    
    return True


def main():
    """Run all tests"""
    print("="*60)
    print("UI AND EVALUATION TEST SUITE")
    print("="*60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Configuration", test_config),
        ("UI Imports", test_imports),
        ("Image Processing", test_image_processing),
        ("Explainability Imports", test_explainability_imports),
        ("Model Loading", test_model_loading)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'-'*20} {test_name} {'-'*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"Test {test_name} crashed: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! UI is ready to use.")
        print("\nNext steps:")
        print("1. Ensure you have a trained model")
        print("2. Launch UI: streamlit run app.py")
        print("3. Or run complete pipeline: python run_complete_pipeline.py")
    else:
        print(f"\n❌ {total - passed} tests failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("- Install missing packages:")
        print("  pip install -r requirements_ui.txt")
        print("  pip install -r requirements_explainability.txt")
        print("- Train a model: python train.py")
        print("- Check file paths and permissions")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)