"""
Test script to verify explainability module installation and basic functionality
"""
import sys
import importlib
import torch
import numpy as np


def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    required_packages = [
        'torch',
        'torchvision', 
        'numpy',
        'matplotlib',
        'cv2',
        'sklearn',
        'shap',
        'lime',
        'skimage'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {str(e)}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nFailed imports: {failed_imports}")
        print("Install missing packages with:")
        print("pip install -r requirements_explainability.txt")
        return False
    
    print("All imports successful!")
    return True


def test_explainability_modules():
    """Test if explainability modules can be imported"""
    print("\nTesting explainability modules...")
    
    try:
        from explainability.grad_cam import GradCAM, GradCAMPlusPlus
        print("✓ Grad-CAM modules")
    except ImportError as e:
        print(f"✗ Grad-CAM modules: {str(e)}")
        return False
    
    try:
        from explainability.shap_explainer import SHAPExplainer
        print("✓ SHAP module")
    except ImportError as e:
        print(f"✗ SHAP module: {str(e)}")
        return False
    
    try:
        from explainability.lime_explainer import LIMEExplainer
        print("✓ LIME module")
    except ImportError as e:
        print(f"✗ LIME module: {str(e)}")
        return False
    
    try:
        from explainability.explainer import ComprehensiveExplainer
        print("✓ Comprehensive explainer")
    except ImportError as e:
        print(f"✗ Comprehensive explainer: {str(e)}")
        return False
    
    print("All explainability modules imported successfully!")
    return True


def test_basic_functionality():
    """Test basic functionality with dummy data"""
    print("\nTesting basic functionality...")
    
    try:
        # Create a dummy model
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.features = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 64, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool2d(1)
                )
                self.features.norm5 = torch.nn.BatchNorm2d(64)  # For Grad-CAM target
                self.classifier = torch.nn.Linear(64, 2)
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        model = DummyModel()
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Test model forward pass
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ Dummy model forward pass: {output.shape}")
        
        # Test Grad-CAM
        from explainability.grad_cam import GradCAM
        gradcam = GradCAM(model, target_layer_name='features.norm5')
        heatmap = gradcam.generate_cam(dummy_input)
        print(f"✓ Grad-CAM generation: {heatmap.shape}")
        
        # Test LIME (basic initialization)
        from explainability.lime_explainer import LIMEExplainer
        lime_explainer = LIMEExplainer(model, 'cpu', num_samples=10)
        print("✓ LIME explainer initialization")
        
        print("Basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_file_structure():
    """Test if all required files exist"""
    print("\nTesting file structure...")
    
    import os
    
    required_files = [
        'explainability/__init__.py',
        'explainability/grad_cam.py',
        'explainability/shap_explainer.py',
        'explainability/lime_explainer.py',
        'explainability/explainer.py',
        'explainability/README.md',
        'run_explainability.py',
        'example_explainability.py',
        'requirements_explainability.txt'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\nMissing files: {missing_files}")
        return False
    
    print("All required files present!")
    return True


def main():
    """Run all tests"""
    print("="*50)
    print("EXPLAINABILITY MODULE TEST SUITE")
    print("="*50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Package Imports", test_imports),
        ("Explainability Modules", test_explainability_modules),
        ("Basic Functionality", test_basic_functionality)
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
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Explainability module is ready to use.")
        print("\nNext steps:")
        print("1. Train your DenseNet model (python train.py)")
        print("2. Run explainability analysis (python run_explainability.py --model_path your_model.pth)")
        print("3. Check example usage (python example_explainability.py)")
    else:
        print(f"\n❌ {total - passed} tests failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("- Install missing packages: pip install -r requirements_explainability.txt")
        print("- Check file paths and permissions")
        print("- Verify Python environment")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)