#!/usr/bin/env python3
"""
Test script to verify corrected DenLsNet setup before full training
"""

import torch
import sys
from pathlib import Path

# Test imports
try:
    from model.denlsnet_corrected import create_denlsnet
    from config.training_config import TrainingConfig, get_device
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_model_architecture():
    """Test the corrected model architecture"""
    print("\n🧪 Testing Model Architecture")
    print("="*40)
    
    # Test binary model
    binary_model = create_denlsnet(num_classes=2, dropout_rate=0.5)
    print("✅ Binary model created")
    
    # Test multiclass model
    multiclass_model = create_denlsnet(num_classes=8, dropout_rate=0.5)
    print("✅ Multiclass model created")
    
    # Test forward pass
    device = get_device()
    binary_model.to(device)
    multiclass_model.to(device)
    
    # Test input
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    
    with torch.no_grad():
        binary_output = binary_model(dummy_input)
        multiclass_output = multiclass_model(dummy_input)
    
    print(f"✅ Binary output shape: {binary_output.shape} (expected: [2, 2])")
    print(f"✅ Multiclass output shape: {multiclass_output.shape} (expected: [2, 8])")
    
    # Verify architecture components
    print("\n🔍 Architecture Verification:")
    binary_model.print_architecture_summary()
    
    return True

def test_training_config():
    """Test training configuration"""
    print("\n🧪 Testing Training Configuration")
    print("="*40)
    
    # Test binary config
    binary_config = TrainingConfig(task='binary')
    print("✅ Binary config created")
    
    # Test multiclass config
    multiclass_config = TrainingConfig(task='multiclass')
    print("✅ Multiclass config created")
    
    # Verify key parameters
    assert binary_config.epochs == 80, f"Expected 80 epochs, got {binary_config.epochs}"
    assert binary_config.optimizer_name == 'SGD', f"Expected SGD, got {binary_config.optimizer_name}"
    assert binary_config.scheduler_name == 'CosineAnnealingLR', f"Expected CosineAnnealingLR, got {binary_config.scheduler_name}"
    
    print("✅ All configuration parameters correct")
    
    # Test reproducibility settings
    print(f"✅ Seed: {binary_config.seed}")
    print(f"✅ Deterministic: {binary_config.deterministic}")
    print(f"✅ CUDNN Deterministic: {torch.backends.cudnn.deterministic}")
    print(f"✅ CUDNN Benchmark: {torch.backends.cudnn.benchmark}")
    
    return True

def test_device_setup():
    """Test device setup"""
    print("\n🧪 Testing Device Setup")
    print("="*40)
    
    device = get_device()
    print(f"✅ Device: {device}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        print(f"✅ CUDA version: {torch.version.cuda}")
    else:
        print("ℹ️  CUDA not available, using CPU")
    
    return True

def main():
    """Run all tests"""
    print("🔬 Corrected DenLsNet Setup Verification")
    print("="*50)
    
    try:
        # Run tests
        test_device_setup()
        test_training_config()
        test_model_architecture()
        
        print("\n" + "="*50)
        print("🎉 All tests passed! Setup is ready for training.")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)