"""
Preprocessing Pipeline for DenseNet and Swin Transformer V2
Minimal implementation for comparative analysis
"""
import torch
import torch.nn as nn
from torchvision import transforms
import timm
from model.swin_transformer import swin_tiny_patch4_window7_224
import config_clean as config

# Import after config to avoid circular import
import sys
sys.path.append('.')
from model.model import class_model

class PreprocessingPipeline:
    """Unified preprocessing for both models"""
    
    def __init__(self, img_size=224):
        self.img_size = img_size
        
        # DenseNet preprocessing (existing)
        self.densenet_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.dataset_mean, std=config.dataset_std)
        ])
        
        # Swin Transformer preprocessing
        self.swin_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
    
    def get_densenet_transform(self):
        return self.densenet_transform
    
    def get_swin_transform(self):
        return self.swin_transform

class SwinTransformerV2(nn.Module):
    """Swin Transformer V2 for medical image classification"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = swin_tiny_patch4_window7_224(num_classes=num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class ModelFactory:
    """Factory for creating models with preprocessing"""
    
    @staticmethod
    def create_densenet():
        """Create DenseNet model"""
        return class_model()
    
    @staticmethod
    def create_swin():
        """Create Swin Transformer V2 model"""
        return SwinTransformerV2(num_classes=config.class_num)
    
    @staticmethod
    def get_preprocessing():
        """Get preprocessing pipeline"""
        return PreprocessingPipeline()

if __name__ == "__main__":
    # Test models
    device = torch.device('cpu')
    
    # Create models
    densenet = ModelFactory.create_densenet()
    swin = ModelFactory.create_swin()
    preprocessing = ModelFactory.get_preprocessing()
    
    # Set models to eval mode for testing
    densenet.eval()
    swin.eval()
    
    # Test input
    x = torch.randn(2, 3, 224, 224)  # Use batch size 2 to avoid BN issues
    
    # Test DenseNet
    densenet_out = densenet(x)
    print(f"DenseNet output shape: {densenet_out.shape}")
    
    # Test Swin
    swin_out = swin(x)
    print(f"Swin output shape: {swin_out.shape}")
    
    print("Models created successfully!")