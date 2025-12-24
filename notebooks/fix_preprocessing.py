# Minimal preprocessing wrapper fix
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class SimplePreprocessingPipeline:
    """Minimal preprocessing pipeline using existing stain normalizer"""
    def __init__(self, config, stain_normalizer=None):
        self.config = config
        self.stain_normalizer = stain_normalizer
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, img_np, is_training=False):
        """Preprocess single image"""
        # Apply stain normalization if enabled
        if self.config.get('stain_normalization') and self.stain_normalizer:
            try:
                img_np = self.stain_normalizer.normalize(img_np)
            except:
                pass  # Skip if normalization fails
        
        # Apply transforms
        return self.transform(img_np)

class PreprocessedWrapper(Dataset):
    """Wrapper to apply preprocessing on-the-fly"""
    def __init__(self, base_dataset, pipeline, is_training=False):
        self.base = base_dataset
        self.pipeline = pipeline
        self.is_training = is_training
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        img, label = self.base[idx]
        # Convert to numpy uint8 RGB
        if isinstance(img, torch.Tensor):
            img_np = (img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        else:
            img_np = np.array(img)
        
        proc = self.pipeline.preprocess_image(img_np, is_training=self.is_training)
        return proc, label
