# Cell 5: Patch Extraction Strategy for BreakHis
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image

class PatchExtractor:
    def __init__(self, patch_size=224, overlap=0.5, min_tissue_ratio=0.7):
        self.patch_size = patch_size
        self.overlap = overlap
        self.min_tissue_ratio = min_tissue_ratio
        self.stride = int(patch_size * (1 - overlap))
        
    def extract_patches(self, image):
        """Extract overlapping patches from image"""
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
        elif isinstance(image, Image.Image):
            image = np.array(image)
            
        h, w = image.shape[:2]
        patches = []
        positions = []
        
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                
                if self._has_sufficient_tissue(patch):
                    patches.append(patch)
                    positions.append((x, y))
                    
        return patches, positions
    
    def _has_sufficient_tissue(self, patch):
        """Check if patch has sufficient tissue content"""
        if len(patch.shape) == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        else:
            gray = patch
            
        tissue_mask = gray < 230
        tissue_ratio = np.sum(tissue_mask) / (patch.shape[0] * patch.shape[1])
        
        return tissue_ratio >= self.min_tissue_ratio

class MultiPatchDataset(Dataset):
    def __init__(self, images, labels, patch_extractor, transform=None):
        self.patch_extractor = patch_extractor
        self.transform = transform
        self.patches = []
        self.labels = []
        
        for img, label in zip(images, labels):
            patches, _ = self.patch_extractor.extract_patches(img)
            for patch in patches:
                self.patches.append(patch)
                self.labels.append(label)
                
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        label = self.labels[idx]
        
        if self.transform:
            patch = self.transform(patch)
            
        return patch, label

patch_extractor = PatchExtractor(patch_size=224, overlap=0.5)
print("Patch extraction strategy initialized")