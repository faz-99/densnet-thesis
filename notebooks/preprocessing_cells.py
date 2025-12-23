# Essential Preprocessing Steps for DenLsNet - BreakHis Dataset
# These cells should be added to the notebook after the setup cells

# Cell 1: Stain Normalization Implementation
stain_normalization_cell = '''
# 1. Stain Normalization (Critical for BreakHis)
import cv2
import numpy as np
from sklearn.decomposition import NMF
from scipy.linalg import lstsq

class StainNormalizer:
    def __init__(self, method='macenko'):
        self.method = method
        self.target_stains = None
        self.target_concentrations = None
        
    def macenko_normalize(self, image, target_image=None):
        """Macenko stain normalization"""
        # Convert to OD space
        od = -np.log((image.astype(np.float64) + 1) / 256.0)
        
        # Remove transparent pixels
        od_hat = od[~np.any(od < 0.15, axis=2)]
        
        # Compute eigenvectors
        eigvals, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
        eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]
        
        # Project on plane
        that = od_hat.dot(eigvecs[:, :2])
        phi = np.arctan2(that[:, 1], that[:, 0])
        
        # Find robust extremes
        min_phi = np.percentile(phi, 1)
        max_phi = np.percentile(phi, 99)
        
        # Stain vectors
        v1 = eigvecs[:, :2].dot(np.array([np.cos(min_phi), np.sin(min_phi)]))
        v2 = eigvecs[:, :2].dot(np.array([np.cos(max_phi), np.sin(max_phi)]))
        
        if v1[0] > v2[0]:
            he = np.array([v1, v2])
        else:
            he = np.array([v2, v1])
            
        # Normalize stain vectors
        he = he / np.linalg.norm(he, axis=1, keepdims=True)
        
        # Target stains (typical H&E)
        if target_image is None:
            target_he = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])
        else:
            target_he = self._extract_stains(target_image)
            
        # Get concentrations
        c = lstsq(he.T, od.reshape(-1, 3).T)[0]
        
        # Normalize concentrations
        max_c = np.percentile(c, 99, axis=1, keepdims=True)
        c = c / max_c * np.percentile(target_he, 99, axis=1, keepdims=True).T
        
        # Reconstruct
        normalized = np.exp(-target_he.T.dot(c)) * 255
        return np.clip(normalized.T.reshape(image.shape), 0, 255).astype(np.uint8)
    
    def reinhard_normalize(self, image, target_stats=None):
        """Reinhard color normalization in LAB space"""
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float64)
        
        # Target statistics (typical H&E values)
        if target_stats is None:
            target_means = np.array([8.63234435, -0.11501964, 0.03868433])
            target_stds = np.array([0.57506023, 0.10403329, 0.01364062])
        else:
            target_means, target_stds = target_stats
            
        # Current statistics
        means = np.mean(lab.reshape(-1, 3), axis=0)
        stds = np.std(lab.reshape(-1, 3), axis=0)
        
        # Normalize
        lab = (lab - means) / stds * target_stds + target_means
        
        # Convert back to RGB
        normalized = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        return np.clip(normalized, 0, 255).astype(np.uint8)
    
    def normalize(self, image):
        if self.method == 'macenko':
            return self.macenko_normalize(image)
        elif self.method == 'reinhard':
            return self.reinhard_normalize(image)
        else:
            return image

# Initialize normalizer
stain_normalizer = StainNormalizer(method=PREPROCESSING_CONFIG['stain_method'])
print(f"Stain normalizer initialized with method: {PREPROCESSING_CONFIG['stain_method']}")
'''

# Cell 2: Color Augmentation
color_augmentation_cell = '''
# 2. Color Augmentation for H&E Staining Variations
import torchvision.transforms as transforms
from torchvision.transforms import ColorJitter
import torch
import random

class ColorAugmentation:
    def __init__(self, hue_range=0.15, saturation_range=0.2, brightness_range=0.2, contrast_range=0.2):
        self.color_jitter = ColorJitter(
            brightness=brightness_range,
            contrast=contrast_range,
            saturation=saturation_range,
            hue=hue_range
        )
        
    def __call__(self, image):
        if random.random() > 0.5:  # Apply with 50% probability
            return self.color_jitter(image)
        return image

class HSVAugmentation:
    def __init__(self, hue_shift=15, sat_shift=20, val_shift=20):
        self.hue_shift = hue_shift
        self.sat_shift = sat_shift
        self.val_shift = val_shift
        
    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
            
        # Convert to HSV
        hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Random shifts
        hsv[:,:,0] += random.uniform(-self.hue_shift, self.hue_shift)
        hsv[:,:,1] *= random.uniform(1-self.sat_shift/100, 1+self.sat_shift/100)
        hsv[:,:,2] *= random.uniform(1-self.val_shift/100, 1+self.val_shift/100)
        
        # Clip values
        hsv[:,:,0] = np.clip(hsv[:,:,0], 0, 179)
        hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
        hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
        
        # Convert back
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return transforms.ToTensor()(rgb)

# Initialize color augmentations
color_aug = ColorAugmentation()
hsv_aug = HSVAugmentation()
print("Color augmentation transforms initialized")
'''

# Cell 3: Geometric Augmentations
geometric_augmentation_cell = '''
# 3. Geometric Augmentations (Histology-specific)
import torchvision.transforms as transforms
from torchvision.transforms import RandomRotation, RandomHorizontalFlip, RandomVerticalFlip
import torch.nn.functional as F

class HistologyAugmentation:
    def __init__(self, img_size=224):
        self.img_size = img_size
        self.transforms = transforms.Compose([
            # Histology is rotation-invariant
            RandomRotation(degrees=[0, 90, 180, 270]),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            # Mild elastic deformation simulation
            transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1))
        ])
        
    def __call__(self, image):
        return self.transforms(image)

class ElasticTransform:
    def __init__(self, alpha=1, sigma=50, alpha_affine=50):
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine
        
    def __call__(self, image):
        if random.random() > 0.3:  # Apply with 30% probability
            return image
            
        # Convert to numpy if tensor
        if isinstance(image, torch.Tensor):
            was_tensor = True
            image_np = image.permute(1, 2, 0).numpy()
        else:
            was_tensor = False
            image_np = np.array(image)
            
        shape = image_np.shape
        shape_size = shape[:2]
        
        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size, 
                          [center_square[0]+square_size, center_square[1]-square_size], 
                          center_square - square_size])
        pts2 = pts1 + np.random.uniform(-self.alpha_affine, self.alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        
        # Apply transformation
        image_np = cv2.warpAffine(image_np, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
        
        # Convert back to tensor if needed
        if was_tensor:
            image = torch.from_numpy(image_np).permute(2, 0, 1)
        else:
            image = image_np
            
        return image

# Initialize geometric augmentations
histology_aug = HistologyAugmentation()
elastic_transform = ElasticTransform()
print("Geometric augmentation transforms initialized")
'''

# Cell 4: Advanced Augmentations
advanced_augmentation_cell = '''
# 4. Advanced Augmentations (MixUp, CutMix, Random Erasing)
import torch
import numpy as np
import random

class MixUp:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def __call__(self, batch_x, batch_y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * batch_x + (1 - lam) * batch_x[index, :]
        y_a, y_b = batch_y, batch_y[index]
        
        return mixed_x, y_a, y_b, lam

class CutMix:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, batch_x, batch_y):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size)
        
        y_a, y_b = batch_y, batch_y[index]
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(batch_x.size(), lam)
        batch_x[:, :, bbx1:bbx2, bby1:bby2] = batch_x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch_x.size()[-1] * batch_x.size()[-2]))
        
        return batch_x, y_a, y_b, lam
    
    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        
        # Uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2

class RandomErasing:
    def __init__(self, probability=0.15, sl=0.02, sh=0.15, r1=0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        
    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
            
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)
            
            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                img[0, x1:x1+h, y1:y1+w] = random.random()
                img[1, x1:x1+h, y1:y1+w] = random.random()
                img[2, x1:x1+h, y1:y1+w] = random.random()
                return img
                
        return img

# Initialize advanced augmentations
mixup = MixUp(alpha=0.2)
cutmix = CutMix(alpha=1.0)
random_erasing = RandomErasing(probability=0.15)
print("Advanced augmentation methods initialized")
'''

# Cell 5: Patch Extraction Strategy
patch_extraction_cell = '''
# 5. Patch Extraction Strategy for BreakHis
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
                
                # Check tissue content (remove mostly white patches)
                if self._has_sufficient_tissue(patch):
                    patches.append(patch)
                    positions.append((x, y))
                    
        return patches, positions
    
    def _has_sufficient_tissue(self, patch):
        """Check if patch has sufficient tissue content"""
        # Convert to grayscale
        if len(patch.shape) == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        else:
            gray = patch
            
        # Threshold for tissue detection (non-white areas)
        tissue_mask = gray < 230
        tissue_ratio = np.sum(tissue_mask) / (patch.shape[0] * patch.shape[1])
        
        return tissue_ratio >= self.min_tissue_ratio

class MultiPatchDataset(Dataset):
    def __init__(self, images, labels, patch_extractor, transform=None):
        self.patch_extractor = patch_extractor
        self.transform = transform
        self.patches = []
        self.labels = []
        
        # Extract patches from all images
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

# Initialize patch extractor
patch_extractor = PatchExtractor(patch_size=224, overlap=0.5)
print("Patch extraction strategy initialized")
'''

# Cell 6: Class Imbalance Handling
class_balancing_cell = '''
# 6. Class Imbalance Handling for BreakHis
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, num_classes=8, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.size_average = size_average
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class ClassBalancer:
    def __init__(self, labels):
        self.labels = labels
        self.class_counts = Counter(labels)
        self.num_classes = len(self.class_counts)
        
    def get_class_weights(self):
        """Calculate class weights for loss function"""
        total_samples = len(self.labels)
        weights = []
        
        for i in range(self.num_classes):
            weight = total_samples / (self.num_classes * self.class_counts.get(i, 1))
            weights.append(weight)
            
        return torch.FloatTensor(weights)
    
    def get_sample_weights(self):
        """Calculate sample weights for WeightedRandomSampler"""
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[label] for label in self.labels]
        return torch.FloatTensor(sample_weights)
    
    def create_balanced_sampler(self):
        """Create WeightedRandomSampler for balanced training"""
        sample_weights = self.get_sample_weights()
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

# Example usage for BreakHis 8-class
class_names = ['Adenosis', 'Fibroadenoma', 'Phyllodes_tumor', 'Tubular_adenoma',
               'Ductal_carcinoma', 'Lobular_carcinoma', 'Mucinous_carcinoma', 'Papillary_carcinoma']

# Initialize focal loss and class balancer
focal_loss = FocalLoss(alpha=1, gamma=2, num_classes=8)
print("Class imbalance handling initialized with Focal Loss")
'''

# Cell 7: Image Quality Enhancement
image_enhancement_cell = '''
# 7. Image Quality Enhancement (CLAHE, Background Removal)
import cv2
import numpy as np
from skimage import morphology, measure
from scipy import ndimage

class ImageEnhancer:
    def __init__(self, clahe_clip_limit=2.0, clahe_tile_size=(8,8)):
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_size)
        
    def apply_clahe(self, image):
        """Apply CLAHE for contrast enhancement"""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            # Apply CLAHE to L channel
            lab[:,:,0] = self.clahe.apply(lab[:,:,0])
            # Convert back to RGB
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            enhanced = self.clahe.apply(image)
        return enhanced
    
    def remove_background(self, image, threshold=230):
        """Remove white background and artifacts"""
        # Convert to grayscale for mask creation
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Create tissue mask (non-white areas)
        tissue_mask = gray < threshold
        
        # Remove small artifacts
        tissue_mask = morphology.remove_small_objects(tissue_mask, min_size=1000)
        tissue_mask = morphology.remove_small_holes(tissue_mask, area_threshold=1000)
        
        # Apply morphological operations
        kernel = np.ones((5,5), np.uint8)
        tissue_mask = cv2.morphologyEx(tissue_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to original image
        if len(image.shape) == 3:
            result = image.copy()
            result[~tissue_mask] = [255, 255, 255]  # Set background to white
        else:
            result = image.copy()
            result[~tissue_mask] = 255
            
        return result, tissue_mask
    
    def enhance_image(self, image):
        """Complete image enhancement pipeline"""
        # Remove background
        enhanced, mask = self.remove_background(image)
        
        # Apply CLAHE only to tissue regions
        enhanced = self.apply_clahe(enhanced)
        
        # Gaussian blur for noise reduction
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return enhanced

class BackgroundRemover:
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        
    def remove_white_background(self, image):
        """Remove white background using color thresholding"""
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
            
        # Convert to HSV for better white detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define range for white color
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        
        # Create mask for white pixels
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Invert mask to get tissue regions
        tissue_mask = cv2.bitwise_not(white_mask)
        
        # Apply mask
        result = image.copy()
        result[white_mask > 0] = [255, 255, 255]
        
        return result

# Initialize image enhancer
image_enhancer = ImageEnhancer()
background_remover = BackgroundRemover()
print("Image quality enhancement tools initialized")
'''

# Cell 8: Complete Preprocessing Pipeline
complete_pipeline_cell = '''
# 8. Complete Preprocessing Pipeline Integration
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class BreakHisPreprocessingPipeline:
    def __init__(self, config=PREPROCESSING_CONFIG, img_size=224):
        self.config = config
        self.img_size = img_size
        
        # Initialize components based on config
        if config['stain_normalization']:
            self.stain_normalizer = StainNormalizer(method=config['stain_method'])
        
        if config['image_enhancement']:
            self.image_enhancer = ImageEnhancer()
            
        # Build transform pipeline
        self.train_transforms = self._build_train_transforms()
        self.val_transforms = self._build_val_transforms()
        
    def _build_train_transforms(self):
        transforms_list = []
        
        # Basic transforms
        transforms_list.extend([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size))
        ])
        
        # Geometric augmentations
        if self.config['geometric_augmentation']:
            transforms_list.extend([
                transforms.RandomRotation(degrees=[0, 90, 180, 270]),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0))
            ])
        
        # Color augmentations
        if self.config['color_augmentation']:
            transforms_list.append(
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.15)
            )
        
        # Convert to tensor
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Advanced augmentations (applied after tensor conversion)
        if self.config['advanced_augmentation']:
            transforms_list.append(RandomErasing(probability=0.15))
            
        return transforms.Compose(transforms_list)
    
    def _build_val_transforms(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image, is_training=True):
        """Complete preprocessing pipeline for a single image"""
        # Image enhancement
        if self.config['image_enhancement']:
            image = self.image_enhancer.enhance_image(image)
        
        # Stain normalization
        if self.config['stain_normalization']:
            image = self.stain_normalizer.normalize(image)
        
        # Apply transforms
        if is_training:
            return self.train_transforms(image)
        else:
            return self.val_transforms(image)
    
    def create_data_loader(self, dataset, batch_size=32, is_training=True, class_balancing=True):
        """Create data loader with optional class balancing"""
        if is_training and class_balancing and self.config['class_balancing']:
            # Extract labels for balancing
            labels = [dataset[i][1] for i in range(len(dataset))]
            balancer = ClassBalancer(labels)
            sampler = balancer.create_balanced_sampler()
            
            return DataLoader(
                dataset, 
                batch_size=batch_size, 
                sampler=sampler,
                num_workers=4,
                pin_memory=True
            )
        else:
            return DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=is_training,
                num_workers=4,
                pin_memory=True
            )

# Initialize complete preprocessing pipeline
preprocessing_pipeline = BreakHisPreprocessingPipeline(config=PREPROCESSING_CONFIG)
print("Complete preprocessing pipeline initialized")
print(f"Configuration: {PREPROCESSING_CONFIG}")
'''

print("All preprocessing cells have been prepared. Add these to your notebook in order:")
print("1. Stain Normalization")
print("2. Color Augmentation") 
print("3. Geometric Augmentations")
print("4. Advanced Augmentations")
print("5. Patch Extraction Strategy")
print("6. Class Imbalance Handling")
print("7. Image Quality Enhancement")
print("8. Complete Preprocessing Pipeline")