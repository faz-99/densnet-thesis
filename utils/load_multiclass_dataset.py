"""
Multi-class data loader for 8-class BreakHis dataset
Supports stain-normalized datasets and class balancing
"""
import os
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional, Dict, List
from collections import Counter
import config_multiclass as config


class MultiClassBreakHisDataset(Dataset):
    """
    Dataset class for multi-class BreakHis histopathology images
    """
    
    def __init__(self, 
                 root_dir: str, 
                 transform: Optional[transforms.Compose] = None,
                 class_names: List[str] = None):
        """
        Initialize dataset
        
        Args:
            root_dir: Root directory containing class subdirectories
            transform: Image transformations
            class_names: List of class names (must match directory names)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = class_names or config.class_names
        
        # Create class to index mapping
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        
        # Load image paths and labels
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples from {root_dir}")
        self._print_class_distribution()
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load all image paths and their corresponding labels"""
        samples = []
        
        for class_name in self.class_names:
            class_dir = os.path.join(self.root_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            # Get all image files in class directory
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                    image_path = os.path.join(class_dir, filename)
                    samples.append((image_path, class_idx))
        
        return samples
    
    def _print_class_distribution(self):
        """Print class distribution statistics"""
        labels = [sample[1] for sample in self.samples]
        class_counts = Counter(labels)
        
        print("Class distribution:")
        for class_name, class_idx in self.class_to_idx.items():
            count = class_counts.get(class_idx, 0)
            percentage = (count / len(self.samples)) * 100 if self.samples else 0
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training"""
        labels = [sample[1] for sample in self.samples]
        class_counts = Counter(labels)
        
        # Calculate weights (inverse frequency)
        total_samples = len(self.samples)
        num_classes = len(self.class_names)
        
        weights = []
        for class_idx in range(num_classes):
            count = class_counts.get(class_idx, 1)  # Avoid division by zero
            weight = total_samples / (num_classes * count)
            weights.append(weight)
        
        return torch.FloatTensor(weights)
    
    def get_sample_weights(self) -> List[float]:
        """Get sample weights for WeightedRandomSampler"""
        class_weights = self.get_class_weights()
        sample_weights = []
        
        for _, label in self.samples:
            sample_weights.append(class_weights[label].item())
        
        return sample_weights
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(img_size: int = 224, 
                  dataset_mean: Tuple[float, float, float] = None,
                  dataset_std: Tuple[float, float, float] = None,
                  augment: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get training and validation transforms
    
    Args:
        img_size: Target image size
        dataset_mean: Dataset mean for normalization
        dataset_std: Dataset std for normalization
        augment: Whether to apply data augmentation
        
    Returns:
        Tuple of (train_transform, val_transform)
    """
    if dataset_mean is None:
        dataset_mean = config.dataset_mean
    if dataset_std is None:
        dataset_std = config.dataset_std
    
    # Base transforms
    base_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ]
    
    # Training transforms with augmentation
    if augment:
        train_transforms = [
            transforms.Resize((int(img_size * 1.1), int(img_size * 1.1))),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean, std=dataset_std)
        ]
    else:
        train_transforms = base_transforms.copy()
    
    # Validation transforms (no augmentation)
    val_transforms = base_transforms.copy()
    
    return (
        transforms.Compose(train_transforms),
        transforms.Compose(val_transforms)
    )


def get_multiclass_data_loader(
    train_path: str = None,
    valid_path: str = None,
    batch_size: int = 32,
    num_workers: int = 0,
    img_size: int = 224,
    use_weighted_sampling: bool = True,
    augment_training: bool = True
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    Create data loaders for multi-class BreakHis dataset
    
    Args:
        train_path: Path to training data
        valid_path: Path to validation data
        batch_size: Batch size
        num_workers: Number of worker processes
        img_size: Target image size
        use_weighted_sampling: Whether to use weighted sampling for class balance
        augment_training: Whether to apply data augmentation to training data
        
    Returns:
        Tuple of (train_loader, val_loader, class_weights)
    """
    if train_path is None:
        train_path = config.train
    if valid_path is None:
        valid_path = config.valid
    
    print(f"Loading multi-class data from:")
    print(f"  Train: {train_path}")
    print(f"  Valid: {valid_path}")
    
    # Get transforms
    train_transform, val_transform = get_transforms(
        img_size=img_size,
        augment=augment_training
    )
    
    # Create datasets
    train_dataset = MultiClassBreakHisDataset(
        root_dir=train_path,
        transform=train_transform,
        class_names=config.class_names
    )
    
    val_dataset = MultiClassBreakHisDataset(
        root_dir=valid_path,
        transform=val_transform,
        class_names=config.class_names
    )
    
    # Get class weights
    class_weights = train_dataset.get_class_weights()
    print(f"Class weights: {class_weights}")
    
    # Create samplers
    train_sampler = None
    if use_weighted_sampling and len(train_dataset) > 0:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        print("Using weighted random sampling for training")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # Don't shuffle if using sampler
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    print(f"Data loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader, class_weights


def create_multiclass_dataset_structure(base_dir: str, output_dir: str):
    """
    Convert binary BreakHis dataset to multi-class structure
    
    Args:
        base_dir: Base directory with binary structure (benign/malignant)
        output_dir: Output directory for multi-class structure
    """
    print(f"Converting dataset structure from {base_dir} to {output_dir}")
    
    # Mapping from filename patterns to class names
    # This is specific to BreakHis dataset naming convention
    class_mapping = {
        'A': 'Adenosis',
        'F': 'Fibroadenoma', 
        'PT': 'Phyllodes_tumor',
        'TA': 'Tubular_adenoma',
        'DC': 'Ductal_carcinoma',
        'LC': 'Lobular_carcinoma',
        'MC': 'Mucinous_carcinoma',
        'PC': 'Papillary_carcinoma'
    }
    
    # Create output directories
    for split in ['train', 'test']:
        for class_name in config.class_names:
            class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(class_dir, exist_ok=True)
    
    # Process each split
    for split in ['train', 'test']:
        split_dir = os.path.join(base_dir, split)
        
        if not os.path.exists(split_dir):
            print(f"Warning: Split directory not found: {split_dir}")
            continue
        
        # Process benign and malignant directories
        for category in ['benign', 'malignant']:
            category_dir = os.path.join(split_dir, category)
            
            if not os.path.exists(category_dir):
                print(f"Warning: Category directory not found: {category_dir}")
                continue
            
            # Process all images in category
            for filename in os.listdir(category_dir):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                    continue
                
                # Extract class from filename (BreakHis naming convention)
                # Example: SOB_B_A_14-22549AB_40_001.png
                parts = filename.split('_')
                if len(parts) >= 3:
                    class_code = parts[2]  # 'A', 'F', 'PT', etc.
                    
                    if class_code in class_mapping:
                        target_class = class_mapping[class_code]
                        
                        # Copy file to appropriate class directory
                        src_path = os.path.join(category_dir, filename)
                        dst_path = os.path.join(output_dir, split, target_class, filename)
                        
                        try:
                            import shutil
                            shutil.copy2(src_path, dst_path)
                        except Exception as e:
                            print(f"Error copying {src_path}: {e}")
                    else:
                        print(f"Unknown class code: {class_code} in {filename}")
    
    print("Dataset structure conversion completed!")
    
    # Print statistics
    for split in ['train', 'test']:
        print(f"\n{split.title()} set statistics:")
        for class_name in config.class_names:
            class_dir = os.path.join(output_dir, split, class_name)
            if os.path.exists(class_dir):
                count = len([f for f in os.listdir(class_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))])
                print(f"  {class_name}: {count} images")


if __name__ == "__main__":
    # Test the data loader
    print("Testing multi-class data loader...")
    
    # Check if multiclass dataset exists, if not create it
    multiclass_train_dir = "datasets/BreaKHis 400X/multiclass/train"
    if not os.path.exists(multiclass_train_dir):
        print("Creating multi-class dataset structure...")
        create_multiclass_dataset_structure(
            base_dir="datasets/BreaKHis 400X",
            output_dir="datasets/BreaKHis 400X/multiclass"
        )
    
    # Test data loader
    try:
        train_loader, val_loader, class_weights = get_multiclass_data_loader(
            train_path="datasets/BreaKHis 400X/multiclass/train",
            valid_path="datasets/BreaKHis 400X/multiclass/test",
            batch_size=8,
            num_workers=0
        )
        
        print("\nTesting data loader...")
        for i, (images, labels) in enumerate(train_loader):
            print(f"Batch {i+1}: Images shape: {images.shape}, Labels: {labels}")
            if i >= 2:  # Test first 3 batches
                break
        
        print("Data loader test completed successfully!")
        
    except Exception as e:
        print(f"Error testing data loader: {e}")
        print("Please ensure the dataset is properly structured.")