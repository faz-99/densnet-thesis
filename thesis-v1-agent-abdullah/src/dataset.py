"""
Dataset and DataLoader utilities for BreakHis histopathology images.

Provides:
  - BreakHisDataset: PyTorch Dataset for 8-class classification
  - get_transforms: training and validation augmentation pipelines
  - create_dataloaders: ready-to-use DataLoaders with class-balanced sampling
"""
import os
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from .config import (
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    BATCH_SIZE, NUM_WORKERS, RANDOM_SEED, CLASS_NAMES,
)


class BreakHisDataset(Dataset):
    """
    PyTorch Dataset for BreakHis 400X histopathology images.

    Expects directory structure:
        root/
        ├── adenosis/
        │   ├── img001.png
        │   └── ...
        ├── ductal_carcinoma/
        │   └── ...
        └── ... (8 class folders)

    Args:
        root_dir: Path to the split directory (e.g., data/processed/none/train)
        transform: torchvision transforms to apply
        class_to_idx: Optional pre-built mapping; built automatically if None
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.root_dir = root_dir
        self.transform = transform

        # Build class mapping (alphabetical = deterministic)
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            self.class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Collect all image paths and labels
        self.samples: List[Tuple[str, int]] = []
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                    self.samples.append((os.path.join(class_dir, fname), class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def targets(self) -> List[int]:
        """All labels - needed by WeightedRandomSampler."""
        return [label for _, label in self.samples]

    def class_distribution(self) -> Dict[str, int]:
        """Count of images per class."""
        counts = Counter(self.targets)
        return {self.idx_to_class[k]: v for k, v in sorted(counts.items())}


def get_transforms(split: str = "train") -> transforms.Compose:
    """
    Get augmentation pipeline for a given split.

    Training augmentations:
      - RandomResizedCrop: simulates different zoom levels
      - RandomHorizontalFlip: tissue orientation is arbitrary
      - RandomVerticalFlip: same reason
      - ColorJitter: handles stain variation (complementary to normalization)
      - RandomRotation: tissue can be at any angle
      - Normalize: ImageNet stats (for pretrained backbones)

    Validation/Test:
      - Resize + CenterCrop: deterministic, no randomness
      - Normalize: same as training
    """
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE + 32),  # slight oversize
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def compute_class_weights(dataset: BreakHisDataset) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for loss function.

    Why: BreakHis is imbalanced (ductal carcinoma has ~2.5x more samples
    than lobular carcinoma). Without weighting, the model would learn to
    always predict the majority class.

    Formula: weight_c = N_total / (N_classes * N_c)
    """
    counts = Counter(dataset.targets)
    total = len(dataset)
    n_classes = len(counts)
    weights = torch.zeros(n_classes)
    for cls_idx, count in counts.items():
        weights[cls_idx] = total / (n_classes * count)
    return weights


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    test_dir: str,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Create DataLoaders with class-balanced sampling for training.

    Returns:
        (train_loader, val_loader, test_loader, class_weights)
    """
    train_dataset = BreakHisDataset(train_dir, transform=get_transforms("train"))
    val_dataset = BreakHisDataset(val_dir, transform=get_transforms("val"))
    test_dataset = BreakHisDataset(test_dir, transform=get_transforms("test"))

    # Weighted sampler for balanced training batches
    class_weights = compute_class_weights(train_dataset)
    sample_weights = [class_weights[label] for label in train_dataset.targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_weights
