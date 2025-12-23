"""
Simple dataset loader for comparative analysis
"""
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
import os

class MyDataset(ImageFolder):
    """Simple wrapper around ImageFolder for compatibility"""
    
    def __init__(self, root, transform=None):
        super(MyDataset, self).__init__(root, transform=transform)
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample, target

def create_data_loaders(train_path, valid_path, train_transform, val_transform, batch_size=32):
    """Create data loaders for training and validation"""
    
    train_dataset = MyDataset(train_path, transform=train_transform)
    val_dataset = MyDataset(valid_path, transform=val_transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test the dataset
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = MyDataset("datasets/BreaKHis 400X/train", transform=transform)
    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.classes}")
    
    # Test loading a sample
    sample, target = dataset[0]
    print(f"Sample shape: {sample.shape}")
    print(f"Target: {target}")