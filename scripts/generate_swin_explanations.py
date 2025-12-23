"""Generate Swin Transformer Explainability Visualizations"""
import torch
import argparse
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from swin_explainability import UnifiedSwinVisualizer
from model.swin_transformer import swin_base_patch4_window7_224
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def load_swin_model(model_path: str, num_classes: int = 8, device: str = 'cuda'):
    """Load trained Swin model"""
    model = swin_base_patch4_window7_224(num_classes=num_classes)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model


def get_dataloader(data_path: str, batch_size: int = 1):
    """Create dataloader for BreakHis"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return loader


def main():
    parser = argparse.ArgumentParser(description='Generate Swin Explainability Visualizations')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained Swin model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='results/swin_explainability', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to process')
    parser.add_argument('--num_classes', type=int, default=8, help='Number of classes')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}")
    model = load_swin_model(args.model_path, args.num_classes, args.device)
    
    print(f"Loading data from {args.data_path}")
    dataloader = get_dataloader(args.data_path)
    
    print("Initializing visualizer")
    visualizer = UnifiedSwinVisualizer(model)
    
    print(f"Processing {args.num_samples} samples...")
    os.makedirs(args.output_dir, exist_ok=True)
    visualizer.batch_process(dataloader, args.output_dir, args.num_samples)
    
    print(f"Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
