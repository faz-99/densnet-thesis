#!/usr/bin/env python3
"""
Run comprehensive XAI evaluation on histopathology images
Integrates with existing DenseNet201 transfer learning model
"""
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import argparse
from pathlib import Path
from PIL import Image
import json

from xai.evaluate_xai import XAIEvaluator


class HistopathologyDataset(Dataset):
    """Simple dataset for histopathology images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, os.path.basename(image_path)


def load_transfer_learning_model(num_classes=2):
    """Load DenseNet201 transfer learning model (same as in app.py)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pre-trained DenseNet201
    model = models.densenet201(weights='IMAGENET1K_V1')  # Updated syntax
    
    # Modify classifier for binary/multi-class classification
    num_features = model.classifier.in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(num_features, num_classes)
    )
    
    model.to(device)
    model.eval()
    
    return model, device


def get_image_transforms():
    """Get image preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def collect_sample_images(data_dir, num_samples=10):
    """Collect sample images for evaluation"""
    image_paths = []
    labels = []
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"⚠️ Data directory {data_dir} not found")
        print("Creating synthetic evaluation with random images...")
        return create_synthetic_evaluation()
    
    # Look for common histopathology dataset structures
    for class_idx, class_folder in enumerate(['benign', 'malignant', 'Benign', 'Malignant']):
        class_path = data_path / class_folder
        if class_path.exists():
            image_files = list(class_path.glob('*.png')) + list(class_path.glob('*.jpg')) + list(class_path.glob('*.jpeg'))
            
            # Take up to num_samples//2 from each class
            selected_files = image_files[:num_samples//2]
            image_paths.extend(selected_files)
            labels.extend([class_idx % 2] * len(selected_files))  # Binary classification
    
    if not image_paths:
        print("No images found in expected structure, creating synthetic evaluation...")
        return create_synthetic_evaluation()
    
    return image_paths, labels


def create_synthetic_evaluation():
    """Create synthetic evaluation with random images for demonstration"""
    print("🎭 Creating synthetic evaluation with random images")
    
    # Create temporary directory for synthetic images
    temp_dir = Path('temp_synthetic_images')
    temp_dir.mkdir(exist_ok=True)
    
    image_paths = []
    labels = []
    
    # Generate synthetic images
    for i in range(10):
        # Create random RGB image
        random_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Add some structure to make it look more like tissue
        # Add some blob-like structures
        center_x, center_y = np.random.randint(50, 174, 2)
        y, x = np.ogrid[:224, :224]
        mask = (x - center_x)**2 + (y - center_y)**2 < 40**2
        random_image[mask] = random_image[mask] * 0.7 + np.array([150, 100, 150]) * 0.3
        
        # Save synthetic image
        img_path = temp_dir / f'synthetic_image_{i}.png'
        Image.fromarray(random_image).save(img_path)
        
        image_paths.append(img_path)
        labels.append(i % 2)  # Alternate between classes
    
    return image_paths, labels


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive XAI evaluation')
    parser.add_argument('--data_dir', type=str, default='data/test',
                       help='Directory containing test images')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of images to evaluate')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes (2 for binary, 8 for multi-class)')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--target_layer', type=str, default='features.norm5',
                       help='Target layer for Grad-CAM')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    print("🔬 Comprehensive XAI Evaluation Pipeline")
    print("="*60)
    print(f"📁 Data directory: {args.data_dir}")
    print(f"🔢 Number of samples: {args.num_samples}")
    print(f"🏷️ Number of classes: {args.num_classes}")
    print(f"💾 Results directory: {args.results_dir}")
    print(f"🎯 Target layer: {args.target_layer}")
    print("="*60)
    
    # Load model
    print("\n🤖 Loading DenseNet201 transfer learning model...")
    model, device = load_transfer_learning_model(args.num_classes)
    print(f"✅ Model loaded on {device}")
    
    # Define class names
    if args.num_classes == 2:
        class_names = ['Benign', 'Malignant']
    else:
        class_names = [f'Class_{i}' for i in range(args.num_classes)]
    
    # Collect images
    print(f"\n📸 Collecting {args.num_samples} sample images...")
    image_paths, labels = collect_sample_images(args.data_dir, args.num_samples)
    print(f"✅ Found {len(image_paths)} images")
    
    # Create dataset and dataloader
    transform = get_image_transforms()
    dataset = HistopathologyDataset(image_paths, labels, transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize XAI evaluator
    print(f"\n🧠 Initializing XAI evaluator...")
    evaluator = XAIEvaluator(
        model=model,
        device=device,
        class_names=class_names,
        target_layer=args.target_layer,
        results_dir=args.results_dir
    )
    
    # Run evaluation
    print(f"\n🔍 Running XAI evaluation...")
    images_batch = []
    labels_batch = []
    ids_batch = []
    
    for batch_idx, (images, batch_labels, image_ids) in enumerate(dataloader):
        images = images.to(device)
        
        for i in range(images.shape[0]):
            images_batch.append(images[i:i+1])  # Keep batch dimension
            labels_batch.append(batch_labels[i].item())
            ids_batch.append(image_ids[i])
    
    # Evaluate batch
    evaluator.evaluate_batch(images_batch, labels_batch, ids_batch)
    
    # Compute and display results
    print(f"\n📊 Computing aggregated results...")
    evaluator.compute_aggregated_results()
    evaluator.print_summary_table()
    
    # Save results
    print(f"\n💾 Saving results...")
    json_path = evaluator.save_results()
    csv_path = evaluator.save_csv_results()
    
    # Create visualization
    print(f"\n📈 Creating visualization...")
    evaluator.create_visualization()
    
    print(f"\n✅ XAI Evaluation Complete!")
    print(f"📄 JSON results: {json_path}")
    print(f"📊 CSV results: {csv_path}")
    print(f"📁 All results saved in: {args.results_dir}")
    
    # Print final summary
    if evaluator.results['aggregated_results']['summary_table']:
        print(f"\n🎯 FINAL SUMMARY:")
        print("-" * 60)
        for row in evaluator.results['aggregated_results']['summary_table']:
            method = row['XAI Method']
            insertion = row['Insertion AUC']
            deletion = row['Deletion AUC']
            iou = row['IoU']
            stability = row['Stability']
            print(f"{method:12} | {insertion:>12} | {deletion:>12} | {iou:>8} | {stability:>9}")
        print("-" * 60)
        print("Higher is better: Insertion AUC, IoU, Stability")
        print("Lower is better: Deletion AUC")


if __name__ == "__main__":
    main()