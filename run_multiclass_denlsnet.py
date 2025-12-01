#!/usr/bin/env python3
"""
Multiclass Classification DenLsNet Training
- DenseNet-121 + Bidirectional LSTM
- 8-class classification (BreakHis subtypes)
- 80 epochs with SGD and CosineAnnealingLR
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Import our corrected modules
from model.denlsnet_corrected import create_denlsnet
from config.training_config import TrainingConfig, EarlyStopping, get_device


class MulticlassDenLsNetTrainer:
    """Multiclass classification trainer for DenLsNet"""
    
    def __init__(self, output_dir='multiclass_denlsnet_results'):
        self.task = 'multiclass'
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"multiclass_experiment_{self.timestamp}"
        
        # Create directory structure
        self.model_dir = self.output_dir / "models"
        self.log_dir = self.output_dir / "logs"
        self.results_dir = self.output_dir / "results"
        self.plots_dir = self.output_dir / "plots"
        
        for dir_path in [self.model_dir, self.log_dir, self.results_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize configuration for multiclass task
        self.config = TrainingConfig(task='multiclass')
        self.device = get_device()
        
        # BreakHis 8-class labels
        self.class_names = [
            'adenosis', 'fibroadenoma', 'phyllodes_tumor', 'tubular_adenoma',  # Benign
            'ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma'  # Malignant
        ]
        
        # Training components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.early_stopping = None
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.class_weights = None
        
        # Metrics tracking
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'learning_rates': []
        }
        
        self.best_metrics = {
            'epoch': 0, 'f1_score': 0.0, 'accuracy': 0.0, 'loss': float('inf')
        }
        
        print(f"🔬 Multiclass DenLsNet Training Pipeline")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🖥️  Device: {self.device}")
        print(f"📊 Classes: {len(self.class_names)} ({', '.join(self.class_names)})")
    
    def create_dummy_multiclass_dataset(self):
        """Create dummy multiclass dataset for demonstration"""
        print("🔧 Creating dummy multiclass dataset...")
        
        base_path = Path("datasets/BreaKHis_multiclass")
        
        # Color schemes for different classes
        class_colors = {
            # Benign classes - lighter, more organized colors
            'adenosis': (200, 180, 220),
            'fibroadenoma': (180, 200, 190),
            'phyllodes_tumor': (190, 190, 200),
            'tubular_adenoma': (210, 190, 180),
            # Malignant classes - darker, more chaotic colors
            'ductal_carcinoma': (120, 80, 100),
            'lobular_carcinoma': (100, 90, 120),
            'mucinous_carcinoma': (110, 100, 90),
            'papillary_carcinoma': (90, 110, 100)
        }
        
        for split in ['train', 'test']:
            for class_name in self.class_names:
                class_dir = base_path / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Create dummy images
                num_images = 150 if split == 'train' else 40
                base_color = class_colors[class_name]
                
                for i in range(num_images):
                    # Add variation to base color
                    color = tuple(max(0, min(255, c + np.random.randint(-40, 40))) for c in base_color)
                    
                    img = Image.new('RGB', (224, 224), color=color)
                    
                    # Add class-specific texture patterns
                    pixels = np.array(img)
                    
                    if 'carcinoma' in class_name:  # Malignant - more noise
                        noise = np.random.normal(0, 25, pixels.shape).astype(np.int16)
                    else:  # Benign - less noise
                        noise = np.random.normal(0, 15, pixels.shape).astype(np.int16)
                    
                    pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                    
                    # Add some class-specific patterns
                    if class_name == 'ductal_carcinoma':
                        # Add some darker streaks
                        for _ in range(5):
                            x, y = np.random.randint(0, 224, 2)
                            pixels[max(0, x-2):min(224, x+3), max(0, y-10):min(224, y+11)] *= 0.7
                    
                    img = Image.fromarray(pixels)
                    img_path = class_dir / f"{class_name}_{i:03d}.png"
                    img.save(img_path)
        
        print(f"✅ Dummy multiclass dataset created at {base_path}")
        return base_path
    
    def setup_data(self):
        """Setup multiclass dataset and data loaders"""
        print("\n" + "="*60)
        print("📊 Setting up Multiclass Dataset")
        print("="*60)
        
        # Check for existing dataset or create dummy
        train_path = "datasets/BreaKHis_multiclass/train"
        val_path = "datasets/BreaKHis_multiclass/test"
        
        if not os.path.exists(train_path):
            dataset_path = self.create_dummy_multiclass_dataset()
            train_path = dataset_path / "train"
            val_path = dataset_path / "test"
        
        # Define transforms with stronger augmentation for multiclass
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = ImageFolder(train_path, transform=train_transform)
        val_dataset = ImageFolder(val_path, transform=val_transform)
        
        # Verify class names match
        print(f"Dataset classes: {train_dataset.classes}")
        print(f"Expected classes: {self.class_names}")
        
        # Calculate class weights for balanced training
        train_labels = [sample[1] for sample in train_dataset.samples]
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(train_labels), y=train_labels
        )
        self.class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True,
            num_workers=self.config.num_workers, pin_memory=torch.cuda.is_available()
        )
        
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False,
            num_workers=self.config.num_workers, pin_memory=torch.cuda.is_available()
        )
        
        print(f"✅ Multiclass dataset loaded")
        print(f"   📊 Training samples: {len(train_dataset)}")
        print(f"   📊 Validation samples: {len(val_dataset)}")
        print(f"   📊 Classes: {len(train_dataset.classes)}")
        print(f"   ⚖️  Class weights: {self.class_weights}")
        
        # Print class distribution
        from collections import Counter
        train_class_counts = Counter(train_labels)
        print(f"   📈 Class distribution:")
        for i, class_name in enumerate(train_dataset.classes):
            count = train_class_counts[i]
            percentage = count / len(train_labels) * 100
            print(f"      {class_name}: {count} ({percentage:.1f}%)")
        
        # Save dataset info
        dataset_info = {
            'task': 'multiclass',
            'classes': train_dataset.classes,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'class_weights': self.class_weights.cpu().tolist(),
            'class_distribution': dict(train_class_counts)
        }
        
        with open(self.log_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
    
    def setup_model(self):
        """Setup multiclass DenLsNet model"""
        print("\n" + "="*60)
        print("🏗️ Setting up Multiclass DenLsNet Model")
        print("="*60)
        
        # Create model for 8 classes
        self.model = create_denlsnet(num_classes=8, dropout_rate=0.5)
        self.model.to(self.device)
        
        # Print architecture summary
        self.model.print_architecture_summary()
        
        # Setup optimizer and scheduler
        self.optimizer = self.config.get_optimizer(self.model.parameters())
        self.scheduler = self.config.get_scheduler(self.optimizer)
        self.criterion = self.config.get_loss_function(self.class_weights)
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            monitor=self.config.monitor_metric
        )
        
        print("✅ Multiclass model setup complete")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_predictions, all_labels = [], []
        
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}/{self.config.epochs}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress
            current_loss = running_loss / (batch_idx + 1)
            current_acc = accuracy_score(all_labels, all_predictions)
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_predictions)
        epoch_f1 = f1_score(all_labels, all_predictions, average='macro')
        
        return epoch_loss, epoch_acc, epoch_f1
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        all_predictions, all_labels, all_probabilities = [], [], []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch+1}")
            
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                current_loss = running_loss / (batch_idx + 1)
                current_acc = accuracy_score(all_labels, all_predictions)
                pbar.set_postfix({'Loss': f'{current_loss:.4f}', 'Acc': f'{current_acc:.4f}'})
        
        # Calculate comprehensive metrics
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_labels, all_predictions)
        epoch_f1_macro = f1_score(all_labels, all_predictions, average='macro')
        epoch_f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
        epoch_precision = precision_score(all_labels, all_predictions, average='macro')
        epoch_recall = recall_score(all_labels, all_predictions, average='macro')
        
        # Per-class metrics
        per_class_f1 = f1_score(all_labels, all_predictions, average=None)
        per_class_precision = precision_score(all_labels, all_predictions, average=None)
        per_class_recall = recall_score(all_labels, all_predictions, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Classification report
        class_report = classification_report(
            all_labels, all_predictions, 
            target_names=self.class_names, 
            output_dict=True
        )
        
        metrics = {
            'loss': epoch_loss, 'accuracy': epoch_acc, 
            'f1_score': epoch_f1_macro, 'f1_weighted': epoch_f1_weighted,
            'precision': epoch_precision, 'recall': epoch_recall,
            'per_class_f1': per_class_f1.tolist(),
            'per_class_precision': per_class_precision.tolist(),
            'per_class_recall': per_class_recall.tolist(),
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'predictions': all_predictions,
            'labels': all_labels, 
            'probabilities': all_probabilities
        }
        
        return metrics
    
    def save_model(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.__dict__,
            'metrics': metrics,
            'history': self.history,
            'best_metrics': self.best_metrics,
            'model': self.model,
            'task': 'multiclass',
            'class_names': self.class_names,
            'architecture_info': {
                'model_name': 'DenLsNet_Multiclass',
                'backbone': 'DenseNet-121',
                'lstm_type': 'Bidirectional',
                'num_classes': 8,
                'final_feature_dim': 1920,
                'class_names': self.class_names
            }
        }
        
        # Save checkpoint
        checkpoint_path = self.model_dir / f"multiclass_denlsnet_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.model_dir / "multiclass_denlsnet_best.pth"
            torch.save(checkpoint, best_path)
            
            # Deployment version
            deployment_checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'model': self.model,
                'config': self.config.__dict__,
                'best_metrics': self.best_metrics,
                'class_names': self.class_names,
                'architecture_info': checkpoint['architecture_info']
            }
            deployment_path = self.model_dir / "multiclass_denlsnet_deployment.pth"
            torch.save(deployment_checkpoint, deployment_path)
            
            print(f"🏆 New best multiclass model saved!")
            print(f"   📊 F1-Score (Macro): {metrics['f1_score']:.4f}")
            print(f"   📊 F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
            print(f"   🎯 Accuracy: {metrics['accuracy']:.4f}")
        
        return checkpoint_path
    
    def log_metrics(self, epoch, train_metrics, val_metrics):
        """Log training metrics"""
        log_data = {
            'epoch': epoch + 1,
            'train_loss': train_metrics[0], 'train_acc': train_metrics[1], 'train_f1': train_metrics[2],
            'val_loss': val_metrics['loss'], 'val_acc': val_metrics['accuracy'], 
            'val_f1_macro': val_metrics['f1_score'], 'val_f1_weighted': val_metrics['f1_weighted'],
            'val_precision': val_metrics['precision'], 'val_recall': val_metrics['recall'],
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        log_file = self.log_dir / "multiclass_training_log.csv"
        if not log_file.exists():
            pd.DataFrame([log_data]).to_csv(log_file, index=False)
        else:
            pd.DataFrame([log_data]).to_csv(log_file, mode='a', header=False, index=False)
    
    def save_plots(self):
        """Save training plots and confusion matrix"""
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Training plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation', linewidth=2)
        ax1.set_title('Multiclass Classification - Loss', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Training', linewidth=2)
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Validation', linewidth=2)
        ax2.set_title('Multiclass Classification - Accuracy', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # F1 Score
        ax3.plot(epochs, self.history['train_f1'], 'b-', label='Training', linewidth=2)
        ax3.plot(epochs, self.history['val_f1'], 'r-', label='Validation', linewidth=2)
        ax3.set_title('Multiclass Classification - F1 Score (Macro)', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Learning Rate
        ax4.plot(epochs, self.history['learning_rates'], 'g-', linewidth=2)
        ax4.set_title('Learning Rate Schedule', fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "multiclass_training_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save final confusion matrix
        if len(self.history['train_loss']) > 0:
            # Get final validation metrics
            final_cm = np.array(self.best_metrics.get('confusion_matrix', np.zeros((8, 8))))
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names, yticklabels=self.class_names)
            plt.title('Final Confusion Matrix - Multiclass Classification', fontweight='bold')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(self.plots_dir / "multiclass_confusion_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def run_training(self):
        """Main training loop"""
        print("\n" + "="*80)
        print("🚀 Starting Multiclass DenLsNet Training")
        print("="*80)
        
        # Setup
        self.setup_data()
        self.setup_model()
        
        # Save config
        with open(self.log_dir / "training_config.json", 'w') as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)
        
        start_time = time.time()
        
        # Training loop
        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch+1}/{self.config.epochs}")
            print("-" * 50)
            
            # Train and validate
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_metrics['f1_score'])
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Log metrics
            self.log_metrics(epoch, (train_loss, train_acc, train_f1), val_metrics)
            
            # Check for best model
            is_best = val_metrics['f1_score'] > self.best_metrics['f1_score']
            if is_best:
                self.best_metrics.update({
                    'epoch': epoch + 1,
                    'f1_score': val_metrics['f1_score'],
                    'f1_weighted': val_metrics['f1_weighted'],
                    'accuracy': val_metrics['accuracy'],
                    'loss': val_metrics['loss'],
                    'confusion_matrix': val_metrics['confusion_matrix'],
                    'per_class_f1': val_metrics['per_class_f1']
                })
            
            # Save model
            self.save_model(epoch + 1, val_metrics, is_best)
            
            # Print summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_score']:.4f}")
            print(f"Best  - F1: {self.best_metrics['f1_score']:.4f}, Acc: {self.best_metrics['accuracy']:.4f} (Epoch {self.best_metrics['epoch']})")
            
            # Print per-class F1 scores for best epoch
            if is_best:
                print(f"Per-class F1 scores:")
                for i, (class_name, f1) in enumerate(zip(self.class_names, val_metrics['per_class_f1'])):
                    print(f"  {class_name}: {f1:.3f}")
            
            # Early stopping
            if self.early_stopping(val_metrics['f1_score']):
                print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
                break
        
        # Training completed
        total_time = time.time() - start_time
        self.save_plots()
        
        # Save final results
        final_results = {
            'task': 'multiclass',
            'timestamp': self.timestamp,
            'total_training_time_hours': total_time / 3600,
            'epochs_trained': len(self.history['train_loss']),
            'class_names': self.class_names,
            'best_metrics': self.best_metrics,
            'final_metrics': {
                'train_loss': self.history['train_loss'][-1],
                'val_loss': self.history['val_loss'][-1],
                'train_acc': self.history['train_acc'][-1],
                'val_acc': self.history['val_acc'][-1],
                'train_f1': self.history['train_f1'][-1],
                'val_f1': self.history['val_f1'][-1]
            },
            'model_paths': {
                'best_model': str(self.model_dir / "multiclass_denlsnet_best.pth"),
                'deployment_model': str(self.model_dir / "multiclass_denlsnet_deployment.pth")
            }
        }
        
        with open(self.results_dir / "multiclass_final_results.json", 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"\n{'='*80}")
        print(f"🎉 Multiclass Training Completed!")
        print(f"⏱️  Total Time: {total_time/3600:.2f} hours")
        print(f"🏆 Best Results (Epoch {self.best_metrics['epoch']}):")
        print(f"   📊 F1-Score (Macro): {self.best_metrics['f1_score']:.4f}")
        print(f"   📊 F1-Score (Weighted): {self.best_metrics['f1_weighted']:.4f}")
        print(f"   🎯 Accuracy: {self.best_metrics['accuracy']:.4f}")
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"{'='*80}")
        
        return final_results


def main():
    """Run multiclass DenLsNet training"""
    print("🔬 Multiclass DenLsNet Training Pipeline")
    print("="*50)
    
    try:
        trainer = MulticlassDenLsNetTrainer()
        results = trainer.run_training()
        
        print(f"\n✅ Multiclass training completed successfully!")
        print(f"📊 Final F1-Score: {results['best_metrics']['f1_score']:.4f}")
        print(f"📁 All files saved to: {trainer.output_dir}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Multiclass training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()