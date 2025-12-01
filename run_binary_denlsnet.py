#!/usr/bin/env python3
"""
Binary Classification DenLsNet Training
- DenseNet-121 + Bidirectional LSTM
- 2-class classification (Benign vs Malignant)
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
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from PIL import Image

# Import our corrected modules
from model.denlsnet_corrected import create_denlsnet
from config.training_config import TrainingConfig, EarlyStopping, get_device


class BinaryDenLsNetTrainer:
    """Binary classification trainer for DenLsNet"""
    
    def __init__(self, output_dir='binary_denlsnet_results'):
        self.task = 'binary'
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"binary_experiment_{self.timestamp}"
        
        # Create directory structure
        self.model_dir = self.output_dir / "models"
        self.log_dir = self.output_dir / "logs"
        self.results_dir = self.output_dir / "results"
        self.plots_dir = self.output_dir / "plots"
        
        for dir_path in [self.model_dir, self.log_dir, self.results_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize configuration for binary task
        self.config = TrainingConfig(task='binary')
        self.device = get_device()
        
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
        
        print(f"🔬 Binary DenLsNet Training Pipeline")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🖥️  Device: {self.device}")
    
    def create_dummy_binary_dataset(self):
        """Create dummy binary dataset for demonstration"""
        print("🔧 Creating dummy binary dataset...")
        
        base_path = Path("datasets/BreaKHis_binary")
        classes = ['benign', 'malignant']
        
        for split in ['train', 'test']:
            for class_name in classes:
                class_dir = base_path / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Create dummy images
                num_images = 200 if split == 'train' else 50
                for i in range(num_images):
                    # Create realistic-looking histopathology-like images
                    if class_name == 'benign':
                        # More organized, regular patterns for benign
                        color_base = (180, 150, 200)  # Lighter purple
                    else:
                        # More chaotic, darker patterns for malignant
                        color_base = (120, 80, 140)   # Darker purple
                    
                    # Add some randomness
                    color = tuple(max(0, min(255, c + np.random.randint(-30, 30))) for c in color_base)
                    
                    img = Image.new('RGB', (224, 224), color=color)
                    
                    # Add some texture-like noise
                    pixels = np.array(img)
                    noise = np.random.normal(0, 15, pixels.shape).astype(np.int16)
                    pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                    img = Image.fromarray(pixels)
                    
                    img_path = class_dir / f"{class_name}_{i:03d}.png"
                    img.save(img_path)
        
        print(f"✅ Dummy binary dataset created at {base_path}")
        return base_path
    
    def setup_data(self):
        """Setup binary dataset and data loaders"""
        print("\n" + "="*60)
        print("📊 Setting up Binary Dataset")
        print("="*60)
        
        # Check for existing dataset or create dummy
        train_path = "datasets/BreaKHis_binary/train"
        val_path = "datasets/BreaKHis_binary/test"
        
        if not os.path.exists(train_path):
            dataset_path = self.create_dummy_binary_dataset()
            train_path = dataset_path / "train"
            val_path = dataset_path / "test"
        
        # Define transforms
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
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
        
        # Calculate class weights
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
        
        print(f"✅ Binary dataset loaded")
        print(f"   📊 Training samples: {len(train_dataset)}")
        print(f"   📊 Validation samples: {len(val_dataset)}")
        print(f"   📊 Classes: {train_dataset.classes}")
        print(f"   ⚖️  Class weights: {self.class_weights}")
        
        # Save dataset info
        dataset_info = {
            'task': 'binary',
            'classes': train_dataset.classes,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'class_weights': self.class_weights.cpu().tolist()
        }
        
        with open(self.log_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
    
    def setup_model(self):
        """Setup binary DenLsNet model"""
        print("\n" + "="*60)
        print("🏗️ Setting up Binary DenLsNet Model")
        print("="*60)
        
        # Create model
        self.model = create_denlsnet(num_classes=2, dropout_rate=0.5)
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
        
        print("✅ Binary model setup complete")
    
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
        epoch_f1 = f1_score(all_labels, all_predictions, average='binary')
        
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
        epoch_f1 = f1_score(all_labels, all_predictions, average='binary')
        epoch_precision = precision_score(all_labels, all_predictions, average='binary')
        epoch_recall = recall_score(all_labels, all_predictions, average='binary')
        
        # ROC-AUC for binary classification
        probs_positive = np.array(all_probabilities)[:, 1]
        epoch_auc = roc_auc_score(all_labels, probs_positive)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        metrics = {
            'loss': epoch_loss, 'accuracy': epoch_acc, 'f1_score': epoch_f1,
            'precision': epoch_precision, 'recall': epoch_recall, 'auc': epoch_auc,
            'confusion_matrix': cm.tolist(), 'predictions': all_predictions,
            'labels': all_labels, 'probabilities': all_probabilities
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
            'task': 'binary',
            'architecture_info': {
                'model_name': 'DenLsNet_Binary',
                'backbone': 'DenseNet-121',
                'lstm_type': 'Bidirectional',
                'num_classes': 2,
                'final_feature_dim': 1920
            }
        }
        
        # Save checkpoint
        checkpoint_path = self.model_dir / f"binary_denlsnet_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.model_dir / "binary_denlsnet_best.pth"
            torch.save(checkpoint, best_path)
            
            # Deployment version
            deployment_checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'model': self.model,
                'config': self.config.__dict__,
                'best_metrics': self.best_metrics,
                'architecture_info': checkpoint['architecture_info']
            }
            deployment_path = self.model_dir / "binary_denlsnet_deployment.pth"
            torch.save(deployment_checkpoint, deployment_path)
            
            print(f"🏆 New best binary model saved!")
            print(f"   📊 F1-Score: {metrics['f1_score']:.4f}")
            print(f"   🎯 Accuracy: {metrics['accuracy']:.4f}")
            print(f"   📈 AUC: {metrics['auc']:.4f}")
        
        return checkpoint_path
    
    def log_metrics(self, epoch, train_metrics, val_metrics):
        """Log training metrics"""
        log_data = {
            'epoch': epoch + 1,
            'train_loss': train_metrics[0], 'train_acc': train_metrics[1], 'train_f1': train_metrics[2],
            'val_loss': val_metrics['loss'], 'val_acc': val_metrics['accuracy'], 'val_f1': val_metrics['f1_score'],
            'val_precision': val_metrics['precision'], 'val_recall': val_metrics['recall'], 'val_auc': val_metrics['auc'],
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        log_file = self.log_dir / "binary_training_log.csv"
        if not log_file.exists():
            pd.DataFrame([log_data]).to_csv(log_file, index=False)
        else:
            pd.DataFrame([log_data]).to_csv(log_file, mode='a', header=False, index=False)
    
    def save_plots(self):
        """Save training plots"""
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation', linewidth=2)
        ax1.set_title('Binary Classification - Loss', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Training', linewidth=2)
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Validation', linewidth=2)
        ax2.set_title('Binary Classification - Accuracy', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # F1 Score
        ax3.plot(epochs, self.history['train_f1'], 'b-', label='Training', linewidth=2)
        ax3.plot(epochs, self.history['val_f1'], 'r-', label='Validation', linewidth=2)
        ax3.set_title('Binary Classification - F1 Score', fontweight='bold')
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
        plt.savefig(self.plots_dir / "binary_training_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_training(self):
        """Main training loop"""
        print("\n" + "="*80)
        print("🚀 Starting Binary DenLsNet Training")
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
                    'accuracy': val_metrics['accuracy'],
                    'auc': val_metrics['auc'],
                    'loss': val_metrics['loss']
                })
            
            # Save model
            self.save_model(epoch + 1, val_metrics, is_best)
            
            # Print summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_score']:.4f}, AUC: {val_metrics['auc']:.4f}")
            print(f"Best  - F1: {self.best_metrics['f1_score']:.4f}, Acc: {self.best_metrics['accuracy']:.4f} (Epoch {self.best_metrics['epoch']})")
            
            # Early stopping
            if self.early_stopping(val_metrics['f1_score']):
                print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
                break
        
        # Training completed
        total_time = time.time() - start_time
        self.save_plots()
        
        # Save final results
        final_results = {
            'task': 'binary',
            'timestamp': self.timestamp,
            'total_training_time_hours': total_time / 3600,
            'epochs_trained': len(self.history['train_loss']),
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
                'best_model': str(self.model_dir / "binary_denlsnet_best.pth"),
                'deployment_model': str(self.model_dir / "binary_denlsnet_deployment.pth")
            }
        }
        
        with open(self.results_dir / "binary_final_results.json", 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"\n{'='*80}")
        print(f"🎉 Binary Training Completed!")
        print(f"⏱️  Total Time: {total_time/3600:.2f} hours")
        print(f"🏆 Best Results (Epoch {self.best_metrics['epoch']}):")
        print(f"   📊 F1-Score: {self.best_metrics['f1_score']:.4f}")
        print(f"   🎯 Accuracy: {self.best_metrics['accuracy']:.4f}")
        print(f"   📈 AUC: {self.best_metrics['auc']:.4f}")
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"{'='*80}")
        
        return final_results


def main():
    """Run binary DenLsNet training"""
    print("🔬 Binary DenLsNet Training Pipeline")
    print("="*50)
    
    try:
        trainer = BinaryDenLsNetTrainer()
        results = trainer.run_training()
        
        print(f"\n✅ Binary training completed successfully!")
        print(f"📊 Final F1-Score: {results['best_metrics']['f1_score']:.4f}")
        print(f"📁 All files saved to: {trainer.output_dir}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Binary training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()