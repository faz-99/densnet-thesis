#!/usr/bin/env python3
"""
Run Corrected DenLsNet Training Pipeline
- Uses corrected DenseNet-121 + Bidirectional LSTM architecture
- Proper 80-epoch training with SGD and CosineAnnealingLR
- Full reproducibility settings
- Saves model in separate folder for later use
"""

import os
import sys
import time
import json
import shutil
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

# Import our corrected modules
from model.denlsnet_corrected import create_denlsnet
from config.training_config import TrainingConfig, EarlyStopping, get_device, save_checkpoint


class CorrectedDenLsNetRunner:
    """
    Complete pipeline runner for corrected DenLsNet implementation
    """
    
    def __init__(self, task='binary', output_base_dir='corrected_denlsnet_results'):
        self.task = task
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_base_dir = Path(output_base_dir)
        self.experiment_dir = self.output_base_dir / f"experiment_{self.timestamp}"
        
        # Create directory structure
        self.model_dir = self.experiment_dir / "models"
        self.log_dir = self.experiment_dir / "logs"
        self.results_dir = self.experiment_dir / "results"
        self.config_dir = self.experiment_dir / "config"
        
        # Create all directories
        for dir_path in [self.model_dir, self.log_dir, self.results_dir, self.config_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize configuration
        self.config = TrainingConfig(task=task)
        self.device = get_device()
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.early_stopping = None
        
        # Data
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.class_weights = None
        
        # Metrics tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': [],
            'learning_rates': []
        }
        
        self.best_metrics = {
            'epoch': 0,
            'f1_score': 0.0,
            'accuracy': 0.0,
            'loss': float('inf')
        }
        
        print(f"🚀 Corrected DenLsNet Training Pipeline")
        print(f"📁 Experiment directory: {self.experiment_dir}")
        print(f"🎯 Task: {task.upper()}")
        print(f"🖥️  Device: {self.device}")
    
    def setup_data(self):
        """Setup dataset and data loaders"""
        print("\n" + "="*60)
        print("📊 Setting up Dataset")
        print("="*60)
        
        # Check if dataset exists
        if self.task == 'binary':
            train_path = "datasets/BreaKHis 400X/train"
            val_path = "datasets/BreaKHis 400X/test"
        else:
            train_path = "datasets/BreaKHis 400X/multiclass/train"
            val_path = "datasets/BreaKHis 400X/multiclass/test"
        
        if not os.path.exists(train_path):
            print(f"❌ Dataset not found at {train_path}")
            print("Creating dummy dataset for demonstration...")
            self._create_dummy_dataset()
            
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
        try:
            train_dataset = ImageFolder(train_path, transform=train_transform)
            val_dataset = ImageFolder(val_path, transform=val_transform)
            
            # Calculate class weights for balanced training
            train_labels = [sample[1] for sample in train_dataset.samples]
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(train_labels),
                y=train_labels
            )
            self.class_weights = torch.FloatTensor(class_weights).to(self.device)
            
            # Create data loaders
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=torch.cuda.is_available()
            )
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=torch.cuda.is_available()
            )
            
            # Use validation set as test set for now
            self.test_loader = self.val_loader
            
            print(f"✅ Dataset loaded successfully")
            print(f"   📊 Training samples: {len(train_dataset)}")
            print(f"   📊 Validation samples: {len(val_dataset)}")
            print(f"   📊 Classes: {train_dataset.classes}")
            print(f"   ⚖️  Class weights: {self.class_weights}")
            
        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            raise
    
    def _create_dummy_dataset(self):
        """Create dummy dataset for demonstration"""
        print("🔧 Creating dummy dataset...")
        
        if self.task == 'binary':
            base_path = Path("datasets/BreaKHis 400X")
            classes = ['benign', 'malignant']
        else:
            base_path = Path("datasets/BreaKHis 400X/multiclass")
            classes = ['adenosis', 'fibroadenoma', 'phyllodes_tumor', 'tubular_adenoma',
                      'ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma']
        
        # Create directory structure
        for split in ['train', 'test']:
            for class_name in classes:
                class_dir = base_path / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Create dummy images
                num_images = 100 if split == 'train' else 25
                for i in range(num_images):
                    # Create a simple colored image
                    from PIL import Image
                    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                    img = Image.new('RGB', (224, 224), color=color)
                    img_path = class_dir / f"{class_name}_{i:03d}.png"
                    img.save(img_path)
        
        print("✅ Dummy dataset created")
    
    def setup_model(self):
        """Setup corrected DenLsNet model"""
        print("\n" + "="*60)
        print("🏗️ Setting up Corrected DenLsNet Model")
        print("="*60)
        
        # Create corrected model
        self.model = create_denlsnet(
            num_classes=self.config.num_classes,
            dropout_rate=self.config.dropout_rate
        )
        self.model.to(self.device)
        
        # Print architecture summary
        self.model.print_architecture_summary()
        
        # Setup optimizer (SGD as specified)
        self.optimizer = self.config.get_optimizer(self.model.parameters())
        
        # Setup scheduler (CosineAnnealingLR as specified)
        self.scheduler = self.config.get_scheduler(self.optimizer)
        
        # Setup loss function with class weights
        self.criterion = self.config.get_loss_function(self.class_weights)
        
        # Setup early stopping
        if self.config.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=self.config.patience,
                min_delta=self.config.min_delta,
                monitor=self.config.monitor_metric
            )
        
        print("✅ Model setup complete")
        
        # Save model architecture info
        arch_info = {
            'model_name': 'DenLsNet_Corrected',
            'backbone': 'DenseNet-121',
            'lstm_type': 'Bidirectional',
            'lstm_hidden_size': 128,
            'final_feature_dim': 1920,
            'num_classes': self.config.num_classes,
            'dropout_rate': self.config.dropout_rate,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        with open(self.config_dir / "model_architecture.json", 'w') as f:
            json.dump(arch_info, f, indent=2)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}/{self.config.epochs}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            current_loss = running_loss / (batch_idx + 1)
            current_acc = accuracy_score(all_labels, all_predictions)
            
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_predictions)
        epoch_f1 = f1_score(all_labels, all_predictions, average='macro')
        
        return epoch_loss, epoch_acc, epoch_f1
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch+1}")
            
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Update progress bar
                current_loss = running_loss / (batch_idx + 1)
                current_acc = accuracy_score(all_labels, all_predictions)
                
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.4f}'
                })
        
        # Calculate comprehensive metrics
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_labels, all_predictions)
        epoch_f1 = f1_score(all_labels, all_predictions, average='macro')
        epoch_precision = precision_score(all_labels, all_predictions, average='macro')
        epoch_recall = recall_score(all_labels, all_predictions, average='macro')
        
        # Confusion matrix and classification report
        cm = confusion_matrix(all_labels, all_predictions)
        class_report = classification_report(all_labels, all_predictions, output_dict=True)
        
        metrics = {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'f1_score': epoch_f1,
            'precision': epoch_precision,
            'recall': epoch_recall,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
        
        return metrics
    
    def save_model_checkpoint(self, epoch, metrics, is_best=False):
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
            'model': self.model,  # Save entire model for easy loading
            'architecture_info': {
                'model_name': 'DenLsNet_Corrected',
                'backbone': 'DenseNet-121',
                'lstm_type': 'Bidirectional',
                'final_feature_dim': 1920,
                'task': self.task
            }
        }
        
        # Save regular checkpoint
        checkpoint_path = self.model_dir / f"denlsnet_corrected_{self.task}_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.model_dir / f"denlsnet_corrected_{self.task}_best.pth"
            torch.save(checkpoint, best_path)
            
            # Also save a deployment-ready version
            deployment_checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'model': self.model,
                'config': self.config.__dict__,
                'best_metrics': self.best_metrics,
                'architecture_info': checkpoint['architecture_info']
            }
            deployment_path = self.model_dir / f"denlsnet_corrected_{self.task}_deployment.pth"
            torch.save(deployment_checkpoint, deployment_path)
            
            print(f"🏆 New best model saved!")
            print(f"   📁 Best model: {best_path}")
            print(f"   🚀 Deployment model: {deployment_path}")
            print(f"   📊 F1-Score: {metrics['f1_score']:.4f}")
        
        return checkpoint_path
    
    def log_training_metrics(self, epoch, train_metrics, val_metrics):
        """Log training metrics"""
        log_data = {
            'epoch': epoch + 1,
            'train_loss': train_metrics[0],
            'train_acc': train_metrics[1],
            'train_f1': train_metrics[2],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'val_f1': val_metrics['f1_score'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save to CSV
        log_file = self.log_dir / f"training_log_{self.task}.csv"
        
        if not log_file.exists():
            pd.DataFrame([log_data]).to_csv(log_file, index=False)
        else:
            pd.DataFrame([log_data]).to_csv(log_file, mode='a', header=False, index=False)
    
    def save_training_plots(self):
        """Save training progress plots"""
        import matplotlib.pyplot as plt
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # F1 Score plot
        ax3.plot(epochs, self.history['train_f1'], 'b-', label='Training F1', linewidth=2)
        ax3.plot(epochs, self.history['val_f1'], 'r-', label='Validation F1', linewidth=2)
        ax3.set_title('Training and Validation F1 Score', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Learning rate plot
        ax4.plot(epochs, self.history['learning_rates'], 'g-', label='Learning Rate', linewidth=2)
        ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plot_path = self.results_dir / f"training_plots_{self.task}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training plots saved: {plot_path}")
    
    def run_training(self):
        """Main training loop"""
        print("\n" + "="*80)
        print(f"🚀 Starting Corrected DenLsNet Training ({self.task.upper()})")
        print("="*80)
        
        # Print configuration
        self.config.print_config()
        self.config.log_environment_info()
        
        # Setup everything
        self.setup_data()
        self.setup_model()
        
        # Save configuration
        config_dict = {
            'task': self.task,
            'timestamp': self.timestamp,
            'device': str(self.device),
            'training_config': self.config.__dict__,
            'experiment_dir': str(self.experiment_dir)
        }
        
        with open(self.config_dir / "experiment_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        print(f"\n🎯 Training for {self.config.epochs} epochs...")
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch+1}/{self.config.epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate_epoch(epoch)
            
            # Update learning rate
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
            self.log_training_metrics(epoch, (train_loss, train_acc, train_f1), val_metrics)
            
            # Check for best model
            is_best = val_metrics['f1_score'] > self.best_metrics['f1_score']
            if is_best:
                self.best_metrics.update({
                    'epoch': epoch + 1,
                    'f1_score': val_metrics['f1_score'],
                    'accuracy': val_metrics['accuracy'],
                    'loss': val_metrics['loss']
                })
            
            # Save model
            self.save_model_checkpoint(epoch + 1, val_metrics, is_best)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_score']:.4f}")
            print(f"Best F1: {self.best_metrics['f1_score']:.4f} (Epoch {self.best_metrics['epoch']})")
            
            # Early stopping check
            if self.early_stopping:
                should_stop = self.early_stopping(val_metrics['f1_score'])
                if should_stop:
                    print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
                    print(f"No improvement in F1-score for {self.config.patience} epochs")
                    break
        
        # Training completed
        total_time = time.time() - start_time
        
        # Save training plots
        self.save_training_plots()
        
        # Save final results
        final_results = {
            'experiment_info': {
                'task': self.task,
                'timestamp': self.timestamp,
                'total_training_time_hours': total_time / 3600,
                'total_epochs_trained': len(self.history['train_loss']),
                'early_stopped': len(self.history['train_loss']) < self.config.epochs
            },
            'best_metrics': self.best_metrics,
            'final_metrics': {
                'train_loss': self.history['train_loss'][-1],
                'val_loss': self.history['val_loss'][-1],
                'train_acc': self.history['train_acc'][-1],
                'val_acc': self.history['val_acc'][-1],
                'train_f1': self.history['train_f1'][-1],
                'val_f1': self.history['val_f1'][-1]
            },
            'training_history': self.history,
            'model_paths': {
                'best_model': str(self.model_dir / f"denlsnet_corrected_{self.task}_best.pth"),
                'deployment_model': str(self.model_dir / f"denlsnet_corrected_{self.task}_deployment.pth")
            }
        }
        
        with open(self.results_dir / f"final_results_{self.task}.json", 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"\n{'='*80}")
        print(f"🎉 Training Completed!")
        print(f"⏱️  Total Time: {total_time/3600:.2f} hours")
        print(f"🏆 Best Results (Epoch {self.best_metrics['epoch']}):")
        print(f"   📊 F1-Score: {self.best_metrics['f1_score']:.4f}")
        print(f"   🎯 Accuracy: {self.best_metrics['accuracy']:.4f}")
        print(f"   📉 Loss: {self.best_metrics['loss']:.4f}")
        print(f"📁 Results saved to: {self.experiment_dir}")
        print(f"🚀 Best model: {self.model_dir / f'denlsnet_corrected_{self.task}_best.pth'}")
        print(f"📦 Deployment model: {self.model_dir / f'denlsnet_corrected_{self.task}_deployment.pth'}")
        print(f"{'='*80}")
        
        return final_results


def main():
    """Main function to run corrected DenLsNet training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Corrected DenLsNet Training')
    parser.add_argument('--task', type=str, choices=['binary', 'multiclass'], default='binary',
                       help='Classification task (default: binary)')
    parser.add_argument('--epochs', type=int, default=80,
                       help='Number of training epochs (default: 80)')
    parser.add_argument('--output_dir', type=str, default='corrected_denlsnet_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("🔬 Corrected DenLsNet Training Pipeline")
    print("="*50)
    print(f"Task: {args.task}")
    print(f"Epochs: {args.epochs}")
    print(f"Output Directory: {args.output_dir}")
    print("="*50)
    
    # Create and run trainer
    try:
        runner = CorrectedDenLsNetRunner(task=args.task, output_base_dir=args.output_dir)
        
        # Override epochs if specified
        if args.epochs != 80:
            runner.config.epochs = args.epochs
            runner.config.T_max = args.epochs
        
        # Run training
        results = runner.run_training()
        
        print(f"\n✅ Training pipeline completed successfully!")
        print(f"📊 Final F1-Score: {results['best_metrics']['f1_score']:.4f}")
        print(f"📁 All files saved to: {runner.experiment_dir}")
        
        return results
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()