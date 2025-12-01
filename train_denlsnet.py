"""
DenLsNet Training Script
Complete training pipeline with proper configuration matching paper specifications
"""
import argparse
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Import our modules
from model.denlsnet_corrected import create_denlsnet
from data.breakhis_dataset import setup_breakhis_dataset
from config.training_config import TrainingConfig, EarlyStopping, get_device, save_checkpoint


class DenLsNetTrainer:
    """
    Complete DenLsNet training pipeline
    """
    
    def __init__(self, task='binary'):
        self.task = task
        self.config = TrainingConfig(task=task)
        self.device = get_device()
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.early_stopping = None
        
        # Data
        self.dataloaders = None
        self.datasets = None
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
    
    def setup_data(self):
        """Setup dataset and dataloaders"""
        print("📊 Setting up dataset...")
        
        # Setup BreakHis dataset
        self.dataset_manager = setup_breakhis_dataset(
            root_dir=self.config.data_root,
            magnification=self.config.magnification
        )
        
        if not self.dataset_manager:
            raise RuntimeError("Failed to setup dataset")
        
        # Create dataloaders
        self.dataloaders, self.datasets = self.dataset_manager.create_dataloaders(
            task=self.task,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers
        )
        
        # Get class weights for balanced training
        self.class_weights = self.datasets['train'].get_class_weights().to(self.device)
        
        print(f"✅ Dataset setup complete")
        print(f"   Train: {len(self.datasets['train'])} samples")
        print(f"   Val: {len(self.datasets['val'])} samples") 
        print(f"   Test: {len(self.datasets['test'])} samples")
        print(f"   Class weights: {self.class_weights}")
    
    def setup_model(self):
        """Setup model, optimizer, scheduler, and loss function"""
        print("🏗️ Setting up model...")
        
        # Create model
        self.model = create_denlsnet(
            num_classes=self.config.num_classes,
            dropout_rate=self.config.dropout_rate
        )
        self.model.to(self.device)
        
        # Print architecture summary
        self.model.print_architecture_summary()
        
        # Setup optimizer
        self.optimizer = self.config.get_optimizer(self.model.parameters())
        
        # Setup scheduler
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
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        # Progress bar
        pbar = tqdm(self.dataloaders['train'], desc=f"Epoch {epoch+1}/{self.config.epochs}")
        
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
        epoch_loss = running_loss / len(self.dataloaders['train'])
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
            pbar = tqdm(self.dataloaders['val'], desc=f"Validation {epoch+1}")
            
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                probabilities = F.softmax(outputs, dim=1)
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
        epoch_loss = running_loss / len(self.dataloaders['val'])
        epoch_acc = accuracy_score(all_labels, all_predictions)
        epoch_f1 = f1_score(all_labels, all_predictions, average='macro')
        epoch_precision = precision_score(all_labels, all_predictions, average='macro')
        epoch_recall = recall_score(all_labels, all_predictions, average='macro')
        
        # Additional metrics for binary classification
        additional_metrics = {}
        if self.config.num_classes == 2:
            # ROC-AUC for binary classification
            probs_positive = np.array(all_probabilities)[:, 1]  # Probability of positive class
            additional_metrics['roc_auc'] = roc_auc_score(all_labels, probs_positive)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Per-class metrics
        per_class_precision = precision_score(all_labels, all_predictions, average=None)
        per_class_recall = recall_score(all_labels, all_predictions, average=None)
        per_class_f1 = f1_score(all_labels, all_predictions, average=None)
        
        metrics = {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'f1_score': epoch_f1,
            'precision': epoch_precision,
            'recall': epoch_recall,
            'confusion_matrix': cm,
            'per_class_precision': per_class_precision,
            'per_class_recall': per_class_recall,
            'per_class_f1': per_class_f1,
            'predictions': np.array(all_predictions),
            'labels': np.array(all_labels),
            'probabilities': np.array(all_probabilities),
            **additional_metrics
        }
        
        return metrics
    
    def save_model(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.__dict__,
            'metrics': metrics,
            'history': self.history,
            'best_metrics': self.best_metrics,
            'model': self.model  # Save entire model for easy loading
        }
        
        # Save regular checkpoint
        checkpoint_path = self.config.model_dir / f"denlsnet_{self.task}_epoch_{epoch}.pth"
        save_checkpoint(
            self.model, self.optimizer, self.scheduler, 
            epoch, metrics, checkpoint_path
        )
        
        # Save best model
        if is_best:
            best_path = self.config.model_dir / f"denlsnet_{self.task}_best.pth"
            save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                epoch, metrics, best_path
            )
            print(f"🏆 New best model saved! F1: {metrics['f1_score']:.4f}")
        
        return checkpoint_path
    
    def log_metrics(self, epoch, train_metrics, val_metrics):
        """Log training metrics to CSV"""
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
        
        # Add ROC-AUC for binary classification
        if 'roc_auc' in val_metrics:
            log_data['val_roc_auc'] = val_metrics['roc_auc']
        
        # Save to CSV
        log_file = self.config.log_dir / f"training_log_{self.task}.csv"
        
        # Create header if file doesn't exist
        if not log_file.exists():
            pd.DataFrame([log_data]).to_csv(log_file, index=False)
        else:
            pd.DataFrame([log_data]).to_csv(log_file, mode='a', header=False, index=False)
    
    def train(self):
        """Main training loop"""
        print(f"🚀 Starting DenLsNet Training ({self.task.upper()})")
        print("=" * 80)
        
        # Print configuration
        self.config.print_config()
        self.config.log_environment_info()
        
        # Setup everything
        self.setup_data()
        self.setup_model()
        
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
            self.log_metrics(epoch, (train_loss, train_acc, train_f1), val_metrics)
            
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
            self.save_model(epoch + 1, val_metrics, is_best)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_score']:.4f}")
            
            if 'roc_auc' in val_metrics:
                print(f"Val   - ROC-AUC: {val_metrics['roc_auc']:.4f}")
            
            print(f"Best F1: {self.best_metrics['f1_score']:.4f} (Epoch {self.best_metrics['epoch']})")
            
            # Early stopping check
            if self.early_stopping:
                should_stop = self.early_stopping(val_metrics['f1_score'])
                if should_stop:
                    print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
                    print(f"No improvement in {self.config.monitor_metric} for {self.config.patience} epochs")
                    break
        
        # Training completed
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"🎉 Training Completed!")
        print(f"Total Time: {total_time/3600:.2f} hours")
        print(f"Best Results (Epoch {self.best_metrics['epoch']}):")
        print(f"  F1-Score: {self.best_metrics['f1_score']:.4f}")
        print(f"  Accuracy: {self.best_metrics['accuracy']:.4f}")
        print(f"  Loss: {self.best_metrics['loss']:.4f}")
        print(f"{'='*80}")
        
        return self.best_metrics


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train DenLsNet for BreakHis Classification')
    parser.add_argument('--task', type=str, choices=['binary', 'multiclass'], default='binary',
                       help='Classification task (binary or multiclass)')
    parser.add_argument('--epochs', type=int, default=80,
                       help='Number of training epochs (default: 80)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.003,
                       help='Learning rate (default: 0.003)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = DenLsNetTrainer(task=args.task)
    
    # Override config if specified
    if args.epochs != 80:
        trainer.config.epochs = args.epochs
        trainer.config.T_max = args.epochs
    if args.batch_size != 32:
        trainer.config.batch_size = args.batch_size
    if args.lr != 0.003:
        trainer.config.learning_rate = args.lr
    
    # Train model
    try:
        best_metrics = trainer.train()
        
        # Save final results
        results_file = trainer.config.results_dir / f"final_results_{args.task}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'task': args.task,
                'best_metrics': best_metrics,
                'config': trainer.config.__dict__,
                'training_completed': True
            }, f, indent=2, default=str)
        
        print(f"📊 Results saved to: {results_file}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()