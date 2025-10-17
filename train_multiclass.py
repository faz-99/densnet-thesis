"""
Multi-class training script for DenLsNet-MC
Supports 8-class BreakHis classification with stain normalization ablation
"""
import argparse
import platform
import pathlib
import os
import sys
import time
import numpy as np
import torch
from torch import nn, optim
from torch.optim import SGD
import warnings
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Handle path issues
plt = platform.system()
if plt != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

warnings.filterwarnings("ignore")

# Import project modules
import config_multiclass as config
from model.multiclass_model import create_multiclass_model, get_loss_function
from utils.load_dataset2 import get_multiclass_data_loader
from utils.confusion_matrix import ConfusionMatrix
from stain_normalization import StainNormalizer, create_stain_normalized_dataset
from evaluation.metrics import ModelEvaluator

# Set display options
np.set_printoptions(suppress=True)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Set device
device = torch.device(config.device if torch.cuda.is_available() else "cpu")


class MultiClassTrainer:
    """
    Trainer for multi-class DenLsNet with comprehensive evaluation
    """
    
    def __init__(self, 
                 stain_method: str = 'none',
                 num_classes: int = 8,
                 experiment_name: str = None):
        """
        Initialize trainer
        
        Args:
            stain_method: Stain normalization method ('none', 'macenko', 'reinhard')
            num_classes: Number of classes
            experiment_name: Name for this experiment
        """
        self.stain_method = stain_method
        self.num_classes = num_classes
        self.experiment_name = experiment_name or f"denlsnet_mc_{stain_method}"
        
        # Create output directories
        self.model_dir = os.path.join(config.output_dirs['models'], stain_method)
        self.log_dir = os.path.join(config.output_dirs['logs'], stain_method)
        self.results_dir = os.path.join(config.output_dirs['results'], stain_method)
        
        for dir_path in [self.model_dir, self.log_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': []
        }
        
        # Best metrics tracking
        self.best_metrics = {
            'accuracy': 0.0,
            'f1_macro': 0.0,
            'epoch': 0,
            'confusion_matrix': None,
            'classification_report': None
        }
    
    def prepare_data(self):
        """Prepare data with optional stain normalization"""
        print(f"Preparing data with stain normalization: {self.stain_method}")
        
        # Create stain-normalized dataset if needed
        if self.stain_method != 'none':
            normalized_dir = os.path.join(config.output_dirs['normalized_data'], self.stain_method)
            
            if not os.path.exists(normalized_dir):
                print(f"Creating {self.stain_method} normalized dataset...")
                create_stain_normalized_dataset(
                    input_dir=config.train,
                    output_dir=config.output_dirs['normalized_data'],
                    method=self.stain_method
                )
                create_stain_normalized_dataset(
                    input_dir=config.valid,
                    output_dir=config.output_dirs['normalized_data'],
                    method=self.stain_method
                )
            
            # Update config paths to use normalized data
            self.train_path = os.path.join(normalized_dir, 'train')
            self.valid_path = os.path.join(normalized_dir, 'test')
        else:
            self.train_path = config.train
            self.valid_path = config.valid
        
        # Load data
        print("Loading data loaders...")
        self.train_loader, self.valid_loader, self.class_weights = get_multiclass_data_loader(
            train_path=self.train_path,
            valid_path=self.valid_path,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            img_size=config.img_s
        )
        
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.valid_loader.dataset)}")
        print(f"Class weights: {self.class_weights}")
    
    def create_model(self):
        """Create and initialize model"""
        print("Creating multi-class DenLsNet model...")
        
        self.model = create_multiclass_model(
            num_classes=self.num_classes,
            dropout_rate=config.drop_path
        )
        self.model.to(device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Model: {self.experiment_name}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def create_optimizer_and_loss(self):
        """Create optimizer and loss function"""
        # Optimizer
        self.optimizer = SGD(
            params=self.model.parameters(),
            lr=config.lr,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=config.max_epoch,
            eta_min=config.min_lr
        )
        
        # Loss function
        loss_config = config.loss_config
        if loss_config['type'] == 'categorical_crossentropy':
            if loss_config.get('class_weights') == 'balanced':
                self.criterion = nn.CrossEntropyLoss(weight=self.class_weights).to(device)
            else:
                self.criterion = nn.CrossEntropyLoss().to(device)
        elif loss_config['type'] == 'focal':
            self.criterion = get_loss_function(
                'focal',
                alpha=loss_config['focal_loss']['alpha'],
                gamma=loss_config['focal_loss']['gamma']
            ).to(device)
        elif loss_config['type'] == 'label_smoothing':
            self.criterion = get_loss_function(
                'label_smoothing',
                smoothing=loss_config.get('label_smoothing', 0.1)
            ).to(device)
        
        print(f"Loss function: {loss_config['type']}")
        print(f"Optimizer: SGD (lr={config.lr}, weight_decay={config.weight_decay})")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        train_loader = tqdm(self.train_loader, desc=f"Training Epoch {epoch}")
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
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
            
            train_loader.set_postfix({
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
        
        val_loader = tqdm(self.valid_loader, desc=f"Validation Epoch {epoch}")
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = self.model(inputs)
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
                
                val_loader.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.4f}'
                })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.valid_loader)
        epoch_acc = accuracy_score(all_labels, all_predictions)
        epoch_f1 = f1_score(all_labels, all_predictions, average='macro')
        
        # Detailed metrics
        precision_macro = precision_score(all_labels, all_predictions, average='macro')
        recall_macro = recall_score(all_labels, all_predictions, average='macro')
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Per-class metrics
        precision_per_class = precision_score(all_labels, all_predictions, average=None)
        recall_per_class = recall_score(all_labels, all_predictions, average=None)
        f1_per_class = f1_score(all_labels, all_predictions, average=None)
        
        metrics = {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'f1_macro': epoch_f1,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'confusion_matrix': cm,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'predictions': np.array(all_predictions),
            'labels': np.array(all_labels),
            'probabilities': np.array(all_probabilities)
        }
        
        return metrics
    
    def save_model(self, epoch, metrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metrics': self.best_metrics,
            'history': self.history,
            'config': {
                'num_classes': self.num_classes,
                'class_names': config.class_names,
                'stain_method': self.stain_method,
                'experiment_name': self.experiment_name,
                'img_size': config.img_s,
                'dataset_mean': config.dataset_mean,
                'dataset_std': config.dataset_std
            },
            'model': self.model  # Save entire model for easy loading
        }
        
        # Save best model
        model_path = os.path.join(self.model_dir, f"{self.experiment_name}_best.pth")
        torch.save(checkpoint, model_path)
        
        # Save epoch model
        epoch_path = os.path.join(self.model_dir, f"{self.experiment_name}_epoch_{epoch}.pth")
        torch.save(checkpoint, epoch_path)
        
        print(f"Model saved: {model_path}")
        return model_path
    
    def log_metrics(self, epoch, train_metrics, val_metrics):
        """Log training metrics"""
        log_file = os.path.join(self.log_dir, f"{self.experiment_name}.csv")
        
        # Create header if file doesn't exist
        if not os.path.exists(log_file):
            header = "epoch,train_loss,train_acc,train_f1,val_loss,val_acc,val_f1,val_precision,val_recall,lr,timestamp\n"
            with open(log_file, 'w') as f:
                f.write(header)
        
        # Log metrics
        with open(log_file, 'a') as f:
            content = f"{epoch},{train_metrics[0]:.6f},{train_metrics[1]:.4f},{train_metrics[2]:.4f}," \
                     f"{val_metrics['loss']:.6f},{val_metrics['accuracy']:.4f},{val_metrics['f1_macro']:.4f}," \
                     f"{val_metrics['precision_macro']:.4f},{val_metrics['recall_macro']:.4f}," \
                     f"{self.optimizer.param_groups[0]['lr']:.8f},{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f.write(content)
    
    def create_plots(self):
        """Create training plots"""
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # F1 Score plot
        ax3.plot(epochs, self.history['train_f1'], 'b-', label='Training F1')
        ax3.plot(epochs, self.history['val_f1'], 'r-', label='Validation F1')
        ax3.set_title('Training and Validation F1 Score')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.legend()
        ax3.grid(True)
        
        # Learning rate plot
        if hasattr(self, 'lr_history'):
            ax4.plot(epochs, self.lr_history, 'g-', label='Learning Rate')
            ax4.set_title('Learning Rate Schedule')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.legend()
            ax4.grid(True)
        else:
            ax4.axis('off')
        
        plt.tight_layout()
        plot_path = os.path.join(self.results_dir, f"{self.experiment_name}_training_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved: {plot_path}")
    
    def train(self):
        """Main training loop"""
        print(f"Starting training: {self.experiment_name}")
        print(f"Device: {device}")
        print(f"Epochs: {config.max_epoch}")
        
        # Prepare everything
        self.prepare_data()
        self.create_model()
        self.create_optimizer_and_loss()
        
        # Training loop
        self.lr_history = []
        
        for epoch in range(1, config.max_epoch + 1):
            print(f"\nEpoch {epoch}/{config.max_epoch}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate_epoch(epoch)
            
            # Update learning rate
            self.scheduler.step()
            self.lr_history.append(self.optimizer.param_groups[0]['lr'])
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_metrics['f1_macro'])
            
            # Log metrics
            self.log_metrics(epoch, (train_loss, train_acc, train_f1), val_metrics)
            
            # Check if best model
            if val_metrics['f1_macro'] > self.best_metrics['f1_macro']:
                self.best_metrics.update({
                    'accuracy': val_metrics['accuracy'],
                    'f1_macro': val_metrics['f1_macro'],
                    'epoch': epoch,
                    'confusion_matrix': val_metrics['confusion_matrix'],
                    'classification_report': {
                        'precision_macro': val_metrics['precision_macro'],
                        'recall_macro': val_metrics['recall_macro'],
                        'precision_per_class': val_metrics['precision_per_class'].tolist(),
                        'recall_per_class': val_metrics['recall_per_class'].tolist(),
                        'f1_per_class': val_metrics['f1_per_class'].tolist()
                    }
                })
                
                # Save best model
                model_path = self.save_model(epoch, val_metrics)
                
                print(f"\n🎉 New best model! F1: {val_metrics['f1_macro']:.4f}")
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_macro']:.4f}")
            print(f"Best F1: {self.best_metrics['f1_macro']:.4f} (Epoch {self.best_metrics['epoch']})")
        
        # Create final plots
        self.create_plots()
        
        # Print final results
        print(f"\n{'='*60}")
        print(f"Training completed: {self.experiment_name}")
        print(f"Best Results (Epoch {self.best_metrics['epoch']}):")
        print(f"  Accuracy: {self.best_metrics['accuracy']:.4f}")
        print(f"  F1-Score (Macro): {self.best_metrics['f1_macro']:.4f}")
        print(f"  Precision (Macro): {self.best_metrics['classification_report']['precision_macro']:.4f}")
        print(f"  Recall (Macro): {self.best_metrics['classification_report']['recall_macro']:.4f}")
        print(f"{'='*60}")
        
        return self.best_metrics


def run_ablation_study():
    """Run complete ablation study with all stain normalization methods"""
    print("Starting Multi-Class DenLsNet Ablation Study")
    print("=" * 60)
    
    methods = ['none', 'macenko', 'reinhard']
    results = {}
    
    for method in methods:
        print(f"\n🔬 Training with stain normalization: {method}")
        print("-" * 40)
        
        trainer = MultiClassTrainer(
            stain_method=method,
            num_classes=config.class_num,
            experiment_name=f"denlsnet_mc_{method}"
        )
        
        try:
            best_metrics = trainer.train()
            results[method] = best_metrics
            
        except Exception as e:
            print(f"❌ Training failed for {method}: {str(e)}")
            results[method] = None
    
    # Compare results
    print(f"\n{'='*80}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*80}")
    
    print(f"{'Method':<15} {'Accuracy':<10} {'F1-Macro':<10} {'Precision':<10} {'Recall':<10} {'Epoch':<6}")
    print("-" * 80)
    
    for method, metrics in results.items():
        if metrics:
            acc = metrics['accuracy']
            f1 = metrics['f1_macro']
            prec = metrics['classification_report']['precision_macro']
            rec = metrics['classification_report']['recall_macro']
            epoch = metrics['epoch']
            
            print(f"{method:<15} {acc:<10.4f} {f1:<10.4f} {prec:<10.4f} {rec:<10.4f} {epoch:<6}")
        else:
            print(f"{method:<15} {'FAILED':<50}")
    
    print(f"{'='*80}")
    
    # Find best method
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        best_method = max(valid_results.keys(), key=lambda k: valid_results[k]['f1_macro'])
        best_f1 = valid_results[best_method]['f1_macro']
        print(f"🏆 Best method: {best_method} (F1: {best_f1:.4f})")
    
    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Multi-class DenLsNet Training')
    parser.add_argument('--stain_method', type=str, default='none',
                       choices=['none', 'macenko', 'reinhard'],
                       help='Stain normalization method')
    parser.add_argument('--ablation', action='store_true',
                       help='Run complete ablation study')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Custom experiment name')
    
    args = parser.parse_args()
    
    if args.ablation:
        run_ablation_study()
    else:
        trainer = MultiClassTrainer(
            stain_method=args.stain_method,
            num_classes=config.class_num,
            experiment_name=args.experiment_name
        )
        trainer.train()


if __name__ == "__main__":
    main()