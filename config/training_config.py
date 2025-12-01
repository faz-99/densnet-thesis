"""
Training Configuration for DenLsNet Reproduction
- SGD optimizer with specified parameters
- CosineAnnealingLR scheduler
- 80 epochs training
- Early stopping with F1-score monitoring
- Reproducibility settings
"""
import torch
import numpy as np
import random
import os
from pathlib import Path


class TrainingConfig:
    """
    Centralized training configuration for DenLsNet reproduction
    """
    
    def __init__(self, task='binary'):
        self.task = task  # 'binary' or 'multiclass'
        
        # Model parameters
        self.num_classes = 2 if task == 'binary' else 8
        self.dropout_rate = 0.5
        
        # Training parameters (as specified in requirements)
        self.epochs = 80
        self.batch_size = 32
        self.num_workers = 4
        
        # Optimizer parameters (SGD as specified)
        self.optimizer_name = 'SGD'
        self.learning_rate = 0.003
        self.momentum = 0.9
        self.weight_decay = 1e-4
        
        # Scheduler parameters (CosineAnnealingLR as specified)
        self.scheduler_name = 'CosineAnnealingLR'
        self.T_max = self.epochs  # 80
        self.eta_min = 1e-6
        
        # Early stopping parameters
        self.early_stopping = True
        self.patience = 10
        self.monitor_metric = 'f1_score'  # Monitor F1-score as specified
        self.min_delta = 0.001
        
        # Data parameters
        self.img_size = 224
        self.data_root = "data"
        self.magnification = "40X"
        
        # Reproducibility settings
        self.seed = 42
        self.deterministic = True
        
        # Output directories
        self.output_root = Path("experiments")
        self.model_dir = self.output_root / "models" / task
        self.log_dir = self.output_root / "logs" / task
        self.results_dir = self.output_root / "results" / task
        
        # Create directories
        self._create_directories()
        
        # Set reproducibility
        self.set_reproducibility()
    
    def _create_directories(self):
        """Create necessary output directories"""
        for dir_path in [self.model_dir, self.log_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def set_reproducibility(self):
        """Set seeds and deterministic behavior for reproducibility"""
        print(f"🔧 Setting reproducibility (seed={self.seed})")
        
        # Set seeds
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        # Set deterministic behavior
        if self.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # For newer PyTorch versions
            if hasattr(torch, 'use_deterministic_algorithms'):
                torch.use_deterministic_algorithms(True, warn_only=True)
        
        # Set environment variables for additional reproducibility
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    def get_optimizer(self, model_parameters):
        """Get configured optimizer"""
        if self.optimizer_name == 'SGD':
            return torch.optim.SGD(
                model_parameters,
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
    
    def get_scheduler(self, optimizer):
        """Get configured learning rate scheduler"""
        if self.scheduler_name == 'CosineAnnealingLR':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.T_max,
                eta_min=self.eta_min
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_name}")
    
    def get_loss_function(self, class_weights=None):
        """Get loss function with optional class weights"""
        if class_weights is not None:
            return torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            return torch.nn.CrossEntropyLoss()
    
    def print_config(self):
        """Print complete configuration"""
        print("=" * 80)
        print(f"DenLsNet Training Configuration ({self.task.upper()})")
        print("=" * 80)
        
        print(f"📊 Model Configuration:")
        print(f"   Task: {self.task}")
        print(f"   Number of Classes: {self.num_classes}")
        print(f"   Dropout Rate: {self.dropout_rate}")
        print(f"   Image Size: {self.img_size}x{self.img_size}")
        
        print(f"\n🎯 Training Configuration:")
        print(f"   Epochs: {self.epochs}")
        print(f"   Batch Size: {self.batch_size}")
        print(f"   Number of Workers: {self.num_workers}")
        
        print(f"\n⚙️ Optimizer Configuration:")
        print(f"   Optimizer: {self.optimizer_name}")
        print(f"   Learning Rate: {self.learning_rate}")
        print(f"   Momentum: {self.momentum}")
        print(f"   Weight Decay: {self.weight_decay}")
        
        print(f"\n📈 Scheduler Configuration:")
        print(f"   Scheduler: {self.scheduler_name}")
        print(f"   T_max: {self.T_max}")
        print(f"   Eta_min: {self.eta_min}")
        
        print(f"\n🛑 Early Stopping Configuration:")
        print(f"   Enabled: {self.early_stopping}")
        print(f"   Patience: {self.patience}")
        print(f"   Monitor Metric: {self.monitor_metric}")
        print(f"   Min Delta: {self.min_delta}")
        
        print(f"\n🔧 Reproducibility Configuration:")
        print(f"   Seed: {self.seed}")
        print(f"   Deterministic: {self.deterministic}")
        print(f"   CUDNN Deterministic: {torch.backends.cudnn.deterministic}")
        print(f"   CUDNN Benchmark: {torch.backends.cudnn.benchmark}")
        
        print(f"\n📁 Output Directories:")
        print(f"   Models: {self.model_dir}")
        print(f"   Logs: {self.log_dir}")
        print(f"   Results: {self.results_dir}")
        
        print("=" * 80)
    
    def log_environment_info(self):
        """Log environment and version information"""
        import sys
        import platform
        
        env_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
            'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A',
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        print("🖥️ Environment Information:")
        for key, value in env_info.items():
            print(f"   {key}: {value}")
        
        # Save to file
        env_file = self.log_dir / "environment_info.txt"
        with open(env_file, 'w') as f:
            for key, value in env_info.items():
                f.write(f"{key}: {value}\n")
        
        return env_info


class EarlyStopping:
    """
    Early stopping utility class
    Monitors F1-score and stops training when no improvement
    """
    
    def __init__(self, patience=10, min_delta=0.001, monitor='f1_score', mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        if mode == 'max':
            self.is_better = lambda current, best: current > best + min_delta
        else:
            self.is_better = lambda current, best: current < best - min_delta
    
    def __call__(self, current_score):
        """
        Check if training should stop
        
        Args:
            current_score: Current epoch's monitored metric value
            
        Returns:
            bool: True if training should stop
        """
        if self.best_score is None:
            self.best_score = current_score
        elif self.is_better(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def reset(self):
        """Reset early stopping state"""
        self.best_score = None
        self.counter = 0
        self.early_stop = False


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
    
    return device


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filepath):
    """
    Save training checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        metrics: Training metrics
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'model': model  # Save entire model for easy loading
    }
    
    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """
    Load training checkpoint
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        
    Returns:
        dict: Loaded checkpoint data
    """
    device = get_device()
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"📂 Checkpoint loaded: {filepath}")
    return checkpoint


if __name__ == "__main__":
    # Test configuration
    print("🧪 Testing Training Configuration")
    
    # Test binary configuration
    binary_config = TrainingConfig(task='binary')
    binary_config.print_config()
    binary_config.log_environment_info()
    
    print("\n" + "="*50 + "\n")
    
    # Test multiclass configuration
    multiclass_config = TrainingConfig(task='multiclass')
    multiclass_config.print_config()
    
    # Test early stopping
    print("\n🧪 Testing Early Stopping")
    early_stopping = EarlyStopping(patience=3, min_delta=0.01)
    
    # Simulate training scores
    scores = [0.85, 0.87, 0.86, 0.85, 0.84, 0.83]  # Decreasing after peak
    
    for epoch, score in enumerate(scores):
        should_stop = early_stopping(score)
        print(f"Epoch {epoch+1}: F1={score:.3f}, Counter={early_stopping.counter}, Stop={should_stop}")
        if should_stop:
            print("🛑 Early stopping triggered!")
            break
    
    print("\n✅ Configuration tests completed!")