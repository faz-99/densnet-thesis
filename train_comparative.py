"""
Training script for DenseNet vs Swin Transformer V2 comparison
Minimal implementation focusing on core training and comparison
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import json
import os
from datetime import datetime

from preprocessing_pipeline import ModelFactory
from simple_dataset import MyDataset
import config_clean as config

class ModelTrainer:
    """Unified trainer for both models"""
    
    def __init__(self, model, model_name, device):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.lr)
        
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        self.val_losses.append(epoch_loss)
        self.val_accuracies.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, epochs):
        """Full training loop"""
        print(f"Training {self.model_name}...")
        start_time = time.time()
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(f'weight/{self.model_name}_best.pth')
            
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        training_time = time.time() - start_time
        
        return {
            'model_name': self.model_name,
            'best_val_acc': best_val_acc,
            'training_time': training_time,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
    
    def save_model(self, path):
        """Save model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

class ComparativeAnalysis:
    """Compare DenseNet vs Swin Transformer performance"""
    
    def __init__(self):
        self.results = {}
    
    def add_result(self, model_name, result):
        """Add training result"""
        self.results[model_name] = result
    
    def generate_comparison(self):
        """Generate comparative analysis"""
        if len(self.results) < 2:
            return "Need at least 2 models for comparison"
        
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'models': self.results,
            'comparison': {}
        }
        
        # Compare best accuracies
        accuracies = {name: result['best_val_acc'] for name, result in self.results.items()}
        best_model = max(accuracies, key=accuracies.get)
        
        comparison['comparison'] = {
            'best_model': best_model,
            'accuracy_comparison': accuracies,
            'training_time_comparison': {name: result['training_time'] for name, result in self.results.items()},
            'performance_summary': f"{best_model} achieved highest accuracy: {accuracies[best_model]:.2f}%"
        }
        
        return comparison
    
    def save_results(self, filename='comparative_analysis.json'):
        """Save comparison results"""
        comparison = self.generate_comparison()
        os.makedirs('results', exist_ok=True)
        
        with open(f'results/{filename}', 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"Comparative analysis saved to results/{filename}")
        return comparison

def main():
    """Main training and comparison function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create preprocessing
    preprocessing = ModelFactory.get_preprocessing()
    
    # Create datasets
    train_dataset = MyDataset(config.train, preprocessing.get_densenet_transform())
    val_dataset = MyDataset(config.valid, preprocessing.get_densenet_transform())
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize comparison
    analysis = ComparativeAnalysis()
    
    # Train DenseNet
    print("="*50)
    print("TRAINING DENSENET")
    print("="*50)
    
    densenet = ModelFactory.create_densenet()
    densenet_trainer = ModelTrainer(densenet, "DenseNet", device)
    densenet_result = densenet_trainer.train(train_loader, val_loader, config.max_epoch)
    analysis.add_result("DenseNet", densenet_result)
    
    # Train Swin Transformer (with Swin preprocessing)
    print("="*50)
    print("TRAINING SWIN TRANSFORMER V2")
    print("="*50)
    
    # Create Swin dataset with appropriate preprocessing
    train_dataset_swin = MyDataset(config.train, preprocessing.get_swin_transform())
    val_dataset_swin = MyDataset(config.valid, preprocessing.get_swin_transform())
    
    train_loader_swin = DataLoader(train_dataset_swin, batch_size=config.batch_size, shuffle=True)
    val_loader_swin = DataLoader(val_dataset_swin, batch_size=config.batch_size, shuffle=False)
    
    swin = ModelFactory.create_swin()
    swin_trainer = ModelTrainer(swin, "SwinTransformerV2", device)
    swin_result = swin_trainer.train(train_loader_swin, val_loader_swin, config.max_epoch)
    analysis.add_result("SwinTransformerV2", swin_result)
    
    # Generate and save comparison
    print("="*50)
    print("COMPARATIVE ANALYSIS")
    print("="*50)
    
    comparison = analysis.save_results()
    
    # Print summary
    print(f"Best Model: {comparison['comparison']['best_model']}")
    print("Accuracy Comparison:")
    for model, acc in comparison['comparison']['accuracy_comparison'].items():
        print(f"  {model}: {acc:.2f}%")
    
    print("Training Time Comparison:")
    for model, time_taken in comparison['comparison']['training_time_comparison'].items():
        print(f"  {model}: {time_taken:.2f} seconds")
    
    print(f"\n{comparison['comparison']['performance_summary']}")

if __name__ == "__main__":
    main()