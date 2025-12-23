#!/usr/bin/env python3
"""
DenLsNet Complete Demonstration Notebook
========================================

This notebook demonstrates the complete DenLsNet project including:
1. Model Architecture Overview
2. Data Loading and Preprocessing
3. Model Training (Binary and Multiclass)
4. Model Evaluation and Metrics
5. Comprehensive Explainability Analysis (Grad-CAM, SHAP, LIME, etc.)
6. Morphological Analysis
7. Clinical Report Generation

Author: DenLsNet Research Team
Date: 2024
Purpose: Master's Thesis Demonstration
"""

# ============================================================================
# SECTION 1: IMPORTS AND SETUP
# ============================================================================

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import json
from datetime import datetime
from pathlib import Path

# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timm

# Metrics and evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve

# Progress bars
from tqdm.auto import tqdm

print("🔬 DenLsNet Complete Demonstration")
print("=" * 50)
print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🐍 Python: {sys.version}")
print(f"🔥 PyTorch: {torch.__version__}")
print(f"💻 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
print("=" * 50)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

print("✅ Setup complete!")
# =
===========================================================================
# SECTION 2: MODEL ARCHITECTURE DEMONSTRATION
# ============================================================================

print("\n" + "="*60)
print("🏗️  SECTION 2: DenLsNet ARCHITECTURE")
print("="*60)

# Import our corrected model
try:
    from model.denlsnet_corrected import create_denlsnet, DenLsNet
    print("✅ Successfully imported DenLsNet architecture")
except ImportError as e:
    print(f"❌ Error importing DenLsNet: {e}")
    print("💡 Make sure you're running from the project root directory")

# Create and examine the model
print("\n📊 Creating DenLsNet Models...")

# Binary classification model
binary_model = create_denlsnet(num_classes=2, dropout_rate=0.5)
print("✅ Binary DenLsNet created (2 classes)")

# Multiclass classification model  
multiclass_model = create_denlsnet(num_classes=8, dropout_rate=0.5)
print("✅ Multiclass DenLsNet created (8 classes)")

# Display architecture summary
print("\n🔍 Binary Model Architecture Summary:")
binary_model.print_architecture_summary()

print("\n🔍 Multiclass Model Architecture Summary:")
multiclass_model.print_architecture_summary()

# Test forward pass
print("\n🧪 Testing Forward Pass...")
dummy_input = torch.randn(2, 3, 224, 224)

with torch.no_grad():
    binary_output = binary_model(dummy_input)
    multiclass_output = multiclass_model(dummy_input)

print(f"✅ Binary model output shape: {binary_output.shape}")
print(f"✅ Multiclass model output shape: {multiclass_output.shape}")

# Calculate model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

binary_params = count_parameters(binary_model)
multiclass_params = count_parameters(multiclass_model)

print(f"\n📊 Model Statistics:")
print(f"   Binary model parameters: {binary_params:,}")
print(f"   Multiclass model parameters: {multiclass_params:,}")
print(f"   Memory usage (approx): {(binary_params * 4) / (1024**2):.1f} MB")# 
============================================================================
# SECTION 3: DATA LOADING AND PREPROCESSING
# ============================================================================

print("\n" + "="*60)
print("📊 SECTION 3: DATA LOADING AND PREPROCESSING")
print("="*60)

# Create sample dataset for demonstration
def create_sample_dataset():
    """Create sample histopathology-like images for demonstration"""
    print("🔧 Creating sample dataset...")
    
    base_path = Path("sample_data")
    classes = ['benign', 'malignant']
    
    for split in ['train', 'test']:
        for class_name in classes:
            class_dir = base_path / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Create sample images if they don't exist
            num_images = 20 if split == 'train' else 5
            for i in range(num_images):
                img_path = class_dir / f"{class_name}_{i:03d}.png"
                if not img_path.exists():
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
                    
                    img.save(img_path)
    
    print(f"✅ Sample dataset created at {base_path}")
    return base_path

# Create sample data
dataset_path = create_sample_dataset()

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
from torchvision.datasets import ImageFolder

train_dataset = ImageFolder(dataset_path / "train", transform=train_transform)
test_dataset = ImageFolder(dataset_path / "test", transform=test_transform)

print(f"\n📊 Dataset Statistics:")
print(f"   Training samples: {len(train_dataset)}")
print(f"   Test samples: {len(test_dataset)}")
print(f"   Classes: {train_dataset.classes}")
print(f"   Class to index: {train_dataset.class_to_idx}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

print(f"   Training batches: {len(train_loader)}")
print(f"   Test batches: {len(test_loader)}")

# Visualize sample data
print("\n🖼️  Sample Data Visualization:")

def show_sample_batch(dataloader, title="Sample Batch"):
    """Display a batch of images"""
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    
    # Denormalize for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images_denorm = images * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)
    
    # Create subplot
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle(title, fontsize=16)
    
    for i in range(min(8, len(images))):
        row, col = i // 4, i % 4
        img = images_denorm[i].permute(1, 2, 0)
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"{train_dataset.classes[labels[i]]}")
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return images, labels

# Show sample training batch
sample_images, sample_labels = show_sample_batch(train_loader, "Training Samples")
print(f"✅ Displayed sample batch with shape: {sample_images.shape}")# ==
==========================================================================
# SECTION 4: MODEL TRAINING DEMONSTRATION
# ============================================================================

print("\n" + "="*60)
print("🚀 SECTION 4: MODEL TRAINING DEMONSTRATION")
print("="*60)

def train_model_demo(model, train_loader, test_loader, num_epochs=5, model_name="DenLsNet"):
    """Demonstrate model training with proper metrics tracking"""
    
    print(f"\n🎯 Training {model_name} for {num_epochs} epochs...")
    
    # Move model to device
    model = model.to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'learning_rates': []
    }
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_pbar = tqdm(train_loader, desc="Training", leave=False)
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct_train/total_train:.2f}%'
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct_train / total_train
        
        # Testing phase
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc="Testing", leave=False)
            for images, labels in test_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                
                test_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct_test/total_test:.2f}%'
                })
        
        test_loss = test_loss / len(test_loader)
        test_acc = 100. * correct_test / total_test
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['learning_rates'].append(current_lr)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            print(f"🏆 New best accuracy: {best_acc:.2f}%")
    
    print(f"\n✅ Training completed!")
    print(f"🏆 Best test accuracy: {best_acc:.2f}%")
    
    return model, history, best_acc

# Train binary model (demo with few epochs)
print("🔬 Training Binary Classification Model...")
trained_binary_model, binary_history, binary_best_acc = train_model_demo(
    binary_model, train_loader, test_loader, num_epochs=3, model_name="Binary DenLsNet"
)

# Plot training history
def plot_training_history(history, title="Training History"):
    """Plot training and validation metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=16)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['test_loss'], 'r-', label='Test Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['test_acc'], 'r-', label='Test Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Learning Rate
    ax3.plot(epochs, history['learning_rates'], 'g-')
    ax3.set_title('Learning Rate')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.grid(True)
    
    # Combined metrics
    ax4.plot(epochs, history['train_acc'], 'b-', label='Train Acc', alpha=0.7)
    ax4.plot(epochs, history['test_acc'], 'r-', label='Test Acc', alpha=0.7)
    ax4.fill_between(epochs, history['train_acc'], alpha=0.3, color='blue')
    ax4.fill_between(epochs, history['test_acc'], alpha=0.3, color='red')
    ax4.set_title('Accuracy Comparison')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

# Plot binary model training history
plot_training_history(binary_history, "Binary DenLsNet Training History")

print(f"\n📊 Binary Model Final Results:")
print(f"   Best Test Accuracy: {binary_best_acc:.2f}%")
print(f"   Final Train Loss: {binary_history['train_loss'][-1]:.4f}")
print(f"   Final Test Loss: {binary_history['test_loss'][-1]:.4f}")# ====
========================================================================
# SECTION 5: MODEL EVALUATION AND METRICS
# ============================================================================

print("\n" + "="*60)
print("📈 SECTION 5: COMPREHENSIVE MODEL EVALUATION")
print("="*60)

def comprehensive_evaluation(model, test_loader, class_names):
    """Perform comprehensive model evaluation"""
    
    print("🔍 Performing comprehensive evaluation...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            
            all_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')
    
    # Per-class metrics
    precision_per_class = precision_score(all_labels, all_predictions, average=None)
    recall_per_class = recall_score(all_labels, all_predictions, average=None)
    f1_per_class = f1_score(all_labels, all_predictions, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Classification report
    report = classification_report(all_labels, all_predictions, target_names=class_names)
    
    print(f"\n📊 Overall Metrics:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision (macro): {precision:.4f}")
    print(f"   Recall (macro): {recall:.4f}")
    print(f"   F1-Score (macro): {f1:.4f}")
    
    print(f"\n📊 Per-Class Metrics:")
    for i, class_name in enumerate(class_names):
        print(f"   {class_name}:")
        print(f"     Precision: {precision_per_class[i]:.4f}")
        print(f"     Recall: {recall_per_class[i]:.4f}")
        print(f"     F1-Score: {f1_per_class[i]:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # ROC curve for binary classification
    if len(class_names) == 2:
        fpr, tpr, _ = roc_curve(all_labels, all_probabilities[:, 1])
        roc_auc = roc_auc_score(all_labels, all_probabilities[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
        
        print(f"   ROC AUC: {roc_auc:.4f}")
    
    # Probability distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    for i, class_name in enumerate(class_names):
        class_probs = all_probabilities[all_labels == i, i]
        plt.hist(class_probs, bins=20, alpha=0.7, label=f'{class_name} (True)', density=True)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Probability Distribution for True Classes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    max_probs = np.max(all_probabilities, axis=1)
    correct_mask = all_predictions == all_labels
    plt.hist(max_probs[correct_mask], bins=20, alpha=0.7, label='Correct', density=True)
    plt.hist(max_probs[~correct_mask], bins=20, alpha=0.7, label='Incorrect', density=True)
    plt.xlabel('Maximum Predicted Probability')
    plt.ylabel('Density')
    plt.title('Confidence Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'classification_report': report
    }

# Evaluate binary model
class_names = ['Benign', 'Malignant']
binary_results = comprehensive_evaluation(trained_binary_model, test_loader, class_names)

print(f"\n✅ Evaluation completed!")
print(f"📊 Binary Model Performance Summary:")
print(f"   Accuracy: {binary_results['accuracy']:.1%}")
print(f"   F1-Score: {binary_results['f1_score']:.4f}")
print(f"   Precision: {binary_results['precision']:.4f}")
print(f"   Recall: {binary_results['recall']:.4f}")#
 ============================================================================
# SECTION 6: EXPLAINABILITY ANALYSIS (GRAD-CAM, SHAP, LIME, etc.)
# ============================================================================

print("\n" + "="*60)
print("🧠 SECTION 6: COMPREHENSIVE EXPLAINABILITY ANALYSIS")
print("="*60)

# Import explainability modules
try:
    from explainability.grad_cam import GradCAM, GradCAMPlusPlus, overlay_heatmap
    print("✅ Grad-CAM modules imported")
except ImportError as e:
    print(f"⚠️ Grad-CAM import failed: {e}")

try:
    import shap
    print("✅ SHAP imported")
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️ SHAP not available")
    SHAP_AVAILABLE = False

try:
    from explainability.lime_explainer import LIMEExplainer
    print("✅ LIME imported")
    LIME_AVAILABLE = True
except ImportError:
    print("⚠️ LIME not available")
    LIME_AVAILABLE = False

# Initialize explainability methods
def initialize_explainers(model, device):
    """Initialize all available explainability methods"""
    
    print("\n🔧 Initializing explainability methods...")
    explainers = {}
    
    # Get model layer names for debugging
    layer_names = [name for name, _ in model.named_modules()]
    print(f"📋 Available layers: {len(layer_names)} total")
    
    # Find suitable target layer for Grad-CAM
    target_layers_to_try = [
        'densenet.features.norm5',
        'features.norm5', 
        'features.denseblock4',
        'norm5'
    ]
    
    target_layer = None
    for layer_name in target_layers_to_try:
        if layer_name in layer_names:
            target_layer = layer_name
            break
    
    if target_layer is None:
        # Fallback: use the last convolutional layer
        conv_layers = [name for name in layer_names if 'conv' in name.lower()]
        if conv_layers:
            target_layer = conv_layers[-1]
    
    print(f"🎯 Using target layer: {target_layer}")
    
    # Initialize Grad-CAM
    try:
        explainers['gradcam'] = GradCAM(model, target_layer_name=target_layer)
        print("✅ Grad-CAM initialized")
    except Exception as e:
        print(f"❌ Grad-CAM failed: {e}")
    
    # Initialize Grad-CAM++
    try:
        explainers['gradcam_plus'] = GradCAMPlusPlus(model, target_layer_name=target_layer)
        print("✅ Grad-CAM++ initialized")
    except Exception as e:
        print(f"❌ Grad-CAM++ failed: {e}")
    
    # Initialize SHAP
    if SHAP_AVAILABLE:
        try:
            # Create model wrapper for SHAP
            class SHAPModelWrapper(nn.Module):
                def __init__(self, model, device):
                    super().__init__()
                    self.model = model
                    self.device = device
                
                def forward(self, x):
                    if not isinstance(x, torch.Tensor):
                        x = torch.tensor(x, dtype=torch.float32, device=self.device)
                    else:
                        x = x.to(self.device).float()
                    outputs = self.model(x)
                    return F.softmax(outputs, dim=1)
            
            wrapped_model = SHAPModelWrapper(model, device)
            wrapped_model.eval()
            
            # Create background data
            background_data = torch.randn(3, 3, 224, 224).to(device).float()
            
            try:
                explainers['shap'] = shap.DeepExplainer(wrapped_model, background_data)
                print("✅ SHAP DeepExplainer initialized")
            except:
                explainers['shap'] = shap.GradientExplainer(wrapped_model, background_data)
                print("✅ SHAP GradientExplainer initialized")
                
        except Exception as e:
            print(f"❌ SHAP failed: {e}")
    
    # Initialize LIME
    if LIME_AVAILABLE:
        try:
            explainers['lime'] = LIMEExplainer(model, str(device), num_samples=50)
            print("✅ LIME initialized")
        except Exception as e:
            print(f"❌ LIME failed: {e}")
    
    return explainers

# Initialize explainers
explainers = initialize_explainers(trained_binary_model, device)

def generate_explanations(model, image_tensor, explainers, target_class=None):
    """Generate explanations using all available methods"""
    
    print(f"\n🔍 Generating explanations...")
    
    model.eval()
    explanations = {}
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    if target_class is None:
        target_class = predicted_class
    
    print(f"🎯 Target class: {target_class}, Confidence: {confidence:.3f}")
    
    # Grad-CAM
    if 'gradcam' in explainers:
        try:
            gradcam_map = explainers['gradcam'].generate_cam(image_tensor, target_class)
            explanations['gradcam'] = gradcam_map
            print("✅ Grad-CAM explanation generated")
        except Exception as e:
            print(f"❌ Grad-CAM failed: {e}")
    
    # Grad-CAM++
    if 'gradcam_plus' in explainers:
        try:
            gradcam_plus_map = explainers['gradcam_plus'].generate_cam(image_tensor, target_class)
            explanations['gradcam_plus'] = gradcam_plus_map
            print("✅ Grad-CAM++ explanation generated")
        except Exception as e:
            print(f"❌ Grad-CAM++ failed: {e}")
    
    # SHAP
    if 'shap' in explainers:
        try:
            shap_values = explainers['shap'].shap_values(image_tensor.cpu().numpy())
            if isinstance(shap_values, list):
                shap_map = shap_values[target_class][0] if len(shap_values) > target_class else shap_values[0][0]
            else:
                shap_map = shap_values[0]
            
            # Convert to single channel if needed
            if len(shap_map.shape) == 3:
                shap_map = np.sum(np.abs(shap_map), axis=0)
            
            # Normalize
            if shap_map.max() > shap_map.min():
                shap_map = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min())
            
            explanations['shap'] = shap_map
            print("✅ SHAP explanation generated")
        except Exception as e:
            print(f"❌ SHAP failed: {e}")
    
    # LIME
    if 'lime' in explainers:
        try:
            # Convert tensor to numpy for LIME
            image_np = image_tensor[0].cpu().numpy().transpose(1, 2, 0)
            
            # Denormalize for LIME
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = image_np * std + mean
            image_np = np.clip(image_np, 0, 1)
            
            explanation, segments = explainers['lime'].explain_image(image_np)
            temp, mask = explanation.get_image_and_mask(
                target_class, positive_only=False, num_features=10, hide_rest=False
            )
            
            lime_map = np.abs(mask).astype(float)
            lime_map = (lime_map - lime_map.min()) / (lime_map.max() - lime_map.min() + 1e-8)
            explanations['lime'] = lime_map
            print("✅ LIME explanation generated")
        except Exception as e:
            print(f"❌ LIME failed: {e}")
    
    return explanations, predicted_class, confidence

def visualize_explanations(original_image, explanations, predicted_class, confidence, class_names):
    """Visualize all explanations in a comprehensive plot"""
    
    num_methods = len(explanations)
    if num_methods == 0:
        print("❌ No explanations to visualize")
        return
    
    # Create subplot grid
    cols = min(4, num_methods + 1)  # +1 for original image
    rows = (num_methods + 1 + cols - 1) // cols  # Ceiling division
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Original image
    axes_flat[0].imshow(original_image)
    axes_flat[0].set_title(f'Original Image\n{class_names[predicted_class]} ({confidence:.1%})', 
                          fontweight='bold')
    axes_flat[0].axis('off')
    
    # Explanations
    idx = 1
    for method_name, explanation_map in explanations.items():
        if idx < len(axes_flat):
            # Heatmap
            im = axes_flat[idx].imshow(explanation_map, cmap='jet')
            axes_flat[idx].set_title(f'{method_name.upper()}\nHeatmap', fontweight='bold')
            axes_flat[idx].axis('off')
            plt.colorbar(im, ax=axes_flat[idx], fraction=0.046, pad=0.04)
            idx += 1
    
    # Hide unused subplots
    for i in range(idx, len(axes_flat)):
        axes_flat[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Create overlay visualizations
    if explanations:
        fig, axes = plt.subplots(1, len(explanations), figsize=(5*len(explanations), 5))
        if len(explanations) == 1:
            axes = [axes]
        
        for idx, (method_name, explanation_map) in enumerate(explanations.items()):
            overlay = overlay_heatmap(original_image, explanation_map, alpha=0.4)
            axes[idx].imshow(overlay)
            axes[idx].set_title(f'{method_name.upper()} Overlay', fontweight='bold')
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()

# Test explainability on sample images
print("\n🧪 Testing Explainability Methods...")

# Get a sample image from test set
sample_images, sample_labels = next(iter(test_loader))
sample_image = sample_images[0:1].to(device)  # Take first image
sample_label = sample_labels[0].item()

# Convert tensor back to displayable format
def tensor_to_image(tensor):
    """Convert normalized tensor to displayable image"""
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    denorm = tensor * std + mean
    denorm = torch.clamp(denorm, 0, 1)
    
    # Convert to numpy
    img_np = denorm[0].cpu().numpy().transpose(1, 2, 0)
    return img_np

original_img = tensor_to_image(sample_image)

print(f"📸 Analyzing sample image (True label: {class_names[sample_label]})")

# Generate explanations
explanations, pred_class, confidence = generate_explanations(
    trained_binary_model, sample_image, explainers
)

print(f"\n📊 Explanation Results:")
print(f"   Predicted: {class_names[pred_class]} (confidence: {confidence:.1%})")
print(f"   True label: {class_names[sample_label]}")
print(f"   Generated {len(explanations)} explanations")

# Visualize explanations
visualize_explanations(original_img, explanations, pred_class, confidence, class_names)# ========
====================================================================
# SECTION 7: MORPHOLOGICAL ANALYSIS AND CLINICAL INTERPRETATION
# ============================================================================

print("\n" + "="*60)
print("🔬 SECTION 7: MORPHOLOGICAL ANALYSIS & CLINICAL INTERPRETATION")
print("="*60)

# Import morphological analysis modules
try:
    from explainability.morphological_analyzer import MorphologicalAnalyzer, ClinicalDescriptorGenerator
    print("✅ Morphological analysis modules imported")
    MORPHOLOGICAL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Morphological analysis not available: {e}")
    MORPHOLOGICAL_AVAILABLE = False

def perform_morphological_analysis(original_image, explanations, predicted_class, confidence, class_names):
    """Perform comprehensive morphological analysis"""
    
    if not MORPHOLOGICAL_AVAILABLE:
        print("❌ Morphological analysis not available")
        return None
    
    print("\n🔍 Performing morphological analysis...")
    
    # Initialize analyzers
    morphological_analyzer = MorphologicalAnalyzer()
    descriptor_generator = ClinicalDescriptorGenerator()
    
    results = {}
    
    for method_name, activation_map in explanations.items():
        print(f"\n📊 Analyzing {method_name.upper()} activation map...")
        
        try:
            # Extract morphological features
            features = morphological_analyzer.analyze_activation_map(
                original_image, activation_map
            )
            
            # Generate clinical description
            description = descriptor_generator.generate_description(
                features, class_names[predicted_class], confidence
            )
            
            # Generate detailed report
            detailed_report = descriptor_generator.generate_detailed_report(
                features, class_names[predicted_class], confidence, f"sample_{method_name}"
            )
            
            results[method_name] = {
                'features': features,
                'description': description,
                'detailed_report': detailed_report
            }
            
            print(f"✅ {method_name.upper()} analysis completed")
            
        except Exception as e:
            print(f"❌ {method_name.upper()} analysis failed: {e}")
    
    return results

def display_morphological_results(morphological_results, class_names):
    """Display morphological analysis results"""
    
    if not morphological_results:
        print("❌ No morphological results to display")
        return
    
    print(f"\n📋 MORPHOLOGICAL ANALYSIS SUMMARY")
    print("=" * 50)
    
    for method_name, results in morphological_results.items():
        print(f"\n🔬 {method_name.upper()} Analysis:")
        print("-" * 30)
        
        features = results['features']
        description = results['description']
        
        # Display key features
        print(f"📊 Quantitative Features:")
        print(f"   Tissue Area Highlighted: {features['tissue_area_percent']:.1f}%")
        print(f"   Dominant Stain: {features['stain_analysis']['dominant_stain'].title()}")
        print(f"   Cellular Entropy: {features['texture_features']['entropy']:.2f}")
        print(f"   Edge Density: {features['texture_features']['edge_density']:.3f}")
        print(f"   Number of Regions: {features['morphological_features']['num_regions']}")
        
        print(f"\n📝 Clinical Description:")
        print(f"   {description}")
        
        # Display detailed findings
        detailed_report = results['detailed_report']
        if detailed_report['key_findings']:
            print(f"\n🔍 Key Findings:")
            for finding in detailed_report['key_findings']:
                print(f"   • {finding}")
    
    # Create comparison table
    print(f"\n📊 COMPARATIVE ANALYSIS TABLE")
    print("=" * 80)
    
    # Prepare data for comparison
    comparison_data = []
    for method_name, results in morphological_results.items():
        features = results['features']
        comparison_data.append({
            'Method': method_name.upper(),
            'Tissue Area (%)': f"{features['tissue_area_percent']:.1f}",
            'Dominant Stain': features['stain_analysis']['dominant_stain'].title(),
            'Entropy': f"{features['texture_features']['entropy']:.2f}",
            'Edge Density': f"{features['texture_features']['edge_density']:.3f}",
            'Regions': features['morphological_features']['num_regions']
        })
    
    # Create DataFrame for better display
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    # Visualize feature comparison
    if len(morphological_results) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Morphological Features Comparison', fontsize=16)
        
        methods = list(morphological_results.keys())
        
        # Tissue area comparison
        tissue_areas = [morphological_results[m]['features']['tissue_area_percent'] for m in methods]
        axes[0, 0].bar(methods, tissue_areas, color='skyblue')
        axes[0, 0].set_title('Tissue Area Highlighted (%)')
        axes[0, 0].set_ylabel('Percentage')
        
        # Entropy comparison
        entropies = [morphological_results[m]['features']['texture_features']['entropy'] for m in methods]
        axes[0, 1].bar(methods, entropies, color='lightcoral')
        axes[0, 1].set_title('Cellular Entropy')
        axes[0, 1].set_ylabel('Entropy')
        
        # Edge density comparison
        edge_densities = [morphological_results[m]['features']['texture_features']['edge_density'] for m in methods]
        axes[1, 0].bar(methods, edge_densities, color='lightgreen')
        axes[1, 0].set_title('Edge Density')
        axes[1, 0].set_ylabel('Density')
        
        # Number of regions comparison
        num_regions = [morphological_results[m]['features']['morphological_features']['num_regions'] for m in methods]
        axes[1, 1].bar(methods, num_regions, color='gold')
        axes[1, 1].set_title('Number of Regions')
        axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()

# Perform morphological analysis on the explanations
if explanations and MORPHOLOGICAL_AVAILABLE:
    morphological_results = perform_morphological_analysis(
        original_img, explanations, pred_class, confidence, class_names
    )
    
    if morphological_results:
        display_morphological_results(morphological_results, class_names)
    else:
        print("❌ Morphological analysis failed")
else:
    print("⚠️ Skipping morphological analysis (no explanations or module not available)")

# ============================================================================
# SECTION 8: COMPREHENSIVE CLINICAL REPORT GENERATION
# ============================================================================

print("\n" + "="*60)
print("📋 SECTION 8: COMPREHENSIVE CLINICAL REPORT GENERATION")
print("="*60)

def generate_comprehensive_report(original_image, explanations, morphological_results, 
                                predicted_class, confidence, true_class, class_names):
    """Generate a comprehensive clinical report"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = {
        'metadata': {
            'analysis_timestamp': timestamp,
            'model_type': 'DenLsNet (DenseNet-121 + Bidirectional LSTM)',
            'predicted_class': class_names[predicted_class],
            'true_class': class_names[true_class],
            'confidence': confidence,
            'methods_used': list(explanations.keys()) if explanations else []
        },
        'prediction_analysis': {
            'prediction_correct': predicted_class == true_class,
            'confidence_level': 'High' if confidence >= 0.8 else 'Medium' if confidence >= 0.6 else 'Low',
            'clinical_significance': 'Reliable' if confidence >= 0.8 else 'Moderate' if confidence >= 0.6 else 'Low confidence'
        },
        'explainability_summary': {},
        'morphological_analysis': morphological_results,
        'clinical_interpretation': '',
        'recommendations': []
    }
    
    # Explainability summary
    if explanations:
        for method_name, explanation_map in explanations.items():
            activation_stats = {
                'mean_activation': float(np.mean(explanation_map)),
                'max_activation': float(np.max(explanation_map)),
                'activation_coverage': float(np.sum(explanation_map > 0.5) / explanation_map.size)
            }
            report['explainability_summary'][method_name] = activation_stats
    
    # Generate clinical interpretation
    interpretation_parts = []
    
    # Confidence assessment
    if confidence >= 0.9:
        interpretation_parts.append(f"High confidence {class_names[predicted_class].lower()} classification")
    elif confidence >= 0.7:
        interpretation_parts.append(f"Moderate confidence {class_names[predicted_class].lower()} classification")
    else:
        interpretation_parts.append(f"Low confidence {class_names[predicted_class].lower()} classification")
    
    # Morphological insights
    if morphological_results:
        # Get consensus from multiple methods
        tissue_areas = []
        dominant_stains = []
        
        for method_results in morphological_results.values():
            features = method_results['features']
            tissue_areas.append(features['tissue_area_percent'])
            dominant_stains.append(features['stain_analysis']['dominant_stain'])
        
        if tissue_areas:
            mean_tissue_area = np.mean(tissue_areas)
            if mean_tissue_area > 40:
                interpretation_parts.append("with extensive model attention across tissue regions")
            elif mean_tissue_area > 20:
                interpretation_parts.append("with moderate model attention to specific tissue areas")
            else:
                interpretation_parts.append("with focal model attention to limited tissue regions")
        
        # Stain analysis
        if dominant_stains:
            most_common_stain = max(set(dominant_stains), key=dominant_stains.count)
            if most_common_stain == 'hematoxylin':
                interpretation_parts.append("Consistent nuclear staining patterns suggest cellular density changes")
            elif most_common_stain == 'eosin':
                interpretation_parts.append("Consistent cytoplasmic staining patterns indicate structural alterations")
    
    # Methods agreement
    if len(explanations) > 1:
        interpretation_parts.append(f"Findings supported by {len(explanations)} independent analysis methods")
    
    report['clinical_interpretation'] = ". ".join(interpretation_parts) + "."
    
    # Generate recommendations
    recommendations = []
    
    if predicted_class != true_class:
        recommendations.append("⚠️ Model prediction differs from ground truth - requires clinical review")
    
    if confidence < 0.7:
        recommendations.append("⚠️ Low confidence prediction - consider additional diagnostic methods")
    
    if morphological_results:
        recommendations.append("✅ Morphological analysis available for detailed tissue characterization")
    
    recommendations.append("📋 Correlate AI analysis with clinical history and additional diagnostic tests")
    recommendations.append("👨‍⚕️ Final diagnosis should always be made by qualified pathologists")
    
    report['recommendations'] = recommendations
    
    return report

def display_clinical_report(report):
    """Display the comprehensive clinical report"""
    
    print(f"\n📋 COMPREHENSIVE CLINICAL ANALYSIS REPORT")
    print("=" * 80)
    
    # Metadata
    metadata = report['metadata']
    print(f"📅 Analysis Date: {metadata['analysis_timestamp']}")
    print(f"🤖 Model: {metadata['model_type']}")
    print(f"🎯 Prediction: {metadata['predicted_class']} (Confidence: {metadata['confidence']:.1%})")
    print(f"✅ Ground Truth: {metadata['true_class']}")
    print(f"🔬 Methods Used: {', '.join(metadata['methods_used'])}")
    
    # Prediction Analysis
    pred_analysis = report['prediction_analysis']
    print(f"\n📊 PREDICTION ANALYSIS")
    print("-" * 30)
    print(f"Prediction Accuracy: {'✅ Correct' if pred_analysis['prediction_correct'] else '❌ Incorrect'}")
    print(f"Confidence Level: {pred_analysis['confidence_level']}")
    print(f"Clinical Significance: {pred_analysis['clinical_significance']}")
    
    # Explainability Summary
    if report['explainability_summary']:
        print(f"\n🧠 EXPLAINABILITY SUMMARY")
        print("-" * 30)
        for method, stats in report['explainability_summary'].items():
            print(f"{method.upper()}:")
            print(f"  Mean Activation: {stats['mean_activation']:.3f}")
            print(f"  Max Activation: {stats['max_activation']:.3f}")
            print(f"  Coverage: {stats['activation_coverage']:.1%}")
    
    # Clinical Interpretation
    print(f"\n🏥 CLINICAL INTERPRETATION")
    print("-" * 30)
    print(report['clinical_interpretation'])
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 30)
    for rec in report['recommendations']:
        print(f"• {rec}")
    
    print("\n" + "=" * 80)
    print("📋 End of Clinical Report")
    print("=" * 80)

# Generate comprehensive report
if explanations:
    comprehensive_report = generate_comprehensive_report(
        original_img, explanations, morphological_results if MORPHOLOGICAL_AVAILABLE else None,
        pred_class, confidence, sample_label, class_names
    )
    
    display_clinical_report(comprehensive_report)
    
    # Save report to JSON
    report_filename = f"clinical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Make report JSON serializable
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_report = make_serializable(comprehensive_report)
    
    with open(report_filename, 'w') as f:
        json.dump(serializable_report, f, indent=2)
    
    print(f"\n💾 Report saved to: {report_filename}")
else:
    print("⚠️ No explanations available for report generation")

print(f"\n🎉 DEMONSTRATION COMPLETED!")
print("=" * 60)
print("📊 Summary of what was demonstrated:")
print("✅ DenLsNet architecture overview and testing")
print("✅ Data loading and preprocessing")
print("✅ Model training with progress tracking")
print("✅ Comprehensive evaluation with metrics")
print("✅ Multi-method explainability analysis")
if MORPHOLOGICAL_AVAILABLE:
    print("✅ Morphological analysis and clinical interpretation")
print("✅ Comprehensive clinical report generation")
print("\n🎓 This notebook demonstrates the complete DenLsNet pipeline")
print("   suitable for master's thesis presentation and evaluation.")
print("=" * 60)