"""
Enhanced SHAP (SHapley Additive exPlanations) implementation for DenLsNet explainability
Includes comprehensive analysis, per-class summaries, and quantitative evaluation
"""
import torch
import torch.nn.functional as F
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from typing import List, Tuple, Optional, Dict
import cv2
from datetime import datetime
from collections import defaultdict
import pandas as pd


class SHAPExplainer:
    """SHAP explainer for image classification models"""
    
    def __init__(self, model, background_data: torch.Tensor, device: str = 'cpu'):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained model
            background_data: Background dataset for SHAP (typically a subset of training data)
            device: Computing device
        """
        self.model = model
        self.device = device
        self.background_data = background_data.to(device)
        
        # Wrap model for SHAP
        def model_wrapper(x):
            x = torch.tensor(x, dtype=torch.float32).to(device)
            with torch.no_grad():
                outputs = model(x)
                return F.softmax(outputs, dim=1).cpu().numpy()
        
        self.model_wrapper = model_wrapper
        
        # Initialize SHAP explainer
        self.explainer = shap.DeepExplainer(model_wrapper, background_data)
    
    def explain_image(self, image: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate SHAP explanation for a single image
        
        Args:
            image: Input image tensor (1, C, H, W)
            class_idx: Target class index (if None, uses predicted class)
            
        Returns:
            SHAP values as numpy array
        """
        image = image.to(self.device)
        
        # Get prediction if class_idx not specified
        if class_idx is None:
            with torch.no_grad():
                output = self.model(image)
                class_idx = output.argmax(dim=1).item()
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(image.cpu().numpy())
        
        # Return SHAP values for the specified class
        return shap_values[class_idx][0]  # [0] to get first (and only) sample
    
    def explain_batch(self, images: torch.Tensor, class_indices: Optional[List[int]] = None) -> List[np.ndarray]:
        """
        Generate SHAP explanations for a batch of images
        
        Args:
            images: Batch of input images (B, C, H, W)
            class_indices: Target class indices for each image
            
        Returns:
            List of SHAP values for each image
        """
        images = images.to(self.device)
        batch_size = images.size(0)
        
        # Get predictions if class_indices not specified
        if class_indices is None:
            with torch.no_grad():
                outputs = self.model(images)
                class_indices = outputs.argmax(dim=1).cpu().tolist()
        
        # Calculate SHAP values for the batch
        shap_values = self.explainer.shap_values(images.cpu().numpy())
        
        # Extract SHAP values for each image and its corresponding class
        explanations = []
        for i in range(batch_size):
            class_idx = class_indices[i]
            explanations.append(shap_values[class_idx][i])
        
        return explanations


def visualize_shap_explanation(
    original_image: np.ndarray,
    shap_values: np.ndarray,
    predicted_class: str,
    true_class: str,
    confidence: float,
    save_path: str
):
    """
    Visualize SHAP explanation
    
    Args:
        original_image: Original input image (H, W, C)
        shap_values: SHAP values (C, H, W)
        predicted_class: Model prediction
        true_class: Ground truth label
        confidence: Prediction confidence
        save_path: Path to save visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # SHAP values for each channel
    for i, channel_name in enumerate(['Red', 'Green', 'Blue']):
        if i < shap_values.shape[0]:
            shap_channel = shap_values[i]
            
            # Normalize SHAP values for visualization
            shap_normalized = (shap_channel - shap_channel.min()) / (shap_channel.max() - shap_channel.min() + 1e-8)
            
            row = i // 2
            col = (i % 2) + 1
            
            im = axes[row, col].imshow(shap_normalized, cmap='RdBu_r', vmin=0, vmax=1)
            axes[row, col].set_title(f'SHAP - {channel_name} Channel')
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    
    # Combined SHAP visualization (sum across channels)
    shap_combined = np.sum(np.abs(shap_values), axis=0)
    shap_combined = (shap_combined - shap_combined.min()) / (shap_combined.max() - shap_combined.min() + 1e-8)
    
    axes[1, 0].imshow(shap_combined, cmap='hot')
    axes[1, 0].set_title('Combined SHAP Importance')
    axes[1, 0].axis('off')
    
    # Prediction info
    axes[1, 2].text(0.1, 0.8, f'Predicted: {predicted_class}', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.6, f'True Label: {true_class}', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.4, f'Confidence: {confidence:.3f}', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.2, f'Correct: {"✓" if predicted_class == true_class else "✗"}', 
                   fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_shap_summary_plot(
    shap_values_list: List[np.ndarray],
    images_list: List[np.ndarray],
    class_names: List[str],
    predictions: List[int],
    save_path: str
):
    """
    Create summary plot showing SHAP explanations for multiple samples
    
    Args:
        shap_values_list: List of SHAP values for each sample
        images_list: List of original images
        class_names: List of class names
        predictions: List of predicted class indices
        save_path: Path to save the summary plot
    """
    num_samples = len(shap_values_list)
    fig, axes = plt.subplots(3, num_samples, figsize=(4*num_samples, 12))
    
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_samples):
        # Original image
        axes[0, i].imshow(images_list[i])
        axes[0, i].set_title(f'Sample {i+1}\nPred: {class_names[predictions[i]]}')
        axes[0, i].axis('off')
        
        # SHAP heatmap (combined channels)
        shap_combined = np.sum(np.abs(shap_values_list[i]), axis=0)
        shap_combined = (shap_combined - shap_combined.min()) / (shap_combined.max() - shap_combined.min() + 1e-8)
        
        axes[1, i].imshow(shap_combined, cmap='hot')
        axes[1, i].set_title('SHAP Importance')
        axes[1, i].axis('off')
        
        # Overlay SHAP on original image
        # Resize SHAP heatmap to match image size
        shap_resized = cv2.resize(shap_combined, (images_list[i].shape[1], images_list[i].shape[0]))
        
        # Create overlay
        overlay = 0.6 * images_list[i] + 0.4 * plt.cm.hot(shap_resized)[:, :, :3]
        axes[2, i].imshow(overlay)
        axes[2, i].set_title('SHAP Overlay')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_per_class_shap_summary(
    shap_values_by_class: Dict[str, List[np.ndarray]],
    class_names: List[str],
    save_dir: str
):
    """
    Create per-class SHAP summary showing average feature importance
    
    Args:
        shap_values_by_class: Dictionary with class names as keys and SHAP values as values
        class_names: List of class names
        save_dir: Directory to save results
    """
    # Calculate average SHAP importance per class
    class_importance = {}
    
    for class_name in class_names:
        if class_name in shap_values_by_class and shap_values_by_class[class_name]:
            # Average across all samples for this class
            class_shap_values = shap_values_by_class[class_name]
            
            # Combine all SHAP values for this class
            combined_shap = np.stack(class_shap_values, axis=0)  # (N, C, H, W)
            
            # Calculate mean absolute SHAP values
            mean_abs_shap = np.mean(np.abs(combined_shap), axis=0)  # (C, H, W)
            
            # Sum across spatial dimensions to get per-channel importance
            channel_importance = np.sum(mean_abs_shap, axis=(1, 2))  # (C,)
            
            # Overall importance (sum across all channels and spatial dimensions)
            overall_importance = np.sum(mean_abs_shap)
            
            class_importance[class_name] = {
                'overall': float(overall_importance),
                'per_channel': channel_importance.tolist(),
                'spatial_map': mean_abs_shap
            }
    
    # Create bar plot of overall importance per class
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Overall importance bar plot
    classes = list(class_importance.keys())
    overall_scores = [class_importance[cls]['overall'] for cls in classes]
    
    bars = ax1.bar(classes, overall_scores, alpha=0.7, color='skyblue')
    ax1.set_title('Average SHAP Importance by Class')
    ax1.set_ylabel('Average SHAP Importance')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars, overall_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(overall_scores)*0.01,
                f'{score:.2f}', ha='center', va='bottom')
    
    # Per-channel importance heatmap
    if classes:
        channel_data = np.array([class_importance[cls]['per_channel'] for cls in classes])
        
        im = ax2.imshow(channel_data, cmap='viridis', aspect='auto')
        ax2.set_title('Per-Channel SHAP Importance')
        ax2.set_xlabel('Channel (R, G, B)')
        ax2.set_ylabel('Class')
        ax2.set_yticks(range(len(classes)))
        ax2.set_yticklabels(classes)
        ax2.set_xticks([0, 1, 2])
        ax2.set_xticklabels(['Red', 'Green', 'Blue'])
        
        # Add colorbar
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        
        # Add text annotations
        for i in range(len(classes)):
            for j in range(3):
                text = ax2.text(j, i, f'{channel_data[i, j]:.1f}',
                               ha="center", va="center", color="white", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_class_shap_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save numerical results
    summary_path = os.path.join(save_dir, 'per_class_shap_importance.json')
    with open(summary_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_data = {}
        for class_name, data in class_importance.items():
            json_data[class_name] = {
                'overall': data['overall'],
                'per_channel': data['per_channel']
                # Skip spatial_map as it's too large for JSON
            }
        json.dump(json_data, f, indent=2)
    
    return class_importance


def create_shap_feature_importance_analysis(
    shap_values_list: List[np.ndarray],
    images_list: List[np.ndarray],
    class_names: List[str],
    predictions: List[int],
    true_labels: List[int],
    save_dir: str
):
    """
    Create comprehensive feature importance analysis
    
    Args:
        shap_values_list: List of SHAP values for each sample
        images_list: List of original images
        class_names: List of class names
        predictions: List of predicted class indices
        true_labels: List of true class indices
        save_dir: Directory to save results
    """
    # Analyze feature importance patterns
    num_samples = len(shap_values_list)
    
    # Calculate statistics
    all_shap_values = np.stack(shap_values_list, axis=0)  # (N, C, H, W)
    
    # Global statistics
    mean_abs_shap = np.mean(np.abs(all_shap_values), axis=0)  # (C, H, W)
    std_shap = np.std(all_shap_values, axis=0)  # (C, H, W)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Row 1: Mean absolute SHAP values per channel
    for c in range(3):
        im = axes[0, c].imshow(mean_abs_shap[c], cmap='hot')
        axes[0, c].set_title(f'Mean |SHAP| - {["Red", "Green", "Blue"][c]} Channel')
        axes[0, c].axis('off')
        plt.colorbar(im, ax=axes[0, c], fraction=0.046, pad=0.04)
    
    # Combined mean absolute SHAP
    combined_mean = np.sum(mean_abs_shap, axis=0)
    im = axes[0, 3].imshow(combined_mean, cmap='hot')
    axes[0, 3].set_title('Combined Mean |SHAP|')
    axes[0, 3].axis('off')
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)
    
    # Row 2: Standard deviation of SHAP values per channel
    for c in range(3):
        im = axes[1, c].imshow(std_shap[c], cmap='viridis')
        axes[1, c].set_title(f'SHAP Std - {["Red", "Green", "Blue"][c]} Channel')
        axes[1, c].axis('off')
        plt.colorbar(im, ax=axes[1, c], fraction=0.046, pad=0.04)
    
    # Combined standard deviation
    combined_std = np.sum(std_shap, axis=0)
    im = axes[1, 3].imshow(combined_std, cmap='viridis')
    axes[1, 3].set_title('Combined SHAP Std')
    axes[1, 3].axis('off')
    plt.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.04)
    
    # Row 3: Analysis plots
    
    # Pixel-level importance distribution
    all_pixel_importance = np.sum(np.abs(all_shap_values), axis=1).flatten()  # Flatten spatial dimensions
    axes[2, 0].hist(all_pixel_importance, bins=50, alpha=0.7, color='blue')
    axes[2, 0].set_title('Pixel Importance Distribution')
    axes[2, 0].set_xlabel('SHAP Importance')
    axes[2, 0].set_ylabel('Frequency')
    
    # Per-channel importance
    channel_importance = np.sum(np.abs(all_shap_values), axis=(0, 2, 3))  # Sum over samples and spatial dims
    axes[2, 1].bar(['Red', 'Green', 'Blue'], channel_importance, color=['red', 'green', 'blue'], alpha=0.7)
    axes[2, 1].set_title('Channel Importance')
    axes[2, 1].set_ylabel('Total SHAP Importance')
    
    # Correct vs Incorrect predictions
    correct_mask = np.array(predictions) == np.array(true_labels)
    correct_importance = np.mean([np.sum(np.abs(shap_values_list[i])) for i in range(num_samples) if correct_mask[i]])
    incorrect_importance = np.mean([np.sum(np.abs(shap_values_list[i])) for i in range(num_samples) if not correct_mask[i]])
    
    axes[2, 2].bar(['Correct', 'Incorrect'], [correct_importance, incorrect_importance], 
                   color=['green', 'red'], alpha=0.7)
    axes[2, 2].set_title('Importance by Prediction Correctness')
    axes[2, 2].set_ylabel('Mean SHAP Importance')
    
    # Class-wise importance
    class_importance_means = []
    class_labels = []
    for class_idx in range(len(class_names)):
        class_mask = np.array(predictions) == class_idx
        if np.any(class_mask):
            class_importance = np.mean([np.sum(np.abs(shap_values_list[i])) for i in range(num_samples) if class_mask[i]])
            class_importance_means.append(class_importance)
            class_labels.append(class_names[class_idx])
    
    if class_importance_means:
        axes[2, 3].bar(range(len(class_labels)), class_importance_means, alpha=0.7)
        axes[2, 3].set_title('Importance by Predicted Class')
        axes[2, 3].set_ylabel('Mean SHAP Importance')
        axes[2, 3].set_xticks(range(len(class_labels)))
        axes[2, 3].set_xticklabels(class_labels, rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'shap_feature_importance_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_comprehensive_shap_analysis(
    model,
    dataloader,
    background_loader,
    device: str,
    class_names: List[str],
    save_dir: str = 'explainability/shap',
    samples_per_class: int = 5,
    background_size: int = 50
) -> Dict:
    """
    Generate comprehensive SHAP analysis with all requested features
    
    Args:
        model: Trained model
        dataloader: Data loader for samples to explain
        background_loader: Data loader for background dataset
        device: Computing device
        class_names: List of class names
        save_dir: Directory to save results
        samples_per_class: Number of samples per class (5 per class = 40 total for 8 classes)
        background_size: Size of background dataset for SHAP
        
    Returns:
        Dictionary with analysis results
    """
    print("Starting comprehensive SHAP analysis...")
    
    # Create output directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'individual'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'per_class'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'summary'), exist_ok=True)
    
    # Prepare background data (50 random samples, 5 per class)
    print("Preparing background dataset...")
    background_images = []
    background_count = 0
    samples_per_bg_class = background_size // len(class_names)
    bg_class_counts = defaultdict(int)
    
    for images, labels in background_loader:
        if background_count >= background_size:
            break
            
        for i in range(images.size(0)):
            if background_count >= background_size:
                break
                
            label = labels[i].item()
            class_name = class_names[label]
            
            if bg_class_counts[class_name] < samples_per_bg_class:
                background_images.append(images[i:i+1])
                bg_class_counts[class_name] += 1
                background_count += 1
    
    background_data = torch.cat(background_images, dim=0)
    print(f"Background dataset prepared: {background_data.shape[0]} samples")
    
    # Initialize SHAP explainer
    shap_explainer = SHAPExplainer(model, background_data, device)
    
    model.eval()
    
    # Storage for analysis
    shap_values_by_class = defaultdict(list)
    all_shap_values = []
    all_images = []
    all_predictions = []
    all_true_labels = []
    all_confidences = []
    
    # Collect samples (5 per class)
    class_sample_counts = defaultdict(int)
    total_samples_needed = len(class_names) * samples_per_class
    
    print(f"Collecting {total_samples_needed} samples ({samples_per_class} per class)...")
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if sample_count >= total_samples_needed:
                break
                
            images = images.to(device)
            labels = labels.to(device)
            
            # Get predictions
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
            
            for i in range(images.size(0)):
                if sample_count >= total_samples_needed:
                    break
                    
                true_label = labels[i].item()
                pred_label = predicted_classes[i].item()
                confidence = probabilities[i, pred_label].item()
                
                true_class_name = class_names[true_label]
                
                # Check if we need more samples for this class
                if class_sample_counts[true_class_name] >= samples_per_class:
                    continue
                
                # Single image processing
                single_image = images[i:i+1]
                
                # Generate SHAP explanation
                print(f"Generating SHAP explanation {sample_count + 1}/{total_samples_needed} (Class: {true_class_name})")
                
                try:
                    shap_values = shap_explainer.explain_image(single_image, pred_label)
                    
                    # Prepare original image for visualization
                    original_img = single_image[0].cpu().numpy().transpose(1, 2, 0)
                    
                    # Denormalize image for visualization
                    mean = np.array([0.5613, 0.5778, 0.6032])  # From config
                    std = np.array([0.2114, 0.1957, 0.1590])
                    original_img = original_img * std + mean
                    original_img = np.clip(original_img, 0, 1)
                    
                    # Store data
                    shap_values_by_class[true_class_name].append(shap_values)
                    all_shap_values.append(shap_values)
                    all_images.append(original_img)
                    all_predictions.append(pred_label)
                    all_true_labels.append(true_label)
                    all_confidences.append(confidence)
                    
                    # Create individual visualization
                    individual_filename = f"{true_class_name}_{class_sample_counts[true_class_name]:02d}_shap.png"
                    individual_save_path = os.path.join(save_dir, 'individual', individual_filename)
                    
                    visualize_shap_explanation(
                        original_img,
                        shap_values,
                        class_names[pred_label],
                        true_class_name,
                        confidence,
                        individual_save_path
                    )
                    
                    class_sample_counts[true_class_name] += 1
                    sample_count += 1
                    
                except Exception as e:
                    print(f"Error generating SHAP explanation: {e}")
                    continue
    
    print(f"Generated SHAP explanations for {sample_count} samples")
    
    # Create per-class summary
    print("Creating per-class SHAP summary...")
    class_importance = create_per_class_shap_summary(
        shap_values_by_class,
        class_names,
        os.path.join(save_dir, 'per_class')
    )
    
    # Create feature importance analysis
    print("Creating feature importance analysis...")
    create_shap_feature_importance_analysis(
        all_shap_values,
        all_images,
        class_names,
        all_predictions,
        all_true_labels,
        os.path.join(save_dir, 'summary')
    )
    
    # Create overall summary plot
    print("Creating overall summary visualization...")
    if len(all_shap_values) >= 5:
        # Select representative samples (1 per class, up to 5)
        summary_indices = []
        classes_used = set()
        
        for i, true_label in enumerate(all_true_labels):
            class_name = class_names[true_label]
            if class_name not in classes_used and len(summary_indices) < 5:
                summary_indices.append(i)
                classes_used.add(class_name)
        
        summary_shap_values = [all_shap_values[i] for i in summary_indices]
        summary_images = [all_images[i] for i in summary_indices]
        summary_predictions = [all_predictions[i] for i in summary_indices]
        
        summary_path = os.path.join(save_dir, 'summary', 'shap_summary.png')
        create_shap_summary_plot(
            summary_shap_values,
            summary_images,
            class_names,
            summary_predictions,
            summary_path
        )
    
    # Generate comprehensive metrics
    analysis_results = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': sample_count,
        'samples_per_class': dict(class_sample_counts),
        'background_size': background_data.shape[0],
        'class_importance': class_importance,
        'overall_statistics': {
            'mean_importance': float(np.mean([np.sum(np.abs(shap)) for shap in all_shap_values])),
            'std_importance': float(np.std([np.sum(np.abs(shap)) for shap in all_shap_values])),
            'channel_importance': {
                'red': float(np.sum([np.sum(np.abs(shap[0])) for shap in all_shap_values])),
                'green': float(np.sum([np.sum(np.abs(shap[1])) for shap in all_shap_values])),
                'blue': float(np.sum([np.sum(np.abs(shap[2])) for shap in all_shap_values]))
            }
        }
    }
    
    # Save analysis results
    results_path = os.path.join(save_dir, 'shap_analysis_results.json')
    with open(results_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Create summary report
    create_shap_summary_report(analysis_results, save_dir)
    
    print(f"Comprehensive SHAP analysis completed!")
    print(f"Results saved to: {save_dir}")
    print(f"- Individual explanations: {save_dir}/individual/")
    print(f"- Per-class summaries: {save_dir}/per_class/")
    print(f"- Overall summaries: {save_dir}/summary/")
    
    return analysis_results


def create_shap_summary_report(analysis_results: Dict, save_dir: str):
    """Create comprehensive SHAP analysis report"""
    report_content = f"""# SHAP Analysis Report

## Analysis Overview
- **Date**: {analysis_results['timestamp']}
- **Total Samples Analyzed**: {analysis_results['total_samples']}
- **Background Dataset Size**: {analysis_results['background_size']}

## Sample Distribution by Class
"""
    
    for class_name, count in analysis_results['samples_per_class'].items():
        report_content += f"- **{class_name}**: {count} samples\n"
    
    overall_stats = analysis_results['overall_statistics']
    
    report_content += f"""
## Overall Statistics
- **Mean Importance**: {overall_stats['mean_importance']:.3f}
- **Std Importance**: {overall_stats['std_importance']:.3f}

### Channel-wise Importance
- **Red Channel**: {overall_stats['channel_importance']['red']:.3f}
- **Green Channel**: {overall_stats['channel_importance']['green']:.3f}
- **Blue Channel**: {overall_stats['channel_importance']['blue']:.3f}

## Per-Class Importance Rankings
"""
    
    # Sort classes by importance
    class_importance = analysis_results['class_importance']
    sorted_classes = sorted(class_importance.items(), key=lambda x: x[1]['overall'], reverse=True)
    
    for i, (class_name, importance_data) in enumerate(sorted_classes, 1):
        report_content += f"{i}. **{class_name}**: {importance_data['overall']:.3f}\n"
    
    report_content += f"""
## Interpretation Guidelines

### SHAP Values
- **Positive values**: Features that increase the prediction probability
- **Negative values**: Features that decrease the prediction probability
- **Magnitude**: Indicates the strength of the feature's contribution

### Per-Class Analysis
- Shows which classes rely more heavily on specific features
- Higher overall importance may indicate more complex decision boundaries
- Channel-wise analysis reveals color sensitivity patterns

## Files Generated
- `individual/`: Individual SHAP explanations for each sample
- `per_class/`: Per-class summary statistics and visualizations
- `summary/`: Overall analysis and feature importance patterns
- `shap_analysis_results.json`: Complete numerical results

## Clinical Insights
Based on the SHAP analysis, the model's decision-making process shows:
"""
    
    # Add insights based on channel importance
    channel_imp = overall_stats['channel_importance']
    max_channel = max(channel_imp.items(), key=lambda x: x[1])
    
    report_content += f"- **Primary color sensitivity**: {max_channel[0].title()} channel shows highest importance\n"
    
    if len(sorted_classes) > 0:
        most_complex = sorted_classes[0][0]
        least_complex = sorted_classes[-1][0]
        report_content += f"- **Most complex class**: {most_complex} (highest feature importance)\n"
        report_content += f"- **Least complex class**: {least_complex} (lowest feature importance)\n"
    
    report_content += """
## Recommendations
- Use SHAP explanations to validate clinical relevance of highlighted regions
- Compare channel importance with known histological staining patterns
- Investigate high-importance regions for potential biomarkers
- Consider class-specific preprocessing based on importance patterns
"""
    
    # Save report
    report_path = os.path.join(save_dir, 'shap_analysis_report.md')
    with open(report_path, 'w') as f:
        f.write(report_content)


# Legacy function for backward compatibility
def generate_shap_explanations(
    model,
    dataloader,
    background_loader,
    device,
    class_names: List[str],
    save_dir: str = 'explainability/shap_results',
    num_samples: int = 20,
    background_size: int = 50
):
    """
    Legacy function - use generate_comprehensive_shap_analysis instead
    """
    samples_per_class = max(1, num_samples // len(class_names))
    return generate_comprehensive_shap_analysis(
        model=model,
        dataloader=dataloader,
        background_loader=background_loader,
        device=device,
        class_names=class_names,
        save_dir=save_dir,
        samples_per_class=samples_per_class,
        background_size=background_size
    )