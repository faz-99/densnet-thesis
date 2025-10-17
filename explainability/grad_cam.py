"""
Enhanced Grad-CAM and Grad-CAM++ implementation for DenLsNet explainability
Includes comprehensive visualization, grid generation, and quantitative analysis
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import json
from datetime import datetime
from collections import defaultdict
import seaborn as sns
from sklearn.metrics import jaccard_score


class GradCAM:
    """Grad-CAM implementation for generating visual explanations"""
    
    def __init__(self, model, target_layer_name: str = 'features.norm5'):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find target layer
        target_layer = self._find_target_layer()
        if target_layer is not None:
            self.hooks.append(target_layer.register_forward_hook(forward_hook))
            self.hooks.append(target_layer.register_backward_hook(backward_hook))
    
    def _find_target_layer(self):
        """Find the target layer by name"""
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                return module
        return None
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            class_idx: Target class index (if None, uses predicted class)
            
        Returns:
            CAM heatmap as numpy array
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()
        
        # Generate CAM
        gradients = self.gradients.detach()
        activations = self.activations.detach()
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize CAM
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def __del__(self):
        """Remove hooks when object is destroyed"""
        for hook in self.hooks:
            hook.remove()


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++ implementation with improved localization"""
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM++ heatmap
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            class_idx: Target class index (if None, uses predicted class)
            
        Returns:
            CAM heatmap as numpy array
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients.detach()
        activations = self.activations.detach()
        
        # Calculate alpha weights (Grad-CAM++ improvement)
        alpha_num = gradients.pow(2)
        alpha_denom = 2.0 * gradients.pow(2) + \
                     torch.sum(activations * gradients.pow(3), dim=(2, 3), keepdim=True)
        alpha = alpha_num / (alpha_denom + 1e-7)
        
        # Calculate weights
        weights = torch.sum(alpha * F.relu(gradients), dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize CAM
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Overlay heatmap on original image
    
    Args:
        image: Original image (H, W, C) in range [0, 1]
        heatmap: CAM heatmap (H, W) in range [0, 1]
        alpha: Transparency factor
        
    Returns:
        Overlayed image
    """
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to RGB
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
    
    # Overlay
    overlayed = alpha * heatmap_colored + (1 - alpha) * image
    
    return overlayed


def visualize_gradcam_results(
    original_image: np.ndarray,
    gradcam_heatmap: np.ndarray,
    gradcam_plus_heatmap: np.ndarray,
    predicted_class: str,
    true_class: str,
    confidence: float,
    save_path: str
):
    """
    Create comprehensive visualization of Grad-CAM results
    
    Args:
        original_image: Original input image
        gradcam_heatmap: Grad-CAM heatmap
        gradcam_plus_heatmap: Grad-CAM++ heatmap
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
    
    # Grad-CAM heatmap
    axes[0, 1].imshow(gradcam_heatmap, cmap='jet')
    axes[0, 1].set_title('Grad-CAM Heatmap')
    axes[0, 1].axis('off')
    
    # Grad-CAM++ heatmap
    axes[0, 2].imshow(gradcam_plus_heatmap, cmap='jet')
    axes[0, 2].set_title('Grad-CAM++ Heatmap')
    axes[0, 2].axis('off')
    
    # Overlayed images
    gradcam_overlay = overlay_heatmap(original_image, gradcam_heatmap)
    gradcam_plus_overlay = overlay_heatmap(original_image, gradcam_plus_heatmap)
    
    axes[1, 0].imshow(gradcam_overlay)
    axes[1, 0].set_title('Grad-CAM Overlay')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(gradcam_plus_overlay)
    axes[1, 1].set_title('Grad-CAM++ Overlay')
    axes[1, 1].axis('off')
    
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


def compute_insertion_deletion_auc(
    model,
    image: torch.Tensor,
    heatmap: np.ndarray,
    target_class: int,
    device: str,
    num_steps: int = 50
) -> Tuple[float, float]:
    """
    Compute insertion and deletion AUC for faithfulness evaluation
    
    Args:
        model: Trained model
        image: Input image tensor (1, C, H, W)
        heatmap: Explanation heatmap
        target_class: Target class index
        device: Computing device
        num_steps: Number of steps for curve computation
        
    Returns:
        Tuple of (insertion_auc, deletion_auc)
    """
    model.eval()
    
    # Get baseline prediction
    with torch.no_grad():
        baseline_output = model(image)
        baseline_prob = F.softmax(baseline_output, dim=1)[0, target_class].item()
    
    # Resize heatmap to match image
    if heatmap.shape != (224, 224):
        heatmap = cv2.resize(heatmap, (224, 224))
    
    # Get pixel importance order
    flat_heatmap = heatmap.flatten()
    sorted_indices = np.argsort(flat_heatmap)
    
    # Insertion curve (start with black image, add important pixels)
    insertion_probs = []
    masked_image = torch.zeros_like(image)
    
    for step in range(num_steps + 1):
        if step == 0:
            prob = 0.0  # Black image
        else:
            # Add most important pixels
            num_pixels = int((step / num_steps) * len(sorted_indices))
            pixels_to_add = sorted_indices[-num_pixels:]
            
            # Create mask
            mask = np.zeros_like(flat_heatmap)
            mask[pixels_to_add] = 1
            mask = mask.reshape(224, 224)
            
            # Apply mask
            for c in range(3):
                masked_image[0, c] = image[0, c] * torch.from_numpy(mask).to(device)
            
            with torch.no_grad():
                output = model(masked_image)
                prob = F.softmax(output, dim=1)[0, target_class].item()
        
        insertion_probs.append(prob)
    
    # Deletion curve (start with full image, remove important pixels)
    deletion_probs = []
    masked_image = image.clone()
    
    for step in range(num_steps + 1):
        if step == 0:
            prob = baseline_prob  # Full image
        else:
            # Remove most important pixels
            num_pixels = int((step / num_steps) * len(sorted_indices))
            pixels_to_remove = sorted_indices[-num_pixels:]
            
            # Create mask
            mask = np.ones_like(flat_heatmap)
            mask[pixels_to_remove] = 0
            mask = mask.reshape(224, 224)
            
            # Apply mask
            for c in range(3):
                masked_image[0, c] = image[0, c] * torch.from_numpy(mask).to(device)
            
            with torch.no_grad():
                output = model(masked_image)
                prob = F.softmax(output, dim=1)[0, target_class].item()
        
        deletion_probs.append(prob)
    
    # Calculate AUC
    x = np.linspace(0, 1, num_steps + 1)
    insertion_auc = np.trapz(insertion_probs, x)
    deletion_auc = np.trapz(deletion_probs, x)
    
    return insertion_auc, deletion_auc


def compute_stability_score(
    model,
    image: torch.Tensor,
    target_class: int,
    device: str,
    num_perturbations: int = 5,
    noise_level: float = 0.1
) -> float:
    """
    Compute stability score under random perturbations
    
    Args:
        model: Trained model
        image: Input image tensor
        target_class: Target class index
        device: Computing device
        num_perturbations: Number of perturbations to test
        noise_level: Noise level for perturbations
        
    Returns:
        Average correlation coefficient
    """
    gradcam = GradCAM(model)
    
    # Generate baseline explanation
    baseline_heatmap = gradcam.generate_cam(image, target_class)
    
    correlations = []
    
    for _ in range(num_perturbations):
        # Add noise to image
        noise = torch.randn_like(image) * noise_level
        perturbed_image = image + noise
        perturbed_image = torch.clamp(perturbed_image, -3, 3)
        
        # Generate explanation for perturbed image
        perturbed_heatmap = gradcam.generate_cam(perturbed_image, target_class)
        
        # Calculate correlation
        corr = np.corrcoef(baseline_heatmap.flatten(), perturbed_heatmap.flatten())[0, 1]
        
        if not np.isnan(corr):
            correlations.append(corr)
    
    return np.mean(correlations) if correlations else 0.0


def create_class_specific_grid(
    samples_by_class: Dict[str, List],
    class_names: List[str],
    save_path: str,
    title: str = "Grad-CAM Analysis by Class"
):
    """
    Create a grid visualization showing samples organized by class
    
    Args:
        samples_by_class: Dictionary with class names as keys and sample data as values
        class_names: List of class names
        save_path: Path to save the grid
        title: Title for the visualization
    """
    num_classes = len(class_names)
    max_samples_per_class = max(len(samples) for samples in samples_by_class.values())
    
    fig, axes = plt.subplots(num_classes, max_samples_per_class, 
                            figsize=(4 * max_samples_per_class, 4 * num_classes))
    
    if num_classes == 1:
        axes = axes.reshape(1, -1)
    elif max_samples_per_class == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for class_idx, class_name in enumerate(class_names):
        samples = samples_by_class.get(class_name, [])
        
        for sample_idx in range(max_samples_per_class):
            ax = axes[class_idx, sample_idx] if max_samples_per_class > 1 else axes[class_idx]
            
            if sample_idx < len(samples):
                sample_data = samples[sample_idx]
                
                # Display overlay image
                ax.imshow(sample_data['overlay'])
                
                # Add prediction info
                pred_class = sample_data['predicted_class']
                confidence = sample_data['confidence']
                is_correct = sample_data['is_correct']
                
                color = 'green' if is_correct else 'red'
                ax.set_title(f'{pred_class}\nConf: {confidence:.3f}', 
                           color=color, fontsize=10, fontweight='bold')
                
                # Add border color based on correctness
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(3)
            else:
                ax.axis('off')
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add class name on the left
            if sample_idx == 0:
                ax.set_ylabel(class_name, fontsize=12, fontweight='bold', rotation=90)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_comprehensive_gradcam_analysis(
    model,
    dataloader,
    device: str,
    class_names: List[str],
    save_dir: str = 'explainability/gradcam',
    samples_per_class: int = 5,
    correct_samples: int = 5,
    incorrect_samples: int = 5
) -> Dict:
    """
    Generate comprehensive Grad-CAM analysis with all requested features
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Computing device
        class_names: List of class names
        save_dir: Directory to save results
        samples_per_class: Number of samples per class for analysis
        correct_samples: Number of correctly classified samples per class
        incorrect_samples: Number of misclassified samples per class
        
    Returns:
        Dictionary with analysis results
    """
    print("Starting comprehensive Grad-CAM analysis...")
    
    # Create output directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'individual'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'grids'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'metrics'), exist_ok=True)
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer_name='densenet.features.norm5')
    gradcam_plus = GradCAMPlusPlus(model, target_layer_name='densenet.features.norm5')
    
    model.eval()
    
    # Storage for samples by class and correctness
    correct_samples_by_class = defaultdict(list)
    incorrect_samples_by_class = defaultdict(list)
    all_samples_by_class = defaultdict(list)
    
    # Metrics storage
    insertion_aucs = []
    deletion_aucs = []
    stability_scores = []
    
    sample_count = 0
    total_samples_needed = len(class_names) * (correct_samples + incorrect_samples)
    
    print(f"Collecting {total_samples_needed} samples ({correct_samples} correct + {incorrect_samples} incorrect per class)")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if sample_count >= total_samples_needed * 2:  # Safety margin
                break
            
            images = images.to(device)
            labels = labels.to(device)
            
            # Get predictions
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
            
            for i in range(images.size(0)):
                single_image = images[i:i+1]
                true_label = labels[i].item()
                pred_label = predicted_classes[i].item()
                confidence = probabilities[i, pred_label].item()
                
                true_class_name = class_names[true_label]
                pred_class_name = class_names[pred_label]
                is_correct = (true_label == pred_label)
                
                # Check if we need more samples for this class
                correct_count = len(correct_samples_by_class[true_class_name])
                incorrect_count = len(incorrect_samples_by_class[true_class_name])
                
                should_collect = False
                if is_correct and correct_count < correct_samples:
                    should_collect = True
                elif not is_correct and incorrect_count < incorrect_samples:
                    should_collect = True
                
                if not should_collect:
                    continue
                
                # Generate explanations
                gradcam_heatmap = gradcam.generate_cam(single_image, pred_label)
                gradcam_plus_heatmap = gradcam_plus.generate_cam(single_image, pred_label)
                
                # Prepare original image for visualization
                original_img = single_image[0].cpu().numpy().transpose(1, 2, 0)
                
                # Denormalize image for visualization
                mean = np.array([0.5613, 0.5778, 0.6032])  # From config
                std = np.array([0.2114, 0.1957, 0.1590])
                original_img = original_img * std + mean
                original_img = np.clip(original_img, 0, 1)
                
                # Create overlays
                gradcam_overlay = overlay_heatmap(original_img, gradcam_heatmap, alpha=0.4)
                gradcam_plus_overlay = overlay_heatmap(original_img, gradcam_plus_heatmap, alpha=0.4)
                
                # Compute quantitative metrics
                ins_auc, del_auc = compute_insertion_deletion_auc(
                    model, single_image, gradcam_heatmap, pred_label, device
                )
                stability = compute_stability_score(
                    model, single_image, pred_label, device
                )
                
                insertion_aucs.append(ins_auc)
                deletion_aucs.append(del_auc)
                stability_scores.append(stability)
                
                # Create sample data
                sample_data = {
                    'original_image': original_img,
                    'gradcam_heatmap': gradcam_heatmap,
                    'gradcam_plus_heatmap': gradcam_plus_heatmap,
                    'gradcam_overlay': gradcam_overlay,
                    'gradcam_plus_overlay': gradcam_plus_overlay,
                    'overlay': gradcam_overlay,  # Default to Grad-CAM for grid
                    'true_class': true_class_name,
                    'predicted_class': pred_class_name,
                    'confidence': confidence,
                    'is_correct': is_correct,
                    'insertion_auc': ins_auc,
                    'deletion_auc': del_auc,
                    'stability': stability
                }
                
                # Store sample
                if is_correct:
                    correct_samples_by_class[true_class_name].append(sample_data)
                else:
                    incorrect_samples_by_class[true_class_name].append(sample_data)
                
                all_samples_by_class[true_class_name].append(sample_data)
                
                # Save individual visualization
                correctness = "correct" if is_correct else "incorrect"
                sample_filename = f"{true_class_name}_{correctness}_{len(correct_samples_by_class[true_class_name]) if is_correct else len(incorrect_samples_by_class[true_class_name]):02d}.png"
                individual_save_path = os.path.join(save_dir, 'individual', sample_filename)
                
                visualize_gradcam_results(
                    original_img,
                    gradcam_heatmap,
                    gradcam_plus_heatmap,
                    pred_class_name,
                    true_class_name,
                    confidence,
                    individual_save_path
                )
                
                sample_count += 1
                
                if sample_count % 10 == 0:
                    print(f"Processed {sample_count} samples...")
    
    print(f"Collected samples for analysis. Generating visualizations...")
    
    # Create grid visualizations
    print("Creating grid visualizations...")
    
    # Correct predictions grid
    create_class_specific_grid(
        correct_samples_by_class,
        class_names,
        os.path.join(save_dir, 'grids', 'correct_predictions_grid.png'),
        "Correctly Classified Samples - Grad-CAM Analysis"
    )
    
    # Incorrect predictions grid
    create_class_specific_grid(
        incorrect_samples_by_class,
        class_names,
        os.path.join(save_dir, 'grids', 'incorrect_predictions_grid.png'),
        "Misclassified Samples - Grad-CAM Analysis"
    )
    
    # Combined grid (mix of correct and incorrect)
    combined_samples = {}
    for class_name in class_names:
        combined = []
        # Add up to 3 correct and 2 incorrect samples
        combined.extend(correct_samples_by_class[class_name][:3])
        combined.extend(incorrect_samples_by_class[class_name][:2])
        combined_samples[class_name] = combined
    
    create_class_specific_grid(
        combined_samples,
        class_names,
        os.path.join(save_dir, 'grids', 'combined_analysis_grid.png'),
        "Grad-CAM Analysis - Mixed Correct/Incorrect Predictions"
    )
    
    # Generate quantitative metrics summary
    print("Computing quantitative metrics...")
    
    metrics_summary = {
        'insertion_auc': {
            'mean': float(np.mean(insertion_aucs)),
            'std': float(np.std(insertion_aucs)),
            'values': insertion_aucs
        },
        'deletion_auc': {
            'mean': float(np.mean(deletion_aucs)),
            'std': float(np.std(deletion_aucs)),
            'values': deletion_aucs
        },
        'stability': {
            'mean': float(np.mean(stability_scores)),
            'std': float(np.std(stability_scores)),
            'values': stability_scores
        },
        'samples_analyzed': sample_count,
        'class_distribution': {
            class_name: {
                'correct': len(correct_samples_by_class[class_name]),
                'incorrect': len(incorrect_samples_by_class[class_name])
            }
            for class_name in class_names
        }
    }
    
    # Save metrics
    metrics_path = os.path.join(save_dir, 'metrics', 'gradcam_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    # Create metrics visualization
    create_metrics_visualization(metrics_summary, os.path.join(save_dir, 'metrics'))
    
    # Create summary report
    create_gradcam_summary_report(
        metrics_summary,
        correct_samples_by_class,
        incorrect_samples_by_class,
        class_names,
        save_dir
    )
    
    print(f"Comprehensive Grad-CAM analysis completed!")
    print(f"Results saved to: {save_dir}")
    print(f"- Individual visualizations: {save_dir}/individual/")
    print(f"- Grid visualizations: {save_dir}/grids/")
    print(f"- Quantitative metrics: {save_dir}/metrics/")
    
    return metrics_summary


def create_metrics_visualization(metrics_summary: Dict, save_dir: str):
    """Create visualization of quantitative metrics"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Insertion AUC distribution
    axes[0].hist(metrics_summary['insertion_auc']['values'], bins=20, alpha=0.7, color='blue')
    axes[0].axvline(metrics_summary['insertion_auc']['mean'], color='red', linestyle='--', 
                   label=f"Mean: {metrics_summary['insertion_auc']['mean']:.3f}")
    axes[0].set_title('Insertion AUC Distribution')
    axes[0].set_xlabel('Insertion AUC')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    
    # Deletion AUC distribution
    axes[1].hist(metrics_summary['deletion_auc']['values'], bins=20, alpha=0.7, color='orange')
    axes[1].axvline(metrics_summary['deletion_auc']['mean'], color='red', linestyle='--',
                   label=f"Mean: {metrics_summary['deletion_auc']['mean']:.3f}")
    axes[1].set_title('Deletion AUC Distribution')
    axes[1].set_xlabel('Deletion AUC')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    
    # Stability distribution
    axes[2].hist(metrics_summary['stability']['values'], bins=20, alpha=0.7, color='green')
    axes[2].axvline(metrics_summary['stability']['mean'], color='red', linestyle='--',
                   label=f"Mean: {metrics_summary['stability']['mean']:.3f}")
    axes[2].set_title('Stability Score Distribution')
    axes[2].set_xlabel('Stability Score')
    axes[2].set_ylabel('Frequency')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_gradcam_summary_report(
    metrics_summary: Dict,
    correct_samples: Dict,
    incorrect_samples: Dict,
    class_names: List[str],
    save_dir: str
):
    """Create a comprehensive summary report"""
    report_content = f"""# Grad-CAM Analysis Report

## Analysis Overview
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Samples Analyzed**: {metrics_summary['samples_analyzed']}
- **Classes**: {len(class_names)} ({', '.join(class_names)})

## Quantitative Metrics

### Faithfulness Metrics
- **Insertion AUC**: {metrics_summary['insertion_auc']['mean']:.3f} ± {metrics_summary['insertion_auc']['std']:.3f}
- **Deletion AUC**: {metrics_summary['deletion_auc']['mean']:.3f} ± {metrics_summary['deletion_auc']['std']:.3f}

### Stability Metrics
- **Stability Score**: {metrics_summary['stability']['mean']:.3f} ± {metrics_summary['stability']['std']:.3f}

## Sample Distribution by Class

| Class | Correct Samples | Incorrect Samples | Total |
|-------|----------------|-------------------|-------|
"""
    
    for class_name in class_names:
        correct_count = metrics_summary['class_distribution'][class_name]['correct']
        incorrect_count = metrics_summary['class_distribution'][class_name]['incorrect']
        total_count = correct_count + incorrect_count
        report_content += f"| {class_name} | {correct_count} | {incorrect_count} | {total_count} |\n"
    
    report_content += f"""
## Interpretation Guidelines

### Insertion AUC
- **Range**: 0.0 - 1.0 (higher is better)
- **Interpretation**: Measures how quickly the model's confidence increases when important pixels are added
- **Good explanations**: Should achieve high AUC (> 0.6)

### Deletion AUC  
- **Range**: 0.0 - 1.0 (lower is better)
- **Interpretation**: Measures how quickly the model's confidence decreases when important pixels are removed
- **Good explanations**: Should achieve low AUC (< 0.4)

### Stability Score
- **Range**: -1.0 - 1.0 (higher is better)
- **Interpretation**: Correlation between explanations under input perturbations
- **Stable explanations**: Should achieve high correlation (> 0.7)

## Files Generated
- `individual/`: Individual Grad-CAM visualizations for each sample
- `grids/`: Grid visualizations organized by class and correctness
- `metrics/`: Quantitative analysis results and distributions

## Recommendations
"""
    
    # Add recommendations based on metrics
    ins_auc = metrics_summary['insertion_auc']['mean']
    del_auc = metrics_summary['deletion_auc']['mean']
    stability = metrics_summary['stability']['mean']
    
    if ins_auc > 0.6:
        report_content += "- ✅ Good insertion AUC indicates faithful explanations\n"
    else:
        report_content += "- ⚠️ Low insertion AUC suggests explanations may not be faithful\n"
    
    if del_auc < 0.4:
        report_content += "- ✅ Good deletion AUC indicates explanations identify important regions\n"
    else:
        report_content += "- ⚠️ High deletion AUC suggests explanations may not identify truly important regions\n"
    
    if stability > 0.7:
        report_content += "- ✅ High stability indicates robust explanations\n"
    else:
        report_content += "- ⚠️ Low stability suggests explanations are sensitive to input perturbations\n"
    
    # Save report
    report_path = os.path.join(save_dir, 'gradcam_analysis_report.md')
    with open(report_path, 'w') as f:
        f.write(report_content)


# Legacy function for backward compatibility
def generate_gradcam_explanations(
    model,
    dataloader,
    device,
    class_names: List[str],
    save_dir: str = 'explainability/gradcam_results',
    num_samples: int = 20
):
    """
    Legacy function - use generate_comprehensive_gradcam_analysis instead
    """
    return generate_comprehensive_gradcam_analysis(
        model=model,
        dataloader=dataloader,
        device=device,
        class_names=class_names,
        save_dir=save_dir,
        samples_per_class=num_samples // len(class_names),
        correct_samples=5,
        incorrect_samples=5
    )