"""Unified Visualizer for Swin Transformer Explainability"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Optional, Tuple
from .attention_rollout import SwinAttentionRollout
from .attention_gradcam import SwinAttentionGradCAM
from .multihead_visualization import SwinMultiHeadVisualization


class UnifiedSwinVisualizer:
    """Unified interface for all Swin explainability methods"""
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.rollout = SwinAttentionRollout(model)
        self.gradcam = SwinAttentionGradCAM(model)
        self.multihead = SwinMultiHeadVisualization(model)
    
    def generate_comprehensive_visualization(self, image: torch.Tensor, 
                                            original_image: np.ndarray,
                                            target_class: Optional[int] = None,
                                            save_path: Optional[str] = None,
                                            dpi: int = 300) -> plt.Figure:
        """Generate comprehensive visualization with all methods
        
        Args:
            image: Preprocessed tensor [1, 3, H, W]
            original_image: Original image for overlay [H, W, 3]
            target_class: Target class for GradCAM
            save_path: Path to save figure
            dpi: Figure DPI
            
        Returns:
            Matplotlib figure
        """
        # Generate all explanations
        rollout_map = self.rollout.generate_rollout(image)
        gradcam_map = self.gradcam.generate_cam(image, target_class)
        head_maps = self.multihead.visualize_heads(image, num_heads=6)
        
        # Create figure
        fig = plt.figure(figsize=(20, 12))
        
        # Original image
        ax1 = plt.subplot(3, 4, 1)
        ax1.imshow(original_image)
        ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Attention Rollout
        ax2 = plt.subplot(3, 4, 2)
        ax2.imshow(original_image)
        ax2.imshow(rollout_map, cmap='jet', alpha=0.5)
        ax2.set_title('Attention Rollout', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # Attention GradCAM
        ax3 = plt.subplot(3, 4, 3)
        ax3.imshow(original_image)
        ax3.imshow(gradcam_map, cmap='jet', alpha=0.5)
        ax3.set_title('Attention GradCAM', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # Comparison
        ax4 = plt.subplot(3, 4, 4)
        combined = (rollout_map + gradcam_map) / 2
        ax4.imshow(original_image)
        ax4.imshow(combined, cmap='jet', alpha=0.5)
        ax4.set_title('Combined', fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        # Multi-head visualization (6 heads)
        for i, head_map in enumerate(head_maps):
            ax = plt.subplot(3, 4, 5 + i)
            ax.imshow(original_image)
            ax.imshow(head_map, cmap='viridis', alpha=0.5)
            ax.set_title(f'Head {i+1}', fontsize=10)
            ax.axis('off')
        
        # Heatmaps only
        ax11 = plt.subplot(3, 4, 11)
        ax11.imshow(rollout_map, cmap='jet')
        ax11.set_title('Rollout Heatmap', fontsize=10)
        ax11.axis('off')
        
        ax12 = plt.subplot(3, 4, 12)
        ax12.imshow(gradcam_map, cmap='jet')
        ax12.set_title('GradCAM Heatmap', fontsize=10)
        ax12.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        
        return fig
    
    def batch_process(self, dataloader, output_dir: str, num_samples: int = 10):
        """Process multiple images and save visualizations
        
        Args:
            dataloader: DataLoader with images
            output_dir: Directory to save results
            num_samples: Number of samples to process
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        device = next(self.model.parameters()).device
        
        for i, (images, labels) in enumerate(dataloader):
            if i >= num_samples:
                break
            
            image = images[0:1].to(device)
            
            # Denormalize for visualization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            original = (image[0].cpu() * std + mean).permute(1, 2, 0).numpy()
            original = np.clip(original, 0, 1)
            
            save_path = os.path.join(output_dir, f'sample_{i}_label_{labels[0].item()}.png')
            self.generate_comprehensive_visualization(image, original, save_path=save_path)
            plt.close()
    
    def compare_methods(self, image: torch.Tensor, original_image: np.ndarray,
                       save_path: Optional[str] = None) -> dict:
        """Compare all methods with metrics
        
        Returns:
            Dictionary with comparison metrics
        """
        rollout_map = self.rollout.generate_rollout(image)
        gradcam_map = self.gradcam.generate_cam(image)
        
        # Compute correlation
        correlation = np.corrcoef(rollout_map.flatten(), gradcam_map.flatten())[0, 1]
        
        # Compute sparsity (how focused the attention is)
        rollout_sparsity = np.sum(rollout_map > 0.5) / rollout_map.size
        gradcam_sparsity = np.sum(gradcam_map > 0.5) / gradcam_map.size
        
        # Compute entropy
        def compute_entropy(map):
            map_norm = map / (map.sum() + 1e-8)
            return -np.sum(map_norm * np.log(map_norm + 1e-8))
        
        rollout_entropy = compute_entropy(rollout_map)
        gradcam_entropy = compute_entropy(gradcam_map)
        
        # Head diversity
        diversity = self.multihead.analyze_head_diversity(image)
        
        return {
            'rollout_gradcam_correlation': float(correlation),
            'rollout_sparsity': float(rollout_sparsity),
            'gradcam_sparsity': float(gradcam_sparsity),
            'rollout_entropy': float(rollout_entropy),
            'gradcam_entropy': float(gradcam_entropy),
            'head_diversity': diversity
        }
