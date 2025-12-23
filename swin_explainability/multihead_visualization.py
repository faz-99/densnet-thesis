"""Multi-Head Attention Visualization for Swin Transformer - Feature diversity analysis"""
import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Optional, List


class SwinMultiHeadVisualization:
    """Visualize individual attention heads to understand feature diversity"""
    
    def __init__(self, model: nn.Module, target_layer: Optional[str] = None):
        self.model = model
        self.target_layer = target_layer or 'layers.3'  # Last stage by default
        self.attention_maps = []
        self.hooks = []
        
    def _register_hooks(self):
        """Register hooks on target layer"""
        def attention_hook(module, input, output):
            if hasattr(module, 'attn'):
                # Store attention: [B, num_heads, N, N]
                self.attention_maps.append(module.attn.detach().cpu())
        
        for name, module in self.model.named_modules():
            if self.target_layer in name and 'attn' in name:
                self.hooks.append(module.register_forward_hook(attention_hook))
    
    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def visualize_heads(self, image: torch.Tensor, num_heads: int = 6, 
                       layer_idx: int = -1) -> List[np.ndarray]:
        """Visualize individual attention heads
        
        Args:
            image: Input tensor [1, 3, H, W]
            num_heads: Number of heads to visualize
            layer_idx: Which layer to visualize (-1 for last)
            
        Returns:
            List of attention maps, one per head
        """
        self.attention_maps = []
        self._register_hooks()
        
        self.model.eval()
        with torch.no_grad():
            _ = self.model(image)
        
        self._remove_hooks()
        
        if not self.attention_maps:
            return [np.zeros((224, 224)) for _ in range(num_heads)]
        
        # Select target layer
        target_attn = self.attention_maps[layer_idx]  # [B, num_heads, N, N]
        
        head_maps = []
        total_heads = target_attn.shape[1]
        selected_heads = np.linspace(0, total_heads - 1, num_heads, dtype=int)
        
        for head_idx in selected_heads:
            # Extract attention for this head
            head_attn = target_attn[0, head_idx]  # [N, N]
            
            # Average attention from CLS token or all tokens
            attn_map = head_attn[0, 1:]  # CLS attention to patches
            
            # Reshape to spatial
            grid_size = int(np.sqrt(attn_map.shape[0]))
            attn_map = attn_map.reshape(grid_size, grid_size).numpy()
            
            # Resize and normalize
            attn_map = cv2.resize(attn_map, (224, 224), interpolation=cv2.INTER_CUBIC)
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            
            head_maps.append(attn_map)
        
        return head_maps
    
    def analyze_head_diversity(self, image: torch.Tensor) -> dict:
        """Analyze what different heads focus on
        
        Returns:
            Dictionary with diversity metrics
        """
        self.attention_maps = []
        self._register_hooks()
        
        self.model.eval()
        with torch.no_grad():
            _ = self.model(image)
        
        self._remove_hooks()
        
        if not self.attention_maps:
            return {}
        
        target_attn = self.attention_maps[-1][0]  # [num_heads, N, N]
        num_heads = target_attn.shape[0]
        
        # Compute pairwise correlation between heads
        head_vectors = target_attn[:, 0, 1:].cpu().numpy()  # [num_heads, num_patches]
        
        correlations = np.corrcoef(head_vectors)
        avg_correlation = (correlations.sum() - num_heads) / (num_heads * (num_heads - 1))
        
        # Compute entropy for each head
        entropies = []
        for h in range(num_heads):
            attn = target_attn[h, 0, 1:].cpu().numpy()
            attn = attn / (attn.sum() + 1e-8)
            entropy = -np.sum(attn * np.log(attn + 1e-8))
            entropies.append(entropy)
        
        return {
            'num_heads': num_heads,
            'avg_correlation': float(avg_correlation),
            'avg_entropy': float(np.mean(entropies)),
            'entropy_std': float(np.std(entropies)),
            'diversity_score': 1.0 - avg_correlation  # Higher is more diverse
        }
    
    def get_head_specialization(self, dataloader, num_samples: int = 50) -> dict:
        """Analyze head specialization across multiple images
        
        Args:
            dataloader: DataLoader with test images
            num_samples: Number of samples to analyze
            
        Returns:
            Dictionary with head specialization patterns
        """
        all_head_attentions = []
        
        for i, (images, _) in enumerate(dataloader):
            if i >= num_samples:
                break
            
            self.attention_maps = []
            self._register_hooks()
            
            self.model.eval()
            with torch.no_grad():
                _ = self.model(images.to(next(self.model.parameters()).device))
            
            self._remove_hooks()
            
            if self.attention_maps:
                all_head_attentions.append(self.attention_maps[-1][0].cpu().numpy())
        
        if not all_head_attentions:
            return {}
        
        # Stack: [num_samples, num_heads, N, N]
        all_attentions = np.stack(all_head_attentions)
        num_heads = all_attentions.shape[1]
        
        # Compute consistency for each head
        head_consistency = []
        for h in range(num_heads):
            head_attn = all_attentions[:, h, 0, 1:]  # [num_samples, num_patches]
            # Compute variance across samples (low variance = consistent)
            consistency = 1.0 / (np.var(head_attn, axis=0).mean() + 1e-8)
            head_consistency.append(consistency)
        
        return {
            'head_consistency': head_consistency,
            'most_consistent_heads': np.argsort(head_consistency)[-3:].tolist(),
            'most_diverse_heads': np.argsort(head_consistency)[:3].tolist()
        }
