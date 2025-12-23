"""Attention Rollout for Swin Transformer - Hierarchical attention aggregation"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
import cv2


class SwinAttentionRollout:
    """Extract and aggregate attention weights across Swin Transformer stages"""
    
    def __init__(self, model: nn.Module, head_fusion: str = 'mean', discard_ratio: float = 0.1):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attentions = []
        self.hooks = []
        
    def _register_hooks(self):
        """Register forward hooks to capture attention weights"""
        def get_attention_hook(name):
            def hook(module, input, output):
                if hasattr(module, 'attn'):
                    # Swin WindowAttention stores attention in forward pass
                    self.attentions.append(module.attn.detach().cpu())
            return hook
        
        for name, module in self.model.named_modules():
            if 'attn' in name and hasattr(module, 'qkv'):
                self.hooks.append(module.register_forward_hook(get_attention_hook(name)))
    
    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _fuse_heads(self, attention: torch.Tensor) -> torch.Tensor:
        """Fuse multi-head attention: [B, num_heads, N, N] -> [B, N, N]"""
        if self.head_fusion == 'mean':
            return attention.mean(dim=1)
        elif self.head_fusion == 'max':
            return attention.max(dim=1)[0]
        elif self.head_fusion == 'min':
            return attention.min(dim=1)[0]
        return attention.mean(dim=1)
    
    def _rollout_attention(self, attentions: list) -> torch.Tensor:
        """Recursively multiply attention matrices with residual connections"""
        result = torch.eye(attentions[0].size(-1))
        
        for attention in attentions:
            # Add identity for residual connection
            attention = attention + torch.eye(attention.size(-1))
            attention = attention / attention.sum(dim=-1, keepdim=True)
            result = torch.matmul(attention, result)
        
        # Normalize
        result = result / result.sum(dim=-1, keepdim=True)
        return result
    
    def generate_rollout(self, image: torch.Tensor, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Generate attention rollout heatmap
        
        Args:
            image: Input tensor [1, 3, H, W]
            target_size: Output heatmap size
            
        Returns:
            Attention heatmap [H, W]
        """
        self.attentions = []
        self._register_hooks()
        
        self.model.eval()
        with torch.no_grad():
            _ = self.model(image)
        
        self._remove_hooks()
        
        if not self.attentions:
            return np.zeros(target_size)
        
        # Fuse heads for each layer
        fused_attentions = [self._fuse_heads(attn) for attn in self.attentions]
        
        # Rollout across layers
        rollout = self._rollout_attention(fused_attentions)
        
        # Extract CLS token attention (or average all tokens)
        mask = rollout[0, 0, 1:].reshape(-1)  # Skip CLS token
        
        # Reshape to spatial dimensions
        grid_size = int(np.sqrt(mask.shape[0]))
        mask = mask.reshape(grid_size, grid_size).numpy()
        
        # Resize to target size
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_CUBIC)
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        
        return mask
    
    def generate_stage_rollout(self, image: torch.Tensor, stage: int = -1) -> np.ndarray:
        """Generate attention rollout for specific Swin stage
        
        Args:
            image: Input tensor
            stage: Stage index (0-3 for Swin-Base), -1 for all stages
        """
        self.attentions = []
        self._register_hooks()
        
        self.model.eval()
        with torch.no_grad():
            _ = self.model(image)
        
        self._remove_hooks()
        
        # Filter attentions by stage if specified
        if stage >= 0 and stage < 4:
            # Approximate stage boundaries (depends on Swin architecture)
            stage_boundaries = [0, 2, 4, 18, 20]  # For Swin-Base
            start, end = stage_boundaries[stage], stage_boundaries[stage + 1]
            stage_attentions = self.attentions[start:end]
        else:
            stage_attentions = self.attentions
        
        fused = [self._fuse_heads(attn) for attn in stage_attentions]
        rollout = self._rollout_attention(fused)
        
        mask = rollout[0, 0, 1:].reshape(-1)
        grid_size = int(np.sqrt(mask.shape[0]))
        mask = mask.reshape(grid_size, grid_size).numpy()
        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_CUBIC)
        
        return (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
