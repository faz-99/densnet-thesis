"""Attention-based GradCAM for Swin Transformer - Class-discriminative explanations"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional


class SwinAttentionGradCAM:
    """Combine attention weights with gradients for class-specific visualization"""
    
    def __init__(self, model: nn.Module, target_layer: Optional[str] = None):
        self.model = model
        self.target_layer = target_layer
        self.gradients = []
        self.activations = []
        self.attention_weights = []
        self.hooks = []
        
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations.append(output.detach())
            if hasattr(module, 'attn'):
                self.attention_weights.append(module.attn.detach())
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients.append(grad_output[0].detach())
        
        # Hook attention layers
        for name, module in self.model.named_modules():
            if 'attn' in name and hasattr(module, 'qkv'):
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_full_backward_hook(backward_hook))
    
    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate_cam(self, image: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Generate class-discriminative attention map
        
        Args:
            image: Input tensor [1, 3, H, W]
            target_class: Target class index (None for predicted class)
            
        Returns:
            CAM heatmap [H, W]
        """
        self.gradients = []
        self.activations = []
        self.attention_weights = []
        
        self._register_hooks()
        
        self.model.eval()
        image.requires_grad = True
        
        # Forward pass
        output = self.model(image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        self._remove_hooks()
        
        if not self.gradients or not self.attention_weights:
            return np.zeros((224, 224))
        
        # Combine gradients with attention weights
        cam = torch.zeros_like(self.attention_weights[0][0, 0])
        
        for grad, attn in zip(self.gradients, self.attention_weights):
            # Weight attention by gradient magnitude
            grad_weights = grad.abs().mean(dim=(0, 1))  # [num_heads]
            weighted_attn = (attn * grad_weights.view(-1, 1, 1)).sum(dim=1)  # [B, N, N]
            cam += weighted_attn[0].mean(dim=0)  # Average over query tokens
        
        # Process CAM
        cam = F.relu(cam)
        cam = cam[1:]  # Remove CLS token
        
        # Reshape to spatial
        grid_size = int(np.sqrt(cam.shape[0]))
        cam = cam.reshape(grid_size, grid_size).cpu().numpy()
        
        # Resize and normalize
        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def generate_multi_scale_cam(self, image: torch.Tensor, target_class: Optional[int] = None) -> dict:
        """Generate CAMs at multiple Swin stages
        
        Returns:
            Dictionary with stage-wise CAMs
        """
        self.gradients = []
        self.activations = []
        self.attention_weights = []
        
        self._register_hooks()
        
        self.model.eval()
        image.requires_grad = True
        
        output = self.model(image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        output[0, target_class].backward()
        
        self._remove_hooks()
        
        # Group by stages (approximate for Swin-Base)
        stage_boundaries = [0, 2, 4, 18, 20]
        stage_cams = {}
        
        for stage_idx in range(4):
            start, end = stage_boundaries[stage_idx], stage_boundaries[stage_idx + 1]
            stage_grads = self.gradients[start:end]
            stage_attns = self.attention_weights[start:end]
            
            if not stage_grads:
                continue
            
            cam = torch.zeros_like(stage_attns[0][0, 0])
            for grad, attn in zip(stage_grads, stage_attns):
                grad_weights = grad.abs().mean(dim=(0, 1))
                weighted_attn = (attn * grad_weights.view(-1, 1, 1)).sum(dim=1)
                cam += weighted_attn[0].mean(dim=0)
            
            cam = F.relu(cam)[1:]
            grid_size = int(np.sqrt(cam.shape[0]))
            cam = cam.reshape(grid_size, grid_size).cpu().numpy()
            cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            stage_cams[f'stage_{stage_idx}'] = cam
        
        return stage_cams
