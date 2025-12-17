"""
Layer-wise Relevance Propagation (LRP) for CNN explainability
Optional advanced explainability method for DenseNet models
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import cv2


class LRPDenseNet:
    """
    Layer-wise Relevance Propagation for DenseNet architectures
    Implements LRP-epsilon rule for CNN layers
    """
    
    def __init__(self, model, device, epsilon: float = 1e-6):
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.activations = {}
        self.relevances = {}
        
    def register_hooks(self):
        """Register forward hooks to capture activations"""
        def hook_fn(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        return hooks
    
    def lrp_linear(self, layer: nn.Linear, activation: torch.Tensor, 
                   relevance: torch.Tensor) -> torch.Tensor:
        """Apply LRP to linear layer"""
        weight = layer.weight.detach()
        bias = layer.bias.detach() if layer.bias is not None else None
        
        # Forward pass
        z = torch.mm(activation, weight.t())
        if bias is not None:
            z += bias
        
        # LRP-epsilon rule
        z += self.epsilon * torch.sign(z)
        s = relevance / z
        
        # Backward pass
        c = torch.mm(s, weight)
        relevance_input = activation * c
        
        return relevance_input
    
    def lrp_conv2d(self, layer: nn.Conv2d, activation: torch.Tensor,
                   relevance: torch.Tensor) -> torch.Tensor:
        """Apply LRP to convolutional layer (simplified)"""
        # This is a simplified LRP for conv layers
        # Full implementation would require more complex relevance propagation
        
        # Use gradient-based approximation for conv layers
        activation.requires_grad_(True)
        z = layer(activation)
        
        # Compute gradients
        z.backward(relevance, retain_graph=True)
        relevance_input = activation * activation.grad
        
        return relevance_input.detach()
    
    def generate_lrp_explanation(self, input_tensor: torch.Tensor,
                                target_class: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """
        Generate LRP explanation (simplified implementation)
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class index
            
        Returns:
            Tuple of (relevance map, metadata)
        """
        # This is a simplified LRP implementation
        # Full LRP requires layer-by-layer relevance propagation
        
        self.model.eval()
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Get target score
        target_score = output[0, target_class]
        
        # Compute gradients (as approximation to LRP)
        self.model.zero_grad()
        target_score.backward()
        
        # Use input * gradient as relevance approximation
        relevance = input_tensor.grad * input_tensor
        relevance_map = relevance.squeeze(0).cpu().numpy()
        
        # Sum across channels
        relevance_map = np.sum(relevance_map, axis=0)
        
        metadata = {
            'method': 'LRP (Simplified)',
            'target_class': target_class,
            'epsilon': self.epsilon,
            'relevance_shape': relevance_map.shape,
            'relevance_range': (float(relevance_map.min()), float(relevance_map.max()))
        }
        
        return relevance_map, metadata