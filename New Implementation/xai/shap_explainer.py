"""
SHAP (DeepExplainer) for model-agnostic cross-verification.
"""
import torch
import numpy as np
import shap


class SHAPExplainer:
    """SHAP DeepExplainer wrapper for the ensemble model."""

    def __init__(self, model, background_data: torch.Tensor):
        """
        Args:
            model: The ensemble model.
            background_data: (N, 3, H, W) background reference images.
        """
        self.model = model
        self.model.eval()

        # DeepExplainer needs a batch of reference (background) images
        self.explainer = shap.DeepExplainer(model, background_data)

    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """Compute SHAP attribution heatmap.

        Args:
            input_tensor: (1, 3, H, W) input image tensor.
            target_class: Class index to explain (default: predicted class).

        Returns:
            heatmap: (H, W) numpy array in [0, 1].
        """
        if target_class is None:
            with torch.no_grad():
                pred = self.model(input_tensor)
            target_class = pred.argmax(dim=1).item()

        shap_values = self.explainer.shap_values(input_tensor)

        # shap_values is a list of arrays (one per class)
        if isinstance(shap_values, list):
            sv = shap_values[target_class]
        else:
            sv = shap_values

        # Aggregate across channels
        sv = np.array(sv).squeeze(0)  # (3, H, W)
        attr_map = np.abs(sv).sum(axis=0)   # (H, W)

        # Normalize
        a_min, a_max = attr_map.min(), attr_map.max()
        if a_max - a_min > 1e-8:
            attr_map = (attr_map - a_min) / (a_max - a_min)
        else:
            attr_map = np.zeros_like(attr_map)
        return attr_map
