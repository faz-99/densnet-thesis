"""
Unified XAI pipeline.

Wraps all explanation methods (Grad-CAM++, SHAP, LIME, Integrated Gradients,
Attention Rollout, Counterfactual) behind a single interface so that the
fidelity evaluator can call them uniformly.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Optional, List

from explainability.grad_cam import GradCAM, GradCAMPlusPlus
from explainability.integrated_gradients import IntegratedGradients
from explainability.counterfactual import CounterfactualExplainer


class XAIPipeline:
    """
    Unified interface for all XAI methods.

    Usage:
        pipeline = XAIPipeline(model, device, target_layer='...')
        heatmap = pipeline.explain('gradcam_plus', image, class_idx)
    """

    METHODS = ['gradcam', 'gradcam_plus', 'shap', 'lime',
               'integrated_gradients', 'attention_rollout', 'counterfactual']

    def __init__(self, model: torch.nn.Module, device,
                 target_layer: str = 'densenet.features.norm5',
                 model_type: str = 'denlsnet',
                 background_data: Optional[torch.Tensor] = None,
                 img_size: int = 224):
        self.model = model
        self.device = device
        self.target_layer = target_layer
        self.model_type = model_type
        self.img_size = img_size
        self._explainers: Dict = {}
        self._init_explainers(background_data)

    # ------------------------------------------------------------------ #
    # Initialisation                                                       #
    # ------------------------------------------------------------------ #
    def _init_explainers(self, background_data):
        # Grad-CAM
        try:
            self._explainers['gradcam'] = GradCAM(self.model, self.target_layer)
        except Exception as e:
            print(f"[XAI] Grad-CAM init failed: {e}")

        # Grad-CAM++
        try:
            self._explainers['gradcam_plus'] = GradCAMPlusPlus(self.model, self.target_layer)
        except Exception as e:
            print(f"[XAI] Grad-CAM++ init failed: {e}")

        # Integrated Gradients
        try:
            self._explainers['integrated_gradients'] = IntegratedGradients(self.model, self.device)
        except Exception as e:
            print(f"[XAI] IG init failed: {e}")

        # Counterfactual
        try:
            self._explainers['counterfactual'] = CounterfactualExplainer(self.model, self.device)
        except Exception as e:
            print(f"[XAI] Counterfactual init failed: {e}")

        # SHAP (optional – needs background data)
        if background_data is not None:
            try:
                from explainability.shap_explainer import SHAPExplainer
                self._explainers['shap'] = SHAPExplainer(
                    self.model, background_data, str(self.device))
            except Exception as e:
                print(f"[XAI] SHAP init failed: {e}")

        # LIME
        try:
            from explainability.lime_explainer import LIMEExplainer
            self._explainers['lime'] = LIMEExplainer(self.model, str(self.device))
        except Exception as e:
            print(f"[XAI] LIME init failed: {e}")

        # Attention Rollout (Swin only)
        if self.model_type in ('swin', 'swin_classifier'):
            try:
                from swin_explainability.attention_rollout import SwinAttentionRollout
                self._explainers['attention_rollout'] = SwinAttentionRollout(self.model)
            except Exception as e:
                print(f"[XAI] Attention Rollout init failed: {e}")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #
    def available_methods(self) -> List[str]:
        return list(self._explainers.keys())

    def explain(self, method: str, image: torch.Tensor,
                class_idx: int) -> Optional[np.ndarray]:
        """
        Generate a (H, W) float32 heatmap in [0,1].

        Args:
            method: one of METHODS
            image: (1, C, H, W) tensor on self.device
            class_idx: target class index

        Returns:
            Normalised heatmap or None on failure.
        """
        if method not in self._explainers:
            return None

        try:
            heatmap = self._dispatch(method, image, class_idx)
            if heatmap is None:
                return None
            heatmap = self._normalise(heatmap)
            heatmap = self._resize(heatmap)
            return heatmap.astype(np.float32)
        except Exception as e:
            print(f"[XAI] {method} explain failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Dispatch                                                             #
    # ------------------------------------------------------------------ #
    def _dispatch(self, method: str, image: torch.Tensor,
                  class_idx: int) -> Optional[np.ndarray]:
        exp = self._explainers[method]

        if method in ('gradcam', 'gradcam_plus'):
            return exp.generate_cam(image, class_idx)

        elif method == 'integrated_gradients':
            attr, _ = exp.generate_integrated_gradients(image, class_idx)
            return attr

        elif method == 'counterfactual':
            # Get source class from model
            with torch.no_grad():
                src = self.model(image).argmax(dim=1).item()
            importance, _ = exp.generate(image, src, class_idx)
            return importance

        elif method == 'shap':
            shap_vals = exp.explain_image(image, class_idx)
            if isinstance(shap_vals, np.ndarray):
                if shap_vals.ndim == 3:
                    return np.sum(np.abs(shap_vals), axis=0)
                return shap_vals
            return None

        elif method == 'lime':
            img_np = self._tensor_to_numpy(image)
            explanation, _ = exp.explain_image(img_np)
            temp, mask = explanation.get_image_and_mask(
                class_idx, positive_only=False, num_features=10, hide_rest=False)
            return mask.astype(np.float32)

        elif method == 'attention_rollout':
            return exp.generate_rollout(image, target_size=(self.img_size, self.img_size))

        return None

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #
    def _tensor_to_numpy(self, image: torch.Tensor) -> np.ndarray:
        """Convert (1,C,H,W) tensor to (H,W,C) numpy in [0,1]."""
        img = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.5613, 0.5778, 0.6032])
        std  = np.array([0.2114, 0.1957, 0.1590])
        img = img * std + mean
        return np.clip(img, 0, 1)

    @staticmethod
    def _normalise(heatmap: np.ndarray) -> np.ndarray:
        mn, mx = heatmap.min(), heatmap.max()
        if mx > mn:
            return (heatmap - mn) / (mx - mn + 1e-8)
        return np.zeros_like(heatmap)

    def _resize(self, heatmap: np.ndarray) -> np.ndarray:
        if heatmap.shape != (self.img_size, self.img_size):
            heatmap = cv2.resize(heatmap, (self.img_size, self.img_size),
                                 interpolation=cv2.INTER_LINEAR)
        return heatmap
