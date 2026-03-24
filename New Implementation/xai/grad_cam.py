"""
Grad-CAM for the ConvNeXt branch of the ensemble.
"""
import torch
import torch.nn.functional as F
import numpy as np


class GradCAM:
    """Gradient-weighted Class Activation Mapping for ConvNeXt branch.

    Hooks into the last convolutional stage to compute spatial importance maps.
    """

    def __init__(self, model, target_layer=None):
        """
        Args:
            model: The full HybridEnsemble or a module with convnext branch.
            target_layer: nn.Module to hook (defaults to ConvNeXt last stage).
        """
        self.model = model
        self.model.eval()

        if target_layer is None:
            target_layer = model.get_convnext_target_layer()

        self.target_layer = target_layer
        self._activations = None
        self._gradients = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    @torch.enable_grad()
    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """Generate Grad-CAM heatmap.

        Args:
            input_tensor: (1, 3, H, W) input image tensor.
            target_class: Class index to explain (default: predicted class).

        Returns:
            heatmap: (H, W) numpy array in [0, 1].
        """
        self.model.zero_grad()
        input_tensor = input_tensor.requires_grad_(True)
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[0, target_class]
        score.backward(retain_graph=True)

        if self._gradients is None or self._activations is None:
            raise RuntimeError("Hooks did not fire. Check target_layer.")

        # Global average pooling of gradients → channel weights
        weights = self._gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def __del__(self):
        try:
            self.remove_hooks()
        except Exception:
            pass
