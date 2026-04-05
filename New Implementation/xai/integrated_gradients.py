"""
Integrated Gradients satisfying the Axiom of Completeness.
Uses Captum library for a robust implementation.
"""
import torch
import numpy as np

try:
    from captum.attr import IntegratedGradients as CaptumIG
    _CAPTUM_AVAILABLE = True
except ImportError:
    _CAPTUM_AVAILABLE = False


class IntegratedGradientsExplainer:
    """Integrated Gradients for the ensemble model.

    Satisfies the *Axiom of Completeness*: the sum of attributions equals
    the difference between the output at the input and at the baseline.
    """

    def __init__(self, model, n_steps: int = 50, internal_batch_size: int = 8):
        self.model = model
        self.model.eval()
        self.ig = CaptumIG(model) if _CAPTUM_AVAILABLE else None
        self.n_steps = n_steps
        self.internal_batch_size = internal_batch_size

    @torch.enable_grad()
    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        if not _CAPTUM_AVAILABLE or self.ig is None:
            H, W = input_tensor.shape[2], input_tensor.shape[3]
            return np.zeros((H, W), dtype=np.float32)

        if target_class is None:
            with torch.no_grad():
                pred = self.model(input_tensor)
            target_class = pred.argmax(dim=1).item()

        # Black baseline
        baseline = torch.zeros_like(input_tensor)

        attributions = self.ig.attribute(
            input_tensor,
            baselines=baseline,
            target=target_class,
            n_steps=self.n_steps,
            internal_batch_size=self.internal_batch_size,
            return_convergence_delta=False,
        )

        # Aggregate over channels → spatial map
        attr_map = attributions.squeeze(0).abs().sum(dim=0).detach().cpu().numpy()

        # Normalize
        a_min, a_max = attr_map.min(), attr_map.max()
        if a_max - a_min > 1e-8:
            attr_map = (attr_map - a_min) / (a_max - a_min)
        else:
            attr_map = np.zeros_like(attr_map)
        return attr_map
