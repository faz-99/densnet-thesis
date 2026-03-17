"""
Counterfactual explanation generator for histopathology classification.

Strategy: gradient-based pixel perturbation that minimally modifies the input
image to flip the model's prediction to a target class (or the nearest
alternative class). Produces a "what would need to change" explanation.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple


class CounterfactualExplainer:
    """
    Generates counterfactual explanations via iterative gradient ascent.

    For each input image the explainer finds the minimal perturbation δ such
    that model(x + δ) predicts a different class, then returns |δ| as the
    counterfactual importance map.
    """

    def __init__(self, model: torch.nn.Module, device,
                 lr: float = 0.01, max_steps: int = 200,
                 l1_lambda: float = 0.1, l2_lambda: float = 0.01):
        self.model = model
        self.device = device
        self.lr = lr
        self.max_steps = max_steps
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def generate(self, image: torch.Tensor,
                 source_class: int,
                 target_class: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """
        Generate counterfactual explanation.

        Args:
            image: Input tensor (1, C, H, W) on self.device
            source_class: Original predicted class
            target_class: Desired target class; if None, uses the second-best class

        Returns:
            (importance_map, metadata)
            importance_map: (H, W) float32 in [0,1], high = region that must change
        """
        self.model.eval()

        # Determine target class
        if target_class is None:
            with torch.no_grad():
                logits = self.model(image)
                probs = F.softmax(logits, dim=1)[0]
                probs[source_class] = -1.0
                target_class = int(probs.argmax().item())

        # Initialise perturbation δ
        delta = torch.zeros_like(image, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([delta], lr=self.lr)

        converged = False
        step = 0
        for step in range(self.max_steps):
            optimizer.zero_grad()
            perturbed = torch.clamp(image + delta, -3.0, 3.0)
            logits = self.model(perturbed)

            # Loss: maximise target class probability + sparsity regularisation
            target_loss = -F.log_softmax(logits, dim=1)[0, target_class]
            l1_reg = self.l1_lambda * delta.abs().mean()
            l2_reg = self.l2_lambda * (delta ** 2).mean()
            loss = target_loss + l1_reg + l2_reg
            loss.backward()
            optimizer.step()

            # Check if prediction flipped
            with torch.no_grad():
                pred = self.model(torch.clamp(image + delta, -3.0, 3.0)).argmax(dim=1).item()
            if pred == target_class:
                converged = True
                break

        # Build importance map from |δ|
        delta_np = delta.detach().squeeze(0).cpu().numpy()          # (C, H, W)
        importance = np.sum(np.abs(delta_np), axis=0)               # (H, W)
        if importance.max() > importance.min():
            importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)

        metadata = {
            'source_class': source_class,
            'target_class': target_class,
            'converged': converged,
            'steps': step + 1,
            'final_delta_l1': float(np.abs(delta_np).mean()),
        }
        return importance.astype(np.float32), metadata
