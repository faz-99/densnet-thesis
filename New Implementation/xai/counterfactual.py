"""
Counterfactual Explanations: adversarial-lite perturbations showing the
minimum tissue change required to flip a diagnosis (e.g., Benign → Malignant).
"""
import torch
import torch.nn.functional as F
import numpy as np
from config.settings import XAI_CONFIG


class CounterfactualExplainer:
    """Generate counterfactual explanations via constrained optimisation.

    Finds the minimal perturbation δ such that the model flips its
    prediction from the original class to a target class, subject to
    an L1 sparsity penalty so δ highlights the most salient tissue region.
    """

    def __init__(
        self,
        model,
        max_iter: int = None,
        lr: float = None,
        lambda_l1: float = None,
        target_confidence: float = None,
    ):
        cfg = XAI_CONFIG["counterfactual"]
        self.model = model
        self.model.eval()
        self.max_iter = max_iter or cfg["max_iter"]
        self.lr = lr or cfg["lr"]
        self.lambda_l1 = lambda_l1 or cfg["lambda_l1"]
        self.target_confidence = target_confidence or cfg["target_confidence"]

    @torch.enable_grad()
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int = None,
    ) -> np.ndarray:
        """Generate a counterfactual perturbation heatmap.

        Args:
            input_tensor: (1, 3, H, W) input image tensor.
            target_class: Class to flip towards. If None, uses the
                          runner-up prediction.

        Returns:
            heatmap: (H, W) absolute perturbation magnitude in [0, 1].
        """
        device = input_tensor.device
        x = input_tensor.detach().clone()

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            original_class = probs.argmax(dim=1).item()

        if target_class is None:
            # Pick the runner-up class
            sorted_probs = probs.squeeze().argsort(descending=True)
            target_class = sorted_probs[1].item()

        # Learnable perturbation δ
        delta = torch.zeros_like(x, requires_grad=True, device=device)
        optimizer = torch.optim.Adam([delta], lr=self.lr)

        for i in range(self.max_iter):
            optimizer.zero_grad()
            perturbed = x + delta
            perturbed = perturbed.clamp(-3, 3)  # stay within reasonable normalized range

            logits = self.model(perturbed)
            probs = F.softmax(logits, dim=1)

            # Loss: push target class probability up + L1 sparsity on delta
            target_prob = probs[0, target_class]
            loss = -torch.log(target_prob + 1e-8) + self.lambda_l1 * delta.abs().mean()
            loss.backward()
            optimizer.step()

            if target_prob.item() >= self.target_confidence:
                break

        # Heatmap: absolute perturbation magnitude summed over channels
        heatmap = delta.detach().abs().sum(dim=1).squeeze(0).cpu().numpy()

        # Normalize
        h_min, h_max = heatmap.min(), heatmap.max()
        if h_max - h_min > 1e-8:
            heatmap = (heatmap - h_min) / (h_max - h_min)
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap

    def generate_with_details(
        self,
        input_tensor: torch.Tensor,
        target_class: int = None,
    ) -> dict:
        """Extended version returning perturbation details for reporting."""
        device = input_tensor.device
        x = input_tensor.detach().clone()

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            original_class = probs.argmax(dim=1).item()
            original_conf = probs[0, original_class].item()

        if target_class is None:
            sorted_probs = probs.squeeze().argsort(descending=True)
            target_class = sorted_probs[1].item()

        delta = torch.zeros_like(x, requires_grad=True, device=device)
        optimizer = torch.optim.Adam([delta], lr=self.lr)

        converged = False
        final_conf = 0.0

        for i in range(self.max_iter):
            optimizer.zero_grad()
            perturbed = x + delta
            perturbed = perturbed.clamp(-3, 3)
            logits = self.model(perturbed)
            probs = F.softmax(logits, dim=1)
            target_prob = probs[0, target_class]
            loss = -torch.log(target_prob + 1e-8) + self.lambda_l1 * delta.abs().mean()
            loss.backward()
            optimizer.step()
            final_conf = target_prob.item()
            if final_conf >= self.target_confidence:
                converged = True
                break

        heatmap = delta.detach().abs().sum(dim=1).squeeze(0).cpu().numpy()
        h_min, h_max = heatmap.min(), heatmap.max()
        if h_max - h_min > 1e-8:
            heatmap = (heatmap - h_min) / (h_max - h_min)
        else:
            heatmap = np.zeros_like(heatmap)

        return {
            "heatmap": heatmap,
            "original_class": original_class,
            "original_confidence": original_conf,
            "target_class": target_class,
            "final_confidence": final_conf,
            "converged": converged,
            "iterations": i + 1,
            "l1_norm": delta.detach().abs().mean().item(),
        }
