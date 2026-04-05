"""
ValidationEngine: rigorous quantitative evaluation of XAI explanations
and generated clinical reports.

Metrics:
  1. Faithfulness  – Insertion AUC, Deletion AUC
  2. Robustness    – Stability Score under ε-noise
  3. Localization  – Localization AUC (vs. pathologist masks / pseudo-masks)
  4. Textual       – RadGraph F1, ROUGE-L
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import auc

from config.settings import EVAL_CONFIG


class ValidationEngine:
    """Quantitative evaluation of XAI heatmaps and generated reports."""

    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.model.eval()
        self.device = device

    # ────────────────────────────────────────
    # 1. FAITHFULNESS: Insertion / Deletion AUC
    # ────────────────────────────────────────

    @torch.no_grad()
    def insertion_auc(
        self,
        input_tensor: torch.Tensor,
        heatmap: np.ndarray,
        target_class: int,
        steps: int = None,
    ) -> float:
        """Insertion AUC: progressively add XAI-prioritised pixels.

        Higher is better — if the XAI map is faithful, adding the
        most-important pixels first should rapidly increase confidence.
        """
        steps = steps or EVAL_CONFIG["faithfulness"]["insertion_steps"]
        return self._pixel_perturbation_auc(
            input_tensor, heatmap, target_class, steps, mode="insertion"
        )

    @torch.no_grad()
    def deletion_auc(
        self,
        input_tensor: torch.Tensor,
        heatmap: np.ndarray,
        target_class: int,
        steps: int = None,
    ) -> float:
        """Deletion AUC: progressively remove XAI-prioritised pixels.

        Lower is better — if faithful, removing important pixels first
        should rapidly decrease confidence.
        """
        steps = steps or EVAL_CONFIG["faithfulness"]["deletion_steps"]
        return self._pixel_perturbation_auc(
            input_tensor, heatmap, target_class, steps, mode="deletion"
        )

    def _pixel_perturbation_auc(
        self, input_tensor, heatmap, target_class, steps, mode
    ) -> float:
        """Core pixel perturbation routine for insertion / deletion."""
        x = input_tensor.to(self.device)
        H, W = heatmap.shape
        n_pixels = H * W

        # Sort pixel indices by importance (descending)
        flat_importance = heatmap.flatten()
        sorted_idx = np.argsort(flat_importance)[::-1]

        # Step sizes
        pixels_per_step = max(1, n_pixels // steps)

        if mode == "insertion":
            # Start from a blank (mean-filled) canvas
            canvas = torch.zeros_like(x)
        else:
            canvas = x.clone()

        scores = []
        for step in range(steps + 1):
            logits = self.model(canvas)
            prob = F.softmax(logits, dim=1)[0, target_class].item()
            scores.append(prob)

            if step < steps:
                start = step * pixels_per_step
                end = min(start + pixels_per_step, n_pixels)
                idx = sorted_idx[start:end]
                rows, cols = np.unravel_index(idx, (H, W))

                if mode == "insertion":
                    canvas[:, :, rows, cols] = x[:, :, rows, cols]
                else:
                    canvas[:, :, rows, cols] = 0.0

        # Compute AUC
        x_axis = np.linspace(0, 1, len(scores))
        return float(auc(x_axis, scores))

    # ────────────────────────────────────────
    # 2. ROBUSTNESS: Stability Score
    # ────────────────────────────────────────

    def stability_score(
        self,
        input_tensor: torch.Tensor,
        explainer_fn,
        target_class: int,
        epsilon: float = None,
        n_perturbations: int = None,
    ) -> float:
        """Stability Score: heatmap consistency under ε-noise perturbations.

        Lower score = more stable/robust.

        Args:
            explainer_fn: callable(input_tensor, target_class) → heatmap np.ndarray
            epsilon: noise magnitude
            n_perturbations: number of noisy samples
        """
        cfg = EVAL_CONFIG["robustness"]
        epsilon = epsilon or cfg["epsilon"]
        n_perturbations = n_perturbations or cfg["n_perturbations"]

        x = input_tensor.to(self.device)
        original_heatmap = explainer_fn(x, target_class)

        differences = []
        for _ in range(n_perturbations):
            noise = torch.randn_like(x) * epsilon
            perturbed = x + noise
            noisy_heatmap = explainer_fn(perturbed, target_class)

            # L2 distance between heatmaps (normalised)
            diff = np.linalg.norm(original_heatmap - noisy_heatmap) / (
                np.linalg.norm(original_heatmap) + 1e-8
            )
            differences.append(diff)

        return float(np.mean(differences))

    # ────────────────────────────────────────
    # 3. LOCALIZATION: Localization AUC
    # ────────────────────────────────────────

    def localization_auc(
        self,
        heatmap: np.ndarray,
        ground_truth_mask: np.ndarray,
    ) -> float:
        """Localization AUC: compare XAI heatmap against ground-truth mask.

        If ground-truth pathologist annotations are unavailable, use
        pseudo-masks from generate_pseudo_mask().

        Returns:
            AUC score in [0, 1]. Higher = better localisation.
        """
        # Flatten both to 1D
        hm = heatmap.flatten()
        gt = ground_truth_mask.flatten().astype(np.float32)
        gt = (gt > 0.5).astype(np.float32)

        if gt.sum() == 0 or gt.sum() == len(gt):
            return 0.5  # degenerate case

        # Compute ROC-style pointwise AUC
        thresholds = np.linspace(0, 1, 100)
        tpr_list, fpr_list = [], []
        for t in thresholds:
            pred = (hm >= t).astype(np.float32)
            tp = (pred * gt).sum()
            fp = (pred * (1 - gt)).sum()
            fn = ((1 - pred) * gt).sum()
            tn = ((1 - pred) * (1 - gt)).sum()
            tpr = tp / (tp + fn + 1e-8)
            fpr = fp / (fp + tn + 1e-8)
            tpr_list.append(tpr)
            fpr_list.append(fpr)

        # Sort by FPR ascending
        sorted_pairs = sorted(zip(fpr_list, tpr_list))
        fpr_sorted = [p[0] for p in sorted_pairs]
        tpr_sorted = [p[1] for p in sorted_pairs]

        return float(auc(fpr_sorted, tpr_sorted))

    @staticmethod
    def generate_pseudo_mask(
        heatmap: np.ndarray,
        threshold_percentile: float = 75,
    ) -> np.ndarray:
        """Generate a pseudo ground-truth mask from a high-quality heatmap.

        Uses the top percentile of a reference XAI method (e.g., Grad-CAM)
        as a proxy for pathologist annotation.
        """
        threshold = np.percentile(heatmap, threshold_percentile)
        return (heatmap >= threshold).astype(np.float32)

    # ────────────────────────────────────────
    # 4. TEXTUAL: ROUGE-L and RadGraph F1
    # ────────────────────────────────────────

    def rouge_l(self, prediction: str, reference: str) -> float:
        """Compute ROUGE-L F1 score between prediction and reference text."""
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()

        if not pred_tokens or not ref_tokens:
            return 0.0

        # LCS computation (dynamic programming)
        m, n = len(ref_tokens), len(pred_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i - 1] == pred_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs_len = dp[m][n]

        precision = lcs_len / n if n > 0 else 0
        recall = lcs_len / m if m > 0 else 0
        if precision + recall == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)
        return float(f1)

    def radgraph_f1(
        self,
        prediction: str,
        reference: str,
        clinical_entities: Optional[List[str]] = None,
    ) -> float:
        """Simplified RadGraph F1: measure overlap of clinical entities.

        Uses a predefined set of clinical entities if none provided.
        Full RadGraph requires the radgraph library / model.
        """
        if clinical_entities is None:
            clinical_entities = [
                "adenosis", "fibroadenoma", "phyllodes", "tubular",
                "ductal", "lobular", "mucinous", "papillary",
                "carcinoma", "benign", "malignant", "tumor",
                "nuclei", "stroma", "glandular", "mitotic",
                "pleomorphic", "necrosis", "calcification",
                "hyperplasia", "atypia", "invasion", "margin",
            ]

        pred_lower = prediction.lower()
        ref_lower = reference.lower()

        pred_entities = {e for e in clinical_entities if e in pred_lower}
        ref_entities = {e for e in clinical_entities if e in ref_lower}

        if not ref_entities:
            return 1.0 if not pred_entities else 0.0
        if not pred_entities:
            return 0.0

        tp = len(pred_entities & ref_entities)
        precision = tp / len(pred_entities) if pred_entities else 0
        recall = tp / len(ref_entities) if ref_entities else 0

        if precision + recall == 0:
            return 0.0
        return float(2 * precision * recall / (precision + recall))

    # ────────────────────────────────────────
    # Comprehensive evaluation
    # ────────────────────────────────────────

    def evaluate_all(
        self,
        input_tensor: torch.Tensor,
        heatmaps: Dict[str, np.ndarray],
        target_class: int,
        ground_truth_mask: Optional[np.ndarray] = None,
        generated_report: Optional[str] = None,
        reference_report: Optional[str] = None,
    ) -> Dict:
        """Run all evaluation metrics on a single sample.

        Returns a nested dict of results.
        """
        results = {"faithfulness": {}, "robustness": {}, "localization": {}, "textual": {}}

        for method, hmap in heatmaps.items():
            # Faithfulness
            ins = self.insertion_auc(input_tensor, hmap, target_class)
            dele = self.deletion_auc(input_tensor, hmap, target_class)
            results["faithfulness"][method] = {
                "insertion_auc": ins,
                "deletion_auc": dele,
            }

            # Localization
            if ground_truth_mask is not None:
                loc = self.localization_auc(hmap, ground_truth_mask)
            else:
                # Use Grad-CAM as pseudo mask reference
                ref_map = heatmaps.get("grad_cam", hmap)
                pseudo = self.generate_pseudo_mask(ref_map)
                loc = self.localization_auc(hmap, pseudo)
            results["localization"][method] = {"localization_auc": loc}

        # Textual
        if generated_report and reference_report:
            results["textual"] = {
                "rouge_l": self.rouge_l(generated_report, reference_report),
                "radgraph_f1": self.radgraph_f1(generated_report, reference_report),
            }

        return results
