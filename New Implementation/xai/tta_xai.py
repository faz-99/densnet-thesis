"""
Test-Time Augmentation (TTA) for XAI Stability Evaluation.

Implements the Augment → Explain → Inverse Transform → Compare loop
to prove that XAI explanations are rotation/flip invariant.

Also provides Rotation-Averaged Ensemble XAI for noise-reduced
"Consensus Heatmaps".

References:
    - Sundararajan et al., Axiomatic Attribution for Deep Networks, 2017
    - Samek et al., Evaluating the Visualization of What a DNN Has Learned, 2017
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Callable, Optional, Tuple
from scipy.stats import pearsonr

from config.settings import XAI_OUTPUT_DIR


# ──────────────────────────────────────────────
# Augmentation definitions
# ──────────────────────────────────────────────

class TTAAugmentation:
    """A geometric augmentation with its corresponding inverse for heatmaps."""

    def __init__(self, name: str, forward_fn, inverse_fn):
        self.name = name
        self.forward_fn = forward_fn   # tensor (1,C,H,W) → tensor (1,C,H,W)
        self.inverse_fn = inverse_fn   # np.ndarray (H,W) → np.ndarray (H,W)


def _get_default_augmentations() -> List[TTAAugmentation]:
    """Return the 6 standard TTA augmentations: original + 3 rotations + 2 flips."""
    return [
        TTAAugmentation(
            name="original",
            forward_fn=lambda x: x,
            inverse_fn=lambda h: h,
        ),
        TTAAugmentation(
            name="rot90",
            forward_fn=lambda x: torch.rot90(x, k=1, dims=[2, 3]),
            inverse_fn=lambda h: np.rot90(h, k=-1),  # undo: rotate back
        ),
        TTAAugmentation(
            name="rot180",
            forward_fn=lambda x: torch.rot90(x, k=2, dims=[2, 3]),
            inverse_fn=lambda h: np.rot90(h, k=-2),
        ),
        TTAAugmentation(
            name="rot270",
            forward_fn=lambda x: torch.rot90(x, k=3, dims=[2, 3]),
            inverse_fn=lambda h: np.rot90(h, k=-3),
        ),
        TTAAugmentation(
            name="hflip",
            forward_fn=lambda x: torch.flip(x, dims=[3]),
            inverse_fn=lambda h: np.fliplr(h),
        ),
        TTAAugmentation(
            name="vflip",
            forward_fn=lambda x: torch.flip(x, dims=[2]),
            inverse_fn=lambda h: np.flipud(h),
        ),
    ]


# ──────────────────────────────────────────────
# Similarity metrics
# ──────────────────────────────────────────────

def ssim_2d(img1: np.ndarray, img2: np.ndarray,
            C1: float = 1e-4, C2: float = 9e-4) -> float:
    """Structural Similarity Index (SSIM) between two 2D heatmaps.

    Both inputs should be in [0, 1].
    """
    mu1, mu2 = img1.mean(), img2.mean()
    sigma1_sq = img1.var()
    sigma2_sq = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

    return float(numerator / denominator)


def pearson_correlation(h1: np.ndarray, h2: np.ndarray) -> float:
    """Pearson correlation between two flattened heatmaps."""
    flat1, flat2 = h1.flatten(), h2.flatten()
    if flat1.std() < 1e-8 or flat2.std() < 1e-8:
        return 0.0
    r, _ = pearsonr(flat1, flat2)
    return float(r)


# ──────────────────────────────────────────────
# Core TTA-XAI Engine
# ──────────────────────────────────────────────

class TTAExplainabilityEngine:
    """Test-Time Augmentation engine for XAI stability evaluation.

    Implements:
        1. Augment-Explain-Inverse-Compare loop for stability scoring
        2. Rotation-Averaged Ensemble XAI for consensus heatmaps
        3. Stability curves (stability vs. augmentation type)

    Usage:
        engine = TTAExplainabilityEngine(xai_manager)
        results = engine.evaluate_stability(input_tensor, target_class, methods=["grad_cam"])
        consensus = engine.consensus_heatmap(input_tensor, target_class, method="grad_cam")
    """

    def __init__(self, xai_manager, augmentations: List[TTAAugmentation] = None):
        """
        Args:
            xai_manager: InterpretabilityManager instance with loaded explainers.
            augmentations: Custom list of TTAAugmentation. Default: 6 standard transforms.
        """
        self.xai_manager = xai_manager
        self.augmentations = augmentations or _get_default_augmentations()

    # ────────────────────────────────────────
    # 1. Augment → Explain → Inverse → Compare
    # ────────────────────────────────────────

    def _explain_single(
        self, input_tensor: torch.Tensor, target_class: int, method: str
    ) -> np.ndarray:
        """Run a single XAI method and return heatmap."""
        result = self.xai_manager.explain(input_tensor, target_class, methods=[method])
        return result.get(method, np.zeros((input_tensor.shape[2], input_tensor.shape[3])))

    def augment_explain_inverse(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        method: str,
    ) -> Dict[str, np.ndarray]:
        """Run the full Augment → Explain → Inverse cycle.

        Returns:
            dict mapping augmentation name → inverse-transformed heatmap
            (all aligned back to the original orientation).
        """
        aligned_heatmaps = {}

        for aug in self.augmentations:
            # 1. Augment the input
            augmented_input = aug.forward_fn(input_tensor)

            # 2. Explain the augmented input
            heatmap = self._explain_single(augmented_input, target_class, method)

            # 3. Inverse-transform the heatmap back to original orientation
            aligned = aug.inverse_fn(heatmap.copy())
            aligned = np.ascontiguousarray(aligned).astype(np.float32)

            # Renormalize after inverse transform
            a_min, a_max = aligned.min(), aligned.max()
            if a_max - a_min > 1e-8:
                aligned = (aligned - a_min) / (a_max - a_min)

            aligned_heatmaps[aug.name] = aligned

        return aligned_heatmaps

    # ────────────────────────────────────────
    # 2. Stability Score
    # ────────────────────────────────────────

    def evaluate_stability(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        methods: List[str] = None,
        metric: str = "ssim",
    ) -> Dict[str, Dict]:
        """Evaluate XAI stability under TTA for each method.

        Stability = (1/N) * Σ Similarity(H_orig, Inverse(H_aug_i))

        Args:
            input_tensor: (1, 3, H, W)
            target_class: class to explain
            methods: list of XAI method names (default: all available)
            metric: "ssim" or "pearson"

        Returns:
            Dict[method_name → {
                "stability_score": float,
                "per_augmentation": {aug_name: similarity},
                "aligned_heatmaps": {aug_name: np.ndarray},
            }]
        """
        if methods is None:
            methods = list(self.xai_manager._explainers.keys())

        sim_fn = ssim_2d if metric == "ssim" else pearson_correlation

        results = {}
        for method in methods:
            print(f"  [TTA] Evaluating stability: {method}...")
            aligned = self.augment_explain_inverse(input_tensor, target_class, method)

            original_hm = aligned.get("original")
            if original_hm is None:
                continue

            per_aug = {}
            for aug_name, hm in aligned.items():
                if aug_name == "original":
                    per_aug[aug_name] = 1.0  # self-similarity
                else:
                    per_aug[aug_name] = sim_fn(original_hm, hm)

            stability = float(np.mean(list(per_aug.values())))

            results[method] = {
                "stability_score": stability,
                "per_augmentation": per_aug,
                "aligned_heatmaps": aligned,
            }

        return results

    # ────────────────────────────────────────
    # 3. Consensus Heatmap (Ensemble XAI Fix)
    # ────────────────────────────────────────

    def consensus_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        method: str,
    ) -> np.ndarray:
        """Generate a Rotation-Averaged Consensus Heatmap.

        Averages inverse-transformed heatmaps from all augmentations
        to reduce explanation noise and increase localization accuracy.

        Returns:
            consensus: (H, W) numpy array in [0, 1]
        """
        aligned = self.augment_explain_inverse(input_tensor, target_class, method)

        # Stack and average all aligned heatmaps
        heatmap_stack = np.stack(list(aligned.values()), axis=0)  # (N, H, W)
        consensus = heatmap_stack.mean(axis=0)

        # Renormalize
        c_min, c_max = consensus.min(), consensus.max()
        if c_max - c_min > 1e-8:
            consensus = (consensus - c_min) / (c_max - c_min)

        return consensus.astype(np.float32)

    def consensus_all_methods(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        methods: List[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Generate consensus heatmaps for multiple methods."""
        if methods is None:
            methods = list(self.xai_manager._explainers.keys())

        results = {}
        for method in methods:
            print(f"  [TTA] Generating consensus heatmap: {method}...")
            results[method] = self.consensus_heatmap(input_tensor, target_class, method)

        return results

    # ────────────────────────────────────────
    # 4. Visualization
    # ────────────────────────────────────────

    def visualize_stability(
        self,
        input_tensor: torch.Tensor,
        stability_results: Dict[str, Dict],
        method: str,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Visualize the TTA heatmaps and stability scores for one method.

        Shows: [Original | Rot90 | Rot180 | Rot270 | HFlip | VFlip | Consensus]
        With similarity scores annotated.
        """
        from data.preprocessing import denormalize

        result = stability_results.get(method)
        if result is None:
            raise ValueError(f"No stability results for method '{method}'")

        aligned = result["aligned_heatmaps"]
        per_aug = result["per_augmentation"]

        # Denormalize original image
        img = denormalize(input_tensor.squeeze(0)).cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # HWC

        aug_order = ["original", "rot90", "rot180", "rot270", "hflip", "vflip"]
        available = [a for a in aug_order if a in aligned]

        ncols = len(available) + 1  # +1 for consensus
        fig, axes = plt.subplots(2, ncols, figsize=(3.5 * ncols, 7))

        # Row 1: Individual aligned heatmaps
        for i, aug_name in enumerate(available):
            axes[0, i].imshow(img)
            axes[0, i].imshow(aligned[aug_name], cmap="jet", alpha=0.5)
            sim_val = per_aug.get(aug_name, 0)
            label = aug_name.replace("_", " ").title()
            axes[0, i].set_title(f"{label}\nSim: {sim_val:.3f}", fontsize=9)
            axes[0, i].axis("off")

        # Consensus in last column of row 1
        consensus = np.stack(list(aligned.values()), axis=0).mean(axis=0)
        c_min, c_max = consensus.min(), consensus.max()
        if c_max - c_min > 1e-8:
            consensus = (consensus - c_min) / (c_max - c_min)
        axes[0, -1].imshow(img)
        axes[0, -1].imshow(consensus, cmap="jet", alpha=0.5)
        axes[0, -1].set_title(f"Consensus\n(Averaged)", fontsize=9)
        axes[0, -1].axis("off")

        # Row 2: Stability bar chart
        ax_bar = plt.subplot(2, 1, 2)
        aug_names = [a for a in available if a != "original"]
        sim_values = [per_aug[a] for a in aug_names]
        colors = plt.cm.RdYlGn([v for v in sim_values])
        bars = ax_bar.bar(
            [a.replace("_", " ").title() for a in aug_names],
            sim_values, color=colors, edgecolor="black", linewidth=0.5,
        )
        ax_bar.axhline(y=result["stability_score"], color="red",
                       linestyle="--", label=f"Mean Stability: {result['stability_score']:.3f}")
        ax_bar.set_ylabel("Similarity to Original")
        ax_bar.set_title(f"TTA Stability: {method.replace('_', ' ').title()}", fontsize=11)
        ax_bar.set_ylim(0, 1.05)
        ax_bar.legend(loc="lower right")

        # Annotate bars
        for bar, val in zip(bars, sim_values):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def visualize_consensus_comparison(
        self,
        input_tensor: torch.Tensor,
        original_heatmaps: Dict[str, np.ndarray],
        consensus_heatmaps: Dict[str, np.ndarray],
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Side-by-side comparison: Original XAI vs Consensus XAI for each method.

        Useful for showing noise reduction from the ensemble approach.
        """
        from data.preprocessing import denormalize

        img = denormalize(input_tensor.squeeze(0)).cpu().numpy()
        img = np.transpose(img, (1, 2, 0))

        methods = list(consensus_heatmaps.keys())
        ncols = len(methods)

        fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 8))
        if ncols == 1:
            axes = axes.reshape(2, 1)

        for j, method in enumerate(methods):
            label = method.replace("_", " ").title()

            # Row 1: Original
            axes[0, j].imshow(img)
            if method in original_heatmaps:
                axes[0, j].imshow(original_heatmaps[method], cmap="jet", alpha=0.5)
            axes[0, j].set_title(f"{label}\n(Original)", fontsize=9)
            axes[0, j].axis("off")

            # Row 2: Consensus
            axes[1, j].imshow(img)
            axes[1, j].imshow(consensus_heatmaps[method], cmap="jet", alpha=0.5)
            axes[1, j].set_title(f"{label}\n(Consensus)", fontsize=9)
            axes[1, j].axis("off")

        axes[0, 0].set_ylabel("Original XAI", fontsize=11, labelpad=10)
        axes[1, 0].set_ylabel("Consensus XAI", fontsize=11, labelpad=10)

        fig.suptitle("Original vs. Rotation-Averaged Consensus Heatmaps", fontsize=13, y=1.02)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_stability_curve(
        self,
        stability_results: Dict[str, Dict],
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Plot Stability Score vs. XAI Method (or vs. Augmentation Type).

        Creates a grouped bar chart showing per-augmentation similarity
        for each XAI method.
        """
        methods = list(stability_results.keys())
        aug_names = ["rot90", "rot180", "rot270", "hflip", "vflip"]

        n_methods = len(methods)
        n_augs = len(aug_names)
        x = np.arange(n_augs)
        width = 0.8 / n_methods

        fig, ax = plt.subplots(figsize=(12, 5))

        for i, method in enumerate(methods):
            per_aug = stability_results[method]["per_augmentation"]
            values = [per_aug.get(a, 0) for a in aug_names]
            label = method.replace("_", " ").title()
            ax.bar(x + i * width, values, width, label=label, edgecolor="black", linewidth=0.3)

        ax.set_xlabel("Augmentation")
        ax.set_ylabel("Similarity to Original (SSIM / Pearson)")
        ax.set_title("TTA-XAI Stability Curve: Similarity vs. Augmentation", fontsize=12)
        ax.set_xticks(x + width * (n_methods - 1) / 2)
        ax.set_xticklabels([a.replace("_", " ").title() for a in aug_names])
        ax.set_ylim(0, 1.1)
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3)

        # Add mean stability line per method
        for i, method in enumerate(methods):
            score = stability_results[method]["stability_score"]
            ax.axhline(y=score, color=f"C{i}", linestyle="--", alpha=0.5, linewidth=0.8)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig
