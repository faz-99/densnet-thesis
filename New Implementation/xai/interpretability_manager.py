"""
Centralized InterpretabilityManager: orchestrates all XAI methods and
produces the 6-column visualization panel.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional

from config.settings import XAI_CONFIG, XAI_OUTPUT_DIR, DATASET_CONFIG
from data.preprocessing import denormalize
from xai.tta_xai import TTAExplainabilityEngine


class InterpretabilityManager:
    """Centralised manager that generates, stores, and visualises explanations
    from all six XAI methods.

    Methods:
        1. Grad-CAM            (gradient-based, ConvNeXt)
        2. Integrated Gradients (gradient-based, axiom-complete)
        3. SHAP DeepExplainer   (model-agnostic)
        4. LIME                 (model-agnostic, superpixel)
        5. Attention Rollout    (structural, Swin)
        6. Counterfactual       (clinical logic)
    """

    def __init__(self, model, background_data: torch.Tensor = None):
        """
        Args:
            model: HybridEnsemble model (eval mode).
            background_data: (N, 3, H, W) reference images for SHAP.
        """
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        self.background_data = background_data
        self._explainers = {}
        self._init_explainers()

    def _init_explainers(self):
        from xai.grad_cam import GradCAM
        from xai.integrated_gradients import IntegratedGradientsExplainer
        from xai.attention_rollout import AttentionRollout
        from xai.counterfactual import CounterfactualExplainer

        cfg = XAI_CONFIG

        self._explainers["grad_cam"] = GradCAM(self.model)
        self._explainers["integrated_gradients"] = IntegratedGradientsExplainer(
            self.model,
            n_steps=cfg["integrated_gradients"]["n_steps"],
            internal_batch_size=cfg["integrated_gradients"]["internal_batch_size"],
        )
        self._explainers["attention_rollout"] = AttentionRollout(
            self.model,
            head_fusion=cfg["attention_rollout"]["head_fusion"],
            discard_ratio=cfg["attention_rollout"]["discard_ratio"],
        )
        self._explainers["counterfactual"] = CounterfactualExplainer(self.model)

        # SHAP and LIME are lazily initialized (need background data / are slow)
        if self.background_data is not None:
            self._init_shap()

        # TTA engine
        self.tta_engine = TTAExplainabilityEngine(self)

    def _init_shap(self):
        from xai.shap_explainer import SHAPExplainer
        self._explainers["shap"] = SHAPExplainer(
            self.model, self.background_data.to(self.device)
        )

    def _init_lime(self):
        from xai.lime_explainer import LIMEExplainer
        cfg = XAI_CONFIG
        self._explainers["lime"] = LIMEExplainer(
            self.model,
            num_samples=cfg["lime"]["num_samples"],
            num_features=cfg["lime"]["num_features"],
        )

    def set_background_data(self, data: torch.Tensor):
        """Set/update SHAP background data and reinitialise."""
        self.background_data = data
        self._init_shap()

    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: int = None,
        methods: list = None,
    ) -> Dict[str, np.ndarray]:
        """Run selected XAI methods and return heatmaps.

        Args:
            input_tensor: (1, 3, H, W) on the correct device.
            target_class: Class to explain (None = predicted).
            methods: Subset of method names. Default: all available.

        Returns:
            dict mapping method name → (H, W) heatmap in [0, 1].
        """
        if methods is None:
            methods = list(self._explainers.keys())

        # Lazy init LIME if needed
        if "lime" in methods and "lime" not in self._explainers:
            self._init_lime()

        results = {}
        for name in methods:
            if name not in self._explainers:
                continue
            try:
                heatmap = self._explainers[name].generate(
                    input_tensor.to(self.device), target_class
                )
                results[name] = heatmap
            except Exception as e:
                print(f"[InterpretabilityManager] {name} failed: {e}")
                H, W = input_tensor.shape[2], input_tensor.shape[3]
                results[name] = np.zeros((H, W), dtype=np.float32)

        return results

    def explain_with_counterfactual_details(
        self, input_tensor: torch.Tensor, target_class: int = None
    ) -> dict:
        """Run counterfactual explainer and return rich details."""
        cf = self._explainers.get("counterfactual")
        if cf is None:
            return {}
        return cf.generate_with_details(input_tensor.to(self.device), target_class)

    # ── TTA: Test-Time Augmentation XAI ──

    def evaluate_tta_stability(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        methods: list = None,
        metric: str = None,
    ) -> Dict:
        """Evaluate XAI stability under geometric augmentations (TTA).

        Returns per-method stability scores and per-augmentation similarities.
        """
        metric = metric or XAI_CONFIG.get("tta", {}).get("similarity_metric", "ssim")
        return self.tta_engine.evaluate_stability(
            input_tensor, target_class, methods=methods, metric=metric,
        )

    def get_consensus_heatmaps(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        methods: list = None,
    ) -> Dict[str, np.ndarray]:
        """Generate Rotation-Averaged Consensus Heatmaps (noise-reduced).

        Thesis argument: 'By using a Rotation-Averaged XAI Ensemble,
        I reduced explanation noise by X% and increased localization accuracy.'
        """
        return self.tta_engine.consensus_all_methods(
            input_tensor, target_class, methods=methods,
        )

    def visualize_tta(
        self,
        input_tensor: torch.Tensor,
        stability_results: Dict,
        method: str,
        save_path: str = None,
        show: bool = False,
    ) -> plt.Figure:
        """Visualize TTA stability for one method."""
        return self.tta_engine.visualize_stability(
            input_tensor, stability_results, method,
            save_path=save_path, show=show,
        )

    def visualize_consensus_comparison(
        self,
        input_tensor: torch.Tensor,
        original_heatmaps: Dict[str, np.ndarray],
        consensus_heatmaps: Dict[str, np.ndarray],
        save_path: str = None,
        show: bool = False,
    ) -> plt.Figure:
        """Side-by-side Original vs Consensus heatmaps."""
        return self.tta_engine.visualize_consensus_comparison(
            input_tensor, original_heatmaps, consensus_heatmaps,
            save_path=save_path, show=show,
        )

    # ── Visualization ──

    def visualize(
        self,
        input_tensor: torch.Tensor,
        heatmaps: Dict[str, np.ndarray],
        class_names: list = None,
        target_class: int = None,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Create the 6-column visualization panel.

        Columns: [Original | Grad-CAM | IG | SHAP | LIME | Attn Rollout | Counterfactual]
        """
        # Denormalize original image
        img = denormalize(input_tensor.squeeze(0)).detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # HWC

        method_titles = {
            "grad_cam": "Grad-CAM",
            "integrated_gradients": "Integrated\nGradients",
            "shap": "SHAP",
            "lime": "LIME",
            "attention_rollout": "Attention\nRollout",
            "counterfactual": "Counterfactual",
        }

        ordered = ["grad_cam", "integrated_gradients", "shap", "lime",
                    "attention_rollout", "counterfactual"]
        available = [m for m in ordered if m in heatmaps]

        ncols = 1 + len(available)
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
        if ncols == 1:
            axes = [axes]

        # Original
        axes[0].imshow(img)
        title = "Original"
        if target_class is not None and class_names:
            title += f"\n{class_names[target_class]}"
        axes[0].set_title(title, fontsize=10)
        axes[0].axis("off")

        # Heatmaps
        for i, name in enumerate(available, 1):
            axes[i].imshow(img)
            axes[i].imshow(heatmaps[name], cmap="jet", alpha=0.5)
            axes[i].set_title(method_titles.get(name, name), fontsize=10)
            axes[i].axis("off")

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig
