"""
Per-method Fidelity evaluation: Insertion AUC and Deletion AUC.

Produces:
  - Insertion / Deletion curves per method
  - AUC summary table
  - Comparative plots
  - Best-method identification
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import auc as sklearn_auc


# ------------------------------------------------------------------ #
# Core curve computation                                               #
# ------------------------------------------------------------------ #

def _sorted_pixel_indices(heatmap: np.ndarray) -> np.ndarray:
    """Return flat pixel indices sorted by importance (descending)."""
    return np.argsort(heatmap.flatten())[::-1]


def _get_confidence(model, image: torch.Tensor, class_idx: int,
                    device) -> float:
    model.eval()
    with torch.no_grad():
        out = model(image)
        return F.softmax(out, dim=1)[0, class_idx].item()


def compute_insertion_curve(model, image: torch.Tensor, heatmap: np.ndarray,
                             class_idx: int, device,
                             num_steps: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Insertion curve: start from blurred baseline, progressively insert
    most-important pixels from the original image.

    Returns:
        x_vals: fraction of pixels inserted (0..1)
        y_vals: model confidence at each step
    """
    H, W = image.shape[2], image.shape[3]
    total_pixels = H * W

    # Blurred baseline
    img_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    blurred = cv2.GaussianBlur(img_np, (51, 51), 10.0)
    baseline = torch.from_numpy(blurred.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    # Resize heatmap
    hm = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)
    sorted_idx = _sorted_pixel_indices(hm)

    current = baseline.clone()
    y_vals = []
    pixels_per_step = max(1, total_pixels // num_steps)

    for step in range(num_steps + 1):
        n_inserted = min(step * pixels_per_step, total_pixels)
        if step > 0:
            idx = sorted_idx[:n_inserted]
            rows, cols = np.unravel_index(idx, (H, W))
            for c in range(image.shape[1]):
                current[0, c, rows, cols] = image[0, c, rows, cols]
        y_vals.append(_get_confidence(model, current, class_idx, device))

    x_vals = np.linspace(0, 1, len(y_vals))
    return x_vals, np.array(y_vals)


def compute_deletion_curve(model, image: torch.Tensor, heatmap: np.ndarray,
                            class_idx: int, device,
                            num_steps: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deletion curve: start from original image, progressively replace
    most-important pixels with blurred values.

    Returns:
        x_vals: fraction of pixels deleted (0..1)
        y_vals: model confidence at each step
    """
    H, W = image.shape[2], image.shape[3]
    total_pixels = H * W

    img_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    blurred = cv2.GaussianBlur(img_np, (51, 51), 10.0)
    replacement = torch.from_numpy(blurred.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    hm = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)
    sorted_idx = _sorted_pixel_indices(hm)

    current = image.clone()
    y_vals = []
    pixels_per_step = max(1, total_pixels // num_steps)

    for step in range(num_steps + 1):
        n_deleted = min(step * pixels_per_step, total_pixels)
        if step > 0:
            idx = sorted_idx[:n_deleted]
            rows, cols = np.unravel_index(idx, (H, W))
            for c in range(image.shape[1]):
                current[0, c, rows, cols] = replacement[0, c, rows, cols]
        y_vals.append(_get_confidence(model, current, class_idx, device))

    x_vals = np.linspace(0, 1, len(y_vals))
    return x_vals, np.array(y_vals)


# ------------------------------------------------------------------ #
# Per-method batch evaluator                                           #
# ------------------------------------------------------------------ #

class FidelityEvaluator:
    """
    Computes Insertion AUC and Deletion AUC for each XAI method
    over a batch of images.
    """

    def __init__(self, model, device, num_steps: int = 50):
        self.model = model
        self.device = device
        self.num_steps = num_steps

    def evaluate(self,
                 images: List[torch.Tensor],
                 heatmaps_by_method: Dict[str, List[np.ndarray]],
                 class_indices: List[int]) -> Dict[str, Dict]:
        """
        Args:
            images: list of (1,C,H,W) tensors
            heatmaps_by_method: {method_name: [heatmap_per_image]}
            class_indices: predicted class per image

        Returns:
            results[method] = {
                'insertion_auc': float,
                'deletion_auc': float,
                'insertion_curves': list of (x,y) arrays,
                'deletion_curves': list of (x,y) arrays,
            }
        """
        results = {}

        for method, heatmaps in heatmaps_by_method.items():
            ins_aucs, del_aucs = [], []
            ins_curves, del_curves = [], []

            for img, hm, cls in zip(images, heatmaps, class_indices):
                if hm is None:
                    continue
                img = img.to(self.device)

                x_ins, y_ins = compute_insertion_curve(
                    self.model, img, hm, cls, self.device, self.num_steps)
                x_del, y_del = compute_deletion_curve(
                    self.model, img, hm, cls, self.device, self.num_steps)

                ins_aucs.append(sklearn_auc(x_ins, y_ins))
                del_aucs.append(sklearn_auc(x_del, y_del))
                ins_curves.append((x_ins, y_ins))
                del_curves.append((x_del, y_del))

            results[method] = {
                'insertion_auc': float(np.mean(ins_aucs)) if ins_aucs else 0.0,
                'deletion_auc':  float(np.mean(del_aucs)) if del_aucs else 0.0,
                'insertion_auc_std': float(np.std(ins_aucs)) if ins_aucs else 0.0,
                'deletion_auc_std':  float(np.std(del_aucs)) if del_aucs else 0.0,
                'insertion_curves': ins_curves,
                'deletion_curves':  del_curves,
            }

        return results

    # ------------------------------------------------------------------ #
    # Plotting                                                             #
    # ------------------------------------------------------------------ #
    def plot_curves(self, results: Dict[str, Dict],
                    save_dir: str = 'results/fidelity') -> None:
        """
        Save insertion curves, deletion curves, and AUC summary bar chart.
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        methods = list(results.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

        # --- Insertion curves ---
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, color in zip(methods, colors):
            curves = results[method]['insertion_curves']
            if not curves:
                continue
            mean_y = np.mean([y for _, y in curves], axis=0)
            x = curves[0][0]
            auc_val = results[method]['insertion_auc']
            ax.plot(x, mean_y, label=f"{method} (AUC={auc_val:.3f})", color=color)
        ax.set_xlabel('Fraction of pixels inserted')
        ax.set_ylabel('Model confidence')
        ax.set_title('Insertion Curves (higher = better faithfulness)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{save_dir}/insertion_curves.png", dpi=150)
        plt.close(fig)

        # --- Deletion curves ---
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, color in zip(methods, colors):
            curves = results[method]['deletion_curves']
            if not curves:
                continue
            mean_y = np.mean([y for _, y in curves], axis=0)
            x = curves[0][0]
            auc_val = results[method]['deletion_auc']
            ax.plot(x, mean_y, label=f"{method} (AUC={auc_val:.3f})", color=color)
        ax.set_xlabel('Fraction of pixels deleted')
        ax.set_ylabel('Model confidence')
        ax.set_title('Deletion Curves (lower = better faithfulness)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{save_dir}/deletion_curves.png", dpi=150)
        plt.close(fig)

        # --- AUC summary bar chart ---
        ins_vals = [results[m]['insertion_auc'] for m in methods]
        del_vals = [results[m]['deletion_auc']  for m in methods]

        x = np.arange(len(methods))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width/2, ins_vals, width, label='Insertion AUC ↑', alpha=0.8)
        ax.bar(x + width/2, del_vals, width, label='Deletion AUC ↓', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha='right')
        ax.set_ylabel('AUC')
        ax.set_title('Fidelity AUC Summary')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(f"{save_dir}/auc_summary.png", dpi=150)
        plt.close(fig)

    def summary_table(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """Return a DataFrame with Insertion/Deletion AUC per method."""
        rows = []
        for method, vals in results.items():
            rows.append({
                'Method': method,
                'Insertion AUC': round(vals['insertion_auc'], 4),
                'Insertion AUC std': round(vals['insertion_auc_std'], 4),
                'Deletion AUC': round(vals['deletion_auc'], 4),
                'Deletion AUC std': round(vals['deletion_auc_std'], 4),
                'Fidelity Score': round(
                    vals['insertion_auc'] + (1 - vals['deletion_auc']), 4),
            })
        df = pd.DataFrame(rows).sort_values('Fidelity Score', ascending=False)
        return df

    def best_method(self, results: Dict[str, Dict]) -> str:
        """Return the method with the highest combined fidelity score."""
        df = self.summary_table(results)
        return df.iloc[0]['Method']
