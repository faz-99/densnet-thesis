"""
Sparsity / Focus metric for XAI heatmaps.

Measures how concentrated the explanation is on a small region of the image.
High sparsity = explanation is focused on a few important pixels (good).
"""
import numpy as np
from typing import Dict


def gini_coefficient(heatmap: np.ndarray) -> float:
    """
    Gini coefficient of the heatmap values.
    0 = perfectly uniform, 1 = perfectly concentrated.
    """
    vals = np.abs(heatmap).flatten()
    vals = np.sort(vals)
    n = len(vals)
    if n == 0 or vals.sum() == 0:
        return 0.0
    cumsum = np.cumsum(vals)
    return float((2 * np.sum(cumsum) - (n + 1) * vals.sum()) / (n * vals.sum()))


def energy_concentration(heatmap: np.ndarray, top_k: float = 0.2) -> float:
    """
    Fraction of total energy (sum of absolute values) contained in the
    top-k% most important pixels.
    """
    vals = np.abs(heatmap).flatten()
    total = vals.sum()
    if total == 0:
        return 0.0
    n = max(1, int(len(vals) * top_k))
    top_vals = np.sort(vals)[-n:]
    return float(top_vals.sum() / total)


def effective_mass(heatmap: np.ndarray, threshold: float = 0.5) -> float:
    """
    Fraction of pixels above threshold (after normalisation to [0,1]).
    Lower = more focused.
    """
    vals = np.abs(heatmap)
    mx = vals.max()
    if mx == 0:
        return 1.0
    norm = vals / mx
    return float((norm >= threshold).mean())


def compute_sparsity(heatmap: np.ndarray) -> Dict[str, float]:
    """Return all sparsity metrics for a single heatmap."""
    return {
        'gini': gini_coefficient(heatmap),
        'energy_concentration_20pct': energy_concentration(heatmap, 0.20),
        'effective_mass_50pct': effective_mass(heatmap, 0.50),
    }
