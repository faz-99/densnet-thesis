"""
Cross-method consistency metrics.

Measures the agreement between different XAI heatmaps using:
  - Pearson correlation
  - Spearman rank correlation
  - Structural Similarity (SSIM)
  - Top-k overlap (Jaccard)
"""
import numpy as np
import cv2
from itertools import combinations
from typing import Dict, List, Optional
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity as ssim
import pandas as pd


def _normalise(hm: np.ndarray) -> np.ndarray:
    mn, mx = hm.min(), hm.max()
    if mx > mn:
        return (hm - mn) / (mx - mn + 1e-8)
    return np.zeros_like(hm)


def _resize_to(hm: np.ndarray, shape: tuple) -> np.ndarray:
    if hm.shape != shape:
        return cv2.resize(hm, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return hm


def pearson_overlap(hm1: np.ndarray, hm2: np.ndarray) -> float:
    h1 = _normalise(hm1).flatten()
    h2 = _normalise(hm2).flatten()
    if np.std(h1) < 1e-8 or np.std(h2) < 1e-8:
        return 0.0
    r, _ = pearsonr(h1, h2)
    return float(r) if not np.isnan(r) else 0.0


def spearman_overlap(hm1: np.ndarray, hm2: np.ndarray) -> float:
    h1 = _normalise(hm1).flatten()
    h2 = _normalise(hm2).flatten()
    r, _ = spearmanr(h1, h2)
    return float(r) if not np.isnan(r) else 0.0


def ssim_overlap(hm1: np.ndarray, hm2: np.ndarray) -> float:
    h1 = _normalise(hm1)
    h2 = _normalise(_resize_to(hm2, h1.shape))
    return float(ssim(h1, h2, data_range=1.0))


def topk_jaccard(hm1: np.ndarray, hm2: np.ndarray, k: float = 0.2) -> float:
    """Jaccard overlap of top-k% pixels."""
    h1 = _normalise(hm1).flatten()
    h2 = _normalise(hm2).flatten()
    n = max(1, int(len(h1) * k))
    top1 = set(np.argsort(h1)[-n:])
    top2 = set(np.argsort(h2)[-n:])
    inter = len(top1 & top2)
    union = len(top1 | top2)
    return inter / union if union > 0 else 0.0


class CrossMethodConsistency:
    """
    Compute pairwise consistency between XAI methods.
    """

    METRICS = ['pearson', 'spearman', 'ssim', 'topk_jaccard']

    def compute(self,
                heatmaps: Dict[str, np.ndarray],
                metrics: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Args:
            heatmaps: {method_name: heatmap (H,W)}
            metrics: subset of METRICS to compute

        Returns:
            results[metric][pair_key] = score
        """
        if metrics is None:
            metrics = self.METRICS

        methods = list(heatmaps.keys())
        results = {m: {} for m in metrics}

        for m1, m2 in combinations(methods, 2):
            hm1 = heatmaps[m1]
            hm2 = heatmaps[m2]
            if hm1 is None or hm2 is None:
                continue
            key = f"{m1} vs {m2}"

            if 'pearson' in metrics:
                results['pearson'][key] = pearson_overlap(hm1, hm2)
            if 'spearman' in metrics:
                results['spearman'][key] = spearman_overlap(hm1, hm2)
            if 'ssim' in metrics:
                results['ssim'][key] = ssim_overlap(hm1, hm2)
            if 'topk_jaccard' in metrics:
                results['topk_jaccard'][key] = topk_jaccard(hm1, hm2)

        return results

    def aggregate(self, per_image_results: List[Dict]) -> pd.DataFrame:
        """
        Aggregate pairwise consistency scores across multiple images.

        Args:
            per_image_results: list of dicts from compute()

        Returns:
            DataFrame with mean ± std per metric per pair
        """
        if not per_image_results:
            return pd.DataFrame()

        # Collect all pair keys
        all_pairs = set()
        for r in per_image_results:
            for metric_dict in r.values():
                all_pairs.update(metric_dict.keys())

        rows = []
        for pair in sorted(all_pairs):
            row = {'Pair': pair}
            for metric in self.METRICS:
                vals = [r.get(metric, {}).get(pair, np.nan)
                        for r in per_image_results]
                vals = [v for v in vals if not np.isnan(v)]
                row[f'{metric}_mean'] = round(np.mean(vals), 4) if vals else np.nan
                row[f'{metric}_std']  = round(np.std(vals), 4)  if vals else np.nan
            rows.append(row)

        return pd.DataFrame(rows)
