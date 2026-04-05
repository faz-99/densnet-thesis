"""
Stain normalization for H&E histopathology images.

Implements two methods used in the ablation study:
  - Macenko: Optical-density based stain separation via SVD
  - Reinhard: LAB color-space statistics transfer

Reference papers:
  Macenko et al. (2009) - "A method for normalizing histology slides for quantitative analysis"
  Reinhard et al. (2001) - "Color transfer between images"
"""
import numpy as np
import cv2
from typing import Optional


class MacenkoNormalizer:
    """
    Macenko stain normalization.

    How it works (simplified):
    1. Convert RGB pixels to Optical Density (OD) space: OD = -log(I / 255)
    2. Use SVD to find the two principal stain directions (Hematoxylin & Eosin)
    3. Project pixels onto those directions to get stain concentrations
    4. Rescale concentrations to match a target (reference) image
    5. Reconstruct the normalized RGB image
    """

    def __init__(self):
        self.target_stain_matrix = None
        self.target_max_conc = None

    def fit(self, target_image: np.ndarray):
        """Learn the stain profile from a reference image."""
        target_od = self._rgb_to_od(target_image)
        self.target_stain_matrix = self._extract_stain_matrix(target_od)
        conc = self._get_concentrations(target_od, self.target_stain_matrix)
        self.target_max_conc = np.percentile(conc, 99, axis=0)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize an image to match the fitted target."""
        if self.target_stain_matrix is None:
            raise ValueError("Call fit() with a reference image first.")

        od = self._rgb_to_od(image)
        h, w = image.shape[:2]

        source_stain_matrix = self._extract_stain_matrix(od)
        source_conc = self._get_concentrations(od, source_stain_matrix)
        source_max_conc = np.percentile(source_conc, 99, axis=0)

        # Avoid division by zero
        source_max_conc = np.maximum(source_max_conc, 1e-6)

        # Rescale concentrations
        normalized_conc = source_conc * (self.target_max_conc / source_max_conc)

        # Reconstruct in OD space using target stain directions
        normalized_od = normalized_conc @ self.target_stain_matrix[:2]
        normalized_rgb = self._od_to_rgb(normalized_od).reshape(h, w, 3)

        return normalized_rgb

    # --- internal helpers ---

    @staticmethod
    def _rgb_to_od(image: np.ndarray) -> np.ndarray:
        img = image.reshape(-1, 3).astype(np.float64)
        img = np.maximum(img, 1.0)
        return -np.log(img / 255.0)

    @staticmethod
    def _od_to_rgb(od: np.ndarray) -> np.ndarray:
        rgb = 255.0 * np.exp(-od)
        return np.clip(rgb, 0, 255).astype(np.uint8)

    @staticmethod
    def _extract_stain_matrix(od_flat: np.ndarray) -> np.ndarray:
        """Extract the 2-component stain matrix via SVD."""
        # Keep only tissue pixels (filter out background)
        tissue_mask = np.sum(od_flat, axis=1) > 0.15
        od_tissue = od_flat[tissue_mask]

        if od_tissue.shape[0] < 10:
            # Fallback: return default H&E stain vectors
            return np.array([
                [0.6442, 0.0928, 0.6339],  # Hematoxylin
                [0.0927, 0.9545, 0.2837],  # Eosin
            ])

        # SVD to find principal stain directions
        od_centered = od_tissue - od_tissue.mean(axis=0)
        _, _, Vt = np.linalg.svd(od_centered, full_matrices=False)
        stain_matrix = Vt[:2]  # top-2 components

        # Ensure consistent orientation (H should have higher blue OD)
        if stain_matrix[0, 2] < stain_matrix[1, 2]:
            stain_matrix = stain_matrix[[1, 0]]

        return stain_matrix

    @staticmethod
    def _get_concentrations(od_flat: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
        """Project OD values onto stain directions to get concentrations."""
        return np.linalg.lstsq(stain_matrix[:2].T, od_flat.T, rcond=None)[0].T


class ReinhardNormalizer:
    """
    Reinhard color normalization.

    How it works:
    1. Convert both target and source images to LAB color space
    2. Compute mean and std of each LAB channel for the target
    3. For each source image, shift and scale its LAB channels
       to match the target's statistics
    4. Convert back to RGB
    """

    def __init__(self):
        self.target_means = None
        self.target_stds = None

    def fit(self, target_image: np.ndarray):
        """Learn color statistics from a reference image."""
        lab = cv2.cvtColor(target_image, cv2.COLOR_RGB2LAB).astype(np.float64)
        self.target_means = lab.reshape(-1, 3).mean(axis=0)
        self.target_stds = lab.reshape(-1, 3).std(axis=0)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize an image to match the fitted target."""
        if self.target_means is None:
            raise ValueError("Call fit() with a reference image first.")

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float64)
        src_means = lab.reshape(-1, 3).mean(axis=0)
        src_stds = lab.reshape(-1, 3).std(axis=0)

        for ch in range(3):
            if src_stds[ch] > 1e-6:
                lab[:, :, ch] = (
                    (lab[:, :, ch] - src_means[ch])
                    * (self.target_stds[ch] / src_stds[ch])
                    + self.target_means[ch]
                )

        lab = np.clip(lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def get_normalizer(method: str) -> Optional[object]:
    """Factory function to create a stain normalizer by name."""
    if method == "macenko":
        return MacenkoNormalizer()
    elif method == "reinhard":
        return ReinhardNormalizer()
    elif method == "none":
        return None
    else:
        raise ValueError(f"Unknown stain normalization method: {method}")
