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
    Macenko stain normalization (Macenko et al., 2009).

    Algorithm:
    1. Convert RGB to Optical Density: OD = -log((I + 1) / 255)
    2. Filter background (low-OD pixels — they carry no stain information)
    3. SVD on UNCENTERED OD; take the plane spanned by the top-2 eigenvectors
    4. Project OD pixels onto that plane and compute each pixel's angle
    5. Stain vectors = the angular extremes (alpha-th and (100-alpha)-th
       percentile angles), reconstructed from angle to 3D OD space. This
       guarantees the stain vectors live in the positive OD orthant — H and E
       are absorption directions and must have non-negative components.
    6. Solve OD = concentrations @ stain_matrix for concentrations
    7. Rescale source concentrations so their per-stain 99th percentile matches
       the target's, then reconstruct RGB using the TARGET stain matrix.

    Why not centered SVD: SVD components on centered data have arbitrary sign
    (an SVD ambiguity), and using them directly as stain vectors produces 38%
    catastrophic green-channel blow-ups on BreakHis. The angular-extreme step
    here is what makes the result orientation-stable.
    """

    # Sensible H&E priors used as a fallback when an image has no tissue.
    _H_E_FALLBACK = np.array(
        [
            [0.5626, 0.7201, 0.4062],  # Hematoxylin (blue-purple)
            [0.2159, 0.8012, 0.5581],  # Eosin       (pink)
        ]
    )

    def __init__(self, od_threshold: float = 0.15, angle_percentile: float = 1.0):
        self.target_stain_matrix: Optional[np.ndarray] = None
        self.target_max_conc: Optional[np.ndarray] = None
        self.od_threshold = od_threshold
        self.angle_percentile = angle_percentile  # uses (alpha, 100-alpha)

    def fit(self, target_image: np.ndarray) -> None:
        """Learn the stain profile from a reference image."""
        target_od = self._rgb_to_od(target_image)
        self.target_stain_matrix = self._extract_stain_matrix(target_od)
        conc = self._get_concentrations(target_od, self.target_stain_matrix)
        self.target_max_conc = np.percentile(conc, 99, axis=0)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize an image to match the fitted target."""
        if self.target_stain_matrix is None:
            raise ValueError("Call fit() with a reference image first.")

        h, w = image.shape[:2]
        od = self._rgb_to_od(image)

        source_stain_matrix = self._extract_stain_matrix(od)
        source_conc = self._get_concentrations(od, source_stain_matrix)
        source_max_conc = np.maximum(np.percentile(source_conc, 99, axis=0), 1e-6)

        normalized_conc = source_conc * (self.target_max_conc / source_max_conc)
        # Reconstruct OD using TARGET stain directions: that's how every image
        # gets re-coloured to look like the reference's stain palette.
        normalized_od = normalized_conc @ self.target_stain_matrix
        return self._od_to_rgb(normalized_od).reshape(h, w, 3)

    # --- internal helpers ---

    @staticmethod
    def _rgb_to_od(image: np.ndarray) -> np.ndarray:
        # +1 (not max(.,1)) keeps the transform smoothly invertible at I=0.
        img = image.reshape(-1, 3).astype(np.float64) + 1.0
        return -np.log(img / 256.0)

    @staticmethod
    def _od_to_rgb(od: np.ndarray) -> np.ndarray:
        # Clip OD before exp to avoid overflow on extreme reconstructions.
        rgb = 256.0 * np.exp(-np.clip(od, 0.0, None)) - 1.0
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def _extract_stain_matrix(self, od_flat: np.ndarray) -> np.ndarray:
        """Macenko 2009 stain-vector extraction via angular extremes in OD plane."""
        tissue = od_flat[od_flat.sum(axis=1) > self.od_threshold]
        if tissue.shape[0] < 10:
            return self._H_E_FALLBACK.copy()

        # Eigendecomposition of the OD Gram matrix. eigh returns ascending
        # eigenvalues; take the top 2 as the dominant OD subspace.
        cov = (tissue.T @ tissue) / tissue.shape[0]
        _, eigvecs = np.linalg.eigh(cov)
        plane = eigvecs[:, [-1, -2]].copy()  # (3, 2)

        # Eigenvector signs are arbitrary. Force the FIRST basis vector into
        # the positive OD orthant so that pixel projections land in the +x
        # half-plane and arctan2 angles don't wrap across ±π. This is the step
        # the original implementation skipped, causing the green-blowup bug.
        if plane[:, 0].sum() < 0:
            plane[:, 0] *= -1
        if plane[:, 1].sum() < 0:
            plane[:, 1] *= -1

        # Project tissue OD onto the plane and compute angles.
        proj = tissue @ plane  # (N, 2)
        angles = np.arctan2(proj[:, 1], proj[:, 0])

        a = self.angle_percentile
        min_angle = np.percentile(angles, a)
        max_angle = np.percentile(angles, 100 - a)

        # Map each extreme angle back into 3D OD space.
        v_min = plane @ np.array([np.cos(min_angle), np.sin(min_angle)])
        v_max = plane @ np.array([np.cos(max_angle), np.sin(max_angle)])

        # Defensive: if any reconstructed stain vector ended up negative-summed
        # (rare, only when angle extremes still straddle the orthant), flip it.
        if v_min.sum() < 0:
            v_min = -v_min
        if v_max.sum() < 0:
            v_max = -v_max

        # Conventional ordering: hematoxylin first. H is the more blue-leaning
        # stain — in OD-space its blue (channel 2) component is larger.
        if v_min[2] > v_max[2]:
            stain_matrix = np.stack([v_min, v_max])
        else:
            stain_matrix = np.stack([v_max, v_min])

        # Unit-normalise so concentration scale is well-defined.
        norms = np.linalg.norm(stain_matrix, axis=1, keepdims=True)
        return stain_matrix / np.maximum(norms, 1e-12)

    @staticmethod
    def _get_concentrations(od_flat: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
        """Solve OD = C @ S for C (least squares). Shape: (N, 2)."""
        return np.linalg.lstsq(stain_matrix.T, od_flat.T, rcond=None)[0].T


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
