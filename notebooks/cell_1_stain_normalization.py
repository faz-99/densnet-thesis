# Cell 1: Stain Normalization Implementation
import cv2
import numpy as np
from scipy.linalg import lstsq

class StainNormalizer:
    def __init__(self, method='macenko'):
        self.method = method
        
    def macenko_normalize(self, image):
        """Macenko stain normalization"""
        od = -np.log((image.astype(np.float64) + 1) / 256.0)
        od_hat = od[~np.any(od < 0.15, axis=2)]
        eigvals, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
        eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]
        that = od_hat.dot(eigvecs[:, :2])
        phi = np.arctan2(that[:, 1], that[:, 0])
        min_phi, max_phi = np.percentile(phi, [1, 99])
        v1 = eigvecs[:, :2].dot([np.cos(min_phi), np.sin(min_phi)])
        v2 = eigvecs[:, :2].dot([np.cos(max_phi), np.sin(max_phi)])
        he = np.array([v1, v2]) if v1[0] > v2[0] else np.array([v2, v1])
        he = he / np.linalg.norm(he, axis=1, keepdims=True)
        target_he = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])
        c = lstsq(he.T, od.reshape(-1, 3).T)[0]
        max_c = np.percentile(c, 99, axis=1, keepdims=True)
        c = c / max_c * np.percentile(target_he, 99, axis=1, keepdims=True).T
        normalized = np.exp(-target_he.T.dot(c)) * 255
        return np.clip(normalized.T.reshape(image.shape), 0, 255).astype(np.uint8)
    
    def reinhard_normalize(self, image):
        """Reinhard color normalization in LAB space"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float64)
        target_means = np.array([8.63234435, -0.11501964, 0.03868433])
        target_stds = np.array([0.57506023, 0.10403329, 0.01364062])
        means = np.mean(lab.reshape(-1, 3), axis=0)
        stds = np.std(lab.reshape(-1, 3), axis=0)
        lab = (lab - means) / stds * target_stds + target_means
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    
    def normalize(self, image):
        if self.method == 'macenko':
            return self.macenko_normalize(image)
        elif self.method == 'reinhard':
            return self.reinhard_normalize(image)
        return image

stain_normalizer = StainNormalizer(method=PREPROCESSING_CONFIG['stain_method'])
print(f"Stain normalizer initialized: {PREPROCESSING_CONFIG['stain_method']}")