"""
Stability/Robustness metric for explanation consistency under perturbations
Higher values indicate more stable explanations
"""
import torch
import numpy as np
import cv2
from typing import Tuple, List, Dict
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
import torch.nn.functional as F


class StabilityMetric:
    """Compute stability metrics for explanation robustness"""
    
    def __init__(self, device, num_perturbations: int = 10, noise_std: float = 0.1):
        self.device = device
        self.num_perturbations = num_perturbations
        self.noise_std = noise_std
    
    def add_gaussian_noise(self, image: torch.Tensor, std: float = None) -> torch.Tensor:
        """Add Gaussian noise to image"""
        if std is None:
            std = self.noise_std
        
        noise = torch.randn_like(image) * std
        perturbed = image + noise
        
        # Clamp to valid range (assuming normalized images)
        perturbed = torch.clamp(perturbed, -3, 3)  # Typical normalization range
        return perturbed
    
    def rotate_image(self, image: torch.Tensor, angle: float) -> torch.Tensor:
        """Rotate image by given angle (degrees)"""
        # Convert to numpy for rotation
        img_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        
        # Get rotation matrix
        h, w = img_np.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(img_np, rotation_matrix, (w, h), 
                               borderMode=cv2.BORDER_REFLECT)
        
        # Convert back to tensor
        rotated_tensor = torch.from_numpy(rotated.transpose(2, 0, 1)).unsqueeze(0).float()
        return rotated_tensor.to(self.device)
    
    def flip_image(self, image: torch.Tensor, horizontal: bool = True) -> torch.Tensor:
        """Flip image horizontally or vertically"""
        if horizontal:
            return torch.flip(image, dims=[3])  # Flip width dimension
        else:
            return torch.flip(image, dims=[2])  # Flip height dimension
    
    def generate_perturbations(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Generate various perturbations of the input image"""
        perturbations = []
        
        # Gaussian noise perturbations
        for i in range(self.num_perturbations // 3):
            noise_std = self.noise_std * (0.5 + i * 0.5)  # Varying noise levels
            perturbed = self.add_gaussian_noise(image, noise_std)
            perturbations.append(perturbed)
        
        # Rotation perturbations
        angles = np.linspace(-10, 10, self.num_perturbations // 3)
        for angle in angles:
            rotated = self.rotate_image(image, angle)
            perturbations.append(rotated)
        
        # Flip perturbations
        perturbations.append(self.flip_image(image, horizontal=True))
        perturbations.append(self.flip_image(image, horizontal=False))
        
        # Fill remaining slots with more noise perturbations
        while len(perturbations) < self.num_perturbations:
            perturbed = self.add_gaussian_noise(image)
            perturbations.append(perturbed)
        
        return perturbations[:self.num_perturbations]
    
    def compute_ssim(self, explanation1: np.ndarray, explanation2: np.ndarray) -> float:
        """Compute Structural Similarity Index between two explanations"""
        # Ensure same shape
        if explanation1.shape != explanation2.shape:
            explanation2 = cv2.resize(explanation2, 
                                    (explanation1.shape[1], explanation1.shape[0]))
        
        # Normalize to [0, 1]
        def normalize(arr):
            if arr.max() > arr.min():
                return (arr - arr.min()) / (arr.max() - arr.min())
            return arr
        
        exp1_norm = normalize(explanation1)
        exp2_norm = normalize(explanation2)
        
        # Compute SSIM
        ssim_score = ssim(exp1_norm, exp2_norm, data_range=1.0)
        return float(ssim_score)
    
    def compute_pearson_correlation(self, explanation1: np.ndarray, 
                                  explanation2: np.ndarray) -> float:
        """Compute Pearson correlation between two explanations"""
        # Flatten arrays
        exp1_flat = explanation1.flatten()
        exp2_flat = explanation2.flatten()
        
        # Ensure same length
        min_len = min(len(exp1_flat), len(exp2_flat))
        exp1_flat = exp1_flat[:min_len]
        exp2_flat = exp2_flat[:min_len]
        
        # Compute correlation
        if np.std(exp1_flat) == 0 or np.std(exp2_flat) == 0:
            return 1.0 if np.array_equal(exp1_flat, exp2_flat) else 0.0
        
        correlation, _ = pearsonr(exp1_flat, exp2_flat)
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def compute_cosine_similarity(self, explanation1: np.ndarray, 
                                explanation2: np.ndarray) -> float:
        """Compute cosine similarity between two explanations"""
        exp1_flat = explanation1.flatten()
        exp2_flat = explanation2.flatten()
        
        # Ensure same length
        min_len = min(len(exp1_flat), len(exp2_flat))
        exp1_flat = exp1_flat[:min_len]
        exp2_flat = exp2_flat[:min_len]
        
        # Compute cosine similarity
        dot_product = np.dot(exp1_flat, exp2_flat)
        norm1 = np.linalg.norm(exp1_flat)
        norm2 = np.linalg.norm(exp2_flat)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0 if np.array_equal(exp1_flat, exp2_flat) else 0.0
        
        cosine_sim = dot_product / (norm1 * norm2)
        return float(cosine_sim)
    
    def evaluate_stability(self, image: torch.Tensor, explanation_generator,
                         target_class: int, similarity_metric: str = 'ssim') -> Tuple[float, List[float]]:
        """
        Evaluate stability of explanations under perturbations
        
        Args:
            image: Original image tensor
            explanation_generator: Function that generates explanations
            target_class: Target class for explanation
            similarity_metric: 'ssim', 'pearson', or 'cosine'
            
        Returns:
            Tuple of (mean stability score, individual similarity scores)
        """
        # Generate original explanation
        original_explanation = explanation_generator(image, target_class)
        
        # Generate perturbations
        perturbations = self.generate_perturbations(image)
        
        similarity_scores = []
        
        for perturbed_image in perturbations:
            try:
                # Generate explanation for perturbed image
                perturbed_explanation = explanation_generator(perturbed_image, target_class)
                
                # Compute similarity
                if similarity_metric == 'ssim':
                    similarity = self.compute_ssim(original_explanation, perturbed_explanation)
                elif similarity_metric == 'pearson':
                    similarity = self.compute_pearson_correlation(original_explanation, perturbed_explanation)
                elif similarity_metric == 'cosine':
                    similarity = self.compute_cosine_similarity(original_explanation, perturbed_explanation)
                else:
                    raise ValueError(f"Unknown similarity metric: {similarity_metric}")
                
                similarity_scores.append(similarity)
                
            except Exception as e:
                print(f"Warning: Failed to compute explanation for perturbation: {e}")
                continue
        
        # Compute mean stability
        mean_stability = np.mean(similarity_scores) if similarity_scores else 0.0
        
        return mean_stability, similarity_scores
    
    def evaluate_batch_stability(self, images: List[torch.Tensor], 
                               explanation_generator, target_classes: List[int],
                               similarity_metric: str = 'ssim') -> Tuple[float, List[float]]:
        """Evaluate stability for a batch of images"""
        stability_scores = []
        
        for img, target in zip(images, target_classes):
            stability, _ = self.evaluate_stability(img, explanation_generator, 
                                                 target, similarity_metric)
            stability_scores.append(stability)
        
        mean_stability = np.mean(stability_scores)
        return mean_stability, stability_scores
    
    def comprehensive_stability_analysis(self, image: torch.Tensor, 
                                       explanation_generator, target_class: int) -> Dict[str, float]:
        """
        Perform comprehensive stability analysis using multiple similarity metrics
        
        Returns:
            Dictionary with stability scores for different metrics
        """
        results = {}
        
        metrics = ['ssim', 'pearson', 'cosine']
        for metric in metrics:
            try:
                stability, _ = self.evaluate_stability(image, explanation_generator, 
                                                     target_class, metric)
                results[f'stability_{metric}'] = stability
            except Exception as e:
                print(f"Warning: Failed to compute {metric} stability: {e}")
                results[f'stability_{metric}'] = 0.0
        
        # Compute average stability across metrics
        stability_values = [v for v in results.values() if v > 0]
        results['stability_average'] = np.mean(stability_values) if stability_values else 0.0
        
        return results