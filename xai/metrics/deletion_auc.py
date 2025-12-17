"""
Deletion AUC metric for measuring faithfulness of explanations
Lower values indicate better faithfulness (confidence should drop when important pixels are removed)
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, List
from sklearn.metrics import auc


class DeletionAUC:
    """Compute Deletion AUC metric for explanation faithfulness"""
    
    def __init__(self, model, device, num_steps: int = 50, blur_sigma: float = 10.0):
        self.model = model
        self.device = device
        self.num_steps = num_steps
        self.blur_sigma = blur_sigma
        
    def create_replacement_value(self, image: torch.Tensor, method: str = 'blur') -> torch.Tensor:
        """Create replacement values for deleted pixels"""
        if method == 'blur':
            # Use blurred version
            img_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            blurred = cv2.GaussianBlur(img_np, (51, 51), self.blur_sigma)
            replacement = torch.from_numpy(blurred.transpose(2, 0, 1)).unsqueeze(0).float()
            return replacement.to(self.device)
        elif method == 'mean':
            # Use mean pixel value
            mean_val = image.mean(dim=(2, 3), keepdim=True)
            return mean_val.expand_as(image)
        else:
            # Use zeros
            return torch.zeros_like(image)
    
    def get_top_k_pixels(self, explanation: np.ndarray, k: int) -> np.ndarray:
        """Get indices of top-k most important pixels"""
        flat_explanation = explanation.flatten()
        top_k_indices = np.argpartition(flat_explanation, -k)[-k:]
        
        # Convert flat indices to 2D coordinates
        h, w = explanation.shape
        coords = np.unravel_index(top_k_indices, (h, w))
        return coords
    
    def compute_deletion_auc(self, image: torch.Tensor, explanation: np.ndarray, 
                           target_class: int) -> Tuple[float, List[float]]:
        """
        Compute Deletion AUC by progressively removing pixels
        
        Args:
            image: Original image tensor (1, C, H, W)
            explanation: Explanation heatmap (H, W)
            target_class: Target class index
            
        Returns:
            Tuple of (AUC score, confidence scores list)
        """
        self.model.eval()
        
        # Create replacement values
        replacement = self.create_replacement_value(image, method='blur')
        
        # Resize explanation to match image spatial dimensions
        img_h, img_w = image.shape[2], image.shape[3]
        if explanation.shape != (img_h, img_w):
            explanation = cv2.resize(explanation, (img_w, img_h))
        
        # Initialize with original image
        current_image = image.clone()
        
        # Get total number of pixels
        total_pixels = img_h * img_w
        pixels_per_step = max(1, total_pixels // self.num_steps)
        
        confidence_scores = []
        
        # Initial confidence with original image
        with torch.no_grad():
            output = self.model(current_image)
            probs = F.softmax(output, dim=1)
            initial_conf = probs[0, target_class].item()
            confidence_scores.append(initial_conf)
        
        # Progressive deletion
        deleted_pixels = 0
        for step in range(self.num_steps):
            # Determine how many pixels to delete this step
            pixels_to_delete = min(pixels_per_step, total_pixels - deleted_pixels)
            if pixels_to_delete <= 0:
                break
            
            # Get top-k pixels for this step
            remaining_explanation = explanation.copy()
            
            # Mask already deleted pixels
            if deleted_pixels > 0:
                # Get previously deleted pixels and set their importance to 0
                top_deleted = self.get_top_k_pixels(explanation, deleted_pixels)
                remaining_explanation[top_deleted] = 0
            
            # Get next most important pixels to delete
            top_coords = self.get_top_k_pixels(remaining_explanation, pixels_to_delete)
            
            # Delete pixels (replace with blurred/mean values)
            for c in range(image.shape[1]):  # For each channel
                current_image[0, c, top_coords[0], top_coords[1]] = replacement[0, c, top_coords[0], top_coords[1]]
            
            deleted_pixels += pixels_to_delete
            
            # Measure confidence
            with torch.no_grad():
                output = self.model(current_image)
                probs = F.softmax(output, dim=1)
                conf = probs[0, target_class].item()
                confidence_scores.append(conf)
        
        # Compute AUC (lower is better for deletion)
        x_values = np.linspace(0, 1, len(confidence_scores))
        auc_score = auc(x_values, confidence_scores)
        
        return auc_score, confidence_scores
    
    def evaluate_batch(self, images: List[torch.Tensor], explanations: List[np.ndarray], 
                      target_classes: List[int]) -> Tuple[float, List[float]]:
        """Evaluate deletion AUC for a batch of images"""
        auc_scores = []
        all_confidence_curves = []
        
        for img, exp, target in zip(images, explanations, target_classes):
            auc_score, conf_scores = self.compute_deletion_auc(img, exp, target)
            auc_scores.append(auc_score)
            all_confidence_curves.append(conf_scores)
        
        mean_auc = np.mean(auc_scores)
        return mean_auc, auc_scores