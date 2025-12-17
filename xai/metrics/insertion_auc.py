"""
Insertion AUC metric for measuring faithfulness of explanations
Higher values indicate better faithfulness
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, List
from sklearn.metrics import auc


class InsertionAUC:
    """Compute Insertion AUC metric for explanation faithfulness"""
    
    def __init__(self, model, device, num_steps: int = 50, blur_sigma: float = 10.0):
        self.model = model
        self.device = device
        self.num_steps = num_steps
        self.blur_sigma = blur_sigma
        
    def create_baseline(self, image: torch.Tensor) -> torch.Tensor:
        """Create blurred baseline image"""
        # Convert to numpy for OpenCV
        img_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(img_np, (51, 51), self.blur_sigma)
        
        # Convert back to tensor
        baseline = torch.from_numpy(blurred.transpose(2, 0, 1)).unsqueeze(0).float()
        return baseline.to(self.device)
    
    def get_top_k_pixels(self, explanation: np.ndarray, k: int) -> np.ndarray:
        """Get indices of top-k most important pixels"""
        flat_explanation = explanation.flatten()
        top_k_indices = np.argpartition(flat_explanation, -k)[-k:]
        
        # Convert flat indices to 2D coordinates
        h, w = explanation.shape
        coords = np.unravel_index(top_k_indices, (h, w))
        return coords
    
    def compute_insertion_auc(self, image: torch.Tensor, explanation: np.ndarray, 
                            target_class: int) -> Tuple[float, List[float]]:
        """
        Compute Insertion AUC by progressively inserting pixels
        
        Args:
            image: Original image tensor (1, C, H, W)
            explanation: Explanation heatmap (H, W)
            target_class: Target class index
            
        Returns:
            Tuple of (AUC score, confidence scores list)
        """
        self.model.eval()
        
        # Create baseline (blurred image)
        baseline = self.create_baseline(image)
        
        # Resize explanation to match image spatial dimensions
        img_h, img_w = image.shape[2], image.shape[3]
        if explanation.shape != (img_h, img_w):
            explanation = cv2.resize(explanation, (img_w, img_h))
        
        # Initialize with baseline
        current_image = baseline.clone()
        
        # Get total number of pixels
        total_pixels = img_h * img_w
        pixels_per_step = max(1, total_pixels // self.num_steps)
        
        confidence_scores = []
        
        # Initial confidence with baseline
        with torch.no_grad():
            output = self.model(current_image)
            probs = F.softmax(output, dim=1)
            initial_conf = probs[0, target_class].item()
            confidence_scores.append(initial_conf)
        
        # Progressive insertion
        inserted_pixels = 0
        for step in range(self.num_steps):
            # Determine how many pixels to insert this step
            pixels_to_insert = min(pixels_per_step, total_pixels - inserted_pixels)
            if pixels_to_insert <= 0:
                break
            
            # Get top-k pixels for this step
            remaining_explanation = explanation.copy()
            
            # Mask already inserted pixels
            if inserted_pixels > 0:
                # Get previously inserted pixels and set their importance to 0
                top_inserted = self.get_top_k_pixels(explanation, inserted_pixels)
                remaining_explanation[top_inserted] = 0
            
            # Get next most important pixels
            top_coords = self.get_top_k_pixels(remaining_explanation, pixels_to_insert)
            
            # Insert pixels from original image
            for c in range(image.shape[1]):  # For each channel
                current_image[0, c, top_coords[0], top_coords[1]] = image[0, c, top_coords[0], top_coords[1]]
            
            inserted_pixels += pixels_to_insert
            
            # Measure confidence
            with torch.no_grad():
                output = self.model(current_image)
                probs = F.softmax(output, dim=1)
                conf = probs[0, target_class].item()
                confidence_scores.append(conf)
        
        # Compute AUC
        x_values = np.linspace(0, 1, len(confidence_scores))
        auc_score = auc(x_values, confidence_scores)
        
        return auc_score, confidence_scores
    
    def evaluate_batch(self, images: List[torch.Tensor], explanations: List[np.ndarray], 
                      target_classes: List[int]) -> Tuple[float, List[float]]:
        """Evaluate insertion AUC for a batch of images"""
        auc_scores = []
        all_confidence_curves = []
        
        for img, exp, target in zip(images, explanations, target_classes):
            auc_score, conf_scores = self.compute_insertion_auc(img, exp, target)
            auc_scores.append(auc_score)
            all_confidence_curves.append(conf_scores)
        
        mean_auc = np.mean(auc_scores)
        return mean_auc, auc_scores