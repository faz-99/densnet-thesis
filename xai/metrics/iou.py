"""
Intersection over Union (IoU) metric for explanation localization accuracy
Higher values indicate better localization
"""
import torch
import numpy as np
import cv2
from typing import Tuple, Optional, List
from sklearn.metrics import jaccard_score


class IoUMetric:
    """Compute IoU metric for explanation localization"""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def create_binary_mask(self, explanation: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Convert explanation heatmap to binary mask"""
        if threshold is None:
            threshold = self.threshold
        
        # Normalize explanation to [0, 1]
        if explanation.max() > explanation.min():
            normalized = (explanation - explanation.min()) / (explanation.max() - explanation.min())
        else:
            normalized = explanation
        
        # Apply threshold
        binary_mask = (normalized >= threshold).astype(np.uint8)
        return binary_mask
    
    def create_pseudo_roi(self, explanation: np.ndarray, top_k_percent: float = 0.2) -> np.ndarray:
        """
        Create pseudo-ROI from explanation by taking top-k% most important pixels
        Used when ground truth ROI is not available
        """
        flat_explanation = explanation.flatten()
        k = int(len(flat_explanation) * top_k_percent)
        
        # Get top-k indices
        top_k_indices = np.argpartition(flat_explanation, -k)[-k:]
        
        # Create binary mask
        pseudo_roi = np.zeros_like(flat_explanation)
        pseudo_roi[top_k_indices] = 1
        
        return pseudo_roi.reshape(explanation.shape)
    
    def compute_iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Compute IoU between predicted and ground truth masks"""
        # Flatten masks
        pred_flat = pred_mask.flatten()
        gt_flat = gt_mask.flatten()
        
        # Compute intersection and union
        intersection = np.logical_and(pred_flat, gt_flat).sum()
        union = np.logical_or(pred_flat, gt_flat).sum()
        
        # Avoid division by zero
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        iou = intersection / union
        return float(iou)
    
    def compute_iou_with_roi(self, explanation: np.ndarray, roi_mask: np.ndarray, 
                           threshold: Optional[float] = None) -> float:
        """
        Compute IoU between explanation and ground truth ROI
        
        Args:
            explanation: Explanation heatmap (H, W)
            roi_mask: Ground truth ROI mask (H, W)
            threshold: Threshold for binarizing explanation
            
        Returns:
            IoU score
        """
        # Resize explanation to match ROI if needed
        if explanation.shape != roi_mask.shape:
            explanation = cv2.resize(explanation, (roi_mask.shape[1], roi_mask.shape[0]))
        
        # Create binary mask from explanation
        pred_mask = self.create_binary_mask(explanation, threshold)
        
        # Ensure ROI mask is binary
        gt_mask = (roi_mask > 0).astype(np.uint8)
        
        return self.compute_iou(pred_mask, gt_mask)
    
    def compute_iou_pseudo_roi(self, explanation1: np.ndarray, explanation2: np.ndarray,
                             top_k_percent: float = 0.2) -> float:
        """
        Compute IoU between two explanations using pseudo-ROI approach
        Useful when ground truth ROI is not available
        """
        # Create pseudo-ROI from first explanation
        pseudo_roi = self.create_pseudo_roi(explanation1, top_k_percent)
        
        # Create binary mask from second explanation
        pred_mask = self.create_binary_mask(explanation2)
        
        return self.compute_iou(pred_mask, pseudo_roi)
    
    def evaluate_multiple_thresholds(self, explanation: np.ndarray, roi_mask: np.ndarray,
                                   thresholds: List[float] = None) -> Tuple[List[float], float]:
        """
        Evaluate IoU at multiple thresholds and return best score
        
        Args:
            explanation: Explanation heatmap
            roi_mask: Ground truth ROI mask
            thresholds: List of thresholds to evaluate
            
        Returns:
            Tuple of (IoU scores list, best IoU score)
        """
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 9).tolist()
        
        iou_scores = []
        for threshold in thresholds:
            iou = self.compute_iou_with_roi(explanation, roi_mask, threshold)
            iou_scores.append(iou)
        
        best_iou = max(iou_scores)
        return iou_scores, best_iou
    
    def evaluate_batch(self, explanations: List[np.ndarray], roi_masks: List[np.ndarray],
                      use_pseudo_roi: bool = False) -> Tuple[float, List[float]]:
        """
        Evaluate IoU for a batch of explanations
        
        Args:
            explanations: List of explanation heatmaps
            roi_masks: List of ROI masks (or reference explanations if use_pseudo_roi=True)
            use_pseudo_roi: Whether to use pseudo-ROI approach
            
        Returns:
            Tuple of (mean IoU, individual IoU scores)
        """
        iou_scores = []
        
        for exp, roi in zip(explanations, roi_masks):
            if use_pseudo_roi:
                iou = self.compute_iou_pseudo_roi(exp, roi)
            else:
                iou = self.compute_iou_with_roi(exp, roi)
            iou_scores.append(iou)
        
        mean_iou = np.mean(iou_scores)
        return mean_iou, iou_scores
    
    def compute_dice_coefficient(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Compute Dice coefficient (alternative to IoU)"""
        pred_flat = pred_mask.flatten()
        gt_flat = gt_mask.flatten()
        
        intersection = np.logical_and(pred_flat, gt_flat).sum()
        dice = (2.0 * intersection) / (pred_flat.sum() + gt_flat.sum())
        
        return float(dice) if (pred_flat.sum() + gt_flat.sum()) > 0 else 1.0