"""
Integrated Gradients implementation for DenseNet-based models
Primary explainability method for histopathology image analysis
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
from scipy import ndimage


class IntegratedGradients:
    """
    Integrated Gradients implementation for deep neural networks
    Primary explainability method for histopathology analysis
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def generate_baseline(self, input_tensor: torch.Tensor, method: str = 'black') -> torch.Tensor:
        """
        Generate baseline for integrated gradients
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            method: Baseline generation method ('black', 'blur', 'noise', 'mean')
            
        Returns:
            Baseline tensor
        """
        if method == 'black':
            return torch.zeros_like(input_tensor)
        elif method == 'blur':
            # Convert to numpy for blurring
            img_np = input_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            blurred = cv2.GaussianBlur(img_np, (51, 51), 10.0)
            baseline = torch.from_numpy(blurred.transpose(2, 0, 1)).unsqueeze(0).float()
            return baseline.to(self.device)
        elif method == 'noise':
            return torch.randn_like(input_tensor) * 0.1
        elif method == 'mean':
            mean_val = input_tensor.mean(dim=(2, 3), keepdim=True)
            return mean_val.expand_as(input_tensor)
        else:
            return torch.zeros_like(input_tensor)
    
    def compute_gradients(self, input_tensor: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        Compute gradients of target class score with respect to input
        
        Args:
            input_tensor: Input tensor with requires_grad=True
            target_class: Target class index
            
        Returns:
            Gradients tensor
        """
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class score
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        target_score = output[0, target_class]
        
        # Backward pass
        self.model.zero_grad()
        target_score.backward()
        
        return input_tensor.grad.clone()
    
    def generate_integrated_gradients(self, input_tensor: torch.Tensor, 
                                    target_class: Optional[int] = None,
                                    baseline_method: str = 'black',
                                    num_steps: int = 50) -> Tuple[np.ndarray, dict]:
        """
        Generate Integrated Gradients explanation
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (if None, uses predicted class)
            baseline_method: Method for baseline generation
            num_steps: Number of integration steps
            
        Returns:
            Tuple of (attribution map, metadata)
        """
        # Generate baseline
        baseline = self.generate_baseline(input_tensor, baseline_method)
        
        # Get predicted class if not specified
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()
        
        # Initialize integrated gradients
        integrated_gradients = torch.zeros_like(input_tensor)
        
        # Integration steps
        for step in range(num_steps):
            # Linear interpolation between baseline and input
            alpha = step / (num_steps - 1) if num_steps > 1 else 1.0
            interpolated_input = baseline + alpha * (input_tensor - baseline)
            interpolated_input.requires_grad_(True)
            
            # Compute gradients
            gradients = self.compute_gradients(interpolated_input, target_class)
            
            # Accumulate gradients
            integrated_gradients += gradients / num_steps
        
        # Multiply by input difference
        attribution = integrated_gradients * (input_tensor - baseline)
        
        # Convert to numpy and aggregate across channels
        attribution_np = attribution.squeeze(0).cpu().numpy()
        
        # Sum across channels for visualization
        attribution_map = np.sum(attribution_np, axis=0)
        
        # Metadata
        metadata = {
            'method': 'Integrated Gradients',
            'target_class': target_class,
            'baseline_method': baseline_method,
            'num_steps': num_steps,
            'attribution_shape': attribution_map.shape,
            'attribution_range': (float(attribution_map.min()), float(attribution_map.max()))
        }
        
        return attribution_map, metadata
    
    def generate_smooth_integrated_gradients(self, input_tensor: torch.Tensor,
                                           target_class: Optional[int] = None,
                                           baseline_method: str = 'black',
                                           num_steps: int = 50,
                                           noise_level: float = 0.1,
                                           num_samples: int = 10) -> Tuple[np.ndarray, dict]:
        """
        Generate Smooth Integrated Gradients (SmoothGrad + IG)
        Reduces noise in attributions by averaging over multiple noisy samples
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class index
            baseline_method: Baseline generation method
            num_steps: Integration steps
            noise_level: Standard deviation of noise
            num_samples: Number of noisy samples
            
        Returns:
            Tuple of (smooth attribution map, metadata)
        """
        smooth_attributions = []
        
        for sample in range(num_samples):
            # Add noise to input
            noise = torch.randn_like(input_tensor) * noise_level
            noisy_input = input_tensor + noise
            
            # Generate IG for noisy input
            attribution, _ = self.generate_integrated_gradients(
                noisy_input, target_class, baseline_method, num_steps
            )
            smooth_attributions.append(attribution)
        
        # Average attributions
        smooth_attribution = np.mean(smooth_attributions, axis=0)
        
        metadata = {
            'method': 'Smooth Integrated Gradients',
            'target_class': target_class,
            'baseline_method': baseline_method,
            'num_steps': num_steps,
            'noise_level': noise_level,
            'num_samples': num_samples,
            'attribution_shape': smooth_attribution.shape,
            'attribution_range': (float(smooth_attribution.min()), float(smooth_attribution.max()))
        }
        
        return smooth_attribution, metadata
    
    def analyze_histopathology_features(self, attribution_map: np.ndarray, 
                                      original_image: np.ndarray,
                                      threshold_percentile: float = 80) -> dict:
        """
        Analyze histopathology-specific features from attribution map
        
        Args:
            attribution_map: IG attribution map
            original_image: Original image (H, W, C)
            threshold_percentile: Percentile for important region threshold
            
        Returns:
            Dictionary with histopathology analysis
        """
        # Normalize attribution map
        attr_norm = np.abs(attribution_map)
        if attr_norm.max() > attr_norm.min():
            attr_norm = (attr_norm - attr_norm.min()) / (attr_norm.max() - attr_norm.min())
        
        # Create binary mask of important regions
        threshold = np.percentile(attr_norm, threshold_percentile)
        important_mask = attr_norm >= threshold
        
        # Analyze tissue proportion
        total_pixels = attribution_map.size
        important_pixels = np.sum(important_mask)
        tissue_proportion = important_pixels / total_pixels
        
        # Analyze spatial distribution
        labeled_regions, num_regions = ndimage.label(important_mask)
        
        # Analyze color/staining patterns in important regions
        if len(original_image.shape) == 3:
            # RGB analysis
            important_rgb = original_image[important_mask]
            if len(important_rgb) > 0:
                mean_rgb = np.mean(important_rgb, axis=0)
                std_rgb = np.std(important_rgb, axis=0)
                
                # H&E staining analysis (simplified)
                # Hematoxylin (blue/purple) vs Eosin (pink/red)
                blue_intensity = mean_rgb[2]  # Blue channel
                red_intensity = mean_rgb[0]   # Red channel
                
                if blue_intensity > red_intensity:
                    dominant_stain = 'hematoxylin'
                    stain_confidence = (blue_intensity - red_intensity) / (blue_intensity + red_intensity + 1e-8)
                else:
                    dominant_stain = 'eosin'
                    stain_confidence = (red_intensity - blue_intensity) / (blue_intensity + red_intensity + 1e-8)
            else:
                mean_rgb = np.array([0, 0, 0])
                std_rgb = np.array([0, 0, 0])
                dominant_stain = 'unknown'
                stain_confidence = 0.0
        else:
            mean_rgb = np.array([0])
            std_rgb = np.array([0])
            dominant_stain = 'grayscale'
            stain_confidence = 0.0
        
        # Analyze structural patterns
        # Compute texture features in important regions
        if np.sum(important_mask) > 0:
            # Local variance (texture measure)
            kernel = np.ones((5, 5)) / 25
            local_mean = cv2.filter2D(attr_norm.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((attr_norm - local_mean)**2, -1, kernel)
            texture_complexity = np.mean(local_variance[important_mask])
            
            # Edge density
            edges = cv2.Canny((attr_norm * 255).astype(np.uint8), 50, 150)
            edge_density = np.sum(edges[important_mask]) / np.sum(important_mask)
        else:
            texture_complexity = 0.0
            edge_density = 0.0
        
        # Compute attribution statistics
        attribution_stats = {
            'mean_attribution': float(np.mean(attribution_map)),
            'std_attribution': float(np.std(attribution_map)),
            'max_attribution': float(np.max(attribution_map)),
            'min_attribution': float(np.min(attribution_map)),
            'positive_attribution_ratio': float(np.sum(attribution_map > 0) / attribution_map.size)
        }
        
        analysis = {
            'tissue_proportion': float(tissue_proportion),
            'num_regions': int(num_regions),
            'dominant_stain': dominant_stain,
            'stain_confidence': float(stain_confidence),
            'mean_rgb': mean_rgb.tolist(),
            'std_rgb': std_rgb.tolist(),
            'texture_complexity': float(texture_complexity),
            'edge_density': float(edge_density),
            'attribution_stats': attribution_stats,
            'threshold_used': float(threshold)
        }
        
        return analysis
    
    def generate_textual_explanation(self, analysis: dict, prediction_info: dict) -> str:
        """
        Generate human-readable textual explanation for histopathology
        
        Args:
            analysis: Histopathology analysis from analyze_histopathology_features
            prediction_info: Model prediction information
            
        Returns:
            Human-readable explanation string
        """
        # Extract key information
        tissue_prop = analysis['tissue_proportion'] * 100
        num_regions = analysis['num_regions']
        dominant_stain = analysis['dominant_stain']
        stain_conf = analysis['stain_confidence']
        texture_complexity = analysis['texture_complexity']
        edge_density = analysis['edge_density']
        
        predicted_class = prediction_info.get('predicted_class', 'Unknown')
        confidence = prediction_info.get('confidence', 0.0)
        class_names = prediction_info.get('class_names', ['Class 0', 'Class 1'])
        
        if predicted_class < len(class_names):
            class_name = class_names[predicted_class]
        else:
            class_name = f'Class {predicted_class}'
        
        # Generate explanation
        explanation = f"""
INTEGRATED GRADIENTS ANALYSIS REPORT

Model Prediction: {class_name} (Confidence: {confidence:.1%})

TISSUE ANALYSIS:
• Highlighted tissue proportion: {tissue_prop:.1f}% of total image area
• Number of distinct regions: {num_regions}
• Spatial distribution: {'Fragmented' if num_regions > 5 else 'Cohesive'} pattern

STAINING PATTERN ANALYSIS:
• Dominant staining: {dominant_stain.title()}
• Staining confidence: {stain_conf:.2f}
• Interpretation: {'Nuclear focus (cell proliferation/chromatin changes)' if dominant_stain == 'hematoxylin' else 'Cytoplasmic focus (structural alterations)' if dominant_stain == 'eosin' else 'Mixed or unclear staining pattern'}

STRUCTURAL IRREGULARITIES:
• Texture complexity: {texture_complexity:.3f}
• Edge density: {edge_density:.3f}
• Structural assessment: {'High heterogeneity' if texture_complexity > 0.1 else 'Moderate heterogeneity' if texture_complexity > 0.05 else 'Low heterogeneity'}
• Boundary definition: {'Sharp cellular boundaries' if edge_density > 0.3 else 'Smooth boundaries'}

MODEL CONFIDENCE ALIGNMENT:
• Attribution strength: {'Strong' if analysis['attribution_stats']['std_attribution'] > 0.1 else 'Moderate' if analysis['attribution_stats']['std_attribution'] > 0.05 else 'Weak'}
• Positive evidence ratio: {analysis['attribution_stats']['positive_attribution_ratio']:.1%}
• Decision confidence: {'High confidence with focused attention' if confidence > 0.8 and tissue_prop < 30 else 'Moderate confidence with distributed attention' if confidence > 0.6 else 'Low confidence with unclear focus'}

CLINICAL INTERPRETATION:
"""
        
        # Add clinical interpretation based on findings
        if dominant_stain == 'hematoxylin' and texture_complexity > 0.1:
            explanation += "• Model focuses on nuclear regions with high cellular heterogeneity, suggesting attention to proliferative activity.\n"
        elif dominant_stain == 'eosin' and edge_density > 0.3:
            explanation += "• Model emphasizes cytoplasmic regions with distinct boundaries, indicating structural architectural focus.\n"
        else:
            explanation += "• Model shows mixed attention patterns across tissue components.\n"
        
        if tissue_prop > 40:
            explanation += "• Extensive tissue involvement suggests widespread pathological changes.\n"
        elif tissue_prop > 20:
            explanation += "• Moderate tissue involvement indicates regional abnormalities.\n"
        else:
            explanation += "• Focal tissue involvement suggests localized changes.\n"
        
        if confidence > 0.8:
            explanation += "• High model confidence supports reliable classification.\n"
        elif confidence > 0.6:
            explanation += "• Moderate model confidence suggests reasonable classification reliability.\n"
        else:
            explanation += "• Low model confidence indicates uncertain classification requiring expert review.\n"
        
        explanation += f"\nNOTE: This analysis is based on Integrated Gradients attribution and should be interpreted by qualified pathologists."
        
        return explanation.strip()
    
    def visualize_attribution(self, attribution_map: np.ndarray, 
                            original_image: np.ndarray,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Create visualization of Integrated Gradients attribution
        
        Args:
            attribution_map: Attribution map from IG
            original_image: Original image
            save_path: Optional path to save visualization
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Attribution heatmap
        im1 = axes[0, 1].imshow(attribution_map, cmap='RdBu_r')
        axes[0, 1].set_title('Integrated Gradients Attribution', fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # Positive attributions only
        positive_attr = np.maximum(attribution_map, 0)
        im2 = axes[1, 0].imshow(positive_attr, cmap='Reds')
        axes[1, 0].set_title('Positive Attributions', fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Overlay on original
        # Normalize attribution for overlay
        attr_norm = np.abs(attribution_map)
        if attr_norm.max() > attr_norm.min():
            attr_norm = (attr_norm - attr_norm.min()) / (attr_norm.max() - attr_norm.min())
        
        # Create overlay
        overlay = original_image.copy()
        if len(overlay.shape) == 3:
            # Add red channel overlay for important regions
            threshold = np.percentile(attr_norm, 80)
            important_mask = attr_norm >= threshold
            overlay[important_mask, 0] = np.minimum(overlay[important_mask, 0] + 0.3, 1.0)
        
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Attribution Overlay', fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig