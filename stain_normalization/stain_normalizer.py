"""
Stain normalization implementation for histopathology images
Supports Macenko and Reinhard methods for ablation studies
"""
import numpy as np
import cv2
from PIL import Image
import os
from typing import Optional, Tuple, Union
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

try:
    import staintools
    STAINTOOLS_AVAILABLE = True
except ImportError:
    STAINTOOLS_AVAILABLE = False
    print("Warning: staintools not available. Using custom implementations.")


class StainNormalizer:
    """
    Comprehensive stain normalization for histopathology images
    """
    
    def __init__(self, method: str = 'macenko', target_image: Optional[np.ndarray] = None):
        """
        Initialize stain normalizer
        
        Args:
            method: Normalization method ('macenko', 'reinhard', 'none')
            target_image: Target image for normalization (RGB, 0-255)
        """
        self.method = method.lower()
        self.target_image = target_image
        self.normalizer = None
        
        if self.method != 'none':
            self._initialize_normalizer()
    
    def _initialize_normalizer(self):
        """Initialize the appropriate normalizer"""
        if self.method == 'macenko':
            if STAINTOOLS_AVAILABLE:
                self.normalizer = staintools.StainNormalizer(method='macenko')
            else:
                self.normalizer = MacenkoNormalizer()
        elif self.method == 'reinhard':
            if STAINTOOLS_AVAILABLE:
                self.normalizer = staintools.StainNormalizer(method='reinhard')
            else:
                self.normalizer = ReinhardNormalizer()
        
        # Fit normalizer with target image if provided
        if self.target_image is not None:
            self.fit_target(self.target_image)
    
    def fit_target(self, target_image: np.ndarray):
        """
        Fit normalizer to target image
        
        Args:
            target_image: Target image (RGB, 0-255)
        """
        self.target_image = target_image
        
        if self.normalizer is not None:
            if STAINTOOLS_AVAILABLE:
                self.normalizer.fit(target_image)
            else:
                self.normalizer.fit(target_image)
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image using fitted normalizer
        
        Args:
            image: Input image (RGB, 0-255)
            
        Returns:
            Normalized image (RGB, 0-255)
        """
        if self.method == 'none' or self.normalizer is None:
            return image
        
        try:
            if STAINTOOLS_AVAILABLE:
                normalized = self.normalizer.transform(image)
            else:
                normalized = self.normalizer.normalize(image)
            
            # Ensure output is in valid range
            normalized = np.clip(normalized, 0, 255).astype(np.uint8)
            return normalized
            
        except Exception as e:
            print(f"Normalization failed: {e}")
            return image
    
    def create_comparison_plot(self, original: np.ndarray, normalized: np.ndarray, 
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comparison plot of original vs normalized image
        
        Args:
            original: Original image
            normalized: Normalized image
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Normalized image
        axes[1].imshow(normalized)
        axes[1].set_title(f'Normalized ({self.method.title()})')
        axes[1].axis('off')
        
        # Target image (if available)
        if self.target_image is not None:
            axes[2].imshow(self.target_image)
            axes[2].set_title('Target Image')
            axes[2].axis('off')
        else:
            axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class MacenkoNormalizer:
    """
    Custom implementation of Macenko stain normalization
    Based on: "A method for normalizing histology slides for quantitative analysis"
    """
    
    def __init__(self):
        self.target_stains = None
        self.target_concentrations = None
        self.maxC_target = None
    
    def fit(self, target_image: np.ndarray):
        """Fit normalizer to target image"""
        # Convert to OD space
        od_target = self._rgb_to_od(target_image)
        
        # Get stain matrix and concentrations
        self.target_stains = self._get_stain_matrix(od_target)
        concentrations = self._get_concentrations(od_target, self.target_stains)
        self.maxC_target = np.percentile(concentrations, 99, axis=0)
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to target"""
        if self.target_stains is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        # Convert to OD space
        od_image = self._rgb_to_od(image)
        
        # Get stain matrix and concentrations for source image
        source_stains = self._get_stain_matrix(od_image)
        source_concentrations = self._get_concentrations(od_image, source_stains)
        maxC_source = np.percentile(source_concentrations, 99, axis=0)
        
        # Normalize concentrations
        normalized_concentrations = source_concentrations * (self.maxC_target / maxC_source)
        
        # Reconstruct image
        normalized_od = normalized_concentrations @ self.target_stains
        normalized_rgb = self._od_to_rgb(normalized_od)
        
        return normalized_rgb
    
    def _rgb_to_od(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to optical density"""
        rgb = rgb.astype(np.float64)
        rgb = np.maximum(rgb, 1)  # Avoid log(0)
        od = -np.log(rgb / 255.0)
        return od
    
    def _od_to_rgb(self, od: np.ndarray) -> np.ndarray:
        """Convert optical density to RGB"""
        rgb = 255 * np.exp(-od)
        rgb = np.clip(rgb, 0, 255)
        return rgb.astype(np.uint8)
    
    def _get_stain_matrix(self, od: np.ndarray) -> np.ndarray:
        """Extract stain matrix using PCA"""
        # Reshape for PCA
        od_flat = od.reshape(-1, 3)
        
        # Remove background (low OD values)
        od_flat = od_flat[np.sum(od_flat, axis=1) > 0.3]
        
        # PCA to get principal directions
        pca = PCA(n_components=2)
        pca.fit(od_flat)
        
        # Get stain directions
        stain_matrix = pca.components_
        
        # Ensure proper orientation (H&E stains)
        if stain_matrix[0, 0] < stain_matrix[0, 1]:
            stain_matrix[[0, 1]] = stain_matrix[[1, 0]]
        
        # Add third component (residual)
        if stain_matrix.shape[0] == 2:
            third_component = np.cross(stain_matrix[0], stain_matrix[1])
            stain_matrix = np.vstack([stain_matrix, third_component])
        
        return stain_matrix
    
    def _get_concentrations(self, od: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
        """Get stain concentrations"""
        od_flat = od.reshape(-1, 3)
        concentrations = np.linalg.lstsq(stain_matrix.T, od_flat.T, rcond=None)[0].T
        return concentrations


class ReinhardNormalizer:
    """
    Custom implementation of Reinhard color normalization
    Based on: "Color transfer between images"
    """
    
    def __init__(self):
        self.target_means = None
        self.target_stds = None
    
    def fit(self, target_image: np.ndarray):
        """Fit normalizer to target image"""
        # Convert to LAB color space
        lab_target = cv2.cvtColor(target_image, cv2.COLOR_RGB2LAB)
        
        # Calculate statistics
        self.target_means = np.mean(lab_target.reshape(-1, 3), axis=0)
        self.target_stds = np.std(lab_target.reshape(-1, 3), axis=0)
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to target statistics"""
        if self.target_means is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        # Convert to LAB
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float64)
        
        # Calculate source statistics
        source_means = np.mean(lab_image.reshape(-1, 3), axis=0)
        source_stds = np.std(lab_image.reshape(-1, 3), axis=0)
        
        # Normalize
        lab_normalized = lab_image.copy()
        for i in range(3):
            if source_stds[i] > 0:
                lab_normalized[:, :, i] = ((lab_normalized[:, :, i] - source_means[i]) * 
                                         (self.target_stds[i] / source_stds[i]) + 
                                         self.target_means[i])
        
        # Convert back to RGB
        lab_normalized = np.clip(lab_normalized, 0, 255).astype(np.uint8)
        rgb_normalized = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2RGB)
        
        return rgb_normalized


def create_stain_normalized_dataset(
    input_dir: str,
    output_dir: str,
    method: str = 'macenko',
    target_image_path: Optional[str] = None
):
    """
    Create stain-normalized version of entire dataset
    
    Args:
        input_dir: Input dataset directory
        output_dir: Output directory for normalized images
        method: Normalization method
        target_image_path: Path to target image
    """
    print(f"Creating {method} normalized dataset...")
    
    # Load target image if provided
    target_image = None
    if target_image_path and os.path.exists(target_image_path):
        target_image = np.array(Image.open(target_image_path))
    
    # Initialize normalizer
    normalizer = StainNormalizer(method=method, target_image=target_image)
    
    # Process all images
    total_processed = 0
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                # Input path
                input_path = os.path.join(root, file)
                
                # Output path (maintain directory structure)
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, method, rel_path)
                
                # Create output directory
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                try:
                    # Load and normalize image
                    image = np.array(Image.open(input_path))
                    
                    # Ensure RGB
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        normalized = normalizer.normalize(image)
                        
                        # Save normalized image
                        Image.fromarray(normalized).save(output_path)
                        total_processed += 1
                        
                        if total_processed % 100 == 0:
                            print(f"Processed {total_processed} images...")
                
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")
    
    print(f"Completed! Processed {total_processed} images.")


def compare_stain_methods(
    image_path: str,
    target_image_path: Optional[str] = None,
    save_dir: str = 'stain_comparison'
):
    """
    Compare different stain normalization methods on a single image
    
    Args:
        image_path: Path to test image
        target_image_path: Path to target image
        save_dir: Directory to save comparison plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load images
    image = np.array(Image.open(image_path))
    target_image = None
    if target_image_path:
        target_image = np.array(Image.open(target_image_path))
    
    # Test different methods
    methods = ['none', 'macenko', 'reinhard']
    results = {'original': image}
    
    for method in methods:
        if method == 'none':
            results[method] = image
        else:
            normalizer = StainNormalizer(method=method, target_image=target_image)
            results[method] = normalizer.normalize(image)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (method, result) in enumerate(results.items()):
        if i < 4:
            axes[i].imshow(result)
            axes[i].set_title(f'{method.title()}')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'stain_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison saved to {save_dir}/stain_comparison.png")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Stain Normalization Module")
    print("Available methods: macenko, reinhard, none")
    
    # Test with sample image if available
    sample_image_path = "datasets/BreaKHis 400X/train/benign/sample.png"
    if os.path.exists(sample_image_path):
        compare_stain_methods(sample_image_path)
    else:
        print("No sample image found for testing.")