"""
Morphological Descriptor Extraction for Histopathology Explainability
Extracts quantitative features from activation maps to generate clinical insights
"""
import numpy as np
import cv2
from skimage import measure, feature, filters
from skimage.color import rgb2hsv
import json
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt


class MorphologicalAnalyzer:
    """Extract morphological descriptors from explainability maps"""
    
    def __init__(self, activation_threshold: float = 0.5):
        self.activation_threshold = activation_threshold
        
    def analyze_activation_map(self, original_image: np.ndarray, 
                             activation_map: np.ndarray) -> Dict:
        """
        Comprehensive analysis of activation maps
        
        Args:
            original_image: Original H&E stained image (H, W, 3)
            activation_map: Normalized activation map (H, W)
            
        Returns:
            Dictionary with morphological descriptors
        """
        # Resize activation map to match image if needed
        if activation_map.shape != original_image.shape[:2]:
            activation_map = cv2.resize(activation_map, 
                                      (original_image.shape[1], original_image.shape[0]))
        
        # Create binary mask for high-activation regions
        high_activation_mask = activation_map > self.activation_threshold
        
        # Calculate tissue area percentage
        tissue_area_percent = self._calculate_tissue_area_percentage(
            original_image, high_activation_mask)
        
        # Extract color features from high-activation zones
        color_features = self._extract_color_features(
            original_image, high_activation_mask)
        
        # Calculate texture features
        texture_features = self._extract_texture_features(
            original_image, high_activation_mask)
        
        # Analyze H&E stain characteristics
        stain_analysis = self._analyze_he_staining(
            original_image, high_activation_mask)
        
        # Morphological characteristics
        morphological_features = self._extract_morphological_features(
            high_activation_mask)
        
        return {
            'tissue_area_percent': tissue_area_percent,
            'color_features': color_features,
            'texture_features': texture_features,
            'stain_analysis': stain_analysis,
            'morphological_features': morphological_features,
            'activation_statistics': {
                'mean_activation': float(np.mean(activation_map)),
                'max_activation': float(np.max(activation_map)),
                'activation_std': float(np.std(activation_map))
            }
        }
    
    def _calculate_tissue_area_percentage(self, image: np.ndarray, 
                                        mask: np.ndarray) -> float:
        """Calculate percentage of tissue area highlighted"""
        # Create tissue mask (non-white regions)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        tissue_mask = gray < 0.9  # Assuming white background
        
        # Calculate percentages
        total_tissue_pixels = np.sum(tissue_mask)
        highlighted_tissue_pixels = np.sum(mask & tissue_mask)
        
        if total_tissue_pixels == 0:
            return 0.0
        
        return (highlighted_tissue_pixels / total_tissue_pixels) * 100
    
    def _extract_color_features(self, image: np.ndarray, 
                              mask: np.ndarray) -> Dict:
        """Extract color statistics from high-activation regions"""
        if not np.any(mask):
            return {
                'mean_rgb': [0, 0, 0],
                'std_rgb': [0, 0, 0],
                'dominant_channel': 'none'
            }
        
        # Extract RGB values from masked regions
        masked_pixels = image[mask]
        
        mean_rgb = np.mean(masked_pixels, axis=0).tolist()
        std_rgb = np.std(masked_pixels, axis=0).tolist()
        
        # Determine dominant color channel
        dominant_channel_idx = np.argmax(mean_rgb)
        dominant_channel = ['red', 'green', 'blue'][dominant_channel_idx]
        
        return {
            'mean_rgb': mean_rgb,
            'std_rgb': std_rgb,
            'dominant_channel': dominant_channel,
            'brightness': float(np.mean(masked_pixels)),
            'contrast': float(np.std(masked_pixels))
        }
    
    def _extract_texture_features(self, image: np.ndarray, 
                                mask: np.ndarray) -> Dict:
        """Extract texture features from high-activation regions"""
        if not np.any(mask):
            return {
                'entropy': 0.0,
                'local_variance': 0.0,
                'edge_density': 0.0
            }
        
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate entropy in masked regions
        masked_gray = gray[mask]
        hist, _ = np.histogram(masked_gray, bins=256, range=(0, 1))
        hist = hist / np.sum(hist)  # Normalize
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Local variance using a sliding window
        local_var = filters.rank.variance(gray, np.ones((5, 5)))
        masked_variance = local_var[mask]
        mean_local_variance = np.mean(masked_variance)
        
        # Edge density using Canny edge detection
        edges = feature.canny(gray, sigma=1.0)
        edge_density = np.sum(edges & mask) / np.sum(mask)
        
        return {
            'entropy': float(entropy),
            'local_variance': float(mean_local_variance),
            'edge_density': float(edge_density)
        }
    
    def _analyze_he_staining(self, image: np.ndarray, 
                           mask: np.ndarray) -> Dict:
        """Analyze H&E staining characteristics"""
        if not np.any(mask):
            return {
                'hematoxylin_intensity': 0.0,
                'eosin_intensity': 0.0,
                'stain_ratio': 0.0,
                'dominant_stain': 'none'
            }
        
        # Convert to HSV for better color analysis
        hsv = rgb2hsv(image)
        masked_pixels = image[mask]
        
        # Hematoxylin (blue/purple) detection
        # Blue channel dominance and low red values
        blue_dominance = masked_pixels[:, 2] - np.mean(masked_pixels[:, :2], axis=1)
        hematoxylin_intensity = np.mean(np.maximum(blue_dominance, 0))
        
        # Eosin (pink/red) detection
        # Red channel dominance
        red_dominance = masked_pixels[:, 0] - np.mean(masked_pixels[:, 1:], axis=1)
        eosin_intensity = np.mean(np.maximum(red_dominance, 0))
        
        # Calculate stain ratio
        total_stain = hematoxylin_intensity + eosin_intensity
        stain_ratio = hematoxylin_intensity / (total_stain + 1e-10)
        
        # Determine dominant stain
        if hematoxylin_intensity > eosin_intensity:
            dominant_stain = 'hematoxylin'
        elif eosin_intensity > hematoxylin_intensity:
            dominant_stain = 'eosin'
        else:
            dominant_stain = 'balanced'
        
        return {
            'hematoxylin_intensity': float(hematoxylin_intensity),
            'eosin_intensity': float(eosin_intensity),
            'stain_ratio': float(stain_ratio),
            'dominant_stain': dominant_stain
        }
    
    def _extract_morphological_features(self, mask: np.ndarray) -> Dict:
        """Extract morphological features from binary mask"""
        if not np.any(mask):
            return {
                'num_regions': 0,
                'largest_region_area': 0,
                'region_compactness': 0.0,
                'region_eccentricity': 0.0
            }
        
        # Label connected components
        labeled_mask = measure.label(mask)
        regions = measure.regionprops(labeled_mask)
        
        if not regions:
            return {
                'num_regions': 0,
                'largest_region_area': 0,
                'region_compactness': 0.0,
                'region_eccentricity': 0.0
            }
        
        # Calculate region statistics
        areas = [region.area for region in regions]
        largest_region = max(regions, key=lambda r: r.area)
        
        # Compactness (circularity)
        compactness = largest_region.area / (largest_region.perimeter ** 2 + 1e-10)
        
        # Eccentricity (elongation)
        eccentricity = largest_region.eccentricity if hasattr(largest_region, 'eccentricity') else 0.0
        
        return {
            'num_regions': len(regions),
            'largest_region_area': int(max(areas)),
            'region_compactness': float(compactness),
            'region_eccentricity': float(eccentricity),
            'total_activated_area': int(np.sum(mask))
        }


class ClinicalDescriptorGenerator:
    """Generate clinical descriptions from morphological features"""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict:
        """Load description templates for different features"""
        return {
            'tissue_area': {
                'low': "focal areas ({:.1f}%)",
                'medium': "moderate regions ({:.1f}%)", 
                'high': "extensive areas ({:.1f}%)"
            },
            'stain_dominance': {
                'hematoxylin': "darkly stained nuclear regions",
                'eosin': "pink cytoplasmic areas",
                'balanced': "mixed nuclear and cytoplasmic staining"
            },
            'texture': {
                'low_entropy': "uniform cellular pattern",
                'high_entropy': "heterogeneous cellular architecture",
                'high_variance': "irregular tissue structure",
                'high_edges': "sharp cellular boundaries"
            },
            'morphology': {
                'compact': "rounded cellular clusters",
                'elongated': "elongated tissue structures",
                'fragmented': "scattered cellular elements",
                'large_regions': "confluent tissue areas"
            }
        }
    
    def generate_description(self, features: Dict, prediction_class: str, 
                           confidence: float) -> str:
        """Generate clinical description from features"""
        
        # Categorize tissue area
        area_percent = features['tissue_area_percent']
        if area_percent < 20:
            area_desc = self.templates['tissue_area']['low'].format(area_percent)
        elif area_percent < 50:
            area_desc = self.templates['tissue_area']['medium'].format(area_percent)
        else:
            area_desc = self.templates['tissue_area']['high'].format(area_percent)
        
        # Stain dominance description
        dominant_stain = features['stain_analysis']['dominant_stain']
        stain_desc = self.templates['stain_dominance'].get(dominant_stain, "mixed staining")
        
        # Texture characteristics
        texture_features = features['texture_features']
        texture_desc = []
        
        if texture_features['entropy'] > 6.0:
            texture_desc.append(self.templates['texture']['high_entropy'])
        elif texture_features['entropy'] < 4.0:
            texture_desc.append(self.templates['texture']['low_entropy'])
        
        if texture_features['local_variance'] > 0.1:
            texture_desc.append(self.templates['texture']['high_variance'])
        
        if texture_features['edge_density'] > 0.3:
            texture_desc.append(self.templates['texture']['high_edges'])
        
        # Morphological characteristics
        morph_features = features['morphological_features']
        morph_desc = []
        
        if morph_features['region_compactness'] > 0.5:
            morph_desc.append(self.templates['morphology']['compact'])
        
        if morph_features['region_eccentricity'] > 0.7:
            morph_desc.append(self.templates['morphology']['elongated'])
        
        if morph_features['num_regions'] > 10:
            morph_desc.append(self.templates['morphology']['fragmented'])
        
        if morph_features['largest_region_area'] > 1000:
            morph_desc.append(self.templates['morphology']['large_regions'])
        
        # Combine descriptions
        texture_str = ", ".join(texture_desc) if texture_desc else "standard cellular pattern"
        morph_str = ", ".join(morph_desc) if morph_desc else "typical tissue organization"
        
        # Generate final description
        description = (
            f"The model highlights {area_desc} of the tissue with {stain_desc}, "
            f"showing {texture_str} and {morph_str}. "
            f"These features are consistent with {prediction_class.lower()} pathology "
            f"(confidence: {confidence:.1%})."
        )
        
        return description
    
    def generate_detailed_report(self, features: Dict, prediction_class: str, 
                               confidence: float, image_id: str) -> Dict:
        """Generate comprehensive report with all details"""
        
        # Basic description
        description = self.generate_description(features, prediction_class, confidence)
        
        # Detailed findings
        findings = []
        
        # Tissue coverage
        area_percent = features['tissue_area_percent']
        if area_percent > 40:
            findings.append(f"Extensive model attention ({area_percent:.1f}% of tissue)")
        elif area_percent < 15:
            findings.append(f"Focal model attention ({area_percent:.1f}% of tissue)")
        
        # Staining patterns
        stain = features['stain_analysis']
        if stain['hematoxylin_intensity'] > stain['eosin_intensity']:
            findings.append("Predominant nuclear staining (hematoxylin)")
        elif stain['eosin_intensity'] > stain['hematoxylin_intensity']:
            findings.append("Predominant cytoplasmic staining (eosin)")
        
        # Texture analysis
        texture = features['texture_features']
        if texture['entropy'] > 6.5:
            findings.append("High cellular heterogeneity")
        if texture['edge_density'] > 0.4:
            findings.append("Sharp cellular boundaries")
        
        # Morphological patterns
        morph = features['morphological_features']
        if morph['num_regions'] > 15:
            findings.append("Multiple discrete cellular clusters")
        if morph['region_eccentricity'] > 0.8:
            findings.append("Elongated tissue structures")
        
        # Clinical interpretation
        clinical_interpretation = self._generate_clinical_interpretation(
            features, prediction_class, confidence)
        
        return {
            'image_id': image_id,
            'predicted_class': prediction_class,
            'confidence': confidence,
            'activated_area_percent': area_percent,
            'dominant_stain': stain['dominant_stain'],
            'key_findings': findings,
            'description': description,
            'clinical_interpretation': clinical_interpretation,
            'quantitative_features': features
        }
    
    def _generate_clinical_interpretation(self, features: Dict, 
                                        prediction_class: str, 
                                        confidence: float) -> str:
        """Generate clinical interpretation based on features"""
        
        interpretations = []
        
        # High confidence interpretations
        if confidence > 0.9:
            if prediction_class.lower() == 'malignant':
                interpretations.append(
                    "High confidence malignant classification supported by "
                    "characteristic morphological features."
                )
            else:
                interpretations.append(
                    "High confidence benign classification with typical "
                    "architectural patterns."
                )
        
        # Feature-based interpretations
        texture = features['texture_features']
        stain = features['stain_analysis']
        morph = features['morphological_features']
        
        if texture['entropy'] > 6.0 and prediction_class.lower() == 'malignant':
            interpretations.append(
                "Cellular heterogeneity consistent with malignant transformation."
            )
        
        if stain['hematoxylin_intensity'] > 0.3 and morph['num_regions'] > 10:
            interpretations.append(
                "Dense nuclear clustering suggesting increased cellular activity."
            )
        
        if morph['region_eccentricity'] > 0.7:
            interpretations.append(
                "Irregular tissue architecture may indicate pathological changes."
            )
        
        return " ".join(interpretations) if interpretations else (
            "Standard morphological features observed."
        )


def save_explainability_report(report: Dict, save_dir: str = "explainability_reports"):
    """Save explainability report in multiple formats"""
    import os
    
    os.makedirs(save_dir, exist_ok=True)
    
    image_id = report['image_id']
    base_filename = os.path.join(save_dir, f"{image_id}_report")
    
    # Save JSON report
    with open(f"{base_filename}.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save text report
    with open(f"{base_filename}.txt", 'w') as f:
        f.write(f"Histopathology Explainability Report\n")
        f.write(f"=" * 40 + "\n\n")
        f.write(f"Image ID: {report['image_id']}\n")
        f.write(f"Predicted Class: {report['predicted_class']}\n")
        f.write(f"Confidence: {report['confidence']:.1%}\n")
        f.write(f"Activated Area: {report['activated_area_percent']:.1f}%\n")
        f.write(f"Dominant Stain: {report['dominant_stain']}\n\n")
        
        f.write("Key Findings:\n")
        for finding in report['key_findings']:
            f.write(f"- {finding}\n")
        f.write("\n")
        
        f.write("Description:\n")
        f.write(f"{report['description']}\n\n")
        
        f.write("Clinical Interpretation:\n")
        f.write(f"{report['clinical_interpretation']}\n")
    
    return base_filename