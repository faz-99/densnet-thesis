"""
Advanced textual explanation generator for histopathology analysis
Generates human-readable pathology reports from XAI attributions
"""
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from scipy import ndimage
import json


class HistopathologyTextualExplainer:
    """
    Generate comprehensive textual explanations for histopathology XAI results
    """
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        
        # Clinical interpretation templates
        self.staining_patterns = {
            'hematoxylin': {
                'description': 'nuclear regions with blue/purple staining',
                'clinical_significance': 'cell proliferation and chromatin changes',
                'pathological_relevance': 'nuclear morphology and mitotic activity'
            },
            'eosin': {
                'description': 'cytoplasmic regions with pink/red staining', 
                'clinical_significance': 'structural alterations and inflammation',
                'pathological_relevance': 'tissue architecture and cellular damage'
            },
            'mixed': {
                'description': 'balanced nuclear and cytoplasmic staining',
                'clinical_significance': 'complex tissue architecture',
                'pathological_relevance': 'multi-component pathological process'
            }
        }
        
        self.tissue_patterns = {
            'focal': {'threshold': 20, 'description': 'localized changes', 'clinical': 'limited pathological involvement'},
            'moderate': {'threshold': 40, 'description': 'regional abnormalities', 'clinical': 'moderate pathological changes'},
            'extensive': {'threshold': 100, 'description': 'widespread involvement', 'clinical': 'extensive pathological process'}
        }
        
        self.confidence_levels = {
            'high': {'threshold': 0.8, 'description': 'high confidence', 'reliability': 'reliable classification'},
            'moderate': {'threshold': 0.6, 'description': 'moderate confidence', 'reliability': 'reasonable classification'},
            'low': {'threshold': 0.0, 'description': 'low confidence', 'reliability': 'uncertain classification requiring expert review'}
        }
    
    def analyze_tissue_distribution(self, attribution_map: np.ndarray, 
                                  threshold_percentile: float = 80) -> Dict:
        """Analyze spatial distribution of tissue attention"""
        # Normalize attribution
        attr_norm = np.abs(attribution_map)
        if attr_norm.max() > attr_norm.min():
            attr_norm = (attr_norm - attr_norm.min()) / (attr_norm.max() - attr_norm.min())
        
        # Create importance mask
        threshold = np.percentile(attr_norm, threshold_percentile)
        important_mask = attr_norm >= threshold
        
        # Analyze spatial properties
        labeled_regions, num_regions = ndimage.label(important_mask)
        
        # Calculate region properties
        region_sizes = []
        region_centroids = []
        
        for region_id in range(1, num_regions + 1):
            region_mask = labeled_regions == region_id
            region_size = np.sum(region_mask)
            region_sizes.append(region_size)
            
            # Calculate centroid
            coords = np.where(region_mask)
            if len(coords[0]) > 0:
                centroid = (np.mean(coords[0]), np.mean(coords[1]))
                region_centroids.append(centroid)
        
        # Tissue proportion
        tissue_proportion = np.sum(important_mask) / important_mask.size * 100
        
        # Spatial distribution metrics
        if len(region_centroids) > 1:
            # Calculate spread of regions
            centroids_array = np.array(region_centroids)
            spatial_spread = np.std(centroids_array, axis=0).mean()
        else:
            spatial_spread = 0.0
        
        return {
            'tissue_proportion': tissue_proportion,
            'num_regions': num_regions,
            'region_sizes': region_sizes,
            'spatial_spread': spatial_spread,
            'largest_region_ratio': max(region_sizes) / sum(region_sizes) if region_sizes else 0,
            'fragmentation_index': num_regions / (tissue_proportion / 100 + 1e-8)
        }
    
    def analyze_staining_patterns(self, original_image: np.ndarray, 
                                attribution_map: np.ndarray,
                                threshold_percentile: float = 80) -> Dict:
        """Analyze H&E staining patterns in important regions"""
        if len(original_image.shape) != 3:
            return {'dominant_stain': 'grayscale', 'confidence': 0.0, 'analysis': 'Grayscale image'}
        
        # Get important regions
        attr_norm = np.abs(attribution_map)
        if attr_norm.max() > attr_norm.min():
            attr_norm = (attr_norm - attr_norm.min()) / (attr_norm.max() - attr_norm.min())
        
        threshold = np.percentile(attr_norm, threshold_percentile)
        important_mask = attr_norm >= threshold
        
        if np.sum(important_mask) == 0:
            return {'dominant_stain': 'unclear', 'confidence': 0.0, 'analysis': 'No significant regions identified'}
        
        # Extract RGB values from important regions
        important_pixels = original_image[important_mask]
        
        # H&E staining analysis
        mean_rgb = np.mean(important_pixels, axis=0)
        std_rgb = np.std(important_pixels, axis=0)
        
        # Hematoxylin detection (blue/purple nuclei)
        # Higher blue channel, lower red/green
        hematoxylin_score = mean_rgb[2] - (mean_rgb[0] + mean_rgb[1]) / 2
        
        # Eosin detection (pink/red cytoplasm)  
        # Higher red channel, lower blue
        eosin_score = mean_rgb[0] - mean_rgb[2]
        
        # Determine dominant stain
        if abs(hematoxylin_score) > abs(eosin_score):
            if hematoxylin_score > 0:
                dominant_stain = 'hematoxylin'
                confidence = hematoxylin_score / (np.sum(mean_rgb) + 1e-8)
            else:
                dominant_stain = 'mixed'
                confidence = 0.5
        else:
            if eosin_score > 0:
                dominant_stain = 'eosin'
                confidence = eosin_score / (np.sum(mean_rgb) + 1e-8)
            else:
                dominant_stain = 'mixed'
                confidence = 0.5
        
        # Color uniformity analysis
        color_uniformity = 1.0 - (np.mean(std_rgb) / (np.mean(mean_rgb) + 1e-8))
        
        return {
            'dominant_stain': dominant_stain,
            'confidence': abs(confidence),
            'mean_rgb': mean_rgb.tolist(),
            'std_rgb': std_rgb.tolist(),
            'color_uniformity': color_uniformity,
            'hematoxylin_score': hematoxylin_score,
            'eosin_score': eosin_score
        }
    
    def analyze_structural_features(self, attribution_map: np.ndarray,
                                  original_image: np.ndarray) -> Dict:
        """Analyze structural irregularities and texture patterns"""
        # Normalize attribution
        attr_norm = np.abs(attribution_map)
        if attr_norm.max() > attr_norm.min():
            attr_norm = (attr_norm - attr_norm.min()) / (attr_norm.max() - attr_norm.min())
        
        # Texture analysis
        # Local variance (texture complexity)
        kernel = np.ones((5, 5)) / 25
        local_mean = cv2.filter2D(attr_norm.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((attr_norm - local_mean)**2, -1, kernel)
        texture_complexity = np.mean(local_variance)
        
        # Edge analysis
        edges = cv2.Canny((attr_norm * 255).astype(np.uint8), 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Gradient analysis
        grad_x = cv2.Sobel(attr_norm, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(attr_norm, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_strength = np.mean(gradient_magnitude)
        
        # Structural regularity
        # Use autocorrelation to measure periodicity
        from scipy.signal import correlate2d
        autocorr = correlate2d(attr_norm, attr_norm, mode='same')
        autocorr_center = autocorr[autocorr.shape[0]//2, autocorr.shape[1]//2]
        autocorr_normalized = autocorr / (autocorr_center + 1e-8)
        
        # Measure how quickly autocorrelation decays (structural regularity)
        center_y, center_x = autocorr.shape[0]//2, autocorr.shape[1]//2
        distances = []
        correlations = []
        
        for r in range(1, min(center_x, center_y, 20)):
            # Sample points at distance r from center
            angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
            for angle in angles:
                y = int(center_y + r * np.sin(angle))
                x = int(center_x + r * np.cos(angle))
                if 0 <= y < autocorr.shape[0] and 0 <= x < autocorr.shape[1]:
                    distances.append(r)
                    correlations.append(autocorr_normalized[y, x])
        
        # Structural regularity score
        if distances:
            structural_regularity = np.mean(correlations)
        else:
            structural_regularity = 0.0
        
        return {
            'texture_complexity': texture_complexity,
            'edge_density': edge_density,
            'gradient_strength': gradient_strength,
            'structural_regularity': structural_regularity,
            'irregularity_score': 1.0 - structural_regularity
        }
    
    def generate_comprehensive_report(self, attribution_map: np.ndarray,
                                    original_image: np.ndarray,
                                    prediction_info: Dict,
                                    method_name: str = "Integrated Gradients") -> str:
        """Generate comprehensive textual pathology report"""
        
        # Perform analyses
        tissue_analysis = self.analyze_tissue_distribution(attribution_map)
        staining_analysis = self.analyze_staining_patterns(original_image, attribution_map)
        structural_analysis = self.analyze_structural_features(attribution_map, original_image)
        
        # Extract prediction information
        predicted_class = prediction_info.get('predicted_class', 0)
        confidence = prediction_info.get('confidence', 0.0)
        
        if predicted_class < len(self.class_names):
            class_name = self.class_names[predicted_class]
        else:
            class_name = f'Class {predicted_class}'
        
        # Generate report sections
        report = f"""
HISTOPATHOLOGY EXPLAINABILITY ANALYSIS REPORT
Method: {method_name}
Generated: Automated XAI Analysis

═══════════════════════════════════════════════════════════════

DIAGNOSTIC PREDICTION:
• Classification: {class_name}
• Model Confidence: {confidence:.1%}
• Reliability Assessment: {self._get_confidence_assessment(confidence)}

═══════════════════════════════════════════════════════════════

TISSUE DISTRIBUTION ANALYSIS:
• Highlighted Tissue Proportion: {tissue_analysis['tissue_proportion']:.1f}% of total image
• Number of Distinct Regions: {tissue_analysis['num_regions']}
• Spatial Pattern: {self._interpret_spatial_pattern(tissue_analysis)}
• Fragmentation Index: {tissue_analysis['fragmentation_index']:.2f}
• Clinical Interpretation: {self._interpret_tissue_distribution(tissue_analysis['tissue_proportion'])}

═══════════════════════════════════════════════════════════════

STAINING PATTERN ANALYSIS:
• Dominant Staining: {staining_analysis['dominant_stain'].title()}
• Staining Confidence: {staining_analysis['confidence']:.2f}
• Pattern Description: {self.staining_patterns.get(staining_analysis['dominant_stain'], {}).get('description', 'Unknown pattern')}
• Clinical Significance: {self.staining_patterns.get(staining_analysis['dominant_stain'], {}).get('clinical_significance', 'Unclear significance')}
• Pathological Relevance: {self.staining_patterns.get(staining_analysis['dominant_stain'], {}).get('pathological_relevance', 'Unknown relevance')}

═══════════════════════════════════════════════════════════════

STRUCTURAL IRREGULARITIES:
• Texture Complexity: {structural_analysis['texture_complexity']:.3f}
• Edge Density: {structural_analysis['edge_density']:.3f}
• Gradient Strength: {structural_analysis['gradient_strength']:.3f}
• Structural Regularity: {structural_analysis['structural_regularity']:.3f}
• Irregularity Assessment: {self._interpret_structural_irregularities(structural_analysis)}

═══════════════════════════════════════════════════════════════

MODEL CONFIDENCE ALIGNMENT:
• Attribution Strength: {self._assess_attribution_strength(attribution_map)}
• Decision Focus: {self._assess_decision_focus(tissue_analysis, confidence)}
• Reliability Indicators: {self._assess_reliability(tissue_analysis, confidence, structural_analysis)}

═══════════════════════════════════════════════════════════════

INTEGRATED CLINICAL INTERPRETATION:
{self._generate_clinical_interpretation(tissue_analysis, staining_analysis, structural_analysis, prediction_info)}

═══════════════════════════════════════════════════════════════

RECOMMENDATIONS:
{self._generate_recommendations(tissue_analysis, staining_analysis, structural_analysis, prediction_info)}

═══════════════════════════════════════════════════════════════

DISCLAIMER:
This automated analysis is generated by explainable AI methods and should be 
interpreted by qualified pathologists. The analysis provides insights into model 
decision-making but does not replace expert clinical judgment.
"""
        
        return report.strip()
    
    def _get_confidence_assessment(self, confidence: float) -> str:
        """Get confidence level assessment"""
        for level, info in self.confidence_levels.items():
            if confidence >= info['threshold']:
                return f"{info['description']} - {info['reliability']}"
        return "very low confidence - requires expert review"
    
    def _interpret_spatial_pattern(self, tissue_analysis: Dict) -> str:
        """Interpret spatial distribution pattern"""
        num_regions = tissue_analysis['num_regions']
        fragmentation = tissue_analysis['fragmentation_index']
        
        if num_regions == 1:
            return "Single cohesive region"
        elif num_regions <= 3:
            return "Few distinct regions"
        elif fragmentation > 2.0:
            return "Highly fragmented distribution"
        else:
            return "Multiple scattered regions"
    
    def _interpret_tissue_distribution(self, proportion: float) -> str:
        """Interpret tissue proportion clinically"""
        if proportion > 40:
            return "Extensive tissue involvement suggesting widespread pathological changes"
        elif proportion > 20:
            return "Moderate tissue involvement indicating regional abnormalities"
        else:
            return "Focal tissue involvement suggesting localized changes"
    
    def _interpret_structural_irregularities(self, structural: Dict) -> str:
        """Interpret structural analysis"""
        texture = structural['texture_complexity']
        edges = structural['edge_density']
        irregularity = structural['irregularity_score']
        
        if irregularity > 0.7:
            return "High structural irregularity with complex tissue architecture"
        elif irregularity > 0.4:
            return "Moderate structural irregularity with some architectural distortion"
        else:
            return "Low structural irregularity with preserved tissue architecture"
    
    def _assess_attribution_strength(self, attribution_map: np.ndarray) -> str:
        """Assess strength of model attributions"""
        std_attr = np.std(attribution_map)
        if std_attr > 0.1:
            return "Strong and focused attributions"
        elif std_attr > 0.05:
            return "Moderate attribution strength"
        else:
            return "Weak or diffuse attributions"
    
    def _assess_decision_focus(self, tissue_analysis: Dict, confidence: float) -> str:
        """Assess how focused the model's decision is"""
        proportion = tissue_analysis['tissue_proportion']
        
        if confidence > 0.8 and proportion < 30:
            return "Highly focused decision with concentrated attention"
        elif confidence > 0.6 and proportion < 50:
            return "Moderately focused decision with regional attention"
        else:
            return "Diffuse decision with distributed attention"
    
    def _assess_reliability(self, tissue_analysis: Dict, confidence: float, 
                          structural_analysis: Dict) -> str:
        """Assess overall reliability indicators"""
        indicators = []
        
        if confidence > 0.8:
            indicators.append("High prediction confidence")
        
        if tissue_analysis['tissue_proportion'] > 15:
            indicators.append("Sufficient tissue coverage")
        
        if structural_analysis['edge_density'] > 0.2:
            indicators.append("Clear structural boundaries")
        
        if len(indicators) >= 2:
            return "Multiple positive reliability indicators"
        elif len(indicators) == 1:
            return "Some reliability indicators present"
        else:
            return "Limited reliability indicators - caution advised"
    
    def _generate_clinical_interpretation(self, tissue_analysis: Dict, 
                                        staining_analysis: Dict,
                                        structural_analysis: Dict,
                                        prediction_info: Dict) -> str:
        """Generate integrated clinical interpretation"""
        interpretation = []
        
        # Combine findings
        dominant_stain = staining_analysis['dominant_stain']
        tissue_prop = tissue_analysis['tissue_proportion']
        irregularity = structural_analysis['irregularity_score']
        confidence = prediction_info.get('confidence', 0.0)
        
        # Staining-based interpretation
        if dominant_stain == 'hematoxylin' and irregularity > 0.5:
            interpretation.append("Model emphasizes nuclear regions with architectural distortion, suggesting focus on cellular proliferation and nuclear morphology changes.")
        elif dominant_stain == 'eosin' and structural_analysis['edge_density'] > 0.3:
            interpretation.append("Model highlights cytoplasmic regions with distinct boundaries, indicating attention to structural and inflammatory changes.")
        
        # Tissue distribution interpretation
        if tissue_prop > 40 and confidence > 0.7:
            interpretation.append("Extensive tissue involvement with high confidence suggests reliable identification of widespread pathological features.")
        elif tissue_prop < 20 and confidence > 0.8:
            interpretation.append("Focal attention with high confidence indicates precise identification of specific pathological regions.")
        
        # Structural interpretation
        if irregularity > 0.7:
            interpretation.append("High structural irregularity suggests complex pathological architecture requiring careful evaluation.")
        
        if not interpretation:
            interpretation.append("Model shows mixed attention patterns across multiple tissue components without clear dominant features.")
        
        return "\n• ".join([""] + interpretation)
    
    def _generate_recommendations(self, tissue_analysis: Dict, 
                                staining_analysis: Dict,
                                structural_analysis: Dict,
                                prediction_info: Dict) -> str:
        """Generate clinical recommendations"""
        recommendations = []
        
        confidence = prediction_info.get('confidence', 0.0)
        tissue_prop = tissue_analysis['tissue_proportion']
        
        # Confidence-based recommendations
        if confidence < 0.6:
            recommendations.append("Low model confidence - recommend expert pathologist review")
        
        # Coverage-based recommendations
        if tissue_prop < 10:
            recommendations.append("Limited tissue coverage - consider additional image analysis")
        
        # Structural-based recommendations
        if structural_analysis['irregularity_score'] > 0.8:
            recommendations.append("High structural complexity - recommend detailed morphological assessment")
        
        # Staining-based recommendations
        if staining_analysis['confidence'] < 0.3:
            recommendations.append("Unclear staining patterns - verify image quality and staining protocol")
        
        # General recommendations
        recommendations.append("Correlate AI analysis with clinical history and additional diagnostic tests")
        recommendations.append("Consider this analysis as supportive evidence alongside traditional pathological evaluation")
        
        return "\n• ".join([""] + recommendations)