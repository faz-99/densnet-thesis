"""
Comprehensive Explainability Pipeline for Histopathology Images
Integrates Grad-CAM, SHAP, LIME with morphological analysis and report generation
"""
import torch
import numpy as np
import cv2
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from PIL import Image

from .grad_cam import GradCAM, GradCAMPlusPlus, overlay_heatmap
from .morphological_analyzer import MorphologicalAnalyzer, ClinicalDescriptorGenerator, save_explainability_report


class ComprehensiveExplainer:
    """Complete explainability pipeline for histopathology analysis"""
    
    def __init__(self, model, device: str, class_names: List[str]):
        self.model = model
        self.device = device
        self.class_names = class_names
        
        # Initialize explainers
        self.gradcam = None
        self.gradcam_plus = None
        self.shap_explainer = None
        self.lime_explainer = None
        
        # Initialize analyzers
        self.morphological_analyzer = MorphologicalAnalyzer()
        self.descriptor_generator = ClinicalDescriptorGenerator()
        
        self._initialize_explainers()
    
    def _initialize_explainers(self):
        """Initialize all explainability methods"""
        try:
            # Grad-CAM initialization
            self.gradcam = GradCAM(self.model, target_layer_name='densenet.features.norm5')
            self.gradcam_plus = GradCAMPlusPlus(self.model, target_layer_name='densenet.features.norm5')
        except Exception as e:
            print(f"Warning: Could not initialize Grad-CAM: {e}")
        
        try:
            # SHAP initialization
            import shap
            
            class SHAPModelWrapper(torch.nn.Module):
                def __init__(self, model, device):
                    super().__init__()
                    self.model = model
                    self.device = device
                
                def forward(self, x):
                    if not isinstance(x, torch.Tensor):
                        x = torch.tensor(x, dtype=torch.float32, device=self.device)
                    else:
                        x = x.to(self.device).float()
                    
                    outputs = self.model(x)
                    return torch.nn.functional.softmax(outputs, dim=1)
            
            wrapped_model = SHAPModelWrapper(self.model, self.device)
            wrapped_model.eval()
            
            background_data = torch.randn(3, 3, 224, 224).to(self.device).float()
            
            try:
                self.shap_explainer = shap.DeepExplainer(wrapped_model, background_data)
            except:
                self.shap_explainer = shap.GradientExplainer(wrapped_model, background_data)
                
        except Exception as e:
            print(f"Warning: Could not initialize SHAP: {e}")
        
        try:
            # LIME initialization
            from .lime_explainer import LIMEExplainer
            self.lime_explainer = LIMEExplainer(self.model, self.device, num_samples=100)
        except Exception as e:
            print(f"Warning: Could not initialize LIME: {e}")
    
    def generate_comprehensive_explanation(self, image_tensor: torch.Tensor, 
                                         original_image: np.ndarray,
                                         image_id: str,
                                         save_dir: str = "explainability_reports") -> Dict:
        """
        Generate comprehensive explainability analysis
        
        Args:
            image_tensor: Preprocessed image tensor (1, C, H, W)
            original_image: Original image array (H, W, C) in [0, 1]
            image_id: Unique identifier for the image
            save_dir: Directory to save results
            
        Returns:
            Comprehensive analysis results
        """
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class_idx].item()
        
        predicted_class = self.class_names[predicted_class_idx]
        
        # Generate explainability maps
        explanation_maps = self._generate_all_explanations(image_tensor, predicted_class_idx)
        
        # Analyze each explanation method
        analysis_results = {}
        
        for method_name, activation_map in explanation_maps.items():
            if activation_map is not None:
                # Extract morphological features
                features = self.morphological_analyzer.analyze_activation_map(
                    original_image, activation_map)
                
                # Generate clinical description
                description = self.descriptor_generator.generate_description(
                    features, predicted_class, confidence)
                
                # Generate detailed report
                detailed_report = self.descriptor_generator.generate_detailed_report(
                    features, predicted_class, confidence, f"{image_id}_{method_name}")
                
                analysis_results[method_name] = {
                    'activation_map': activation_map,
                    'features': features,
                    'description': description,
                    'detailed_report': detailed_report
                }
        
        # Create visual outputs
        visual_outputs = self._create_visual_outputs(
            original_image, explanation_maps, image_id, save_dir)
        
        # Generate comprehensive report
        comprehensive_report = self._generate_comprehensive_report(
            image_id, predicted_class, confidence, analysis_results, visual_outputs)
        
        # Save all outputs
        self._save_all_outputs(comprehensive_report, save_dir)
        
        return comprehensive_report
    
    def _generate_all_explanations(self, image_tensor: torch.Tensor, 
                                 predicted_class: int) -> Dict[str, np.ndarray]:
        """Generate explanation maps from all available methods"""
        explanation_maps = {}
        
        # Grad-CAM
        if self.gradcam is not None:
            try:
                gradcam_map = self.gradcam.generate_cam(image_tensor, predicted_class)
                explanation_maps['gradcam'] = gradcam_map
            except Exception as e:
                print(f"Grad-CAM failed: {e}")
        
        # Grad-CAM++
        if self.gradcam_plus is not None:
            try:
                gradcam_plus_map = self.gradcam_plus.generate_cam(image_tensor, predicted_class)
                explanation_maps['gradcam_plus'] = gradcam_plus_map
            except Exception as e:
                print(f"Grad-CAM++ failed: {e}")
        
        # SHAP
        if self.shap_explainer is not None:
            try:
                shap_values = self.shap_explainer.shap_values(image_tensor.cpu().numpy())
                if isinstance(shap_values, list):
                    shap_map = shap_values[predicted_class][0]
                else:
                    shap_map = shap_values[0]
                
                # Convert to single channel if needed
                if len(shap_map.shape) == 3:
                    shap_map = np.sum(np.abs(shap_map), axis=0)
                
                # Normalize
                shap_map = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min() + 1e-8)
                explanation_maps['shap'] = shap_map
            except Exception as e:
                print(f"SHAP failed: {e}")
        
        # LIME
        if self.lime_explainer is not None:
            try:
                explanation, segments = self.lime_explainer.explain_image(
                    image_tensor[0].cpu().numpy().transpose(1, 2, 0))
                
                # Extract LIME mask
                temp, mask = explanation.get_image_and_mask(
                    predicted_class, positive_only=False, num_features=10, hide_rest=False)
                
                # Convert mask to activation map
                lime_map = np.abs(mask).astype(float)
                lime_map = (lime_map - lime_map.min()) / (lime_map.max() - lime_map.min() + 1e-8)
                explanation_maps['lime'] = lime_map
            except Exception as e:
                print(f"LIME failed: {e}")
        
        return explanation_maps
    
    def _create_visual_outputs(self, original_image: np.ndarray, 
                             explanation_maps: Dict[str, np.ndarray],
                             image_id: str, save_dir: str) -> Dict:
        """Create and save visual outputs"""
        
        os.makedirs(os.path.join(save_dir, "visualizations"), exist_ok=True)
        
        visual_outputs = {}
        
        # Create comprehensive visualization
        num_methods = len(explanation_maps)
        if num_methods > 0:
            fig, axes = plt.subplots(3, num_methods + 1, figsize=(4 * (num_methods + 1), 12))
            
            if num_methods == 1:
                axes = axes.reshape(3, -1)
            
            # Original image in first column
            for row in range(3):
                axes[row, 0].imshow(original_image)
                axes[row, 0].set_title('Original Image' if row == 0 else '')
                axes[row, 0].axis('off')
            
            # Process each explanation method
            for col, (method_name, activation_map) in enumerate(explanation_maps.items(), 1):
                # Raw activation map
                im1 = axes[0, col].imshow(activation_map, cmap='jet')
                axes[0, col].set_title(f'{method_name.upper()} Heatmap')
                axes[0, col].axis('off')
                plt.colorbar(im1, ax=axes[0, col], fraction=0.046, pad=0.04)
                
                # Overlay
                overlay = overlay_heatmap(original_image, activation_map, alpha=0.4)
                axes[1, col].imshow(overlay)
                axes[1, col].set_title(f'{method_name.upper()} Overlay')
                axes[1, col].axis('off')
                
                # Binary mask (high activation regions)
                binary_mask = activation_map > 0.5
                axes[2, col].imshow(binary_mask, cmap='gray')
                axes[2, col].set_title(f'{method_name.upper()} Mask')
                axes[2, col].axis('off')
                
                # Save individual maps
                individual_save_path = os.path.join(
                    save_dir, "visualizations", f"{image_id}_{method_name}")
                
                # Save raw activation map
                np.save(f"{individual_save_path}_raw.npy", activation_map)
                
                # Save overlay
                overlay_pil = Image.fromarray((overlay * 255).astype(np.uint8))
                overlay_pil.save(f"{individual_save_path}_overlay.png")
                
                # Save binary mask
                mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
                mask_pil.save(f"{individual_save_path}_mask.png")
                
                visual_outputs[method_name] = {
                    'raw_map_path': f"{individual_save_path}_raw.npy",
                    'overlay_path': f"{individual_save_path}_overlay.png",
                    'mask_path': f"{individual_save_path}_mask.png"
                }
            
            plt.tight_layout()
            comprehensive_viz_path = os.path.join(
                save_dir, "visualizations", f"{image_id}_comprehensive.png")
            plt.savefig(comprehensive_viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visual_outputs['comprehensive_visualization'] = comprehensive_viz_path
        
        return visual_outputs
    
    def _generate_comprehensive_report(self, image_id: str, predicted_class: str,
                                     confidence: float, analysis_results: Dict,
                                     visual_outputs: Dict) -> Dict:
        """Generate comprehensive analysis report"""
        
        # Aggregate findings across all methods
        all_findings = []
        all_descriptions = []
        method_summaries = {}
        
        for method_name, results in analysis_results.items():
            method_summaries[method_name] = {
                'tissue_area_percent': results['features']['tissue_area_percent'],
                'dominant_stain': results['features']['stain_analysis']['dominant_stain'],
                'description': results['description'],
                'key_findings': results['detailed_report']['key_findings']
            }
            
            all_findings.extend(results['detailed_report']['key_findings'])
            all_descriptions.append(f"{method_name.upper()}: {results['description']}")
        
        # Generate consensus findings
        consensus_findings = self._generate_consensus_findings(analysis_results)
        
        # Create final clinical interpretation
        clinical_interpretation = self._generate_final_clinical_interpretation(
            predicted_class, confidence, consensus_findings)
        
        comprehensive_report = {
            'metadata': {
                'image_id': image_id,
                'analysis_timestamp': datetime.now().isoformat(),
                'predicted_class': predicted_class,
                'confidence': confidence,
                'methods_used': list(analysis_results.keys())
            },
            'consensus_findings': consensus_findings,
            'clinical_interpretation': clinical_interpretation,
            'method_summaries': method_summaries,
            'detailed_descriptions': all_descriptions,
            'visual_outputs': visual_outputs,
            'detailed_analysis': analysis_results
        }
        
        return comprehensive_report
    
    def _generate_consensus_findings(self, analysis_results: Dict) -> Dict:
        """Generate consensus findings across all methods"""
        
        if not analysis_results:
            return {}
        
        # Aggregate tissue area percentages
        tissue_areas = [results['features']['tissue_area_percent'] 
                       for results in analysis_results.values()]
        
        # Aggregate stain analysis
        stain_votes = {}
        for results in analysis_results.values():
            stain = results['features']['stain_analysis']['dominant_stain']
            stain_votes[stain] = stain_votes.get(stain, 0) + 1
        
        # Aggregate texture features
        texture_features = []
        for results in analysis_results.values():
            texture_features.append(results['features']['texture_features'])
        
        consensus = {
            'mean_tissue_area_percent': np.mean(tissue_areas),
            'tissue_area_range': [min(tissue_areas), max(tissue_areas)],
            'consensus_stain': max(stain_votes.items(), key=lambda x: x[1])[0],
            'stain_agreement': max(stain_votes.values()) / len(analysis_results),
            'mean_entropy': np.mean([tf['entropy'] for tf in texture_features]),
            'mean_edge_density': np.mean([tf['edge_density'] for tf in texture_features]),
            'methods_agreement': len(analysis_results)
        }
        
        return consensus
    
    def _generate_final_clinical_interpretation(self, predicted_class: str,
                                              confidence: float,
                                              consensus_findings: Dict) -> str:
        """Generate final clinical interpretation"""
        
        if not consensus_findings:
            return f"Model predicts {predicted_class} with {confidence:.1%} confidence."
        
        interpretation_parts = []
        
        # Confidence assessment
        if confidence > 0.9:
            interpretation_parts.append(f"High confidence {predicted_class.lower()} classification")
        elif confidence > 0.7:
            interpretation_parts.append(f"Moderate confidence {predicted_class.lower()} classification")
        else:
            interpretation_parts.append(f"Low confidence {predicted_class.lower()} classification")
        
        # Tissue coverage
        mean_area = consensus_findings['mean_tissue_area_percent']
        if mean_area > 40:
            interpretation_parts.append("with extensive model attention across tissue regions")
        elif mean_area > 20:
            interpretation_parts.append("with moderate model attention to specific tissue areas")
        else:
            interpretation_parts.append("with focal model attention to limited tissue regions")
        
        # Stain pattern
        consensus_stain = consensus_findings['consensus_stain']
        stain_agreement = consensus_findings['stain_agreement']
        
        if stain_agreement > 0.7:
            if consensus_stain == 'hematoxylin':
                interpretation_parts.append("Consistent nuclear staining patterns suggest cellular density changes")
            elif consensus_stain == 'eosin':
                interpretation_parts.append("Consistent cytoplasmic staining patterns indicate structural alterations")
        
        # Texture analysis
        mean_entropy = consensus_findings['mean_entropy']
        if mean_entropy > 6.0:
            interpretation_parts.append("High cellular heterogeneity observed")
        elif mean_entropy < 4.0:
            interpretation_parts.append("Uniform cellular architecture noted")
        
        # Methods agreement
        methods_count = consensus_findings['methods_agreement']
        if methods_count > 2:
            interpretation_parts.append(f"Findings supported by {methods_count} independent analysis methods")
        
        return ". ".join(interpretation_parts) + "."
    
    def _save_all_outputs(self, comprehensive_report: Dict, save_dir: str):
        """Save comprehensive report and all outputs"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        image_id = comprehensive_report['metadata']['image_id']
        
        # Save comprehensive JSON report
        json_path = os.path.join(save_dir, f"{image_id}_comprehensive_report.json")
        with open(json_path, 'w') as f:
            # Create a serializable version
            serializable_report = self._make_serializable(comprehensive_report)
            json.dump(serializable_report, f, indent=2)
        
        # Save human-readable text report
        txt_path = os.path.join(save_dir, f"{image_id}_comprehensive_report.txt")
        with open(txt_path, 'w') as f:
            self._write_text_report(f, comprehensive_report)
        
        # Save individual method reports
        for method_name, results in comprehensive_report['detailed_analysis'].items():
            method_report_path = os.path.join(save_dir, f"{image_id}_{method_name}_detailed.json")
            with open(method_report_path, 'w') as f:
                serializable_results = self._make_serializable(results)
                json.dump(serializable_results, f, indent=2)
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to serializable format"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def _write_text_report(self, f, report: Dict):
        """Write human-readable text report"""
        metadata = report['metadata']
        consensus = report['consensus_findings']
        
        f.write("COMPREHENSIVE HISTOPATHOLOGY EXPLAINABILITY REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Metadata
        f.write("ANALYSIS METADATA\n")
        f.write("-" * 20 + "\n")
        f.write(f"Image ID: {metadata['image_id']}\n")
        f.write(f"Analysis Date: {metadata['analysis_timestamp']}\n")
        f.write(f"Predicted Class: {metadata['predicted_class']}\n")
        f.write(f"Confidence: {metadata['confidence']:.1%}\n")
        f.write(f"Methods Used: {', '.join(metadata['methods_used'])}\n\n")
        
        # Clinical interpretation
        f.write("CLINICAL INTERPRETATION\n")
        f.write("-" * 25 + "\n")
        f.write(f"{report['clinical_interpretation']}\n\n")
        
        # Consensus findings
        if consensus:
            f.write("CONSENSUS FINDINGS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Average Tissue Area Highlighted: {consensus['mean_tissue_area_percent']:.1f}%\n")
            f.write(f"Dominant Staining Pattern: {consensus['consensus_stain']}\n")
            f.write(f"Methods Agreement: {consensus['stain_agreement']:.1%}\n")
            f.write(f"Cellular Heterogeneity (Entropy): {consensus['mean_entropy']:.2f}\n")
            f.write(f"Edge Density: {consensus['mean_edge_density']:.3f}\n\n")
        
        # Method-specific summaries
        f.write("METHOD-SPECIFIC SUMMARIES\n")
        f.write("-" * 30 + "\n")
        for method_name, summary in report['method_summaries'].items():
            f.write(f"\n{method_name.upper()}:\n")
            f.write(f"  Tissue Area: {summary['tissue_area_percent']:.1f}%\n")
            f.write(f"  Dominant Stain: {summary['dominant_stain']}\n")
            f.write(f"  Description: {summary['description']}\n")
            if summary['key_findings']:
                f.write("  Key Findings:\n")
                for finding in summary['key_findings']:
                    f.write(f"    - {finding}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("End of Report\n")


def process_dataset_batch(model, device: str, class_names: List[str],
                         image_paths: List[str], save_dir: str = "explainability_reports"):
    """Process a batch of images for comprehensive explainability analysis"""
    
    explainer = ComprehensiveExplainer(model, device, class_names)
    
    results = []
    
    for i, image_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
        
        try:
            # Load and preprocess image
            from PIL import Image as PILImage
            import torchvision.transforms as transforms
            
            # Load image
            pil_image = PILImage.open(image_path).convert('RGB')
            original_image = np.array(pil_image) / 255.0
            
            # Preprocess for model
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5613, 0.5778, 0.6032], 
                                   std=[0.2114, 0.1957, 0.1590])
            ])
            
            image_tensor = transform(pil_image).unsqueeze(0).to(device)
            
            # Generate image ID
            image_id = os.path.splitext(os.path.basename(image_path))[0]
            
            # Process image
            result = explainer.generate_comprehensive_explanation(
                image_tensor, original_image, image_id, save_dir)
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    # Generate batch summary
    batch_summary_path = os.path.join(save_dir, "batch_summary.json")
    with open(batch_summary_path, 'w') as f:
        summary = {
            'processed_images': len(results),
            'total_images': len(image_paths),
            'success_rate': len(results) / len(image_paths),
            'analysis_timestamp': datetime.now().isoformat(),
            'results': [r['metadata'] for r in results]
        }
        json.dump(summary, f, indent=2)
    
    print(f"Batch processing complete. Results saved to {save_dir}")
    return results