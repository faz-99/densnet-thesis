"""
Enhanced LIME (Local Interpretable Model-agnostic Explanations) implementation for DenLsNet explainability
Includes comprehensive analysis, JSON export, and quantitative evaluation
"""
import torch
import torch.nn.functional as F
import numpy as np
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import matplotlib.pyplot as plt
import os
import json
from typing import List, Tuple, Optional, Dict
from skimage.segmentation import mark_boundaries
import cv2
from datetime import datetime
from collections import defaultdict


class LIMEExplainer:
    """LIME explainer for image classification models"""
    
    def __init__(self, model, device: str = 'cpu', num_samples: int = 1000):
        """
        Initialize LIME explainer
        
        Args:
            model: Trained model
            device: Computing device
            num_samples: Number of samples for LIME perturbation
        """
        self.model = model
        self.device = device
        self.num_samples = num_samples
        
        # Initialize LIME image explainer
        self.explainer = lime_image.LimeImageExplainer()
        
        # Segmentation algorithm for superpixels
        self.segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4, max_dist=200, ratio=0.2)
    
    def _predict_fn(self, images: np.ndarray) -> np.ndarray:
        """
        Prediction function for LIME
        
        Args:
            images: Batch of images (B, H, W, C)
            
        Returns:
            Prediction probabilities (B, num_classes)
        """
        # Convert to torch tensor and adjust dimensions
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)
        
        # Convert from (B, H, W, C) to (B, C, H, W)
        images_tensor = torch.tensor(images.transpose(0, 3, 1, 2), dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(images_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def explain_image(
        self, 
        image: np.ndarray, 
        top_labels: int = 2,
        num_features: int = 10,
        hide_color: int = 0
    ) -> Tuple[object, np.ndarray]:
        """
        Generate LIME explanation for a single image
        
        Args:
            image: Input image as numpy array (H, W, C) in range [0, 1]
            top_labels: Number of top predicted labels to explain
            num_features: Number of superpixels to include in explanation
            hide_color: Color to use for hiding superpixels (0=black, 1=white)
            
        Returns:
            Tuple of (explanation object, segmentation mask)
        """
        # Ensure image is in correct format
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            image_uint8,
            self._predict_fn,
            top_labels=top_labels,
            hide_color=hide_color,
            num_samples=self.num_samples,
            segmentation_fn=self.segmentation_fn
        )
        
        # Get segmentation
        segments = self.segmentation_fn(image_uint8)
        
        return explanation, segments
    
    def get_image_and_mask(
        self, 
        explanation, 
        label: int, 
        positive_only: bool = True,
        negative_only: bool = False,
        hide_rest: bool = False,
        num_features: int = 10,
        min_weight: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get image and mask from LIME explanation
        
        Args:
            explanation: LIME explanation object
            label: Class label to explain
            positive_only: Show only positive contributions
            negative_only: Show only negative contributions
            hide_rest: Hide non-contributing regions
            num_features: Number of features to show
            min_weight: Minimum weight threshold
            
        Returns:
            Tuple of (image, mask)
        """
        return explanation.get_image_and_mask(
            label,
            positive_only=positive_only,
            negative_only=negative_only,
            hide_rest=hide_rest,
            num_features=num_features,
            min_weight=min_weight
        )


def visualize_lime_explanation(
    original_image: np.ndarray,
    explanation,
    segments: np.ndarray,
    predicted_class: str,
    true_class: str,
    confidence: float,
    class_idx: int,
    save_path: str
):
    """
    Visualize LIME explanation
    
    Args:
        original_image: Original input image (H, W, C)
        explanation: LIME explanation object
        segments: Segmentation mask
        predicted_class: Model prediction
        true_class: Ground truth label
        confidence: Prediction confidence
        class_idx: Class index for explanation
        save_path: Path to save visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Superpixel segmentation
    axes[0, 1].imshow(mark_boundaries(original_image, segments))
    axes[0, 1].set_title('Superpixel Segmentation')
    axes[0, 1].axis('off')
    
    # LIME explanation - positive contributions
    temp_pos, mask_pos = explanation.get_image_and_mask(
        class_idx, positive_only=True, num_features=10, hide_rest=False
    )
    axes[0, 2].imshow(mark_boundaries(temp_pos, mask_pos))
    axes[0, 2].set_title('Positive Contributions')
    axes[0, 2].axis('off')
    
    # LIME explanation - negative contributions
    temp_neg, mask_neg = explanation.get_image_and_mask(
        class_idx, negative_only=True, num_features=10, hide_rest=False
    )
    axes[1, 0].imshow(mark_boundaries(temp_neg, mask_neg))
    axes[1, 0].set_title('Negative Contributions')
    axes[1, 0].axis('off')
    
    # Combined explanation
    temp_combined, mask_combined = explanation.get_image_and_mask(
        class_idx, positive_only=False, num_features=10, hide_rest=True
    )
    axes[1, 1].imshow(mark_boundaries(temp_combined, mask_combined))
    axes[1, 1].set_title('Combined Explanation')
    axes[1, 1].axis('off')
    
    # Prediction info and feature importance
    axes[1, 2].text(0.1, 0.9, f'Predicted: {predicted_class}', fontsize=10, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.8, f'True Label: {true_class}', fontsize=10, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.7, f'Confidence: {confidence:.3f}', fontsize=10, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.6, f'Correct: {"✓" if predicted_class == true_class else "✗"}', 
                   fontsize=10, transform=axes[1, 2].transAxes)
    
    # Show top contributing superpixels
    local_exp = explanation.local_exp[class_idx]
    top_features = sorted(local_exp, key=lambda x: abs(x[1]), reverse=True)[:5]
    
    axes[1, 2].text(0.1, 0.4, 'Top Superpixels:', fontsize=10, fontweight='bold', transform=axes[1, 2].transAxes)
    for i, (feature, weight) in enumerate(top_features):
        color = 'green' if weight > 0 else 'red'
        axes[1, 2].text(0.1, 0.35 - i*0.05, f'#{feature}: {weight:.3f}', 
                       fontsize=8, color=color, transform=axes[1, 2].transAxes)
    
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_lime_comparison_plot(
    images_list: List[np.ndarray],
    explanations_list: List[object],
    class_names: List[str],
    predictions: List[int],
    true_labels: List[int],
    save_path: str
):
    """
    Create comparison plot showing LIME explanations for multiple samples
    
    Args:
        images_list: List of original images
        explanations_list: List of LIME explanation objects
        class_names: List of class names
        predictions: List of predicted class indices
        true_labels: List of true class indices
        save_path: Path to save the comparison plot
    """
    num_samples = len(images_list)
    fig, axes = plt.subplots(3, num_samples, figsize=(4*num_samples, 12))
    
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_samples):
        pred_class = predictions[i]
        true_class = true_labels[i]
        
        # Original image
        axes[0, i].imshow(images_list[i])
        axes[0, i].set_title(f'Sample {i+1}\nPred: {class_names[pred_class]}\nTrue: {class_names[true_class]}')
        axes[0, i].axis('off')
        
        # Positive contributions
        temp_pos, mask_pos = explanations_list[i].get_image_and_mask(
            pred_class, positive_only=True, num_features=8, hide_rest=False
        )
        axes[1, i].imshow(mark_boundaries(temp_pos, mask_pos))
        axes[1, i].set_title('Positive Contributions')
        axes[1, i].axis('off')
        
        # Combined explanation
        temp_combined, mask_combined = explanations_list[i].get_image_and_mask(
            pred_class, positive_only=False, num_features=8, hide_rest=True
        )
        axes[2, i].imshow(mark_boundaries(temp_combined, mask_combined))
        axes[2, i].set_title('Full Explanation')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def extract_lime_features(explanation, class_idx: int) -> Dict:
    """
    Extract detailed LIME features and weights
    
    Args:
        explanation: LIME explanation object
        class_idx: Class index to extract features for
        
    Returns:
        Dictionary with superpixel features and weights
    """
    local_exp = explanation.local_exp[class_idx]
    
    features = {
        'superpixel_weights': {},
        'positive_features': [],
        'negative_features': [],
        'total_positive_weight': 0.0,
        'total_negative_weight': 0.0,
        'num_features': len(local_exp),
        'feature_statistics': {
            'mean_weight': 0.0,
            'std_weight': 0.0,
            'max_positive': 0.0,
            'max_negative': 0.0
        }
    }
    
    weights = []
    
    for feature_id, weight in local_exp:
        features['superpixel_weights'][str(feature_id)] = float(weight)
        weights.append(weight)
        
        if weight > 0:
            features['positive_features'].append({
                'superpixel_id': int(feature_id),
                'weight': float(weight)
            })
            features['total_positive_weight'] += weight
        else:
            features['negative_features'].append({
                'superpixel_id': int(feature_id),
                'weight': float(weight)
            })
            features['total_negative_weight'] += weight
    
    # Calculate statistics
    if weights:
        features['feature_statistics'] = {
            'mean_weight': float(np.mean(weights)),
            'std_weight': float(np.std(weights)),
            'max_positive': float(max([w for w in weights if w > 0], default=0.0)),
            'max_negative': float(min([w for w in weights if w < 0], default=0.0))
        }
    
    # Sort features by absolute weight
    features['positive_features'].sort(key=lambda x: abs(x['weight']), reverse=True)
    features['negative_features'].sort(key=lambda x: abs(x['weight']), reverse=True)
    
    return features


def create_per_class_lime_analysis(
    explanations_by_class: Dict[str, List],
    class_names: List[str],
    save_dir: str
):
    """
    Create per-class LIME analysis showing superpixel patterns
    
    Args:
        explanations_by_class: Dictionary with class names as keys and explanation data as values
        class_names: List of class names
        save_dir: Directory to save results
    """
    # Analyze superpixel usage patterns per class
    class_analysis = {}
    
    for class_name in class_names:
        if class_name in explanations_by_class and explanations_by_class[class_name]:
            class_data = explanations_by_class[class_name]
            
            # Aggregate statistics
            all_positive_weights = []
            all_negative_weights = []
            superpixel_usage = defaultdict(list)
            
            for sample_data in class_data:
                features = sample_data['features']
                
                # Collect weights
                for feature in features['positive_features']:
                    all_positive_weights.append(feature['weight'])
                    superpixel_usage[feature['superpixel_id']].append(feature['weight'])
                
                for feature in features['negative_features']:
                    all_negative_weights.append(feature['weight'])
                    superpixel_usage[feature['superpixel_id']].append(feature['weight'])
            
            # Calculate class-level statistics
            class_analysis[class_name] = {
                'num_samples': len(class_data),
                'avg_positive_weight': float(np.mean(all_positive_weights)) if all_positive_weights else 0.0,
                'avg_negative_weight': float(np.mean(all_negative_weights)) if all_negative_weights else 0.0,
                'total_superpixels_used': len(superpixel_usage),
                'avg_superpixels_per_sample': float(np.mean([len(sample['features']['superpixel_weights']) for sample in class_data])),
                'weight_distribution': {
                    'positive_mean': float(np.mean(all_positive_weights)) if all_positive_weights else 0.0,
                    'positive_std': float(np.std(all_positive_weights)) if all_positive_weights else 0.0,
                    'negative_mean': float(np.mean(all_negative_weights)) if all_negative_weights else 0.0,
                    'negative_std': float(np.std(all_negative_weights)) if all_negative_weights else 0.0
                }
            }
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Average positive weights per class
    classes = list(class_analysis.keys())
    pos_weights = [class_analysis[cls]['avg_positive_weight'] for cls in classes]
    neg_weights = [abs(class_analysis[cls]['avg_negative_weight']) for cls in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, pos_weights, width, label='Positive', color='green', alpha=0.7)
    axes[0, 0].bar(x + width/2, neg_weights, width, label='Negative', color='red', alpha=0.7)
    axes[0, 0].set_title('Average Superpixel Weights by Class')
    axes[0, 0].set_ylabel('Average Weight')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(classes, rotation=45)
    axes[0, 0].legend()
    
    # Number of superpixels used per class
    superpixel_counts = [class_analysis[cls]['avg_superpixels_per_sample'] for cls in classes]
    axes[0, 1].bar(classes, superpixel_counts, alpha=0.7, color='blue')
    axes[0, 1].set_title('Average Superpixels per Sample by Class')
    axes[0, 1].set_ylabel('Number of Superpixels')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Weight distribution comparison
    pos_means = [class_analysis[cls]['weight_distribution']['positive_mean'] for cls in classes]
    pos_stds = [class_analysis[cls]['weight_distribution']['positive_std'] for cls in classes]
    
    axes[1, 0].errorbar(classes, pos_means, yerr=pos_stds, fmt='o-', capsize=5, color='green')
    axes[1, 0].set_title('Positive Weight Distribution by Class')
    axes[1, 0].set_ylabel('Weight (Mean ± Std)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Sample count per class
    sample_counts = [class_analysis[cls]['num_samples'] for cls in classes]
    axes[1, 1].bar(classes, sample_counts, alpha=0.7, color='orange')
    axes[1, 1].set_title('Number of Samples Analyzed per Class')
    axes[1, 1].set_ylabel('Sample Count')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_class_lime_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save numerical analysis
    analysis_path = os.path.join(save_dir, 'per_class_lime_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(class_analysis, f, indent=2)
    
    return class_analysis


def generate_comprehensive_lime_analysis(
    model,
    dataloader,
    device: str,
    class_names: List[str],
    save_dir: str = 'explainability/lime',
    samples_per_class: int = 2,
    lime_samples: int = 500
) -> Dict:
    """
    Generate comprehensive LIME analysis with all requested features
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Computing device
        class_names: List of class names
        save_dir: Directory to save results
        samples_per_class: Number of samples per class (2 representative samples per class)
        lime_samples: Number of samples for LIME perturbation
        
    Returns:
        Dictionary with analysis results
    """
    print("Starting comprehensive LIME analysis...")
    
    # Create output directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'individual'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'per_class'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'json_exports'), exist_ok=True)
    
    # Initialize LIME explainer
    lime_explainer = LIMEExplainer(model, device, lime_samples)
    
    model.eval()
    
    # Storage for analysis
    explanations_by_class = defaultdict(list)
    all_explanations = []
    
    # Collect samples (2 per class)
    class_sample_counts = defaultdict(int)
    total_samples_needed = len(class_names) * samples_per_class
    
    print(f"Collecting {total_samples_needed} samples ({samples_per_class} per class)...")
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if sample_count >= total_samples_needed:
                break
                
            images = images.to(device)
            labels = labels.to(device)
            
            # Get predictions
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
            
            for i in range(images.size(0)):
                if sample_count >= total_samples_needed:
                    break
                    
                true_label = labels[i].item()
                pred_label = predicted_classes[i].item()
                confidence = probabilities[i, pred_label].item()
                
                true_class_name = class_names[true_label]
                pred_class_name = class_names[pred_label]
                
                # Check if we need more samples for this class
                if class_sample_counts[true_class_name] >= samples_per_class:
                    continue
                
                # Prepare image for LIME
                image_np = images[i].cpu().numpy().transpose(1, 2, 0)
                
                # Denormalize image for LIME
                mean = np.array([0.5613, 0.5778, 0.6032])  # From config
                std = np.array([0.2114, 0.1957, 0.1590])
                image_np = image_np * std + mean
                image_np = np.clip(image_np, 0, 1)
                
                # Generate LIME explanation
                print(f"Generating LIME explanation {sample_count + 1}/{total_samples_needed} (Class: {true_class_name})")
                
                try:
                    explanation, segments = lime_explainer.explain_image(
                        image_np, 
                        top_labels=2, 
                        num_features=10
                    )
                    
                    # Extract features and weights
                    features = extract_lime_features(explanation, pred_label)
                    
                    # Create sample data
                    sample_data = {
                        'sample_id': f"{true_class_name}_{class_sample_counts[true_class_name]:02d}",
                        'true_class': true_class_name,
                        'predicted_class': pred_class_name,
                        'confidence': float(confidence),
                        'is_correct': true_label == pred_label,
                        'features': features,
                        'image_shape': image_np.shape,
                        'num_segments': int(np.max(segments)) + 1
                    }
                    
                    # Store data
                    explanations_by_class[true_class_name].append(sample_data)
                    all_explanations.append(sample_data)
                    
                    # Create individual visualization
                    individual_filename = f"{true_class_name}_{class_sample_counts[true_class_name]:02d}_lime.png"
                    individual_save_path = os.path.join(save_dir, 'individual', individual_filename)
                    
                    visualize_lime_explanation(
                        image_np,
                        explanation,
                        segments,
                        pred_class_name,
                        true_class_name,
                        confidence,
                        pred_label,
                        individual_save_path
                    )
                    
                    # Export explanation as JSON
                    json_filename = f"{true_class_name}_{class_sample_counts[true_class_name]:02d}_lime.json"
                    json_save_path = os.path.join(save_dir, 'json_exports', json_filename)
                    
                    with open(json_save_path, 'w') as f:
                        json.dump(sample_data, f, indent=2)
                    
                    class_sample_counts[true_class_name] += 1
                    sample_count += 1
                    
                except Exception as e:
                    print(f"Error generating LIME explanation: {e}")
                    continue
    
    print(f"Generated LIME explanations for {sample_count} samples")
    
    # Create per-class analysis
    print("Creating per-class LIME analysis...")
    class_analysis = create_per_class_lime_analysis(
        explanations_by_class,
        class_names,
        os.path.join(save_dir, 'per_class')
    )
    
    # Create comparison visualization
    print("Creating comparison visualization...")
    if len(all_explanations) >= 2:
        # Select one sample per class for comparison (up to 5)
        comparison_data = []
        classes_used = set()
        
        for sample_data in all_explanations:
            class_name = sample_data['true_class']
            if class_name not in classes_used and len(comparison_data) < 5:
                comparison_data.append(sample_data)
                classes_used.add(class_name)
        
        if len(comparison_data) >= 2:
            # Note: This would require re-loading images and explanations for visualization
            # For now, we'll create a summary instead
            create_lime_summary_visualization(comparison_data, save_dir)
    
    # Generate comprehensive results
    analysis_results = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': sample_count,
        'samples_per_class': dict(class_sample_counts),
        'lime_parameters': {
            'num_samples': lime_samples,
            'samples_per_class': samples_per_class
        },
        'class_analysis': class_analysis,
        'overall_statistics': {
            'avg_superpixels_per_sample': float(np.mean([exp['num_segments'] for exp in all_explanations])),
            'avg_positive_features': float(np.mean([len(exp['features']['positive_features']) for exp in all_explanations])),
            'avg_negative_features': float(np.mean([len(exp['features']['negative_features']) for exp in all_explanations])),
            'avg_positive_weight': float(np.mean([exp['features']['total_positive_weight'] for exp in all_explanations])),
            'avg_negative_weight': float(np.mean([abs(exp['features']['total_negative_weight']) for exp in all_explanations]))
        }
    }
    
    # Save comprehensive results
    results_path = os.path.join(save_dir, 'lime_analysis_results.json')
    with open(results_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Create summary report
    create_lime_summary_report(analysis_results, save_dir)
    
    print(f"Comprehensive LIME analysis completed!")
    print(f"Results saved to: {save_dir}")
    print(f"- Individual explanations: {save_dir}/individual/")
    print(f"- JSON exports: {save_dir}/json_exports/")
    print(f"- Per-class analysis: {save_dir}/per_class/")
    
    return analysis_results


def create_lime_summary_visualization(comparison_data: List[Dict], save_dir: str):
    """Create summary visualization of LIME analysis"""
    fig, axes = plt.subplots(2, len(comparison_data), figsize=(4*len(comparison_data), 8))
    
    if len(comparison_data) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, sample_data in enumerate(comparison_data):
        # Sample info
        class_name = sample_data['true_class']
        pred_class = sample_data['predicted_class']
        confidence = sample_data['confidence']
        is_correct = sample_data['is_correct']
        
        # Feature statistics
        features = sample_data['features']
        num_positive = len(features['positive_features'])
        num_negative = len(features['negative_features'])
        
        # Top row: Feature count visualization
        axes[0, i].bar(['Positive', 'Negative'], [num_positive, num_negative], 
                      color=['green', 'red'], alpha=0.7)
        axes[0, i].set_title(f'{class_name}\n({"✓" if is_correct else "✗"} {pred_class})')
        axes[0, i].set_ylabel('Number of Features')
        
        # Bottom row: Weight distribution
        pos_weights = [f['weight'] for f in features['positive_features']]
        neg_weights = [abs(f['weight']) for f in features['negative_features']]
        
        if pos_weights:
            axes[1, i].hist(pos_weights, bins=10, alpha=0.7, color='green', label='Positive')
        if neg_weights:
            axes[1, i].hist(neg_weights, bins=10, alpha=0.7, color='red', label='Negative')
        
        axes[1, i].set_title('Weight Distribution')
        axes[1, i].set_xlabel('Weight Magnitude')
        axes[1, i].set_ylabel('Frequency')
        axes[1, i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lime_summary_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_lime_summary_report(analysis_results: Dict, save_dir: str):
    """Create comprehensive LIME analysis report"""
    report_content = f"""# LIME Analysis Report

## Analysis Overview
- **Date**: {analysis_results['timestamp']}
- **Total Samples Analyzed**: {analysis_results['total_samples']}
- **LIME Parameters**: {analysis_results['lime_parameters']['num_samples']} perturbation samples

## Sample Distribution by Class
"""
    
    for class_name, count in analysis_results['samples_per_class'].items():
        report_content += f"- **{class_name}**: {count} samples\n"
    
    overall_stats = analysis_results['overall_statistics']
    
    report_content += f"""
## Overall Statistics
- **Average Superpixels per Sample**: {overall_stats['avg_superpixels_per_sample']:.1f}
- **Average Positive Features**: {overall_stats['avg_positive_features']:.1f}
- **Average Negative Features**: {overall_stats['avg_negative_features']:.1f}
- **Average Positive Weight**: {overall_stats['avg_positive_weight']:.3f}
- **Average Negative Weight**: {overall_stats['avg_negative_weight']:.3f}

## Per-Class Analysis
"""
    
    class_analysis = analysis_results['class_analysis']
    
    for class_name, class_data in class_analysis.items():
        report_content += f"""
### {class_name}
- **Samples**: {class_data['num_samples']}
- **Avg Positive Weight**: {class_data['avg_positive_weight']:.3f}
- **Avg Negative Weight**: {class_data['avg_negative_weight']:.3f}
- **Avg Superpixels per Sample**: {class_data['avg_superpixels_per_sample']:.1f}
"""
    
    report_content += f"""
## Interpretation Guidelines

### LIME Superpixels
- **Positive weights**: Superpixels that support the predicted class
- **Negative weights**: Superpixels that oppose the predicted class
- **Magnitude**: Indicates the strength of the superpixel's contribution

### Superpixel Analysis
- Higher number of superpixels may indicate more complex decision boundaries
- Balanced positive/negative features suggest nuanced decision-making
- Extreme weights may indicate critical diagnostic regions

## Files Generated
- `individual/`: Individual LIME visualizations for each sample
- `json_exports/`: JSON files with superpixel IDs and contribution weights
- `per_class/`: Per-class analysis and statistics
- `lime_analysis_results.json`: Complete numerical results

## JSON Export Format
Each JSON file contains:
```json
{{
  "sample_id": "class_name_XX",
  "true_class": "class_name",
  "predicted_class": "predicted_class",
  "confidence": 0.XXX,
  "features": {{
    "superpixel_weights": {{"superpixel_id": weight}},
    "positive_features": [{{superpixel_id, weight}}],
    "negative_features": [{{superpixel_id, weight}}]
  }}
}}
```

## Clinical Insights
Based on the LIME analysis:
"""
    
    # Add insights based on analysis
    if class_analysis:
        most_complex_class = max(class_analysis.items(), key=lambda x: x[1]['avg_superpixels_per_sample'])
        least_complex_class = min(class_analysis.items(), key=lambda x: x[1]['avg_superpixels_per_sample'])
        
        report_content += f"- **Most complex class**: {most_complex_class[0]} (avg {most_complex_class[1]['avg_superpixels_per_sample']:.1f} superpixels)\n"
        report_content += f"- **Least complex class**: {least_complex_class[0]} (avg {least_complex_class[1]['avg_superpixels_per_sample']:.1f} superpixels)\n"
    
    report_content += """
## Recommendations
- Examine superpixels with highest positive weights for potential biomarkers
- Investigate negative-weight regions for confounding factors
- Use JSON exports for quantitative analysis of feature importance patterns
- Compare superpixel patterns across different stain normalization methods
"""
    
    # Save report
    report_path = os.path.join(save_dir, 'lime_analysis_report.md')
    with open(report_path, 'w') as f:
        f.write(report_content)


# Legacy function for backward compatibility
def generate_lime_explanations(
    model,
    dataloader,
    device,
    class_names: List[str],
    save_dir: str = 'explainability/lime_results',
    num_samples: int = 10,
    lime_samples: int = 1000
):
    """
    Legacy function - use generate_comprehensive_lime_analysis instead
    """
    samples_per_class = max(1, num_samples // len(class_names))
    return generate_comprehensive_lime_analysis(
        model=model,
        dataloader=dataloader,
        device=device,
        class_names=class_names,
        save_dir=save_dir,
        samples_per_class=samples_per_class,
        lime_samples=lime_samples
    )