#!/usr/bin/env python3
"""
Test script for XAI evaluation pipeline
Quick verification that all components work
"""
import torch
import torchvision.models as models
import numpy as np
from PIL import Image
import tempfile
import os

def test_xai_evaluation():
    """Test the XAI evaluation pipeline"""
    print("🧪 Testing XAI Evaluation Pipeline")
    print("="*50)
    
    # Create a simple model
    print("1. Loading DenseNet201 model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.densenet201(weights='IMAGENET1K_V1')
    
    # Modify for binary classification
    num_features = model.classifier.in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(num_features, 2)
    )
    
    model.to(device)
    model.eval()
    print(f"✅ Model loaded on {device}")
    
    # Test individual metrics
    print("\n2. Testing individual metrics...")
    
    # Create synthetic image
    synthetic_image = torch.randn(1, 3, 224, 224).to(device)
    synthetic_explanation = np.random.rand(224, 224)
    target_class = 0
    
    # Test Insertion AUC
    try:
        from xai.metrics.insertion_auc import InsertionAUC
        insertion_metric = InsertionAUC(model, device, num_steps=10)  # Fewer steps for testing
        auc_score, _ = insertion_metric.compute_insertion_auc(synthetic_image, synthetic_explanation, target_class)
        print(f"✅ Insertion AUC: {auc_score:.3f}")
    except Exception as e:
        print(f"❌ Insertion AUC failed: {e}")
    
    # Test Deletion AUC
    try:
        from xai.metrics.deletion_auc import DeletionAUC
        deletion_metric = DeletionAUC(model, device, num_steps=10)
        auc_score, _ = deletion_metric.compute_deletion_auc(synthetic_image, synthetic_explanation, target_class)
        print(f"✅ Deletion AUC: {auc_score:.3f}")
    except Exception as e:
        print(f"❌ Deletion AUC failed: {e}")
    
    # Test IoU
    try:
        from xai.metrics.iou import IoUMetric
        iou_metric = IoUMetric()
        pseudo_roi = iou_metric.create_pseudo_roi(synthetic_explanation)
        iou_score = iou_metric.compute_iou_with_roi(synthetic_explanation, pseudo_roi)
        print(f"✅ IoU: {iou_score:.3f}")
    except Exception as e:
        print(f"❌ IoU failed: {e}")
    
    # Test Stability
    try:
        from xai.metrics.stability import StabilityMetric
        stability_metric = StabilityMetric(device, num_perturbations=3)  # Fewer perturbations for testing
        
        def dummy_explanation_generator(img, target):
            return np.random.rand(224, 224)
        
        stability_score, _ = stability_metric.evaluate_stability(
            synthetic_image, dummy_explanation_generator, target_class)
        print(f"✅ Stability: {stability_score:.3f}")
    except Exception as e:
        print(f"❌ Stability failed: {e}")
    
    # Test full evaluator
    print("\n3. Testing full XAI evaluator...")
    try:
        from xai.evaluate_xai import XAIEvaluator
        
        class_names = ['Benign', 'Malignant']
        evaluator = XAIEvaluator(model, device, class_names, results_dir='temp_results')
        
        # Test single image evaluation
        result = evaluator.evaluate_single_image(synthetic_image, true_class=0, image_id='test_image')
        print(f"✅ Single image evaluation completed")
        print(f"   Available explainers: {list(result['explainer_results'].keys())}")
        
        # Test aggregation
        evaluator.compute_aggregated_results()
        print(f"✅ Aggregation completed")
        
        # Test saving
        json_path = evaluator.save_results('test_results.json')
        csv_path = evaluator.save_csv_results('test_results.csv')
        print(f"✅ Results saved: {json_path}, {csv_path}")
        
    except Exception as e:
        print(f"❌ Full evaluator failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 XAI Evaluation Pipeline Test Complete!")
    print("="*50)


if __name__ == "__main__":
    test_xai_evaluation()