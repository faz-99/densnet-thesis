"""Evaluate Swin Explainability Methods with Quantitative Metrics"""
import torch
import numpy as np
import argparse
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent))

from swin_explainability import SwinAttentionRollout, SwinAttentionGradCAM
from model.swin_transformer import swin_base_patch4_window7_224
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn.functional as F


class ExplainabilityEvaluator:
    """Evaluate explainability methods with quantitative metrics"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        
    def insertion_auc(self, image, explanation, num_steps=50):
        """Measure performance when adding pixels by importance"""
        h, w = explanation.shape
        total_pixels = h * w
        
        # Sort pixels by importance
        flat_exp = explanation.flatten()
        sorted_indices = np.argsort(flat_exp)[::-1]
        
        scores = []
        masked_image = torch.zeros_like(image)
        
        for step in range(num_steps):
            # Add top pixels
            num_pixels = int((step + 1) / num_steps * total_pixels)
            indices = sorted_indices[:num_pixels]
            
            mask = np.zeros(total_pixels)
            mask[indices] = 1
            mask = mask.reshape(h, w)
            
            # Resize mask to image size
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
            mask_tensor = F.interpolate(mask_tensor, size=image.shape[2:], mode='nearest')
            
            masked_image = image * mask_tensor
            
            with torch.no_grad():
                output = self.model(masked_image)
                prob = F.softmax(output, dim=1).max().item()
            
            scores.append(prob)
        
        # Compute AUC
        auc = np.trapz(scores, dx=1/num_steps)
        return auc
    
    def deletion_auc(self, image, explanation, num_steps=50):
        """Measure performance when removing pixels by importance"""
        h, w = explanation.shape
        total_pixels = h * w
        
        flat_exp = explanation.flatten()
        sorted_indices = np.argsort(flat_exp)[::-1]
        
        scores = []
        
        for step in range(num_steps):
            # Remove top pixels
            num_pixels = int((step + 1) / num_steps * total_pixels)
            indices = sorted_indices[:num_pixels]
            
            mask = np.ones(total_pixels)
            mask[indices] = 0
            mask = mask.reshape(h, w)
            
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
            mask_tensor = F.interpolate(mask_tensor, size=image.shape[2:], mode='nearest')
            
            masked_image = image * mask_tensor
            
            with torch.no_grad():
                output = self.model(masked_image)
                prob = F.softmax(output, dim=1).max().item()
            
            scores.append(prob)
        
        auc = np.trapz(scores, dx=1/num_steps)
        return auc
    
    def stability_score(self, image, explainer, num_perturbations=10, noise_level=0.1):
        """Measure consistency under input perturbations"""
        base_explanation = explainer.generate_rollout(image) if hasattr(explainer, 'generate_rollout') else explainer.generate_cam(image)
        
        correlations = []
        
        for _ in range(num_perturbations):
            # Add noise
            noise = torch.randn_like(image) * noise_level
            perturbed = image + noise
            
            # Generate explanation
            perturbed_exp = explainer.generate_rollout(perturbed) if hasattr(explainer, 'generate_rollout') else explainer.generate_cam(perturbed)
            
            # Compute correlation
            corr = np.corrcoef(base_explanation.flatten(), perturbed_exp.flatten())[0, 1]
            correlations.append(corr)
        
        return np.mean(correlations)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Swin Explainability')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/swin_evaluation')
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Load model
    model = swin_base_patch4_window7_224(num_classes=8)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint.get('model', checkpoint))
    model = model.to(args.device)
    model.eval()
    
    # Load data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Initialize explainers
    rollout = SwinAttentionRollout(model)
    gradcam = SwinAttentionGradCAM(model)
    evaluator = ExplainabilityEvaluator(model, args.device)
    
    results = {
        'rollout': {'insertion': [], 'deletion': [], 'stability': []},
        'gradcam': {'insertion': [], 'deletion': [], 'stability': []}
    }
    
    print(f"Evaluating {args.num_samples} samples...")
    
    for i, (images, _) in enumerate(dataloader):
        if i >= args.num_samples:
            break
        
        images = images.to(args.device)
        
        # Rollout metrics
        rollout_map = rollout.generate_rollout(images)
        results['rollout']['insertion'].append(evaluator.insertion_auc(images, rollout_map))
        results['rollout']['deletion'].append(evaluator.deletion_auc(images, rollout_map))
        results['rollout']['stability'].append(evaluator.stability_score(images, rollout))
        
        # GradCAM metrics
        gradcam_map = gradcam.generate_cam(images)
        results['gradcam']['insertion'].append(evaluator.insertion_auc(images, gradcam_map))
        results['gradcam']['deletion'].append(evaluator.deletion_auc(images, gradcam_map))
        results['gradcam']['stability'].append(evaluator.stability_score(images, gradcam))
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{args.num_samples}")
    
    # Compute averages
    summary = {}
    for method in ['rollout', 'gradcam']:
        summary[method] = {
            'insertion_auc': float(np.mean(results[method]['insertion'])),
            'deletion_auc': float(np.mean(results[method]['deletion'])),
            'stability': float(np.mean(results[method]['stability']))
        }
    
    # Save results
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump({'summary': summary, 'detailed': results}, f, indent=2)
    
    print("\nEvaluation Results:")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
