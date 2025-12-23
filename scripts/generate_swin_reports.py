"""Generate Medical Reports from Swin Transformer"""
import torch
import argparse
import os
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from swin_report_generation import SwinMedicalReportGenerator
from model.swin_transformer import swin_base_patch4_window7_224
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image


def load_model_and_generator(model_path: str, num_classes: int = 8, 
                             llm_name: str = 'template', device: str = 'cuda'):
    """Load Swin model and report generator"""
    swin_model = swin_base_patch4_window7_224(num_classes=num_classes)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model' in checkpoint:
        swin_model.load_state_dict(checkpoint['model'])
    elif 'model_state_dict' in checkpoint:
        swin_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        swin_model.load_state_dict(checkpoint)
    
    swin_model = swin_model.to(device)
    
    generator = SwinMedicalReportGenerator(swin_model, llm_name=llm_name)
    generator = generator.to(device)
    
    return generator


def main():
    parser = argparse.ArgumentParser(description='Generate Medical Reports')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained Swin model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='results/swin_reports', help='Output directory')
    parser.add_argument('--llm_name', type=str, default='template', help='LLM name or "template"')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples')
    parser.add_argument('--num_classes', type=int, default=8, help='Number of classes')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print(f"Loading model and generator...")
    generator = load_model_and_generator(args.model_path, args.num_classes, 
                                        args.llm_name, args.device)
    
    print(f"Loading data from {args.data_path}")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_reports = []
    
    print(f"Generating reports for {args.num_samples} samples...")
    for i, (images, labels) in enumerate(dataloader):
        if i >= args.num_samples:
            break
        
        images = images.to(args.device)
        
        # Generate report
        result = generator(images)
        
        # Save individual report
        report_path = os.path.join(args.output_dir, f'report_{i}.txt')
        with open(report_path, 'w') as f:
            f.write(generator.format_report_text(result))
        
        # Save JSON
        json_path = os.path.join(args.output_dir, f'report_{i}.json')
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        all_reports.append(result)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{args.num_samples}")
    
    # Save summary
    summary = {
        'total_reports': len(all_reports),
        'class_distribution': {},
        'avg_confidence': sum(r['confidence'] for r in all_reports) / len(all_reports)
    }
    
    for report in all_reports:
        pred = report['prediction']
        summary['class_distribution'][pred] = summary['class_distribution'].get(pred, 0) + 1
    
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nReports saved to {args.output_dir}")
    print(f"Average confidence: {summary['avg_confidence']:.3f}")


if __name__ == '__main__':
    main()
