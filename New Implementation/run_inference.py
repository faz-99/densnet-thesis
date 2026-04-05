"""
Inference / demo pipeline: run the full Vision-XAI-Report pipeline on
a single image or a batch, producing XAI visualisations and clinical reports.

Usage:
    python run_inference.py --checkpoint outputs/checkpoints/best_model.pth --image path/to/image.png
    python run_inference.py --checkpoint outputs/checkpoints/best_model.pth --batch-dir path/to/images/
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image

from config.settings import (
    DATASET_CONFIG, MODEL_CONFIG, XAI_OUTPUT_DIR, REPORT_OUTPUT_DIR,
)
from data.preprocessing import get_val_transforms, denormalize
from models.ensemble import HybridEnsemble
from xai.interpretability_manager import InterpretabilityManager
from report.report_generator import ReportGenerator
from evaluation.validation_engine import ValidationEngine
from training.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="Run Vision-XAI-Report inference")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    p.add_argument("--image", type=str, default=None, help="Single image path")
    p.add_argument("--batch-dir", type=str, default=None, help="Directory of images")
    p.add_argument("--task", choices=["binary", "multiclass"], default="multiclass")
    p.add_argument("--magnification", default="400X")
    p.add_argument("--load-llm", action="store_true", help="Load Med-LLM for report generation")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--xai-methods", nargs="+", default=None,
                   help="Subset of XAI methods to run")
    return p.parse_args()


def load_image(path: str, transform):
    """Load and transform a single image."""
    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    return tensor


def run_single(
    image_path: str,
    model,
    xai_manager,
    report_gen,
    val_engine,
    transform,
    class_names,
    device,
    magnification,
    xai_methods=None,
):
    """Run full pipeline on a single image."""
    img_name = Path(image_path).stem

    # 1. Load & predict
    input_tensor = load_image(image_path, transform).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()

    class_label = class_names[pred_class] if pred_class < len(class_names) else str(pred_class)
    print(f"\n[{img_name}] Prediction: {class_label} ({confidence:.1%})")

    # 2. XAI explanations
    print(f"  Generating XAI explanations...")
    heatmaps = xai_manager.explain(input_tensor, pred_class, methods=xai_methods)

    # 2b. Counterfactual with details
    cf_details = xai_manager.explain_with_counterfactual_details(input_tensor, None)

    # 3. Visualisation
    vis_path = str(XAI_OUTPUT_DIR / f"{img_name}_xai_panel.png")
    xai_manager.visualize(
        input_tensor, heatmaps, class_names=class_names,
        target_class=pred_class, save_path=vis_path,
    )
    print(f"  XAI panel saved: {vis_path}")

    # 4. Report generation
    report = report_gen.generate_report(
        input_tensor=input_tensor,
        predicted_class=pred_class,
        confidence=confidence,
        heatmaps=heatmaps,
        cf_details=cf_details,
        magnification=magnification,
        class_names=class_names,
    )
    report_path = REPORT_OUTPUT_DIR / f"{img_name}_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Report saved: {report_path}")

    # 5. Validation metrics
    print(f"  Computing validation metrics...")
    eval_results = val_engine.evaluate_all(
        input_tensor=input_tensor,
        heatmaps=heatmaps,
        target_class=pred_class,
    )

    # Print summary
    print(f"  Faithfulness (Insertion/Deletion AUC):")
    for method, scores in eval_results["faithfulness"].items():
        print(f"    {method}: Ins={scores['insertion_auc']:.4f}, Del={scores['deletion_auc']:.4f}")

    # 6. TTA Stability evaluation
    tta_methods = [m for m in (xai_methods or ["grad_cam", "integrated_gradients", "attention_rollout"])
                   if m not in ("counterfactual", "shap", "lime")]  # skip expensive methods
    print(f"  TTA stability evaluation ({len(tta_methods)} methods)...")
    tta_results = xai_manager.evaluate_tta_stability(
        input_tensor, pred_class, methods=tta_methods,
    )
    for method, res in tta_results.items():
        print(f"    {method}: Stability={res['stability_score']:.4f}")

    # TTA stability visualisation
    for method in tta_methods:
        if method in tta_results:
            tta_vis_path = str(XAI_OUTPUT_DIR / f"{img_name}_tta_{method}.png")
            xai_manager.visualize_tta(
                input_tensor, tta_results, method, save_path=tta_vis_path,
            )

    # 7. Consensus heatmaps
    print(f"  Generating consensus heatmaps...")
    consensus_heatmaps = xai_manager.get_consensus_heatmaps(
        input_tensor, pred_class, methods=tta_methods,
    )
    consensus_vis_path = str(XAI_OUTPUT_DIR / f"{img_name}_consensus_comparison.png")
    xai_manager.visualize_consensus_comparison(
        input_tensor, heatmaps, consensus_heatmaps, save_path=consensus_vis_path,
    )
    print(f"  Consensus comparison saved: {consensus_vis_path}")

    return {
        "image": image_path,
        "prediction": class_label,
        "confidence": confidence,
        "heatmaps": heatmaps,
        "consensus_heatmaps": consensus_heatmaps,
        "report": report,
        "eval_results": eval_results,
        "tta_stability": {m: r["stability_score"] for m, r in tta_results.items()},
    }


def main():
    args = parse_args()
    device = args.device

    num_classes = 2 if args.task == "binary" else 8
    class_names = (
        DATASET_CONFIG["binary_classes"] if args.task == "binary"
        else DATASET_CONFIG["multiclass_classes"]
    )

    # ── Load model ──
    print("[Pipeline] Loading model...")
    model = HybridEnsemble(num_classes=num_classes)
    model = Trainer.load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)
    model.eval()

    # ── XAI Manager ──
    print("[Pipeline] Initialising XAI manager...")
    xai_manager = InterpretabilityManager(model)

    # ── Report Generator ──
    report_gen = ReportGenerator(model=model, device=device)
    if args.load_llm:
        report_gen.load_llm()

    # ── Validation Engine ──
    val_engine = ValidationEngine(model, device=device)

    # ── Transform ──
    transform = get_val_transforms()

    # ── Run inference ──
    if args.image:
        run_single(
            args.image, model, xai_manager, report_gen, val_engine,
            transform, class_names, device, args.magnification, args.xai_methods,
        )
    elif args.batch_dir:
        batch_dir = Path(args.batch_dir)
        image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        image_paths = sorted(
            p for p in batch_dir.rglob("*") if p.suffix.lower() in image_exts
        )
        print(f"[Pipeline] Processing {len(image_paths)} images from {batch_dir}")

        results = []
        for img_path in image_paths:
            try:
                res = run_single(
                    str(img_path), model, xai_manager, report_gen, val_engine,
                    transform, class_names, device, args.magnification, args.xai_methods,
                )
                results.append(res)
            except Exception as e:
                print(f"  ERROR on {img_path}: {e}")

        print(f"\n[Pipeline] Processed {len(results)}/{len(image_paths)} images successfully.")
    else:
        print("ERROR: Provide either --image or --batch-dir")
        sys.exit(1)


if __name__ == "__main__":
    main()
