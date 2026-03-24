"""
Comprehensive evaluation entry point.
Runs all validation metrics on the test set and produces a full report.

Usage:
    python run_evaluate.py --checkpoint outputs/checkpoints/best_model.pth
    python run_evaluate.py --checkpoint outputs/checkpoints/best_model.pth --n-samples 50
"""
import argparse
import sys
import os
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config.settings import (
    DATASET_CONFIG, MODEL_CONFIG, OUTPUT_DIR,
    XAI_OUTPUT_DIR, REPORT_OUTPUT_DIR,
)
from data.breakhis_dataset import get_dataloaders
from data.preprocessing import get_val_transforms
from models.ensemble import HybridEnsemble
from xai.interpretability_manager import InterpretabilityManager
from evaluation.validation_engine import ValidationEngine
from training.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the trained model")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", choices=["binary", "multiclass"], default="multiclass")
    p.add_argument("--magnification", default="400X")
    p.add_argument("--n-samples", type=int, default=20,
                   help="Number of samples for XAI evaluation (expensive)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def evaluate_classification(model, val_loader, device, class_names):
    """Standard classification metrics on the full validation set."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    results = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision_weighted": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall_weighted": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "classification_report": classification_report(
            all_labels, all_preds, target_names=class_names, zero_division=0,
        ),
    }

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return results


def evaluate_xai(model, val_loader, device, n_samples, class_names):
    """Run XAI evaluation metrics on a subset of samples."""
    xai_manager = InterpretabilityManager(model)
    val_engine = ValidationEngine(model, device=device)

    dataset = val_loader.dataset
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    all_results = {
        "faithfulness": {},
        "localization": {},
    }

    # Stability: pick one method and measure
    stability_scores = []

    for i, idx in enumerate(indices):
        img, label, mag = dataset[idx]
        input_tensor = img.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            pred = logits.argmax(dim=1).item()
            conf = F.softmax(logits, dim=1)[0, pred].item()

        print(f"  XAI eval [{i+1}/{len(indices)}] "
              f"True: {class_names[label]}, Pred: {class_names[pred]} ({conf:.1%})")

        # Generate heatmaps (fast methods only for batch eval)
        heatmaps = xai_manager.explain(
            input_tensor, pred,
            methods=["grad_cam", "integrated_gradients", "attention_rollout"],
        )

        # Evaluate
        sample_results = val_engine.evaluate_all(
            input_tensor=input_tensor,
            heatmaps=heatmaps,
            target_class=pred,
        )

        # Aggregate
        for method, scores in sample_results["faithfulness"].items():
            if method not in all_results["faithfulness"]:
                all_results["faithfulness"][method] = {"insertion_auc": [], "deletion_auc": []}
            all_results["faithfulness"][method]["insertion_auc"].append(scores["insertion_auc"])
            all_results["faithfulness"][method]["deletion_auc"].append(scores["deletion_auc"])

        for method, scores in sample_results["localization"].items():
            if method not in all_results["localization"]:
                all_results["localization"][method] = {"localization_auc": []}
            all_results["localization"][method]["localization_auc"].append(scores["localization_auc"])

        # Stability for Grad-CAM
        if "grad_cam" in xai_manager._explainers:
            stability = val_engine.stability_score(
                input_tensor,
                lambda x, tc: xai_manager._explainers["grad_cam"].generate(x, tc),
                pred,
            )
            stability_scores.append(stability)

        # TTA Stability (Augment-Explain-Inverse-Compare)
        if i < 10:  # TTA is expensive, run on a subset
            tta_methods = ["grad_cam", "integrated_gradients", "attention_rollout"]
            tta_result = xai_manager.evaluate_tta_stability(
                input_tensor, pred, methods=tta_methods,
            )
            for method, res in tta_result.items():
                if method not in all_results.get("tta_stability", {}):
                    all_results.setdefault("tta_stability", {})[method] = []
                all_results["tta_stability"][method].append(res["stability_score"])

        # Save XAI panel for first few samples
        if i < 5:
            vis_path = str(XAI_OUTPUT_DIR / f"eval_sample_{i}_xai.png")
            xai_manager.visualize(
                input_tensor, heatmaps, class_names=class_names,
                target_class=pred, save_path=vis_path,
            )

            # Also save TTA stability visualisation and consensus comparison
            if i < 3:
                tta_methods_vis = ["grad_cam", "integrated_gradients", "attention_rollout"]
                tta_res = xai_manager.evaluate_tta_stability(
                    input_tensor, pred, methods=tta_methods_vis,
                )
                for m in tta_methods_vis:
                    if m in tta_res:
                        tta_vis = str(XAI_OUTPUT_DIR / f"eval_sample_{i}_tta_{m}.png")
                        xai_manager.visualize_tta(
                            input_tensor, tta_res, m, save_path=tta_vis,
                        )

                consensus = xai_manager.get_consensus_heatmaps(
                    input_tensor, pred, methods=tta_methods_vis,
                )
                cons_vis = str(XAI_OUTPUT_DIR / f"eval_sample_{i}_consensus.png")
                xai_manager.visualize_consensus_comparison(
                    input_tensor, heatmaps, consensus, save_path=cons_vis,
                )

    # Compute means
    summary = {"faithfulness": {}, "localization": {}, "robustness": {}}

    for method, scores in all_results["faithfulness"].items():
        summary["faithfulness"][method] = {
            "insertion_auc_mean": float(np.mean(scores["insertion_auc"])),
            "insertion_auc_std": float(np.std(scores["insertion_auc"])),
            "deletion_auc_mean": float(np.mean(scores["deletion_auc"])),
            "deletion_auc_std": float(np.std(scores["deletion_auc"])),
        }

    for method, scores in all_results["localization"].items():
        summary["localization"][method] = {
            "localization_auc_mean": float(np.mean(scores["localization_auc"])),
            "localization_auc_std": float(np.std(scores["localization_auc"])),
        }

    if stability_scores:
        summary["robustness"]["grad_cam_stability"] = {
            "mean": float(np.mean(stability_scores)),
            "std": float(np.std(stability_scores)),
        }

    # TTA stability summary
    summary["tta_stability"] = {}
    for method, scores in all_results.get("tta_stability", {}).items():
        summary["tta_stability"][method] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
        }

    return summary


def main():
    args = parse_args()
    device = args.device

    num_classes = 2 if args.task == "binary" else 8
    class_names = (
        DATASET_CONFIG["binary_classes"] if args.task == "binary"
        else DATASET_CONFIG["multiclass_classes"]
    )

    # Load model
    print("[Evaluate] Loading model...")
    model = HybridEnsemble(num_classes=num_classes)
    model = Trainer.load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)
    model.eval()

    # Data
    _, val_loader = get_dataloaders(
        task=args.task, magnification=args.magnification,
    )

    # 1. Classification metrics
    print("\n[Evaluate] Classification metrics...")
    cls_results = evaluate_classification(model, val_loader, device, class_names)
    print(f"  Accuracy:  {cls_results['accuracy']:.4f}")
    print(f"  F1 (wt):   {cls_results['f1_weighted']:.4f}")
    print(f"  Precision: {cls_results['precision_weighted']:.4f}")
    print(f"  Recall:    {cls_results['recall_weighted']:.4f}")
    print(f"\n{cls_results['classification_report']}")

    # 2. XAI metrics
    print(f"\n[Evaluate] XAI evaluation on {args.n_samples} samples...")
    xai_results = evaluate_xai(model, val_loader, device, args.n_samples, class_names)

    print("\n[Evaluate] XAI Faithfulness Summary:")
    for method, scores in xai_results["faithfulness"].items():
        print(f"  {method}: "
              f"Ins AUC={scores['insertion_auc_mean']:.4f}±{scores['insertion_auc_std']:.4f}, "
              f"Del AUC={scores['deletion_auc_mean']:.4f}±{scores['deletion_auc_std']:.4f}")

    if xai_results["robustness"]:
        print("\n[Evaluate] Robustness (ε-noise):")
        for method, scores in xai_results["robustness"].items():
            print(f"  {method}: {scores['mean']:.4f}±{scores['std']:.4f}")

    if xai_results.get("tta_stability"):
        print("\n[Evaluate] TTA Stability (Augment-Explain-Inverse-Compare):")
        for method, scores in xai_results["tta_stability"].items():
            print(f"  {method}: {scores['mean']:.4f}±{scores['std']:.4f}")
        print("  (Higher = more rotation/flip invariant explanations)")

    # Save all results
    full_results = {
        "classification": {k: v for k, v in cls_results.items() if k != "classification_report"},
        "classification_report_text": cls_results["classification_report"],
        "xai": xai_results,
        "config": {
            "task": args.task,
            "magnification": args.magnification,
            "n_xai_samples": args.n_samples,
        },
    }

    results_path = OUTPUT_DIR / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"\n[Evaluate] Results saved to {results_path}")


if __name__ == "__main__":
    main()
