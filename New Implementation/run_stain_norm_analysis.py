"""
Stain Normalization Analysis
============================
Qualitative + Quantitative comparison of XAI explanations on
original vs. Macenko-normalized images.

Outputs (saved to outputs/stain_analysis/):
  - {stem}_comparison.png      : Original | Normalized | Difference map
  - {stem}_xai_original.png    : Grad-CAM + Attention Rollout on original
  - {stem}_xai_normalized.png  : Grad-CAM + Attention Rollout on normalized
  - {stem}_counterfactual.png  : Counterfactual heatmaps side-by-side
  - stain_analysis_results.json: IoU, Faithfulness Delta, all metrics

Usage:
    python run_stain_norm_analysis.py --checkpoint outputs/checkpoints/best_model.pth
    python run_stain_norm_analysis.py --checkpoint outputs/checkpoints/best_model.pth --n-samples 10
"""
import argparse
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.settings import DATASET_CONFIG, OUTPUT_DIR
from data.preprocessing import MacenkoNormalizer, get_val_transforms, denormalize
from models.ensemble import HybridEnsemble
from training.trainer import Trainer
from xai.grad_cam import GradCAM
from xai.attention_rollout import AttentionRollout
from xai.counterfactual import CounterfactualExplainer
from evaluation.validation_engine import ValidationEngine

# ── Output directory ──
ANALYSIS_DIR = OUTPUT_DIR / "stain_analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", default="multiclass")
    p.add_argument("--n-samples", type=int, default=5,
                   help="Number of images to analyse")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def load_image_pair(image_path: str, device: str):
    """Load an image as both original tensor and Macenko-normalized tensor."""
    pil_img = Image.open(image_path).convert("RGB")

    # Shared final transforms (resize + normalize stats only, no augmentation)
    from torchvision import transforms
    mean, std = DATASET_CONFIG["mean"], DATASET_CONFIG["std"]
    to_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Original (no stain norm)
    t_original = to_tensor(pil_img).unsqueeze(0).to(device)

    # Macenko normalized
    normalizer = MacenkoNormalizer()
    try:
        pil_norm = normalizer(pil_img)
    except Exception:
        pil_norm = pil_img  # fallback
    t_normalized = to_tensor(pil_norm).unsqueeze(0).to(device)

    return t_original, t_normalized, pil_img


def tensor_to_rgb(tensor):
    """Denormalize tensor → HWC numpy [0,1]."""
    img = denormalize(tensor.squeeze(0)).detach().cpu().numpy()
    return np.transpose(img, (1, 2, 0))


def iou(heatmap_a: np.ndarray, heatmap_b: np.ndarray, threshold: float = 0.5) -> float:
    """Intersection over Union between two binary heatmaps."""
    mask_a = (heatmap_a >= threshold).astype(np.float32)
    mask_b = (heatmap_b >= threshold).astype(np.float32)
    intersection = (mask_a * mask_b).sum()
    union = np.clip(mask_a + mask_b, 0, 1).sum()
    return float(intersection / (union + 1e-8))


# ─────────────────────────────────────────────
# Plot functions
# ─────────────────────────────────────────────

def plot_comparison(img_orig, img_norm, stem):
    """Original | Normalized | Difference map."""
    diff = np.abs(img_orig - img_norm)
    diff_scaled = diff / (diff.max() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_orig); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(img_norm); axes[1].set_title("Macenko Normalized"); axes[1].axis("off")
    im = axes[2].imshow(diff_scaled, cmap="hot"); axes[2].set_title("Difference Map"); axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    fig.suptitle(f"Stain Normalization Effect — {stem}", fontsize=11)
    plt.tight_layout()
    out = ANALYSIS_DIR / f"{stem}_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_xai_overlay(img_rgb, heatmap_gc, heatmap_ar, title, stem, suffix):
    """Grad-CAM + Attention Rollout overlaid on image."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_rgb); axes[0].set_title("Image"); axes[0].axis("off")

    axes[1].imshow(img_rgb)
    axes[1].imshow(heatmap_gc, cmap="jet", alpha=0.5)
    axes[1].set_title("Grad-CAM"); axes[1].axis("off")

    axes[2].imshow(img_rgb)
    axes[2].imshow(heatmap_ar, cmap="jet", alpha=0.5)
    axes[2].set_title("Attention Rollout"); axes[2].axis("off")

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    out = ANALYSIS_DIR / f"{stem}_xai_{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_counterfactual(img_orig, img_norm, cf_orig, cf_norm, stem):
    """Counterfactual heatmaps side-by-side."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].imshow(img_orig); axes[0, 0].set_title("Original Image"); axes[0, 0].axis("off")
    axes[0, 1].imshow(cf_orig, cmap="hot"); axes[0, 1].set_title("CF — Original"); axes[0, 1].axis("off")
    axes[1, 0].imshow(img_norm); axes[1, 0].set_title("Normalized Image"); axes[1, 0].axis("off")
    axes[1, 1].imshow(cf_norm, cmap="hot"); axes[1, 1].set_title("CF — Normalized"); axes[1, 1].axis("off")

    fig.suptitle(f"Counterfactual Comparison — {stem}\n"
                 "If CF on original highlights color regions → model is color-biased", fontsize=10)
    plt.tight_layout()
    out = ANALYSIS_DIR / f"{stem}_counterfactual.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────

def analyse_image(image_path, model, grad_cam, attn_rollout, cf_explainer,
                  val_engine, device, class_names):
    stem = Path(image_path).stem
    print(f"\n[Analysis] {stem}")

    # Load both versions
    t_orig, t_norm, pil_orig = load_image_pair(image_path, device)
    img_orig = tensor_to_rgb(t_orig)
    img_norm = tensor_to_rgb(t_norm)

    # Predictions
    with torch.no_grad():
        logits_o = model(t_orig)
        logits_n = model(t_norm)
        pred_o = logits_o.argmax(1).item()
        pred_n = logits_n.argmax(1).item()
        conf_o = F.softmax(logits_o, 1)[0, pred_o].item()
        conf_n = F.softmax(logits_n, 1)[0, pred_n].item()

    print(f"  Original  → {class_names[pred_o]} ({conf_o:.1%})")
    print(f"  Normalized→ {class_names[pred_n]} ({conf_n:.1%})")

    # 1. Comparison plot
    plot_comparison(img_orig, img_norm, stem)

    # 2. XAI on both
    gc_orig = grad_cam.generate(t_orig, pred_o)
    gc_norm = grad_cam.generate(t_norm, pred_n)
    ar_orig = attn_rollout.generate(t_orig, pred_o)
    ar_norm = attn_rollout.generate(t_norm, pred_n)

    plot_xai_overlay(img_orig, gc_orig, ar_orig, f"XAI on Original — {stem}", stem, "original")
    plot_xai_overlay(img_norm, gc_norm, ar_norm, f"XAI on Normalized — {stem}", stem, "normalized")

    # 3. Counterfactual
    cf_orig = cf_explainer.generate(t_orig, None)
    cf_norm = cf_explainer.generate(t_norm, None)
    plot_counterfactual(img_orig, img_norm, cf_orig, cf_norm, stem)

    # 4. IoU between original and normalized heatmaps
    iou_gc = iou(gc_orig, gc_norm)
    iou_ar = iou(ar_orig, ar_norm)
    iou_cf = iou(cf_orig, cf_norm)
    print(f"  IoU Grad-CAM:         {iou_gc:.4f}")
    print(f"  IoU Attention Rollout:{iou_ar:.4f}")
    print(f"  IoU Counterfactual:   {iou_cf:.4f}")

    # 5. Faithfulness delta (Insertion AUC)
    heatmaps_o = {"grad_cam": gc_orig, "attention_rollout": ar_orig}
    heatmaps_n = {"grad_cam": gc_norm, "attention_rollout": ar_norm}

    faith_o = val_engine.evaluate_all(t_orig, heatmaps_o, pred_o)
    faith_n = val_engine.evaluate_all(t_norm, heatmaps_n, pred_n)

    faith_delta = {}
    for method in ["grad_cam", "attention_rollout"]:
        ins_o = faith_o["faithfulness"][method]["insertion_auc"]
        ins_n = faith_n["faithfulness"][method]["insertion_auc"]
        del_o = faith_o["faithfulness"][method]["deletion_auc"]
        del_n = faith_n["faithfulness"][method]["deletion_auc"]
        delta_ins = ins_n - ins_o
        delta_del = del_n - del_o
        faith_delta[method] = {
            "insertion_auc_original":   ins_o,
            "insertion_auc_normalized": ins_n,
            "insertion_delta":          delta_ins,
            "deletion_auc_original":    del_o,
            "deletion_auc_normalized":  del_n,
            "deletion_delta":           delta_del,
        }
        print(f"  {method} Insertion AUC: orig={ins_o:.4f} norm={ins_n:.4f} Δ={delta_ins:+.4f}")
        if delta_ins > 0:
            print(f"    → Normalized features are MORE faithful (+{delta_ins:.4f})")
        else:
            print(f"    → Original features are more faithful ({delta_ins:.4f})")

    return {
        "image": str(image_path),
        "stem": stem,
        "prediction_original":   class_names[pred_o],
        "confidence_original":   conf_o,
        "prediction_normalized": class_names[pred_n],
        "confidence_normalized": conf_n,
        "prediction_changed":    pred_o != pred_n,
        "iou": {"grad_cam": iou_gc, "attention_rollout": iou_ar, "counterfactual": iou_cf},
        "faithfulness_delta": faith_delta,
    }


def main():
    args = parse_args()
    device = args.device
    class_names = (DATASET_CONFIG["binary_classes"] if args.task == "binary"
                   else DATASET_CONFIG["multiclass_classes"])

    # Load model
    print("[StainAnalysis] Loading model...")
    model = HybridEnsemble(num_classes=2 if args.task == "binary" else 8)
    model = Trainer.load_checkpoint(model, args.checkpoint, device)
    model = model.to(device).eval()

    # Init XAI
    grad_cam     = GradCAM(model)
    attn_rollout = AttentionRollout(model)
    cf_explainer = CounterfactualExplainer(model)
    val_engine   = ValidationEngine(model, device=device)

    # Collect test images (one per class)
    test_dir = Path(DATASET_CONFIG["test_dir"])
    samples = []
    for cls in class_names:
        imgs = sorted((test_dir / cls).glob("*.png")) if (test_dir / cls).exists() else []
        if imgs:
            samples.append(str(imgs[0]))
    samples = samples[:args.n_samples]
    print(f"[StainAnalysis] Analysing {len(samples)} images...")

    all_results = []
    for img_path in samples:
        try:
            result = analyse_image(
                img_path, model, grad_cam, attn_rollout, cf_explainer,
                val_engine, device, class_names
            )
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR on {img_path}: {e}")

    # Aggregate summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    avg_iou_gc = np.mean([r["iou"]["grad_cam"] for r in all_results])
    avg_iou_ar = np.mean([r["iou"]["attention_rollout"] for r in all_results])
    pred_changed = sum(r["prediction_changed"] for r in all_results)

    print(f"Avg IoU Grad-CAM:          {avg_iou_gc:.4f}")
    print(f"Avg IoU Attention Rollout: {avg_iou_ar:.4f}")
    print(f"Prediction changed after normalization: {pred_changed}/{len(all_results)}")

    if avg_iou_gc < 0.4:
        print("\n→ LOW IoU: Model was using stain color shortcuts in original images.")
        print("  Normalization significantly shifts where the model 'looks'.")
    else:
        print("\n→ HIGH IoU: Model focuses on similar regions regardless of stain color.")
        print("  Normalization has minimal effect on explanation localization.")

    # Faithfulness delta summary
    for method in ["grad_cam", "attention_rollout"]:
        deltas = [r["faithfulness_delta"][method]["insertion_delta"]
                  for r in all_results if method in r["faithfulness_delta"]]
        if deltas:
            avg_delta = np.mean(deltas)
            print(f"\n{method} avg Insertion AUC delta (norm - orig): {avg_delta:+.4f}")
            if avg_delta > 0:
                print("  → Normalized images produce MORE faithful explanations.")
            else:
                print("  → Original images produce more faithful explanations.")

    # Save results
    out_json = ANALYSIS_DIR / "stain_analysis_results.json"
    with open(out_json, "w") as f:
        json.dump({"samples": all_results, "summary": {
            "avg_iou_grad_cam": avg_iou_gc,
            "avg_iou_attention_rollout": avg_iou_ar,
            "predictions_changed": pred_changed,
            "total_samples": len(all_results),
        }}, f, indent=2)
    print(f"\nResults saved to: {out_json}")
    print(f"Plots saved to:   {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
