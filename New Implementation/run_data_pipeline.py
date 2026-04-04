"""
run_data_pipeline.py
====================
End-to-end data pipeline:
  1. Download BreaKHis from Kaggle
  2. Run comprehensive EDA (figures → outputs/eda/)
  3. Patient-level split (no leakage)
  4. Organize split data with stain normalization preview
  5. Verify dataloaders work

Usage:
    python run_data_pipeline.py                       # full pipeline
    python run_data_pipeline.py --skip-download       # if data already exists
    python run_data_pipeline.py --skip-eda            # skip EDA plots
    python run_data_pipeline.py --mag 400X            # filter magnification
"""
import sys
import argparse
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import DATASET_CONFIG, DATA_DIR, OUTPUT_DIR


def step_download(dataset_type: str = "full"):
    """Step 1: Download BreaKHis from Kaggle."""
    print("\n" + "=" * 60)
    print("STEP 1: Download BreaKHis from Kaggle")
    print("=" * 60)
    from data.download_breakhis import download_breakhis
    raw_path = download_breakhis(dataset=dataset_type)
    print(f"[OK] Raw data at: {raw_path}")
    return raw_path


def step_eda(root_dir: str = None):
    """Step 2: Comprehensive Exploratory Data Analysis."""
    print("\n" + "=" * 60)
    print("STEP 2: Exploratory Data Analysis")
    print("=" * 60)
    from data.eda import run_eda
    records, patient_images, patient_class = run_eda(root_dir)
    return records


def step_patient_split(root_dir: str, output_dir: str, mag: str = None):
    """Step 3: Patient-level train/test/val split."""
    print("\n" + "=" * 60)
    print("STEP 3: Patient-Level Data Split")
    print("=" * 60)
    from data.patient_split import (
        patient_level_split,
        materialize_split,
        print_split_statistics,
    )

    train_recs, test_recs, val_recs = patient_level_split(
        root_dir,
        test_size=DATASET_CONFIG.get("test_size", 0.2),
        val_size=DATASET_CONFIG.get("val_size", 0.1),
        random_state=DATASET_CONFIG.get("split_seed", 42),
    )

    # Filter magnification if requested
    if mag:
        train_recs = [r for r in train_recs if r["magnification"] == mag]
        test_recs = [r for r in test_recs if r["magnification"] == mag]
        if val_recs:
            val_recs = [r for r in val_recs if r["magnification"] == mag]
        print(f"[INFO] Filtered to {mag}: train={len(train_recs)}, test={len(test_recs)}")

    print_split_statistics(train_recs, test_recs, val_recs)

    # Save to disk
    materialize_split(
        train_recs, test_recs, output_dir,
        val_records=val_recs, structure="multiclass"
    )
    return train_recs, test_recs, val_recs


def step_stain_preview():
    """Step 4: Show stain normalization before/after on sample images."""
    print("\n" + "=" * 60)
    print("STEP 4: Stain Normalization Preview")
    print("=" * 60)
    import numpy as np
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from data.preprocessing import MacenkoNormalizer, ReinhardNormalizer

    train_dir = Path(DATASET_CONFIG["train_dir"])
    if not train_dir.exists():
        print("[SKIP] Train directory not found – run split first")
        return

    sample_imgs = list(train_dir.rglob("*.png"))[:6]
    if not sample_imgs:
        print("[SKIP] No images found in train directory")
        return

    macenko = MacenkoNormalizer()
    reinhard = ReinhardNormalizer()

    fig, axes = plt.subplots(len(sample_imgs), 3, figsize=(12, 4 * len(sample_imgs)))
    if len(sample_imgs) == 1:
        axes = axes[np.newaxis, :]

    for i, img_path in enumerate(sample_imgs):
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            mac_norm = macenko.normalize(img_rgb)
        except Exception:
            mac_norm = img_rgb
        try:
            rein_norm = reinhard.normalize(img_rgb)
        except Exception:
            rein_norm = img_rgb

        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title("Original" if i == 0 else "")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mac_norm)
        axes[i, 1].set_title("Macenko" if i == 0 else "")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(rein_norm)
        axes[i, 2].set_title("Reinhard" if i == 0 else "")
        axes[i, 2].axis("off")

    fig.suptitle("Stain Normalization Comparison", fontsize=14)
    fig.tight_layout()
    out = OUTPUT_DIR / "eda" / "stain_normalization_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out}")


def step_verify_dataloaders():
    """Step 5: Quick verification that dataloaders work."""
    print("\n" + "=" * 60)
    print("STEP 5: Verify DataLoaders")
    print("=" * 60)
    from data.breakhis_dataset import get_dataloaders

    try:
        train_loader, val_loader = get_dataloaders(task="multiclass")
        batch = next(iter(train_loader))
        imgs, labels, mags = batch
        print(f"[OK] Train batch: images {imgs.shape}, labels {labels.shape}")
        print(f"     Label values: {labels[:8].tolist()}")
        print(f"     Magnifications: {mags[:4]}")
        print(f"     Train samples: {len(train_loader.dataset)}")
        print(f"     Val   samples: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"[WARN] DataLoader check failed: {e}")
        print("       This is OK if data hasn't been split yet.")


def main():
    parser = argparse.ArgumentParser(description="BreaKHis data pipeline")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-eda", action="store_true")
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument("--mag", default=None, help="Filter magnification (e.g. 400X)")
    parser.add_argument("--dataset", default="full", choices=["full", "400x"])
    args = parser.parse_args()

    # Determine raw data path
    raw_root = str(DATA_DIR / "raw" / "breakhis")

    # 1. Download
    if not args.skip_download:
        raw_root = step_download(args.dataset)
    else:
        # Try to find existing data
        candidates = [
            DATA_DIR / "raw" / "breakhis",
            DATA_DIR / "raw",
            DATA_DIR / "BreaKHis 400X",
        ]
        for c in candidates:
            if c.exists():
                raw_root = str(c)
                break
        print(f"[INFO] Using existing data at: {raw_root}")

    # 2. EDA
    if not args.skip_eda:
        step_eda(raw_root)

    # 3. Patient-level split
    if not args.skip_split:
        split_output = str(DATA_DIR / "BreaKHis 400X")
        step_patient_split(raw_root, split_output, mag=args.mag)

    # 4. Stain normalization preview
    step_stain_preview()

    # 5. Verify dataloaders
    step_verify_dataloaders()

    print("\n" + "=" * 60)
    print("DATA PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
