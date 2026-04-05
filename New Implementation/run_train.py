"""
Training entry point for the Multimodal Histopathology XAI Framework.
Supports magnification-aware training across 40X, 100X, 200X, 400X.

Usage:
    python run_train.py                         # default 400X, 8-class
    python run_train.py --task binary            # binary classification
    python run_train.py --magnification all      # train on all magnifications
    python run_train.py --no-wandb               # disable WandB logging
"""
import argparse
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import Subset
from config.settings import DATASET_CONFIG, MODEL_CONFIG, TRAIN_CONFIG
from data.breakhis_dataset import get_dataloaders
from models.ensemble import HybridEnsemble
from training.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="Train Hybrid Ensemble on BreaKHis")
    p.add_argument("--task", choices=["binary", "multiclass"], default="multiclass")
    p.add_argument("--magnification", default="400X",
                   help="40X|100X|200X|400X|all")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--fusion", choices=["mlp", "weighted_avg"], default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--subset", type=int, default=None,
                   help="Limit training to N images for quick CPU testing (e.g. 800)")
    p.add_argument("--val-subset", type=int, default=None,
                   help="Limit validation to N images (e.g. 200)")
    return p.parse_args()


def main():
    args = parse_args()

    # Override config
    if args.epochs:
        TRAIN_CONFIG["epochs"] = args.epochs
    if args.lr:
        TRAIN_CONFIG["lr"] = args.lr

    num_classes = 2 if args.task == "binary" else 8
    MODEL_CONFIG["num_classes"] = num_classes
    MODEL_CONFIG["task"] = args.task
    if args.fusion:
        MODEL_CONFIG["fusion"]["method"] = args.fusion

    magnifications = (
        DATASET_CONFIG["magnifications"]
        if args.magnification == "all"
        else [args.magnification]
    )

    for mag in magnifications:
        print(f"\n{'='*60}")
        print(f"Training on magnification: {mag} | Task: {args.task} | Classes: {num_classes}")
        print(f"{'='*60}\n")

        train_loader, val_loader = get_dataloaders(
            task=args.task,
            magnification=mag,
            batch_size=args.batch_size,
        )

        # get class weights from the full dataset before any subsetting
        class_weights = train_loader.dataset.get_class_weights()

        # Subset for quick CPU testing — balanced across classes
        if args.subset:
            import numpy as np
            from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
            ds = train_loader.dataset
            labels = np.array([s[1] for s in ds.samples])
            per_class = args.subset // num_classes
            indices = []
            for c in range(num_classes):
                cls_idx = np.where(labels == c)[0]
                chosen = cls_idx[:per_class] if len(cls_idx) >= per_class else cls_idx
                indices.extend(chosen.tolist())
            sub_ds = Subset(ds, indices)
            sub_weights = ds.get_sample_weights()[indices]
            sampler = WeightedRandomSampler(sub_weights, len(sub_weights), replacement=True)
            bs = args.batch_size or DATASET_CONFIG["batch_size"]
            train_loader = DataLoader(sub_ds, batch_size=bs, sampler=sampler,
                                      num_workers=0, pin_memory=False, drop_last=True)
            print(f"[Subset] Using {len(sub_ds)} training images ({per_class} per class)")

        # Val subset for faster CPU iteration
        if args.val_subset:
            import numpy as np
            from torch.utils.data import DataLoader, Subset
            val_ds = val_loader.dataset
            val_indices = list(range(min(args.val_subset, len(val_ds))))
            val_sub = Subset(val_ds, val_indices)
            bs = args.batch_size or DATASET_CONFIG["batch_size"]
            val_loader = DataLoader(val_sub, batch_size=bs, shuffle=False,
                                    num_workers=0, pin_memory=False)
            print(f"[Val Subset] Using {len(val_sub)} validation images")

        model = HybridEnsemble(num_classes=num_classes)

        if args.resume:
            Trainer.load_checkpoint(model, args.resume, args.device)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=args.device,
            use_wandb=not args.no_wandb,
            class_weights=class_weights,
            run_name=f"hybrid_{args.task}_{mag}",
        )

        history = trainer.train()
        print(f"[Done] Magnification {mag} training complete.\n")


if __name__ == "__main__":
    main()
