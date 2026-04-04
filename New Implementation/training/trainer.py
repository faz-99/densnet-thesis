"""
Training loop with:
  - Mixed precision (AMP)
  - Cosine / step LR scheduling with warmup
  - Focal loss & label smoothing
  - Early stopping
  - Gradient clipping
  - WandB logging (images, metrics, XAI panels)
  - Magnification-aware training
"""
import os
import time
import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

from config.settings import TRAIN_CONFIG, MODEL_CONFIG, DATASET_CONFIG, WANDB_CONFIG, CHECKPOINT_DIR


# ────────────────────────────────
# Loss functions
# ────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced datasets."""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()


class LabelSmoothingCE(nn.Module):
    """Cross-entropy with label smoothing."""

    def __init__(self, smoothing: float = 0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, inputs, targets):
        n_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        if self.weight is not None:
            w = self.weight[targets].unsqueeze(1)
            loss = -(smooth_targets * log_probs * w).sum(dim=-1).mean()
        else:
            loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
        return loss


# ────────────────────────────────
# Trainer
# ────────────────────────────────

class Trainer:
    """End-to-end training manager for the HybridEnsemble."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device: str = "cuda",
        use_wandb: bool = True,
        class_weights: torch.Tensor = None,
        run_name: str = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_wandb = use_wandb

        cfg = TRAIN_CONFIG

        # ── Loss ──
        cw = class_weights.to(device) if class_weights is not None else None
        if cfg["use_focal_loss"]:
            self.criterion = FocalLoss(
                alpha=cfg["focal_alpha"], gamma=cfg["focal_gamma"], weight=cw
            )
        else:
            self.criterion = LabelSmoothingCE(
                smoothing=cfg["label_smoothing"], weight=cw
            )

        # ── Optimizer ──
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )

        # ── Scheduler ──
        if cfg["scheduler"] == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=cfg["epochs"] - cfg["warmup_epochs"],
                eta_min=cfg["min_lr"],
            )
        else:
            self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.5)

        self.warmup_epochs = cfg["warmup_epochs"]
        self.warmup_lr_init = cfg["lr"] / 10

        # ── AMP ──
        self.use_amp = cfg["mixed_precision"] and device == "cuda"
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # ── Early stopping ──
        self.patience = cfg["early_stopping_patience"]
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.epochs_no_improve = 0

        self.grad_clip = cfg["grad_clip_norm"]
        self.epochs = cfg["epochs"]

        # ── WandB ──
        self.wandb_run = None
        if self.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=WANDB_CONFIG["project"],
                    entity=WANDB_CONFIG["entity"],
                    name=run_name or f"hybrid_ensemble_{time.strftime('%Y%m%d_%H%M%S')}",
                    config={
                        "model": MODEL_CONFIG,
                        "train": TRAIN_CONFIG,
                        "dataset": DATASET_CONFIG,
                    },
                )
            except Exception as e:
                print(f"[Trainer] WandB init failed: {e}. Proceeding without logging.")
                self.use_wandb = False

    def _warmup_lr(self, epoch: int):
        """Linear warmup learning rate."""
        if epoch >= self.warmup_epochs:
            return
        lr = self.warmup_lr_init + (TRAIN_CONFIG["lr"] - self.warmup_lr_init) * (
            epoch / self.warmup_epochs
        )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def train_one_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        from tqdm import tqdm
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=True)

        for step, (images, labels, _mags) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{accuracy_score(all_labels, all_preds):.4f}")

            # Logging
            if self.use_wandb and step % WANDB_CONFIG["log_interval"] == 0:
                import wandb
                wandb.log({
                    "train/step_loss": loss.item(),
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                })

        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        return epoch_loss, epoch_acc, epoch_f1

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for images, labels, _mags in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)

        return {
            "loss": epoch_loss,
            "accuracy": epoch_acc,
            "f1": epoch_f1,
            "precision": precision,
            "recall": recall,
            "preds": all_preds,
            "labels": all_labels,
        }

    def train(self):
        """Full training loop."""
        print(f"[Trainer] Starting training for {self.epochs} epochs on {self.device}")
        print(f"[Trainer] Train size: {len(self.train_loader.dataset)}, "
              f"Val size: {len(self.val_loader.dataset)}")

        history = {"train_loss": [], "train_acc": [], "val_loss": [],
                    "val_acc": [], "val_f1": []}

        for epoch in range(self.epochs):
            t0 = time.time()
            self._warmup_lr(epoch)

            train_loss, train_acc, train_f1 = self.train_one_epoch(epoch)

            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            val_metrics = self.validate()

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch+1}/{self.epochs} ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
                f"F1: {val_metrics['f1']:.4f}"
            )

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_f1"].append(val_metrics["f1"])

            # WandB epoch logging
            if self.use_wandb:
                import wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "train/epoch_loss": train_loss,
                    "train/epoch_acc": train_acc,
                    "train/epoch_f1": train_f1,
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/f1": val_metrics["f1"],
                    "val/precision": val_metrics["precision"],
                    "val/recall": val_metrics["recall"],
                })

            # Checkpointing
            improved = False
            if val_metrics["f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1"]
                self.best_val_acc = val_metrics["accuracy"]
                self.epochs_no_improve = 0
                improved = True
                self._save_checkpoint(epoch, val_metrics, "best_model.pth")
                print(f"  → New best F1: {self.best_val_f1:.4f}")
            else:
                self.epochs_no_improve += 1

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, val_metrics, f"checkpoint_epoch_{epoch+1}.pth")

            # Early stopping
            if self.epochs_no_improve >= self.patience:
                print(f"[Trainer] Early stopping at epoch {epoch+1} "
                      f"(no improvement for {self.patience} epochs)")
                break

        # Final report
        print(f"\n[Trainer] Training complete. Best Val Acc: {self.best_val_acc:.4f}, "
              f"Best Val F1: {self.best_val_f1:.4f}")

        # Save history
        history_path = CHECKPOINT_DIR / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        if self.use_wandb:
            import wandb
            wandb.finish()

        return history

    def _save_checkpoint(self, epoch, val_metrics, filename):
        path = CHECKPOINT_DIR / filename
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
        }, path)

    @staticmethod
    def load_checkpoint(model, path: str, device: str = "cuda"):
        """Load a saved checkpoint."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Trainer] Loaded checkpoint from {path} "
              f"(epoch {ckpt['epoch']+1}, val_f1={ckpt.get('val_f1', 'N/A')})")
        return model
