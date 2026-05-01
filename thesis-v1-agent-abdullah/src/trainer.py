"""
Training engine for the thesis experiments.

Handles:
  - Training loop with validation
  - Early stopping
  - Checkpoint saving/loading
  - Metric logging (loss, accuracy, F1)
  - Learning curve tracking
"""
import os
import time
import json
import copy
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score

from .config import DEVICE, WEIGHTS_DIR, RESULTS_DIR


class Trainer:
    """
    Training engine for classification models.

    Usage:
        trainer = Trainer(model, train_loader, val_loader, ...)
        history = trainer.fit(max_epochs=100)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[object] = None,
        device: torch.device = DEVICE,
        experiment_name: str = "experiment",
        patience: int = 15,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.experiment_name = experiment_name
        self.patience = patience

        # State
        self.best_val_f1 = 0.0
        self.best_model_state = None
        self.epochs_without_improvement = 0
        self.history = defaultdict(list)

        # Paths
        self.save_dir = os.path.join(WEIGHTS_DIR, experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)

    def fit(self, max_epochs: int = 100) -> Dict:
        """
        Run the full training loop.

        Returns:
            history dict with per-epoch metrics
        """
        print(f"\n{'='*60}")
        print(f"Training: {self.experiment_name}")
        print(f"  Device: {self.device}")
        print(f"  Max epochs: {max_epochs}")
        print(f"  Early stopping patience: {self.patience}")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(1, max_epochs + 1):
            # Train
            train_loss, train_acc, train_f1 = self._train_epoch()

            # Validate
            val_loss, val_acc, val_f1 = self._validate_epoch()

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["train_f1"].append(train_f1)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)
            self.history["lr"].append(current_lr)

            # Print progress
            print(
                f"Epoch {epoch:3d}/{max_epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
                f"LR: {current_lr:.2e}"
            )

            # Early stopping check (based on validation F1)
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, is_best=True)
                print(f"  ** New best val F1: {val_f1:.4f} - model saved **")
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    print(f"\n  Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                    break

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time/60:.1f} minutes")
        print(f"Best validation F1: {self.best_val_f1:.4f}")

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        # Save history
        self._save_history()

        return dict(self.history)

    def _train_epoch(self) -> Tuple[float, float, float]:
        """Run one training epoch. Returns (loss, accuracy, f1)."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.train_loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")
        return avg_loss, acc, f1

    @torch.no_grad()
    def _validate_epoch(self) -> Tuple[float, float, float]:
        """Run one validation epoch. Returns (loss, accuracy, f1)."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")
        return avg_loss, acc, f1

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_f1": self.best_val_f1,
            "experiment_name": self.experiment_name,
        }

        path = os.path.join(self.save_dir, "best_model.pth" if is_best else "last_model.pth")
        torch.save(checkpoint, path)

    def _save_history(self):
        """Save training history to CSV and JSON."""
        results_dir = os.path.join(RESULTS_DIR, self.experiment_name)
        os.makedirs(results_dir, exist_ok=True)

        # CSV
        df = pd.DataFrame(self.history)
        df.to_csv(os.path.join(results_dir, "training_history.csv"), index=False)

        # JSON summary
        summary = {
            "experiment_name": self.experiment_name,
            "best_val_f1": self.best_val_f1,
            "best_epoch": int(self.history["epoch"][int(np.argmax(self.history["val_f1"]))]),
            "total_epochs": len(self.history["epoch"]),
            "final_train_loss": self.history["train_loss"][-1],
            "final_val_loss": self.history["val_loss"][-1],
            "final_train_acc": self.history["train_acc"][-1],
            "final_val_acc": self.history["val_acc"][-1],
        }
        with open(os.path.join(results_dir, "training_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)


def train_swin_progressive(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    experiment_name: str,
    device: torch.device = DEVICE,
    weight_decay: float = 1e-4,
    patience: int = 15,
    phase_epochs: Tuple[int, int, int] = (20, 40, 40),
    phase_lrs: Tuple[float, float, float] = (1e-4, 3e-5, 1e-5),
    layerwise_decay: float = 0.7,
) -> Tuple[Dict, float]:
    """
    Three-stage progressive fine-tuning loop for Swin-Base.

    Phase 1 (head + last 2 blocks): protects pretrained features while the head warms up.
    Phase 2 (+ deep stage layers.2): bulk of attention adapts to H&E texture.
    Phase 3 (all + layer-wise LR decay): full model fine-tune with shallow layers
    learning slowest.

    A fresh AdamW optimizer + CosineAnnealingLR is built per phase. Validation F1 is
    tracked across all phases; the best checkpoint and a single concatenated history
    are saved. Early stopping uses one global counter.
    """
    from .models import freeze_swin_for_stage, get_swin_layerwise_param_groups

    save_dir = os.path.join(WEIGHTS_DIR, experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    results_dir = os.path.join(RESULTS_DIR, experiment_name)
    os.makedirs(results_dir, exist_ok=True)

    model = model.to(device)
    history: Dict[str, list] = defaultdict(list)
    best_val_f1 = 0.0
    best_model_state = None
    epochs_without_improvement = 0
    global_epoch = 0
    early_stopped = False

    print(f"\n{'='*60}")
    print(f"Swin progressive training: {experiment_name}")
    print(f"  Phase epochs: {phase_epochs}  LRs: {phase_lrs}")
    print(f"  Layer-wise decay (phase 3): {layerwise_decay}")
    print(f"{'='*60}")

    start_time = time.time()

    for phase_idx, (n_epochs, lr) in enumerate(zip(phase_epochs, phase_lrs), start=1):
        if early_stopped:
            break

        n_trainable = freeze_swin_for_stage(model, phase_idx)

        if phase_idx == 3:
            param_groups = get_swin_layerwise_param_groups(
                model, base_lr=lr, decay=layerwise_decay
            )
        else:
            trainable = [p for p in model.parameters() if p.requires_grad]
            param_groups = [{"params": trainable, "lr": lr}]

        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

        print(
            f"\n--- Phase {phase_idx}: {n_epochs} epochs, "
            f"base_lr={lr:.0e}, trainable={n_trainable:,} params ---"
        )

        for phase_epoch in range(1, n_epochs + 1):
            global_epoch += 1

            model.train()
            total_loss = 0.0
            preds_all, labels_all = [], []
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * images.size(0)
                preds_all.extend(outputs.argmax(dim=1).cpu().numpy())
                labels_all.extend(labels.cpu().numpy())
            train_loss = total_loss / len(train_loader.dataset)
            train_acc = accuracy_score(labels_all, preds_all)
            train_f1 = f1_score(labels_all, preds_all, average="macro")

            model.eval()
            total_loss = 0.0
            preds_all, labels_all = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item() * images.size(0)
                    preds_all.extend(outputs.argmax(dim=1).cpu().numpy())
                    labels_all.extend(labels.cpu().numpy())
            val_loss = total_loss / len(val_loader.dataset)
            val_acc = accuracy_score(labels_all, preds_all)
            val_f1 = f1_score(labels_all, preds_all, average="macro")

            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            history["epoch"].append(global_epoch)
            history["phase"].append(phase_idx)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["train_f1"].append(train_f1)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)
            history["lr"].append(current_lr)

            print(
                f"P{phase_idx} Epoch {phase_epoch:3d}/{n_epochs} "
                f"(global {global_epoch:3d}) | "
                f"Train L: {train_loss:.4f} A: {train_acc:.4f} F1: {train_f1:.4f} | "
                f"Val L: {val_loss:.4f} A: {val_acc:.4f} F1: {val_f1:.4f} | "
                f"LR: {current_lr:.2e}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                torch.save(
                    {
                        "epoch": global_epoch,
                        "phase": phase_idx,
                        "model_state_dict": model.state_dict(),
                        "best_val_f1": best_val_f1,
                        "experiment_name": experiment_name,
                    },
                    os.path.join(save_dir, "best_model.pth"),
                )
                print(f"  ** New best val F1: {val_f1:.4f} - model saved **")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(
                        f"\n  Early stopping at global epoch {global_epoch} "
                        f"(no improvement for {patience} epochs)"
                    )
                    early_stopped = True
                    break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    total_time = time.time() - start_time
    print(f"\nSwin progressive training complete in {total_time/60:.1f} minutes")
    print(f"Best validation F1: {best_val_f1:.4f}")

    df = pd.DataFrame(history)
    df.to_csv(os.path.join(results_dir, "training_history.csv"), index=False)
    summary = {
        "experiment_name": experiment_name,
        "best_val_f1": best_val_f1,
        "best_epoch": int(history["epoch"][int(np.argmax(history["val_f1"]))]),
        "total_epochs": len(history["epoch"]),
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "training_strategy": "swin_progressive_3stage",
        "phase_epochs": list(phase_epochs),
        "phase_lrs": list(phase_lrs),
        "layerwise_decay": layerwise_decay,
    }
    with open(os.path.join(results_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return dict(history), best_val_f1


def create_criterion(class_weights: torch.Tensor, label_smoothing: float = 0.1, device=DEVICE) -> nn.Module:
    """
    Create a weighted cross-entropy loss with label smoothing.

    Why class weights?
        BreakHis is imbalanced. Without weights, the model optimizes
        for the majority class. Weights penalize mistakes on rare
        classes more heavily.

    Why label smoothing?
        Instead of hard labels [0, 0, 0, 1, 0, 0, 0, 0], we use
        soft labels [0.014, 0.014, 0.014, 0.9, 0.014, ...].
        This prevents the model from becoming overconfident and
        improves generalization.
    """
    return nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=label_smoothing,
    )
