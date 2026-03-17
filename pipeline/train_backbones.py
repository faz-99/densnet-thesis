"""
Training script for ConvNeXt and Swin Transformer on BreaKHis.

Usage:
  python pipeline/train_backbones.py --model convnext --task binary --epochs 50
  python pipeline/train_backbones.py --model swin     --task binary --epochs 50
"""
import sys
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.breakhis_dataset import setup_breakhis_dataset, BINARY_CLASSES, MULTICLASS_NAMES
from model.convnext_model import create_convnext
from model.swin_classifier import create_swin


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for imgs, labels in tqdm(loader, desc='Train', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / len(loader), acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device, n_classes):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []
    for imgs, labels in tqdm(loader, desc='Eval', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item()
        probs = F.softmax(out, dim=1)
        all_preds.extend(probs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='macro', zero_division=0)
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except Exception:
        auc = 0.0

    return total_loss / len(loader), acc, f1, auc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',    choices=['convnext', 'swin'], default='convnext')
    p.add_argument('--task',     choices=['binary', 'multiclass'], default='binary')
    p.add_argument('--data_root', default='datasets/BreaKHis 400X')
    p.add_argument('--magnification', default='400X')
    p.add_argument('--epochs',   type=int, default=50)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr',       type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--dropout',  type=float, default=0.3)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--save_dir', default='checkpoints')
    args = p.parse_args()

    device = get_device()
    class_names = BINARY_CLASSES if args.task == 'binary' else MULTICLASS_NAMES
    n_classes = len(class_names)

    print(f"[Train] Model={args.model}  Task={args.task}  "
          f"Classes={n_classes}  Device={device}")

    # Data
    dm = setup_breakhis_dataset(args.data_root, args.magnification)
    if dm is None:
        raise RuntimeError(f"Dataset not found at {args.data_root}")
    dataloaders, datasets = dm.create_dataloaders(
        task=args.task, batch_size=args.batch_size, num_workers=args.num_workers)

    # Model
    if args.model == 'convnext':
        model = create_convnext(n_classes, args.dropout)
    else:
        model = create_swin(n_classes, args.dropout)
    model.to(device)

    # Loss with class weights
    class_weights = datasets['train'].dataset.get_class_weights().to(device) \
        if hasattr(datasets['train'], 'dataset') else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    save_dir = Path(args.save_dir) / args.model / args.task
    save_dir.mkdir(parents=True, exist_ok=True)

    best_f1, history = 0.0, []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, dataloaders['train'], optimizer, criterion, device)
        val_loss, val_acc, val_f1, val_auc = evaluate(
            model, dataloaders['val'], criterion, device, n_classes)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} tr_f1={tr_f1:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
              f"val_auc={val_auc:.4f} | {elapsed:.1f}s")

        history.append({'epoch': epoch, 'val_f1': val_f1, 'val_auc': val_auc})

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_auc': val_auc,
                'class_names': class_names,
            }, save_dir / 'best.pth')
            print(f"  ✓ Best model saved (F1={best_f1:.4f})")

    # Save training history
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n[Train] Done. Best val F1 = {best_f1:.4f}")
    print(f"[Train] Checkpoint → {save_dir / 'best.pth'}")


if __name__ == '__main__':
    main()
