"""
Comprehensive classification metrics for binary and multiclass settings.

Computes:
  Accuracy, Precision, Recall/Sensitivity, Specificity, F1,
  ROC-AUC, PR-AUC, Confusion Matrix, per-class metrics.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)


def compute_specificity(y_true: np.ndarray, y_pred: np.ndarray,
                        average: str = 'macro') -> float:
    """Specificity = TN / (TN + FP) per class, then averaged."""
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    specificities = []
    for i in range(n_classes):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        denom = tn + fp
        specificities.append(tn / denom if denom > 0 else 0.0)
    if average == 'macro':
        return float(np.mean(specificities))
    return specificities


def evaluate_classification(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             y_prob: np.ndarray,
                             class_names: List[str],
                             save_dir: Optional[str] = None) -> Dict:
    """
    Full classification evaluation.

    Args:
        y_true: ground-truth labels (N,)
        y_pred: predicted labels (N,)
        y_prob: predicted probabilities (N, C)
        class_names: list of class name strings
        save_dir: if given, saves confusion matrix and ROC plots

    Returns:
        metrics dict
    """
    n_classes = len(class_names)
    is_binary = n_classes == 2

    metrics = {
        'accuracy':    float(accuracy_score(y_true, y_pred)),
        'precision':   float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall':      float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'specificity': float(compute_specificity(y_true, y_pred, average='macro')),
        'f1':          float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
    }

    # ROC-AUC
    try:
        if is_binary:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            metrics['roc_auc'] = float(roc_auc_score(
                y_true, y_prob, multi_class='ovr', average='macro'))
    except Exception:
        metrics['roc_auc'] = 0.0

    # PR-AUC
    try:
        if is_binary:
            metrics['pr_auc'] = float(average_precision_score(y_true, y_prob[:, 1]))
        else:
            metrics['pr_auc'] = float(average_precision_score(
                y_true, y_prob, average='macro'))
    except Exception:
        metrics['pr_auc'] = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    # Per-class metrics
    per_class = {}
    prec_arr = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_arr  = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_arr   = f1_score(y_true, y_pred, average=None, zero_division=0)
    spec_arr = compute_specificity(y_true, y_pred, average=None)

    for i, name in enumerate(class_names):
        per_class[name] = {
            'precision':   float(prec_arr[i]) if i < len(prec_arr) else 0.0,
            'recall':      float(rec_arr[i])  if i < len(rec_arr)  else 0.0,
            'f1':          float(f1_arr[i])   if i < len(f1_arr)   else 0.0,
            'specificity': float(spec_arr[i]) if i < len(spec_arr) else 0.0,
        }
    metrics['per_class'] = per_class

    if save_dir:
        _save_plots(y_true, y_pred, y_prob, cm, class_names, save_dir)

    return metrics


def _save_plots(y_true, y_pred, y_prob, cm, class_names, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names))))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    fig.tight_layout()
    fig.savefig(save_dir / 'confusion_matrix.png', dpi=150)
    plt.close(fig)

    # ROC curves (one-vs-rest)
    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
            auc_val = roc_auc_score((y_true == i).astype(int), y_prob[:, i])
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC Curves (OvR)')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(save_dir / 'roc_curves.png', dpi=150)
    plt.close(fig)

    # PR curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(class_names):
        try:
            prec, rec, _ = precision_recall_curve((y_true == i).astype(int), y_prob[:, i])
            ap = average_precision_score((y_true == i).astype(int), y_prob[:, i])
            ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
        except Exception:
            pass
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(save_dir / 'pr_curves.png', dpi=150)
    plt.close(fig)
