"""
XAI metrics package
Quantitative evaluation metrics for explainability methods
"""

from .insertion_auc import InsertionAUC
from .deletion_auc import DeletionAUC
from .iou import IoUMetric
from .stability import StabilityMetric
from .fidelity import FidelityEvaluator, compute_insertion_curve, compute_deletion_curve
from .cross_method_consistency import CrossMethodConsistency
from .sparsity import compute_sparsity, gini_coefficient, energy_concentration

__all__ = [
    'InsertionAUC', 'DeletionAUC', 'IoUMetric', 'StabilityMetric',
    'FidelityEvaluator', 'compute_insertion_curve', 'compute_deletion_curve',
    'CrossMethodConsistency',
    'compute_sparsity', 'gini_coefficient', 'energy_concentration',
]