"""
ConvNeXt backbone for breast cancer histopathology classification.
Supports binary (benign/malignant) and 8-class multiclass settings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional


class ConvNeXtClassifier(nn.Module):
    """
    ConvNeXt-Base fine-tuned for BreaKHis classification.
    Adds a custom head with dropout for regularization.
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.3,
                 pretrained: bool = True, model_name: str = 'convnext_base'):
        super().__init__()
        self.num_classes = num_classes

        # Load ConvNeXt backbone via timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,          # Remove default head
            global_pool='avg'
        )
        in_features = self.backbone.num_features  # 1024 for convnext_base

        # Custom classification head
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)   # (B, in_features)
        return self.head(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate features (before final linear)."""
        return self.backbone(x)

    # ------------------------------------------------------------------ #
    # Grad-CAM compatible: expose the last conv stage as target layer     #
    # ------------------------------------------------------------------ #
    def get_target_layer(self) -> nn.Module:
        """Return the last feature stage for Grad-CAM hooks."""
        # convnext_base stages: backbone.stages[-1].blocks[-1].norm
        try:
            return self.backbone.stages[-1].blocks[-1].norm
        except AttributeError:
            # Fallback: last norm layer
            return self.backbone.norm_pre


def create_convnext(num_classes: int = 2, dropout_rate: float = 0.3,
                    pretrained: bool = True) -> ConvNeXtClassifier:
    return ConvNeXtClassifier(num_classes=num_classes,
                              dropout_rate=dropout_rate,
                              pretrained=pretrained)
