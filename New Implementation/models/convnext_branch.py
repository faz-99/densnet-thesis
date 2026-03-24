"""
Branch A: ConvNeXt-Base for local texture / inductive-bias features.
"""
import torch
import torch.nn as nn
import timm


class ConvNeXtBranch(nn.Module):
    """ConvNeXt-Base feature extractor (Branch A).

    Captures local inductive biases and texture patterns critical for
    histopathological tissue micro-structure analysis.
    """

    def __init__(self, pretrained: bool = True, drop_path_rate: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_base",
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            num_classes=0,          # remove classifier head → feature extractor
        )
        self.feature_dim = self.backbone.num_features  # 1024 for convnext_base

    # ── public helpers for XAI hooks ──

    def get_target_layer(self):
        """Return the last convolutional stage for Grad-CAM hooking."""
        return self.backbone.stages[-1]

    def get_feature_dim(self) -> int:
        return self.feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, feature_dim) pooled features."""
        return self.backbone(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, C, H, W) spatial feature maps before pooling."""
        return self.backbone.forward_features(x)
