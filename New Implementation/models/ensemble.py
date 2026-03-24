"""
Hybrid dual-branch ensemble: ConvNeXt + Swin Transformer with fusion head.
"""
import torch
import torch.nn as nn

from models.convnext_branch import ConvNeXtBranch
from models.swin_branch import SwinBranch
from models.fusion import MLPFusion, WeightedAverageFusion
from config.settings import MODEL_CONFIG


class HybridEnsemble(nn.Module):
    """Dual-branch ensemble for histopathology classification.

    Branch A  – ConvNeXt-Base  (local texture, inductive biases)
    Branch B  – Swin-Base      (global dependencies, hierarchical features)
    Fusion    – MLP or Weighted Average
    """

    def __init__(self, num_classes: int = None, fusion_method: str = None):
        super().__init__()
        nc = num_classes or MODEL_CONFIG["num_classes"]
        fm = fusion_method or MODEL_CONFIG["fusion"]["method"]

        # ── Branches ──
        cfg_c = MODEL_CONFIG["convnext"]
        cfg_s = MODEL_CONFIG["swin"]
        self.branch_a = ConvNeXtBranch(
            pretrained=cfg_c["pretrained"],
            drop_path_rate=cfg_c["drop_path_rate"],
        )
        self.branch_b = SwinBranch(
            pretrained=cfg_s["pretrained"],
            drop_path_rate=cfg_s["drop_path_rate"],
        )

        dim_a = self.branch_a.get_feature_dim()
        dim_b = self.branch_b.get_feature_dim()

        # ── Fusion ──
        if fm == "mlp":
            self.fusion = MLPFusion(
                dim_a=dim_a, dim_b=dim_b,
                hidden_dim=MODEL_CONFIG["fusion"]["hidden_dim"],
                num_classes=nc,
                dropout=MODEL_CONFIG["fusion"]["dropout"],
            )
        else:
            assert dim_a == dim_b, "Weighted average requires equal feature dims"
            self.fusion = WeightedAverageFusion(
                feature_dim=dim_a, num_classes=nc,
                dropout=MODEL_CONFIG["fusion"]["dropout"],
            )

        self.num_classes = nc
        self._fusion_method = fm

    # ── Forward ──

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_a = self.branch_a(x)
        feat_b = self.branch_b(x)
        return self.fusion(feat_a, feat_b)

    # ── Helpers for XAI & reporting ──

    def get_branch_features(self, x: torch.Tensor):
        """Return (feat_a, feat_b) pooled vectors."""
        feat_a = self.branch_a(x)
        feat_b = self.branch_b(x)
        return feat_a, feat_b

    def get_fused_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return fused features before the final classifier."""
        feat_a, feat_b = self.get_branch_features(x)
        return self.fusion.get_fused_features(feat_a, feat_b)

    def get_convnext_target_layer(self):
        """Target layer for Grad-CAM on the CNN branch."""
        return self.branch_a.get_target_layer()

    def get_spatial_features(self, x: torch.Tensor):
        """Return spatial feature maps from both branches."""
        spatial_a = self.branch_a.forward_features(x)
        spatial_b = self.branch_b.forward_features(x)
        return spatial_a, spatial_b

    def freeze_branches(self, freeze: bool = True):
        """Freeze/unfreeze both backbones (fine-tune only fusion head)."""
        for param in self.branch_a.parameters():
            param.requires_grad = not freeze
        for param in self.branch_b.parameters():
            param.requires_grad = not freeze
