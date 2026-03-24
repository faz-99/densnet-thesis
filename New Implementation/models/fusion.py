"""
Fusion heads: Weighted Average Ensemble and MLP fusion.
"""
import torch
import torch.nn as nn


class WeightedAverageFusion(nn.Module):
    """Learn a scalar weight per branch and combine feature vectors."""

    def __init__(self, feature_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        # Learnable branch weights (softmaxed before use)
        self.branch_weights = nn.Parameter(torch.ones(2))
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.branch_weights, dim=0)
        fused = w[0] * feat_a + w[1] * feat_b
        return self.classifier(fused)

    def get_fused_features(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.branch_weights, dim=0)
        return w[0] * feat_a + w[1] * feat_b


class MLPFusion(nn.Module):
    """MLP fusion head: concatenate branch features and project."""

    def __init__(
        self,
        dim_a: int,
        dim_b: int,
        hidden_dim: int = 512,
        num_classes: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        concat_dim = dim_a + dim_b
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        fused = self.fusion_mlp(torch.cat([feat_a, feat_b], dim=-1))
        return self.classifier(fused)

    def get_fused_features(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        return self.fusion_mlp(torch.cat([feat_a, feat_b], dim=-1))
