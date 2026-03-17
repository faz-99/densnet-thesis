"""
Swin Transformer wrapper for BreaKHis classification.
Wraps the existing SwinTransformer with a custom head and
exposes attention weights for Attention Rollout.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.swin_transformer import SwinTransformer


class SwinClassifier(nn.Module):
    """
    Swin-Tiny / Swin-Base wrapper with:
    - Custom classification head
    - Attention weight capture for Attention Rollout
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.3,
                 variant: str = 'tiny'):
        super().__init__()
        self.num_classes = num_classes
        self.variant = variant
        self._attention_weights = []   # populated by hooks

        if variant == 'tiny':
            self.backbone = SwinTransformer(
                patch_size=4, in_chans=3, num_classes=0,
                embed_dim=96, depths=(2, 2, 6, 2),
                num_heads=(3, 6, 12, 24), window_size=7
            )
            in_features = 768   # 96 * 2^3
        elif variant == 'base':
            self.backbone = SwinTransformer(
                patch_size=4, in_chans=3, num_classes=0,
                embed_dim=128, depths=(2, 2, 18, 2),
                num_heads=(4, 8, 16, 32), window_size=7
            )
            in_features = 1024  # 128 * 2^3
        else:
            raise ValueError(f"Unknown variant: {variant}")

        # Remove the default head (set num_classes=0 above keeps Identity)
        self.backbone.head = nn.Identity()

        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # backbone returns (B, num_features) after avgpool + flatten
        features = self.backbone(x)
        return self.head(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def register_attention_hooks(self):
        """Register hooks to capture attention weights for Attention Rollout."""
        self._attention_weights = []
        self._hooks = []

        def _hook(module, inp, out):
            # WindowAttention stores softmax output in forward; we capture it
            self._attention_weights.append(out.detach().cpu())

        for name, module in self.backbone.named_modules():
            if module.__class__.__name__ == 'WindowAttention':
                self._hooks.append(module.register_forward_hook(_hook))

    def remove_attention_hooks(self):
        for h in getattr(self, '_hooks', []):
            h.remove()
        self._hooks = []

    def get_attention_weights(self):
        return self._attention_weights


def create_swin(num_classes: int = 2, dropout_rate: float = 0.3,
                variant: str = 'tiny') -> SwinClassifier:
    return SwinClassifier(num_classes=num_classes,
                          dropout_rate=dropout_rate,
                          variant=variant)
