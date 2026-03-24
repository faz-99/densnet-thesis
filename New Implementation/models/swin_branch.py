"""
Branch B: Swin-Transformer-Base for global dependencies and hierarchical features.
"""
import torch
import torch.nn as nn
import timm


class SwinBranch(nn.Module):
    """Swin-Transformer-Base feature extractor (Branch B).

    Captures global dependencies and hierarchical features via shifted-window
    self-attention, complementing the CNN branch.
    """

    def __init__(self, pretrained: bool = True, drop_path_rate: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            num_classes=0,          # feature extractor mode
        )
        self.feature_dim = self.backbone.num_features  # 1024 for swin_base

        # Storage for attention weights (populated by hooks)
        self._attention_weights = []
        self._hooks = []

    def get_feature_dim(self) -> int:
        return self.feature_dim

    # ── Attention capture for Attention Rollout ──

    def register_attention_hooks(self):
        """Register forward hooks on all attention modules to capture weights."""
        self.remove_attention_hooks()
        self._attention_weights = []

        for name, module in self.backbone.named_modules():
            if hasattr(module, "attn_drop"):  # Swin WindowAttention
                hook = module.register_forward_hook(self._attn_hook_fn)
                self._hooks.append(hook)

    def _attn_hook_fn(self, module, inp, out):
        """Hook function to capture attention weights from qkv softmax."""
        # For Swin's WindowAttention, the attention probs are computed inside
        # We need to recompute from q, k.  Alternatively, patch timm to expose them.
        # We'll monkey-patch at runtime in attention_rollout.py for cleanliness.
        pass

    def remove_attention_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._attention_weights = []

    def get_attention_weights(self):
        return self._attention_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, feature_dim) pooled features."""
        return self.backbone(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return spatial feature maps before final pooling."""
        return self.backbone.forward_features(x)
