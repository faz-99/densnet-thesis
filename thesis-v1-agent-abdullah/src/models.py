"""
Model definitions for breast histopathology classification.

Two backbone architectures for comparative study:
  - ConvNeXt-Base: A modernized CNN (2022) that matches transformer performance
  - Swin-Base: The leading Vision Transformer with shifted-window attention

Both use ImageNet-pretrained weights with a custom 8-class classification head.
"""
import torch
import torch.nn as nn
import timm
from typing import Dict, List, Optional, Tuple


def create_convnext(num_classes: int = 8, pretrained: bool = True, drop_rate: float = 0.3) -> nn.Module:
    """
    Create a ConvNeXt-Base model for classification.

    Architecture (simplified):
        Input (3x224x224)
        -> Stem: 4x4 conv, stride 4 (patchify)
        -> Stage 1: 3 blocks  (128 channels,  56x56)
        -> Stage 2: 3 blocks  (256 channels,  28x28)
        -> Stage 3: 27 blocks (512 channels,  14x14)   <-- most computation here
        -> Stage 4: 3 blocks  (1024 channels, 7x7)
        -> Global Average Pool -> LayerNorm -> Linear(1024 -> num_classes)

    Key ideas:
        - Uses depthwise separable convolutions (like MobileNet)
        - GELU activation instead of ReLU
        - LayerNorm instead of BatchNorm
        - Inverted bottleneck (expand then compress)
        - Basically a CNN that borrows all the good ideas from transformers

    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet-pretrained weights
        drop_rate: Dropout rate before the classifier head
    """
    model = timm.create_model(
        "convnext_base.fb_in22k_ft_in1k",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model


def create_swin(num_classes: int = 8, pretrained: bool = True, drop_rate: float = 0.3) -> nn.Module:
    """
    Create a Swin Transformer Base model for classification.

    Architecture (simplified):
        Input (3x224x224)
        -> Patch Embedding: 4x4 patches -> 128-dim tokens (56x56 = 3136 tokens)
        -> Stage 1: 2 Swin blocks  (128-dim,  56x56 tokens)
        -> Stage 2: 2 Swin blocks  (256-dim,  28x28 tokens)  <- patch merging halves resolution
        -> Stage 3: 18 Swin blocks (512-dim,  14x14 tokens)  <- most computation here
        -> Stage 4: 2 Swin blocks  (1024-dim, 7x7 tokens)
        -> Global Average Pool -> LayerNorm -> Linear(1024 -> num_classes)

    Key ideas:
        - Self-attention computed within LOCAL WINDOWS (7x7 by default)
        - Windows SHIFT every other layer to allow cross-window communication
        - Hierarchical (like a CNN) - resolution decreases, channels increase
        - This is what makes it better than ViT for dense tasks

    Attention Rollout (for XAI):
        - Each Swin block produces attention weights
        - We can extract and multiply them across layers to see what the model "looks at"
        - This is a NATIVE interpretability advantage over ConvNeXt

    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet-pretrained weights
        drop_rate: Dropout rate before the classifier head
    """
    model = timm.create_model(
        "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model


def get_model(name: str, num_classes: int = 8, pretrained: bool = True) -> nn.Module:
    """Factory function to create a model by name."""
    if name == "convnext":
        return create_convnext(num_classes=num_classes, pretrained=pretrained)
    elif name == "swin":
        return create_swin(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {name}. Choose 'convnext' or 'swin'.")


def get_parameter_groups(model: nn.Module, backbone_lr: float, head_lr: float) -> List[Dict]:
    """
    Create parameter groups with different learning rates.

    Why discriminative learning rates?
        The backbone was pretrained on ImageNet - its features are already good.
        The classification head is randomly initialized - it needs to learn from scratch.
        So we use a LOWER learning rate for the backbone (gentle fine-tuning)
        and a HIGHER learning rate for the head (fast learning).

    Typical ratio: head_lr = 10x backbone_lr

    Args:
        model: The model
        backbone_lr: Learning rate for pretrained backbone layers
        head_lr: Learning rate for the classification head
    """
    # timm models use different attribute names for the head
    head_params = []
    backbone_params = []

    head_names = {"head", "classifier", "fc", "head.fc"}

    for name, param in model.named_parameters():
        if any(h in name for h in head_names):
            head_params.append(param)
        else:
            backbone_params.append(param)

    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ]


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def freeze_backbone(model: nn.Module, freeze: bool = True):
    """
    Freeze/unfreeze backbone parameters (keep head trainable).

    Useful for:
        - Stage 1 of training: freeze backbone, only train head (few epochs)
        - Stage 2 of training: unfreeze all, fine-tune with low LR

    This 2-stage approach prevents the randomly-initialized head from
    sending wild gradients through the pretrained backbone early on.
    """
    head_names = {"head", "classifier", "fc"}

    for name, param in model.named_parameters():
        if any(h in name for h in head_names):
            param.requires_grad = True  # head always trainable
        else:
            param.requires_grad = not freeze


# ---------------------------------------------------------------------------
# Swin-specific progressive fine-tuning helpers
# ---------------------------------------------------------------------------

# timm Swin-Base structure:
#   patch_embed (stem)
#   layers.0  (stage 1, 2 blocks)
#   layers.1  (stage 2, 2 blocks)
#   layers.2  (stage 3, 18 blocks)  <- bulk of computation
#   layers.3  (stage 4, 2 blocks)   <- "last 2 blocks"
#   norm      (final LayerNorm)
#   head.fc   (classifier)

_SWIN_STAGE_TRAINABLE = {
    1: ("head.", "norm.", "layers.3."),
    2: ("head.", "norm.", "layers.3.", "layers.2."),
    3: None,  # everything
}


def freeze_swin_for_stage(model: nn.Module, stage: int) -> int:
    """
    Configure which parts of a Swin model are trainable for a given fine-tuning stage.

    Stage 1: head + final norm + last 2 transformer blocks (layers.3).
             Pretrained features are protected; only the classifier and the deepest
             attention blocks adapt. Prevents the random head from corrupting backbone.
    Stage 2: + layers.2 (Swin-Base's deep stage with 18 blocks).
             Now the bulk of attention adapts to histopathology textures.
    Stage 3: full model. Combine with layer-wise LR decay so shallow layers
             (closer to ImageNet features) get smaller updates than deep layers.

    Returns the number of trainable parameters after configuration.
    """
    if stage not in _SWIN_STAGE_TRAINABLE:
        raise ValueError(f"stage must be 1, 2, or 3; got {stage}")

    prefixes = _SWIN_STAGE_TRAINABLE[stage]

    if prefixes is None:
        for p in model.parameters():
            p.requires_grad = True
    else:
        for name, p in model.named_parameters():
            p.requires_grad = any(name.startswith(pref) for pref in prefixes)

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Order matters: head first so head.fc.* doesn't get swallowed by a more general prefix.
# Decay exponents are the layer's "depth" from the head (head=0, deepest stage=1, ...).
_SWIN_LAYERWISE_DEPTH = [
    ("head.", 0),
    ("norm.", 1),
    ("layers.3.", 1),
    ("layers.2.", 2),
    ("layers.1.", 3),
    ("layers.0.", 4),
    ("patch_embed.", 5),
]


def get_swin_layerwise_param_groups(
    model: nn.Module, base_lr: float, decay: float = 0.7
) -> List[Dict]:
    """
    Build AdamW parameter groups with per-layer LR decay for Swin.

    LR for a parameter at depth d is base_lr * decay**d, so the head learns at base_lr,
    each stage one step shallower learns at base_lr * decay, and the patch stem learns
    slowest. This is the standard recipe for full-model fine-tuning of ViTs/Swin and
    keeps shallow ImageNet features from drifting under the gradient pressure of a
    smaller downstream dataset.
    """
    groups: List[Dict] = []
    seen: set = set()

    for prefix, depth in _SWIN_LAYERWISE_DEPTH:
        params = []
        for name, p in model.named_parameters():
            if name in seen or not p.requires_grad:
                continue
            if name.startswith(prefix):
                params.append(p)
                seen.add(name)
        if params:
            groups.append({"params": params, "lr": base_lr * (decay ** depth)})

    # Catch-all for any parameter we didn't anticipate (none expected for Swin-Base).
    leftovers = [
        p for name, p in model.named_parameters()
        if name not in seen and p.requires_grad
    ]
    if leftovers:
        groups.append({"params": leftovers, "lr": base_lr * (decay ** 5)})

    return groups
