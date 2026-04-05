"""
Attention Rollout for the Swin Transformer branch.
Visualizes patch-level importance by aggregating attention across layers.
"""
import torch
import numpy as np
import torch.nn.functional as F
from functools import partial


class AttentionRollout:
    """Attention Rollout for Swin-Transformer.

    Captures attention weights from all WindowAttention layers and
    recursively multiplies them (with residual connections) to produce
    a single input-resolution importance map.
    """

    def __init__(self, model, head_fusion: str = "mean", discard_ratio: float = 0.1):
        """
        Args:
            model: HybridEnsemble (will hook into branch_b / Swin).
            head_fusion: How to fuse multi-head attention ("mean"|"max"|"min").
            discard_ratio: Fraction of lowest-attention tokens to zero out per layer.
        """
        self.model = model
        self.swin = model.branch_b.backbone
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio

        self._attention_maps = []
        self._hooks = []

    def _hook_fn(self, module, inp, out, attn_storage):
        """Capture attention probabilities from WindowAttention.

        We monkey-patch the forward to also store the attention matrix.
        """
        attn_storage.append(module._attn_probs.detach().cpu())

    def _patch_attention_modules(self):
        """Patch Swin WindowAttention to expose attn_probs."""
        self._originals = {}
        for name, module in self.swin.named_modules():
            cls_name = module.__class__.__name__
            if "Attention" in cls_name and hasattr(module, "qkv"):
                original_forward = module.forward
                self._originals[name] = original_forward

                def patched_forward(self_mod, x, mask=None, _orig=original_forward):
                    B_, N, C = x.shape
                    qkv = self_mod.qkv(x).reshape(B_, N, 3, self_mod.num_heads, C // self_mod.num_heads)
                    qkv = qkv.permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    q = q * self_mod.scale
                    attn = q @ k.transpose(-2, -1)

                    if hasattr(self_mod, "relative_position_bias_table"):
                        relative_position_bias = self_mod.relative_position_bias_table[
                            self_mod.relative_position_index.view(-1)
                        ].view(
                            self_mod.window_size[0] * self_mod.window_size[1],
                            self_mod.window_size[0] * self_mod.window_size[1],
                            -1,
                        )
                        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
                        attn = attn + relative_position_bias.unsqueeze(0)

                    if mask is not None:
                        nW = mask.shape[0]
                        attn = attn.view(B_ // nW, nW, self_mod.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                        attn = attn.view(-1, self_mod.num_heads, N, N)

                    attn = attn.softmax(dim=-1)
                    self_mod._attn_probs = attn.detach()  # store for hook
                    attn = self_mod.attn_drop(attn)
                    x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
                    x = self_mod.proj(x)
                    x = self_mod.proj_drop(x)
                    return x

                module.forward = partial(patched_forward, module)

    def _unpatch_attention_modules(self):
        for name, orig in self._originals.items():
            parts = name.split(".")
            mod = self.swin
            for p in parts:
                mod = getattr(mod, p)
            mod.forward = orig
        self._originals = {}

    def _register_hooks(self):
        self._attention_maps = []
        for name, module in self.swin.named_modules():
            cls_name = module.__class__.__name__
            if "Attention" in cls_name and hasattr(module, "qkv"):
                hook = module.register_forward_hook(
                    partial(self._hook_fn, attn_storage=self._attention_maps)
                )
                self._hooks.append(hook)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _fuse_heads(self, attn: torch.Tensor) -> torch.Tensor:
        """Fuse multi-head attention: (B, heads, N, N) → (B, N, N)."""
        if self.head_fusion == "mean":
            return attn.mean(dim=1)
        elif self.head_fusion == "max":
            return attn.max(dim=1).values
        elif self.head_fusion == "min":
            return attn.min(dim=1).values
        return attn.mean(dim=1)

    @torch.no_grad()
    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """Generate attention rollout heatmap.

        Args:
            input_tensor: (1, 3, H, W).
            target_class: Not used (rollout is class-agnostic).

        Returns:
            heatmap: (H, W) numpy array in [0, 1].
        """
        self.model.eval()
        self._patch_attention_modules()
        self._register_hooks()

        try:
            _ = self.model(input_tensor)
        finally:
            self._remove_hooks()
            self._unpatch_attention_modules()

        if not self._attention_maps:
            # Fallback: return uniform map
            H, W = input_tensor.shape[2], input_tensor.shape[3]
            return np.ones((H, W), dtype=np.float32) * 0.5

        # Rollout: recursively multiply attention matrices with residual
        rollout = None
        for attn in self._attention_maps:
            # attn: (num_windows*B, heads, win_size^2, win_size^2)
            attn_fused = self._fuse_heads(attn)  # (num_windows*B, N, N)

            # Add residual (identity)
            eye = torch.eye(attn_fused.size(-1), device=attn_fused.device)
            attn_fused = attn_fused + eye
            attn_fused = attn_fused / attn_fused.sum(dim=-1, keepdim=True)

            # Discard low-attention
            if self.discard_ratio > 0:
                flat = attn_fused.view(-1)
                threshold = torch.quantile(flat, self.discard_ratio)
                attn_fused = attn_fused * (attn_fused >= threshold).float()
                attn_fused = attn_fused / (attn_fused.sum(dim=-1, keepdim=True) + 1e-8)

            # Average across windows
            avg_attn = attn_fused.mean(dim=0)  # (N, N)

            if rollout is None:
                rollout = avg_attn
            else:
                # Align sizes (Swin has varying window sizes across stages)
                if rollout.shape != avg_attn.shape:
                    # Interpolate rollout to match current stage
                    s = avg_attn.shape[0]
                    rollout = F.interpolate(
                        rollout.unsqueeze(0).unsqueeze(0),
                        size=(s, s), mode="bilinear", align_corners=False,
                    ).squeeze()
                rollout = avg_attn @ rollout

        # Convert 1D rollout to 2D spatial map
        n_patches = rollout.shape[0]
        grid_size = int(np.sqrt(n_patches))
        if grid_size * grid_size != n_patches:
            grid_size = int(np.ceil(np.sqrt(n_patches)))

        # Take diagonal (self-attention importance) or first row
        importance = rollout.mean(dim=0).numpy()
        importance = importance[:grid_size * grid_size]
        heatmap = importance.reshape(grid_size, grid_size)

        # Upsample
        H, W = input_tensor.shape[2], input_tensor.shape[3]
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).float()
        heatmap_up = F.interpolate(heatmap_tensor, size=(H, W), mode="bilinear", align_corners=False)
        heatmap = heatmap_up.squeeze().numpy()

        # Normalize
        h_min, h_max = heatmap.min(), heatmap.max()
        if h_max - h_min > 1e-8:
            heatmap = (heatmap - h_min) / (h_max - h_min)
        else:
            heatmap = np.ones_like(heatmap) * 0.5
        return heatmap
