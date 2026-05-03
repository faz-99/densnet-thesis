"""
Figure 3.3: Swin three-phase progressive unfreezing schedule.
Saves to figures/swin_progressive_unfreezing.png.
"""
import os, sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import FIGURES_DIR

OUT = os.path.join(FIGURES_DIR, "swin_progressive_unfreezing.png")

fig, ax = plt.subplots(figsize=(13, 8))
ax.set_xlim(0, 13); ax.set_ylim(0, 9); ax.axis("off")

C_FROZEN = "#d6d6d6"
C_TRAIN  = "#7ec97e"
C_HEAD   = "#ffd0d0"
EDGE = "#333"

def box(x, y, w, h, text, color, fs=9, weight="normal", text_color="#222"):
    p = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2, edgecolor=EDGE, facecolor=color)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, color=text_color)

LAYERS = [
    ("Linear(1024 → 8)  head", 6.6, C_HEAD),
    ("LayerNorm",                5.9, C_HEAD),
    ("Stage 4  (layers.3, 2 blocks)", 5.2, None),
    ("Stage 3  (layers.2, 18 blocks)", 4.5, None),
    ("Stage 2  (layers.1, 2 blocks)", 3.8, None),
    ("Stage 1  (layers.0, 2 blocks)", 3.1, None),
    ("Patch embedding (patch_embed)", 2.4, None),
]

phases = [
    {
        "x_center": 2.5,
        "title": "Phase 1   (epochs 1–20)",
        "subtitle": "lr = 1e-4   •   trainable: head + final norm + layers.3",
        "trainable": {"head", "norm", "layers.3"},
    },
    {
        "x_center": 6.5,
        "title": "Phase 2   (epochs 21–60)",
        "subtitle": "lr = 3e-5   •   + layers.2 (deep stage, 18 blocks)",
        "trainable": {"head", "norm", "layers.3", "layers.2"},
    },
    {
        "x_center": 10.5,
        "title": "Phase 3   (epochs 61–100)",
        "subtitle": "lr = 1e-5   •   full model + layer-wise decay γ = 0.7",
        "trainable": {"head", "norm", "layers.3", "layers.2", "layers.1", "layers.0", "patch_embed"},
    },
]

key_map = {
    "Linear(1024 → 8)  head": "head",
    "LayerNorm": "norm",
    "Stage 4  (layers.3, 2 blocks)": "layers.3",
    "Stage 3  (layers.2, 18 blocks)": "layers.2",
    "Stage 2  (layers.1, 2 blocks)": "layers.1",
    "Stage 1  (layers.0, 2 blocks)": "layers.0",
    "Patch embedding (patch_embed)": "patch_embed",
}

ax.text(6.5, 8.5,
    "Figure 3.3  Swin Transformer three-phase progressive unfreezing schedule",
    ha="center", va="center", fontsize=12, fontweight="bold")

W = 3.6
H = 0.55

for phase in phases:
    cx = phase["x_center"]
    ax.text(cx, 7.8, phase["title"],
            ha="center", fontsize=11, fontweight="bold")
    ax.text(cx, 7.45, phase["subtitle"],
            ha="center", fontsize=9, style="italic", color="#333")

    for label, y, base in LAYERS:
        key = key_map[label]
        is_trainable = key in phase["trainable"]
        color = C_TRAIN if is_trainable else (C_HEAD if base == C_HEAD and is_trainable else C_FROZEN)
        if base == C_HEAD and not is_trainable:
            color = C_FROZEN
        weight = "bold" if is_trainable else "normal"
        box(cx - W/2, y, W, H, label, color, fs=9, weight=weight)

legend_y = 1.4
box(2.0, legend_y, 0.5, 0.4, "", C_TRAIN); ax.text(2.65, legend_y + 0.2, "trainable", va="center", fontsize=10)
box(4.5, legend_y, 0.5, 0.4, "", C_FROZEN); ax.text(5.15, legend_y + 0.2, "frozen", va="center", fontsize=10)

ax.text(6.5, 0.5,
    "A fresh AdamW optimiser and CosineAnnealingLR are constructed at the start of each phase. "
    "Early-stopping is global across all three phases.\n"
    "Phase 3 additionally applies layer-wise LR decay: lr_d = base_lr · γ^d, where d = 0 for the head and "
    "d = 5 for patch_embed.",
    ha="center", va="center", fontsize=9, color="#333")

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {OUT}")
