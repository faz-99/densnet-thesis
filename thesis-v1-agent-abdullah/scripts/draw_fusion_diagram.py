"""
Generates Figure 3.1: feature-level fusion architecture block diagram.

Saves to figures/fusion_architecture_diagram.png.
"""
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import FIGURES_DIR

os.makedirs(FIGURES_DIR, exist_ok=True)
OUT_PATH = os.path.join(FIGURES_DIR, "fusion_architecture_diagram.png")

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis("off")

C_INPUT   = "#dbe7ff"
C_SWIN    = "#ffe0b3"
C_CONV    = "#cfe8cf"
C_FEAT    = "#f4e4f9"
C_NORM    = "#fff3b3"
C_MLP     = "#d9d9d9"
C_OUTPUT  = "#ffd0d0"
EDGE      = "#333333"
ARROW     = "#222222"

def box(x, y, w, h, text, color, fontsize=10, weight="normal"):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.4, edgecolor=EDGE, facecolor=color,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, fontweight=weight)

def arrow(x1, y1, x2, y2, label=None, label_offset=(0, 0.18), label_fs=9):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=14,
        linewidth=1.4, color=ARROW,
    )
    ax.add_patch(a)
    if label:
        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=label_fs, style="italic", color="#333")

box(0.2, 2.85, 1.6, 1.3, "Input image\nx\n(3 × 224 × 224)",
    C_INPUT, fontsize=10, weight="bold")

box(2.6, 4.5, 2.4, 1.2, "Swin-B\n(frozen)", C_SWIN, fontsize=11, weight="bold")
box(2.6, 1.3, 2.4, 1.2, "ConvNeXt-B\n(frozen)", C_CONV, fontsize=11, weight="bold")

box(5.6, 4.5, 1.6, 1.2, r"$f_{swin}$" + "\n∈ ℝ^1024", C_FEAT, fontsize=10)
box(5.6, 1.3, 1.6, 1.2, r"$f_{conv}$" + "\n∈ ℝ^1024", C_FEAT, fontsize=10)

box(7.7, 2.85, 1.6, 1.3, "Concat\n2048-D", C_FEAT, fontsize=10, weight="bold")

box(9.7, 2.85, 1.7, 1.3, "Standardise\n(train μ, σ)", C_NORM, fontsize=10)

box(11.8, 4.4, 2.0, 0.85, "Linear\n2048 → 512", C_MLP, fontsize=9)
box(11.8, 3.45, 2.0, 0.85, "ReLU + Dropout(0.3)", C_MLP, fontsize=9)
box(11.8, 2.5, 2.0, 0.85, "Linear\n512 → 8", C_MLP, fontsize=9)

box(11.8, 0.85, 2.0, 1.2,
    r"$\hat{y}$" + "\n8-class logits", C_OUTPUT, fontsize=10, weight="bold")

arrow(1.8, 4.0, 2.6, 5.1)
arrow(1.8, 3.0, 2.6, 1.9)

arrow(5.0, 5.1, 5.6, 5.1)
arrow(5.0, 1.9, 5.6, 1.9)

arrow(7.2, 5.1, 7.7, 4.0)
arrow(7.2, 1.9, 7.7, 3.0)

arrow(9.3, 3.5, 9.7, 3.5)

arrow(11.4, 3.5, 11.8, 4.8)

arrow(12.8, 4.4, 12.8, 4.3)
arrow(12.8, 3.45, 12.8, 3.35)

arrow(11.8, 2.92, 11.8, 2.05)

ax.text(7.0, 6.5, "Figure 3.1  Late-fusion architecture for breast histopathology classification",
        ha="center", va="center", fontsize=13, fontweight="bold")

ax.text(6.4, 0.3,
        "Backbones are first fine-tuned on the 8-class task, then frozen. "
        "Only the MLP head is trained on the standardised concatenated features.",
        ha="center", va="center", fontsize=9, style="italic", color="#444")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {OUT_PATH}")
