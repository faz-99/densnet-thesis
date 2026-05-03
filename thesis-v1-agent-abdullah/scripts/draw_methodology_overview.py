"""
Figure 3.1: end-to-end methodology pipeline overview.
Saves to figures/methodology_overview.png.
"""
import os, sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import FIGURES_DIR

OUT = os.path.join(FIGURES_DIR, "methodology_overview.png")

fig, ax = plt.subplots(figsize=(15, 6.5))
ax.set_xlim(0, 15); ax.set_ylim(0, 6.5); ax.axis("off")

C_DATA  = "#dbe7ff"
C_PREP  = "#fff3b3"
C_SWIN  = "#ffe0b3"
C_CONV  = "#cfe8cf"
C_FUSE  = "#f4e4f9"
C_EVAL  = "#ffd0d0"
C_XAI   = "#d9d9d9"
EDGE = "#333"

def box(x, y, w, h, text, color, fs=10, weight="normal"):
    p = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.4, edgecolor=EDGE, facecolor=color)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight)

def arrow(x1, y1, x2, y2):
    a = FancyArrowPatch((x1,y1),(x2,y2), arrowstyle="-|>",
        mutation_scale=14, linewidth=1.4, color="#222")
    ax.add_patch(a)

box(0.2, 2.7, 1.9, 1.3,
    "BreaKHis 400×\n1,693 images\n8 subtypes", C_DATA, fs=10, weight="bold")

box(2.5, 2.7, 2.0, 1.3,
    "Preprocessing\n• Resize 224\n• ImageNet norm\n• Augment (train)", C_PREP, fs=9)

box(4.9, 2.7, 1.9, 1.3,
    "Stratified split\n70 / 15 / 15\nseed = 42", C_PREP, fs=10)

box(7.2, 4.3, 2.4, 1.3,
    "Swin-B\nfine-tune\n(3-phase progressive)", C_SWIN, fs=10, weight="bold")
box(7.2, 1.0, 2.4, 1.3,
    "ConvNeXt-B\nfine-tune\n(discriminative LR)", C_CONV, fs=10, weight="bold")

box(10.0, 2.7, 2.1, 1.3,
    "Feature fusion\n[2048-D] → MLP\n(backbones frozen)", C_FUSE, fs=10, weight="bold")

box(12.5, 4.3, 2.3, 1.3,
    "Classification\nF1, AUC, ECE,\nMcNemar", C_EVAL, fs=9, weight="bold")
box(12.5, 1.0, 2.3, 1.3,
    "Explainability\nGrad-CAM++ / HiResCAM\nIG • Deletion-AUC", C_XAI, fs=9, weight="bold")

arrow(2.1, 3.35, 2.5, 3.35)
arrow(4.5, 3.35, 4.9, 3.35)
arrow(6.8, 3.6, 7.2, 4.7)
arrow(6.8, 3.1, 7.2, 1.7)
arrow(9.6, 4.9, 10.0, 3.6)
arrow(9.6, 1.7, 10.0, 3.1)
arrow(12.1, 3.6, 12.5, 4.7)
arrow(12.1, 3.1, 12.5, 1.7)

ax.text(7.5, 6.0,
    "Figure 3.1  Methodology pipeline: data preparation → backbone fine-tuning → fusion → evaluation",
    ha="center", va="center", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {OUT}")
