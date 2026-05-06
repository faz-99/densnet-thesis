"""Figure 3.4 — two-head v3.6 fusion architecture (2 experts + learned gate)."""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "figures", "fusion_architecture_diagram.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

C_INPUT  = "#dbe7ff"
C_SWIN   = "#ffe0b3"
C_CONV   = "#cfe8cf"
C_FEAT   = "#f4e4f9"
C_BIN    = "#ffd0d0"
C_8C     = "#d0e8ff"
C_GATE   = "#fff3b3"
C_OUT    = "#ffe6cc"
EDGE     = "#222222"

fig, ax = plt.subplots(figsize=(15, 8.5))
ax.set_xlim(0, 15)
ax.set_ylim(0, 9)
ax.axis("off")

def box(x, y, w, h, text, color, fs=10, weight="normal"):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.12",
                       linewidth=1.4, edgecolor=EDGE, facecolor=color)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight)

def arrow(x1, y1, x2, y2, color="#222"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>", mutation_scale=14,
                        linewidth=1.3, color=color)
    ax.add_patch(a)

# Input
box(0.2, 4.0, 1.5, 1.2, "Input\n3×224×224", C_INPUT, fs=10, weight="bold")

# Backbones
box(2.2, 6.0, 1.9, 1.1, "Swin-B\n(frozen)", C_SWIN, fs=11, weight="bold")
box(2.2, 2.3, 1.9, 1.1, "ConvNeXt-B\n(frozen)", C_CONV, fs=11, weight="bold")

# Features
box(4.6, 6.05, 1.5, 1.0, r"$f_{swin}$"+"\n∈ ℝ^1024", C_FEAT, fs=10)
box(4.6, 2.35, 1.5, 1.0, r"$f_{cnx}$"+"\n∈ ℝ^1024", C_FEAT, fs=10)

# Concat
box(6.6, 4.1, 1.5, 1.0, "Concat\nx ∈ ℝ²⁰⁴⁸", C_FEAT, fs=10, weight="bold")

# Branches: binary expert (top), gate (middle), 8-class expert (bottom)
# All take x as input

# Binary expert
box(8.7, 6.7, 2.6, 1.6,
    "Expert$_{bin}$\n2048 → 256 → 128 → 1\nBCE",
    C_BIN, fs=10, weight="bold")

# Gate
box(8.7, 4.1, 2.6, 1.0,
    "Gate\n2048 → 128 → 2 (softmax)",
    C_GATE, fs=10, weight="bold")

# 8-class expert
box(8.7, 1.4, 2.6, 1.6,
    "Expert$_{8c}$\n2048 → 256 → 128 → 8\nLA-CE + LS",
    C_8C, fs=10, weight="bold")

# Combination
box(12.1, 4.1, 2.5, 1.0,
    r"$\hat{y}_{final} = w_{bin}·\hat{y}_{bin} + w_{8c}·\hat{y}_{8c}$",
    C_OUT, fs=10, weight="bold")

# Final output
box(12.1, 6.5, 2.5, 0.9, "P(malignant)", C_BIN, fs=10, weight="bold")
box(12.1, 1.7, 2.5, 0.9, "P(8-subtype)", C_8C, fs=10, weight="bold")

# Arrows: input -> backbones
arrow(1.7, 4.7, 2.2, 6.55)
arrow(1.7, 4.5, 2.2, 2.85)
# backbones -> features
arrow(4.1, 6.55, 4.6, 6.55)
arrow(4.1, 2.85, 4.6, 2.85)
# features -> concat
arrow(6.1, 6.55, 6.6, 4.85)
arrow(6.1, 2.85, 6.6, 4.35)
# concat -> three branches
arrow(8.1, 4.85, 8.7, 7.4)   # to binary expert
arrow(8.1, 4.6, 8.7, 4.6)    # to gate
arrow(8.1, 4.35, 8.7, 2.2)   # to 8-class expert
# experts + gate -> combination
arrow(11.3, 7.5, 12.1, 5.0)
arrow(11.3, 4.6, 12.1, 4.6)
arrow(11.3, 2.2, 12.1, 4.2)
# combination -> outputs
arrow(13.35, 5.1, 13.35, 6.5)
arrow(13.35, 4.1, 13.35, 2.6)

# Title
ax.text(7.5, 8.55,
        "Figure 3.4  Two-head v3.6 fusion: two experts + learned gate over frozen backbone features",
        ha="center", va="center", fontsize=13, fontweight="bold")

# Param annotation
ax.text(7.5, 0.25,
        "4.27 M trainable params  |  ~131 K active when w_bin → 1, ~263 K when w_8c → 1  |  20× smaller than fine-tuning Swin-B (88 M)",
        ha="center", va="center", fontsize=9.5, style="italic", color="#333")

# Loss annotation
ax.text(10.0, 4.85, "softmax\n(w_bin, w_8c)", ha="center", va="center",
        fontsize=8, color="#5a4a00", style="italic")

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {OUT}")
