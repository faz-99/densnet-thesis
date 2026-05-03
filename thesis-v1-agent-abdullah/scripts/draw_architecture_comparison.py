"""
Figure 3.2: side-by-side block-stack comparison of ConvNeXt-Base and Swin-Base.
Saves to figures/architecture_comparison.png.
"""
import os, sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import FIGURES_DIR

OUT = os.path.join(FIGURES_DIR, "architecture_comparison.png")

fig, ax = plt.subplots(figsize=(13, 9.5))
ax.set_xlim(0, 13); ax.set_ylim(0, 10); ax.axis("off")

EDGE = "#333"
C_INPUT = "#dbe7ff"
C_STEM  = "#fff3b3"
C_S1    = "#cde6ff"
C_S2    = "#9fcfff"
C_S3    = "#6fb6ff"
C_S4    = "#3f9bff"
C_HEAD  = "#ffd0d0"

def box(x, y, w, h, text, color, fs=9, weight="normal"):
    p = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.2, edgecolor=EDGE, facecolor=color)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight)

def arrow(x1, y1, x2, y2):
    a = FancyArrowPatch((x1,y1),(x2,y2), arrowstyle="-|>",
        mutation_scale=10, linewidth=1.0, color="#444")
    ax.add_patch(a)

ax.text(6.5, 9.7,
    "Figure 3.2  ConvNeXt-Base vs Swin Transformer-Base — capacity-matched architectures",
    ha="center", va="center", fontsize=12, fontweight="bold")

ax.text(3.0, 9.1, "ConvNeXt-Base  (87.6 M params)",
        ha="center", fontsize=11, fontweight="bold", color="#1a4d8a")
ax.text(10.0, 9.1, "Swin Transformer-Base  (86.8 M params)",
        ha="center", fontsize=11, fontweight="bold", color="#a04030")

cx = 3.0  # ConvNeXt center x
sx = 10.0 # Swin center x
W = 4.0
y = 8.4

def stage_row(cx, y, label_left, color, h=0.55, fs=9, weight="normal"):
    box(cx - W/2, y - h, W, h, label_left, color, fs=fs, weight=weight)

stage_row(cx, y, "Input  (3 × 224 × 224)", C_INPUT, fs=10, weight="bold")
stage_row(sx, y, "Input  (3 × 224 × 224)", C_INPUT, fs=10, weight="bold")
y -= 0.8

stage_row(cx, y, "Patchify stem  (4×4 conv, stride 4)  →  56×56×128", C_STEM, fs=9)
stage_row(sx, y, "Patch embedding  (4×4 patches)  →  56×56 tokens, 128-D", C_STEM, fs=9)
y -= 0.7

stage_row(cx, y, "Stage 1  •  3 ConvNeXt blocks  •  56×56×128", C_S1, fs=9)
stage_row(sx, y, "Stage 1  •  2 Swin blocks  •  56×56, 128-D", C_S1, fs=9)
y -= 0.7

stage_row(cx, y, "Downsample  (2×2 conv, stride 2)", C_STEM, fs=8)
stage_row(sx, y, "Patch merging  (2×2 → 1)", C_STEM, fs=8)
y -= 0.7

stage_row(cx, y, "Stage 2  •  3 ConvNeXt blocks  •  28×28×256", C_S2, fs=9)
stage_row(sx, y, "Stage 2  •  2 Swin blocks  •  28×28, 256-D", C_S2, fs=9)
y -= 0.7

stage_row(cx, y, "Downsample", C_STEM, fs=8)
stage_row(sx, y, "Patch merging", C_STEM, fs=8)
y -= 0.7

stage_row(cx, y, "Stage 3  •  27 ConvNeXt blocks  •  14×14×512  ★", C_S3, fs=9, weight="bold")
stage_row(sx, y, "Stage 3  •  18 Swin blocks  •  14×14, 512-D  ★", C_S3, fs=9, weight="bold")
y -= 0.7

stage_row(cx, y, "Downsample", C_STEM, fs=8)
stage_row(sx, y, "Patch merging", C_STEM, fs=8)
y -= 0.7

stage_row(cx, y, "Stage 4  •  3 ConvNeXt blocks  •  7×7×1024", C_S4, fs=9)
stage_row(sx, y, "Stage 4  •  2 Swin blocks  •  7×7, 1024-D", C_S4, fs=9)
y -= 0.7

stage_row(cx, y, "GAP → LayerNorm → Linear(1024 → 8)", C_HEAD, fs=9, weight="bold")
stage_row(sx, y, "Final LN → GAP → Linear(1024 → 8)", C_HEAD, fs=9, weight="bold")
y -= 0.6

ax.text(cx, y - 0.1,
        "Block: depthwise 7×7 conv  •  inverted bottleneck  •  GELU  •  LayerNorm",
        ha="center", fontsize=8.5, style="italic", color="#222")
ax.text(sx, y - 0.1,
        "Block: shifted-window self-attention (7×7, shift (3,3))  •  MLP  •  LayerNorm",
        ha="center", fontsize=8.5, style="italic", color="#222")

ax.text(6.5, 0.7,
        "★  Stage 3 holds the bulk of the computation in both architectures.\n"
        "Channel widths (128 → 256 → 512 → 1024) and resolutions are matched, isolating "
        "the architectural inductive bias.",
        ha="center", va="center", fontsize=9.5, color="#333")

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {OUT}")
