"""Figure 3.4 — late-fusion block diagram for both final models.

Top panel:    Feature-ensemble baseline (single-head MLP).
Bottom panel: Two-head v3.6 (parallel experts + learned gate).

Both panels share the same frozen backbones and concatenated 2048-D feature
vector; they differ only in the head architecture, which is the contribution
of this thesis.
"""
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
C_MLP    = "#e6e6e6"
C_PANEL_TOP = "#fafafa"
C_PANEL_BOT = "#f3f7ff"
EDGE     = "#222222"

fig, ax = plt.subplots(figsize=(16, 11))
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.axis("off")


def box(x, y, w, h, text, color, fs=10, weight="normal"):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.12",
                       linewidth=1.4, edgecolor=EDGE, facecolor=color)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight)


def panel(x, y, w, h, color):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.18",
                       linewidth=1.0, edgecolor="#bbb", facecolor=color)
    ax.add_patch(p)


def arrow(x1, y1, x2, y2, color="#222"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>", mutation_scale=14,
                        linewidth=1.3, color=color)
    ax.add_patch(a)


# ------------------------------------------------------------------
# Title
# ------------------------------------------------------------------
ax.text(8.0, 10.55,
        "Figure 3.4   Late-fusion block diagram for the two final models",
        ha="center", va="center", fontsize=14, fontweight="bold")
ax.text(8.0, 10.15,
        "Both models share the same frozen ConvNeXt-B + Swin-B backbones; they differ only in the head over the concatenated 2048-D feature vector.",
        ha="center", va="center", fontsize=10, style="italic", color="#444")

# ------------------------------------------------------------------
# TOP PANEL — Feature-ensemble baseline
# ------------------------------------------------------------------
panel(0.15, 5.45, 15.7, 4.45, C_PANEL_TOP)
ax.text(0.45, 9.55, "(a)  Feature-ensemble baseline   —   single-head MLP, 562 K trainable params",
        ha="left", va="center", fontsize=12, fontweight="bold", color="#1a3a5e")

# Input
box(0.6, 7.1, 1.3, 1.0, "Input\n3×224×224", C_INPUT, fs=9.5, weight="bold")

# Backbones
box(2.6, 8.1, 1.7, 0.95, "Swin-B\n(frozen)", C_SWIN, fs=10, weight="bold")
box(2.6, 6.15, 1.7, 0.95, "ConvNeXt-B\n(frozen)", C_CONV, fs=10, weight="bold")

# Features
box(4.9, 8.1, 1.4, 0.95, r"$f_{swin}$" + "\n∈ ℝ¹⁰²⁴", C_FEAT, fs=9.5)
box(4.9, 6.15, 1.4, 0.95, r"$f_{cnx}$" + "\n∈ ℝ¹⁰²⁴", C_FEAT, fs=9.5)

# Concat
box(6.9, 7.1, 1.4, 1.0, "Concat\nx ∈ ℝ²⁰⁴⁸", C_FEAT, fs=10, weight="bold")

# MLP layers
box(8.9, 7.6, 1.6, 0.7, "Linear\n2048 → 256", C_MLP, fs=9)
box(8.9, 6.85, 1.6, 0.7, "ReLU + Dropout(0.3)", C_MLP, fs=9)
box(8.9, 6.10, 1.6, 0.7, "Linear\n256 → 128", C_MLP, fs=9)

box(11.0, 7.6, 1.6, 0.7, "ReLU + Dropout(0.3)", C_MLP, fs=9)
box(11.0, 6.85, 1.6, 0.7, "Linear\n128 → 8", C_MLP, fs=9)

# Output
box(13.2, 7.1, 2.4, 1.0, r"$\hat{y}_{8c}$" + "\n8-class logits", C_OUT, fs=10, weight="bold")

# Arrows top
arrow(1.9, 7.6, 2.6, 8.55)
arrow(1.9, 7.6, 2.6, 6.6)
arrow(4.3, 8.55, 4.9, 8.55)
arrow(4.3, 6.6, 4.9, 6.6)
arrow(6.3, 8.55, 6.9, 7.85)
arrow(6.3, 6.6, 6.9, 7.35)
arrow(8.3, 7.6, 8.9, 7.95)
arrow(10.5, 7.95, 11.0, 7.95)
arrow(11.8, 6.85, 11.8, 6.05)
arrow(12.6, 7.2, 13.2, 7.55)

# Loss / training note
ax.text(8.0, 5.7,
        "Loss: cross-entropy + label smoothing (ε = 0.1)        Selection: max val 8-class macro-F1",
        ha="center", va="center", fontsize=9, style="italic", color="#444")


# ------------------------------------------------------------------
# BOTTOM PANEL — Two-head v3.6
# ------------------------------------------------------------------
panel(0.15, 0.20, 15.7, 5.05, C_PANEL_BOT)
ax.text(0.45, 4.95, "(b)  Two-head v3.6   —   parallel experts + learned gate, 4.27 M trainable params (≈ 200 K active per image)",
        ha="left", va="center", fontsize=12, fontweight="bold", color="#0b3a64")

# Input + backbones (compressed)
box(0.6, 2.3, 1.3, 1.0, "Input\n3×224×224", C_INPUT, fs=9.5, weight="bold")
box(2.6, 3.4, 1.7, 0.85, "Swin-B (frozen)", C_SWIN, fs=10, weight="bold")
box(2.6, 1.4, 1.7, 0.85, "ConvNeXt-B (frozen)", C_CONV, fs=10, weight="bold")
box(4.9, 3.4, 1.4, 0.85, r"$f_{swin}$" + " ∈ ℝ¹⁰²⁴", C_FEAT, fs=9.5)
box(4.9, 1.4, 1.4, 0.85, r"$f_{cnx}$" + " ∈ ℝ¹⁰²⁴", C_FEAT, fs=9.5)
box(6.9, 2.4, 1.4, 0.85, "Concat\nx ∈ ℝ²⁰⁴⁸", C_FEAT, fs=10, weight="bold")

arrow(1.9, 2.8, 2.6, 3.8)
arrow(1.9, 2.8, 2.6, 1.8)
arrow(4.3, 3.8, 4.9, 3.8)
arrow(4.3, 1.8, 4.9, 1.8)
arrow(6.3, 3.8, 6.9, 3.0)
arrow(6.3, 1.8, 6.9, 2.6)

# Three parallel branches from x: binary expert (top), gate (mid), 8c expert (bottom)
box(8.9, 4.05, 2.4, 0.95,
    r"Expert$_{bin}$" + "\n2048→256→128→1   (BCE)",
    C_BIN, fs=9.5, weight="bold")

box(8.9, 2.40, 2.4, 0.85,
    "Gate\n2048→128→2   (softmax)",
    C_GATE, fs=9.5, weight="bold")

box(8.9, 0.65, 2.4, 0.95,
    r"Expert$_{8c}$" + "\n2048→256→128→8   (LA-CE + LS)",
    C_8C, fs=9.5, weight="bold")

# Convex combination
box(12.0, 2.40, 2.6, 0.85,
    r"$\hat{y}_{final} = w_{bin}\hat{y}_{bin}+w_{8c}\hat{y}_{8c}$",
    C_OUT, fs=9.5, weight="bold")

# Final outputs
box(12.0, 4.05, 2.6, 0.95, "P(malignant)\nfrom Expert$_{bin}$", C_BIN, fs=9.5, weight="bold")
box(12.0, 0.70, 2.6, 0.95, "P(subtype) ∈ Δ⁷\nfrom $\\hat{y}_{final}$", C_8C, fs=9.5, weight="bold")

# Arrows: concat -> three branches
arrow(8.3, 3.0, 8.9, 4.5)   # to binary expert
arrow(8.3, 2.8, 8.9, 2.85)  # to gate
arrow(8.3, 2.6, 8.9, 1.1)   # to 8c expert

# Branches into combination
arrow(11.3, 4.5, 12.0, 3.05)
arrow(11.3, 2.82, 12.0, 2.82)
arrow(11.3, 1.1, 12.0, 2.6)

# Combination -> final outputs
arrow(13.3, 3.25, 13.3, 4.05)
arrow(13.3, 2.40, 13.3, 1.65)

# Gate weight annotation
ax.text(11.65, 2.05, "(w_bin, w_8c)",
        ha="center", va="center", fontsize=8, color="#5a4a00", style="italic")

# Loss / training note for v3.6
ax.text(8.0, 0.30,
        "Loss: BCE(binary) + λ·LA-CE(8-class, ε = 0.05)        Sampler: WeightedRandom 1/√freq        "
        "Selection: 0.3·val_binF1 + 0.7·val_macroF1, gated > 0.970",
        ha="center", va="center", fontsize=9, style="italic", color="#444")

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {OUT}")
