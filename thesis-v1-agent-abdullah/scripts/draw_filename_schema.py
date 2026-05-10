"""
Annotated diagram of the BreaKHis filename schema.

Filename layout:
    SOB_<B|M>_<subtype>-<year>-<PATIENT_ID>-400-<tile_index>.png

Concrete example used in the figure:
    SOB_M_DC-14-17614-400-025.png

Token roles:
  SOB           - "Slide-Of-Biopsy" prefix, common to every file in the corpus.
  <B|M>         - benign (B) vs malignant (M) tier.
  <subtype>     - 2-letter subtype code: A, F, PT, TA, DC, LC, MC, PC.
  <year>        - 2-digit year of acquisition.
  <PATIENT_ID>  - patient identifier; tiles sharing this token are
                  from the same biopsy. Used as the group key in
                  StratifiedGroupKFold for patient-disjoint CV.
  400           - magnification level (400X) - constant in this thesis.
  <tile_index>  - 3-digit tile index within the slide.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# (token_text, role_label, role_description, fill_colour)
TOKENS = [
    ("SOB",   "prefix",      "fixed corpus prefix",                    "#cfd8dc"),
    ("_",     None,          None,                                     None),
    ("M",     "tier",        "B = benign,  M = malignant",             "#ef9a9a"),
    ("_",     None,          None,                                     None),
    ("DC",    "subtype",     "A / F / PT / TA / DC / LC / MC / PC",    "#ef9a9a"),
    ("-",     None,          None,                                     None),
    ("14",    "year",        "2-digit acquisition year",               "#fff59d"),
    ("-",     None,          None,                                     None),
    ("17614", "PATIENT_ID",  "group key for patient-disjoint CV",      "#a5d6a7"),
    ("-",     None,          None,                                     None),
    ("400",   "magnification","fixed at 400X for this thesis",         "#cfd8dc"),
    ("-",     None,          None,                                     None),
    ("025",   "tile_index",  "tile index within the slide",            "#90caf9"),
    (".png",  "extension",   None,                                     "#eceff1"),
]

fig, ax = plt.subplots(figsize=(17, 6.0))
ax.set_xlim(0, 100)
ax.set_ylim(0, 56)
ax.axis("off")

# ------------------------------------------------------------------
# Title
# ------------------------------------------------------------------
ax.text(50, 53.5, "BreaKHis filename schema", ha="center", va="center",
        fontsize=16, fontweight="bold")
ax.text(50, 50, r"$\mathtt{SOB\_<B|M>\_<subtype>\!-\!<year>\!-\!<PATIENT\_ID>\!-\!400\!-\!<tile\_index>.png}$",
        ha="center", va="center", fontsize=11, color="#37474f")

# ------------------------------------------------------------------
# Token boxes
# ------------------------------------------------------------------
# Compute token widths proportional to text length
char_w = 1.55
gap = 0.4
total_w = sum(max(len(t[0]), 2) * char_w for t in TOKENS) + gap * (len(TOKENS) - 1)
x = (100 - total_w) / 2
box_y = 25
box_h = 6

token_centres = []
for tok, role, _, colour in TOKENS:
    w = max(len(tok), 2) * char_w
    if colour is None:
        ax.text(x + w / 2, box_y + box_h / 2, tok, ha="center", va="center",
                fontsize=14, family="monospace", color="#455a64")
    else:
        box = FancyBboxPatch((x, box_y), w, box_h,
                             boxstyle="round,pad=0.05,rounding_size=0.6",
                             linewidth=1.2, edgecolor="#37474f",
                             facecolor=colour)
        ax.add_patch(box)
        ax.text(x + w / 2, box_y + box_h / 2, tok, ha="center", va="center",
                fontsize=13, family="monospace", fontweight="bold")
    token_centres.append((x + w / 2, role))
    x += w + gap

# ------------------------------------------------------------------
# Annotation arrows + role labels (alternating above / below)
# ------------------------------------------------------------------
ANNOT = {
    "prefix":        ("fixed corpus prefix",                  "above", 41),
    "tier":          ("B = benign  /  M = malignant",         "below", 16),
    "subtype":       ("subtype code\n(A · F · PT · TA · DC · LC · MC · PC)",  "above", 44),
    "year":          ("2-digit acquisition year",             "below", 11),
    "PATIENT_ID":    ("group key for patient-disjoint CV\n(StratifiedGroupKFold)", "above", 41),
    "magnification": ("fixed at 400X (this thesis)",          "below", 16),
    "tile_index":    ("tile index within slide",              "above", 44),
}

for (cx, role), (tok, *_rest) in zip(token_centres, TOKENS):
    if role not in ANNOT:
        continue
    text, side, ty = ANNOT[role]
    if side == "above":
        ax.annotate("", xy=(cx, box_y + box_h + 0.2), xytext=(cx, ty - 1.0),
                    arrowprops=dict(arrowstyle="-", lw=0.9, color="#546e7a"))
        ax.text(cx, ty, role, ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#1a237e")
        ax.text(cx, ty - 1.6, text, ha="center", va="top",
                fontsize=8.5, color="#37474f")
    else:
        ax.annotate("", xy=(cx, box_y - 0.2), xytext=(cx, ty + 1.0),
                    arrowprops=dict(arrowstyle="-", lw=0.9, color="#546e7a"))
        ax.text(cx, ty, role, ha="center", va="top",
                fontsize=10, fontweight="bold", color="#1a237e")
        ax.text(cx, ty - 1.6, text, ha="center", va="top",
                fontsize=8.5, color="#37474f")

# ------------------------------------------------------------------
# Footer: patient-ID extraction one-liner
# ------------------------------------------------------------------
ax.text(50, 4.5,
        r"$\mathtt{patient\_id\ =\ filename.split('-')[2]}$"
        "    →    used as the group key in `StratifiedGroupKFold` for the patient-CV protocol",
        ha="center", va="center", fontsize=10.5, color="#1b5e20",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f5e9",
                  edgecolor="#66bb6a", linewidth=1.0))

out = os.path.join(FIG_DIR, "filename_schema.png")
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"wrote {out}")
