# PROJECT RECAP — v3.6 Thesis (BreakHis, ConvNeXt + Swin + 8-expert gate)

> Single source of truth so we don't re-explain context every session.
> Last updated: 2026-06-10. All numbers below were read off disk / notebook outputs and are marked **VERIFIED** or **⚠️ UNVERIFIED**.

---

## 0. Scope (LOCKED)

- This thesis is **YOUR v3.6 only**: frozen **ConvNeXt + Swin** features → fusion → **8-expert gate**, on **BreakHis** (8-class).
- **NOT in scope** (do not mix in): DINO, SAM, MedGemma report generation, Llama comparison, or any friend's-thesis material.
- **Honesty rule:** real, computed numbers only. Never fabricate. An earlier pasted "plan" from another thread contained **fabricated** numbers (IoU=0.68, Deletion-AUC 0.445/0.421/0.381, Deferral-AUC=0.963, p=0.023, "47/28 wins") that **contradict the real data** — never reuse them.

---

## 1. Terminology / artifact map (these names collide — be careful)

| Name in repo | What it is | Params | Protocol | 8-class macro-F1 |
|---|---|---|---|---|
| **`feature_ensemble`** = **"our fusion model"** | concat Swin(1024)+ConvNeXt(1024) → MLP 2048→256→128→8, **NO gate** | 562K | patient-CV (5 folds) | 0.8211 ± 0.1599 |
| **`v3.6`** (two-head) | same fusion + **8-expert gate** + SE + EMA + logit-adjusted loss | ≈4.27M | patient-CV (5 folds) | **0.8352 ± 0.0876** ← headline |
| Binary-opt fusion (Variant A) | `fusion_mlp_binary_cv_patient`, 1.05M | 1.05M | patient-CV | 0.8208 |
| Macro-opt fusion (Variant B) | `fusion_mlp_macro` | 1.05M | **single-split ONLY** | 0.8434 (single-split, optimistic) |

**Traps:**
- Notebook **cell 13** labels the 1.05M Variant A/B as "Feature Ens." too — *different* artifact from the 562K `feature_ensemble`. Cell 13 is a **single train/val split** (X_val=253), optimistically biased — **never compare its numbers to patient-CV numbers.**
- **Pooled** macro-F1 (0.863 / 0.892) ≠ **mean-of-folds** macro-F1 (0.8211 / 0.8352). Both valid; **label which aggregation** and never mix them in one column.

---

## 2. The contribution — stability + rare classes, NOT mean F1

| Measure | feature_ensemble | v3.6 | Gain | Status |
|---|---|---|---|---|
| Patient-CV macro-F1 **mean** | 0.8211 | 0.8352 | +0.0141 (+1.41pp) | VERIFIED |
| Patient-CV macro-F1 **std** | 0.1599 | 0.0876 | −0.0723 (**45% less variance**) | VERIFIED |
| Pooled macro-F1 | 0.863 | 0.892 | +0.029 (+2.9pp) | VERIFIED |
| Pooled macro-**recall** | 0.837 | 0.909 | **+0.072 (+7.2pp)** | VERIFIED |
| ECE (calibration) | see ⚠️ below | 0.0605 | — | ⚠️ ownership unclear |

**One-line verdict:** the mean F1 gain is small; the real win is **~2× more stable across patients** and **+7.2pp macro-recall**, concentrated in rare malignant subtypes.

---

## 3. Subclass table (pooled patient-CV, N=1693, predictions in identical order — VERIFIED)

| Subtype | n | FE  P / R / F1 | v3.6  P / R / F1 | ΔF1 |
|---|---|---|---|---|
| adenosis | 96 | 0.989 / 0.958 / 0.974 | 0.958 / 0.958 / 0.958 | −0.015 |
| ductal_carcinoma | 734 | 0.837 / 0.935 / 0.883 | 0.931 / 0.866 / 0.898 | +0.015 |
| fibroadenoma | 223 | 0.938 / 0.888 / 0.912 | 0.940 / 0.915 / 0.927 | +0.015 |
| **lobular_carcinoma** | 128 | 0.743 / 0.633 / 0.684 | 0.699 / **0.945** / **0.804** | **+0.120** |
| mucinous_carcinoma | 159 | 0.948 / 0.912 / 0.929 | 0.909 / 0.937 / 0.923 | −0.007 |
| **papillary_carcinoma** | 125 | 0.918 / 0.720 / 0.807 | 0.844 / **0.912** / **0.877** | **+0.070** |
| phyllodes_tumor | 106 | 0.839 / 0.689 / 0.756 | 0.828 / 0.774 / 0.800 | +0.044 |
| tubular_adenoma | 122 | 0.959 / 0.959 / 0.959 | 0.937 / 0.967 / 0.952 | −0.007 |
| **Macro avg** | 1693 | 0.896 / 0.837 / 0.863 | 0.881 / **0.909** / 0.892 | +0.029 |
| **Accuracy** | | 0.875 | 0.895 | |

**Where the +7.2pp recall comes from** — 3 rare malignant classes (359 samples = 21% of data):
- lobular_carcinoma recall **+31.2pp** (0.633→0.945), F1 +0.120
- papillary_carcinoma recall **+19.2pp** (0.720→0.912), F1 +0.070
- phyllodes_tumor recall +8.5pp (0.689→0.774), F1 +0.044

**IMPORTANT correction:** the gain is **NOT on adenosis** (already solved by plain fusion at F1 0.974; v3.6 marginally lower 0.958). Do not claim "the gate fixed adenosis."

---

## 4. The cost — ductal carcinoma tradeoff (and how to defend it)

| Subtype | n | FE R | v3.6 R | Loss | FE F1 | v3.6 F1 | ΔF1 |
|---|---|---|---|---|---|---|---|
| ductal_carcinoma | 734 (43% of data) | 0.935 | 0.866 | −6.9pp | 0.883 | 0.898 | **+0.015** |

The gate reallocates capacity away from the over-represented majority (ductal) toward rare classes. **Ductal F1 still rises** (0.883→0.898) because precision jumps 0.837→0.931 — it trades recall for precision on the majority class.

**Viva defense — "you made the majority class worse, why is that good?":**
1. **Not worse overall:** ductal F1 improved (recall down, precision up — net positive).
2. **Clinical priority:** missing lobular (infiltrative, easily missed) is worse than missing ductal (obvious). +31pp lobular for −7pp ductal recall is a good trade.
3. **By design + controllable:** the gate targets class imbalance; ductal recall is recoverable via the logit-adjustment bias or abstention threshold. Plain fusion gives no such control.
4. **Predictability:** v3.6 ductal recall 0.866 ± low-var beats FE's high-variance behavior (±0.16 macro std) — better for deployment.

---

## 5. Thesis contribution paragraph (drop-in, numbers VERIFIED except ECE)

> **Contribution:** We propose an 8-expert gated fusion architecture for BreakHis 8-class histopathology. Compared to plain feature-level fusion, the gate reduces cross-patient variance by 45% (std 0.160→0.088) and improves macro-recall by +7.2pp (0.837→0.909). The gain is concentrated in under-represented malignant subtypes: lobular carcinoma recall +31.2pp (0.633→0.945) and papillary carcinoma recall +19.2pp (0.720→0.912). This comes from explicit expert routing that reallocates capacity from the majority ductal class (−6.9pp recall, but F1 still +1.5pp via a precision gain) to rare classes.

(Add the ECE/"first calibrated" sentence only after resolving §6 ⚠️.)

---

## 6. Verification status of every claimed number

**VERIFIED (read off disk / notebook):**
- All §2–§4 F1 / recall / precision / std / accuracy numbers.
- Predictions of feature_ensemble and v3.6 are in the **same sample order** → a v3.6-vs-feature_ensemble McNemar is **valid**.
- Binary ROC-AUC (patient-CV): v3.6 0.9901±0.0057, ConvNeXt-only 0.9920 (higher!), Swin 0.9852. **Differences are noise (saturated easy axis) — do NOT lead the viva with AUC.**
- Deletion-AUC (`tables/deletion_auc_stats.tex`): IG ConvNeXt 0.130 < Fusion 0.178 < Swin 0.409 (ConvNeXt-only best; fusion NOT best).
- **McNemar v3.6 vs feature_ensemble ALREADY DONE** (`tables/mcnemar_results.tex`, Table 3.4 / 5.4): discordant b+c=130, **b=101 (v3.6 wins), c=29 (FE wins), p<10⁻⁷**. v3.6 significantly beats the fusion model. (The p=8.6e-5 in `stat_tests_patient_cv` was the v3.6-vs-binary-opt pair, b+c=75.)
- **Gating ablation ALREADY DONE** (`tables/gating_ablation.txt`, Table 3.2, patient-CV 8-class macro-F1): No-gate/feature_ensemble 0.8211 · **Hard gate (argmax) 0.7959** · **Soft average (fixed w) 0.8129** · **Learned gate (v3.6) 0.8352** — all 4.27M params except no-gate. KEY PROOF: at *equal* 4.27M params, learned gate (0.8352) > soft-average (0.8129) > hard gate (0.7959). The gate's value is NOT just extra params.

**⚠️ UNVERIFIED / needs resolving before use:**
- **ECE = 0.0605** is real (`results/table_4_1.json`, netcal/15-bin, after **per-class temperature scaling**) BUT the source CSV labels that row **"Feature Ensemble"**, while thesis-writing scripts attribute 0.0605 to **"v3.6"**. **CONFIRM which model owns ECE=0.0605 before claiming "v3.6 is calibrated."** Note it is a **binary** calibration metric (table 4.1 is BA/Sens/Spec), likely single-split — not 8-class patient-CV calibration.
- "First/only calibrated BreakHis model" — soften to the defensible version your own scripts use: **"none of the cited 2022+ BreakHis work reports ECE."** Do not claim "first ever."

**MISSING (must run to produce):**
- `v36_gate_weights_1693x8.npy` — **does not exist**; blocks all gate analysis. Produced by cell **2.5.18**.
- 8-class / multiclass ROC-AUC — not computed (only binary AUC + argmax cached). Comes from 2.5.18's saved (N,8) probs.

---

## 7. What still needs running (to make it bulletproof)

**Already done (found on disk — do NOT re-run):** gating ablation (Table 3.2) and McNemar v3.6-vs-feature_ensemble (Table 3.4/5.4). The "is it just the params?" and "is the gap significant?" questions are already answered.

**Genuinely still missing — needs cell 2.5.18 → 2.5.19 on the ROCm GPU:**
1. Per-sample 8-way gate softmax weights (`v36_gate_weights_1693x8.npy`) — for gate **entropy, per-class routing, deferral/abstain curve** (these are the only gate analyses NOT yet produced; the F1-level gate proof already exists as Table 3.2).
2. 8-class / multiclass ROC-AUC (needs the (N,8) probs 2.5.18 saves).

---

## 8. Environment

- Python with numpy for quick analysis: **`/home/abdullah/venvs/torch-rocm/bin/python`** (no sklearn installed — per-class metrics computed manually).
- ROCm: AMD RX 6800 XT, `HSA_OVERRIDE_GFX_VERSION=10.3.0`.
- Notebook backup before edits: `03_training.ipynb.bak_20260610_023159`.

---

**Bottom line:** the contribution is **stable, rare-class-sensitive fusion**. The ductal recall loss is the documented, intentional, controllable cost of the gate's capacity reallocation. Resolve the ECE ownership (§6) and run cell 2.5.18 for the gate-behavior evidence (§7) and the methodology is defensible.

---

## 9. Methodology section — what to write, add, correct (with real artifacts)

### 9.1 The narrative arc (write the methodology as a 3-stage "why" story, not a model list)

**Stage A — Single backbones (the problem).** Train frozen Swin-B and ConvNeXt-B with linear heads.
- *Why:* establish that even strong modern CNN/Transformer backbones hit a ceiling on the hard task (8-class, patient-level CV): Swin 0.747, ConvNeXt 0.816 macro-F1, with high variance and weak rare-class recall.
- *What it exposes:* large image-CV → patient-CV generalisation gap (memorising patients), and poor recall on rare malignant subtypes.
- *Tables/figs:* `architecture_comparison.png`, `confusion_8class_per_model.png`, `learning_curves_*.png`, `model_parameter_comparison.png`. Numbers: Ablation Table 1 top rows.

**Stage B — Simple ensembles (partial fix).** Logit ensemble (weighted logit avg) and Feature ensemble (concat 2048-d → MLP).
- *Why:* CNN and Transformer are **complementary** (different inductive biases) → combining should help.
- *What it fixes:* feature fusion lifts image-CV F1 to 0.938; but patient-CV stays unstable (0.8211 ± **0.1599**) and rare-class recall still poor (lobular 0.633).
- *Proof of complementarity:* `backbone_attribution_comparison.png`, `fig_4_1_spatial_complementarity.png`. *Efficiency:* `params_vs_f1.png`. *Ensemble tuning:* `ensemble_*_weight_sweep.png`.

**Stage C — v3.6 gated two-head (the fix).** Add an 8-expert gate + SE + EMA + logit-adjusted CE + per-class temperature scaling on top of fusion.
- *Why each piece, what it fixes:*
  | Component | Problem it targets | Proof artifact |
  |---|---|---|
  | **8-expert learned gate** | rare-class recall + capacity allocation | **Table 3.2 `gating_ablation`** (learned 0.8352 > soft-avg 0.8129 > hard 0.7959 at equal 4.27M params), subclass table |
  | **Squeeze-Excitation (SE)** | channel feature recalibration | `architecture_ablation_full.tex` |
  | **EMA weights** | training/fold stability | `regularisation_ablation.tex` |
  | **Logit-adjusted CE** | class imbalance (43% ductal) | `loss_sampler_ablation.tex` |
  | **Per-class temp. scaling** | calibration / trustworthy confidence | `reliability_diagram_v36.png`, `tta_calibration.png` |
- *Headline proofs:* stability std 0.160→0.088 (`perfold_macro_f1_v36_vs_featens.png`, `bootstrap_ci.png`); significance `mcnemar_results.tex` (v3.6 vs feature_ensemble p<10⁻⁷); rare-class gains (subclass table, `cv_per_class_f1_v36.png`).

### 9.2 What to ADD to methodology
1. **Gating ablation paragraph (Table 3.2)** — the single strongest argument: *at equal 4.27M parameters*, learned routing (0.8352) beats fixed soft-average (0.8129) and hard argmax routing (0.7959). This pre-empts "it's just more params."
2. **Side-by-side per-class table (feature_ensemble vs v3.6)** — currently only v3.6's per-class table is saved; add the FE column (see §3) so the lobular/papillary recall story is explicit.
3. **Stability as a first-class result** — state the 45% variance reduction (0.160→0.088) as a contribution, not a footnote.
4. **Calibration method sentence** — "per-class temperature scaling (LBFGS on val NLL), netcal, 15 bins" (`tables/hyperparameters.tex`).
5. **Capacity-reallocation explanation** for the ductal tradeoff (recall −6.9pp but F1 +1.5pp via precision) — frame as intentional design, controllable via logit-adjustment bias / abstention threshold.

### 9.3 What to CORRECT
1. **Adenosis is NOT where the gate helps** — fix any text claiming so. Gains are lobular + papillary + phyllodes.
2. **Don't lead with AUC** — binary ROC-AUC is saturated (~0.99 for everyone; ConvNeXt-only 0.992 actually > v3.6 0.990). Lead with 8-class macro-F1; present binary AUC only as "easy axis already solved."
3. **ECE ownership (§6)** — confirm whether 0.0605 is feature_ensemble or v3.6 before claiming "v3.6 is calibrated." Source CSV says "Feature Ensemble"; prose says v3.6.
4. **Soften the novelty claim** — "no cited 2022+ BreakHis work reports ECE," not "first calibrated model."
5. **Never mix pooled (0.863/0.892) with fold-mean (0.8211/0.8352)** in one table; label aggregation.

### 9.4 Tables to cite
- Methodology/ablation: `gating_ablation` (3.2), `architecture_ablation_full`, `loss_sampler_ablation`, `regularisation_ablation`, `finetune_ablation_detailed`, `hyperparameters`.
- Results: Ablation Table 1, `patient_cv_full_metrics`, `per_fold_breakdown`, `mcnemar_results`/`mcnemar_detailed`, `sota_patient_cv_comparison`, `deletion_auc_stats`.

### 9.5 Figures for methodology
`methodology_overview.png`, `fusion_architecture_diagram.png`, `patient_cv_diagram.png`, `class_distribution.png` (imbalance motivation), `stain_normalization_comparison.png`, `augmentation_examples.png`, `cv_confusion_matrix_v36_pub.png`, `cv_per_class_f1_v36.png`, `perfold_macro_f1_v36_vs_featens.png`, `gate_distribution.png`, `gate_entropy_by_fold.png`, `gate_correlation.png`, `reliability_diagram_v36.png`, `bootstrap_ci.png`, `ablation_barplot.png`, `ensemble_fixes_matrix.png`.

### 9.6 Explainability images to attach
- **Complementarity (the "why fuse" visual):** `backbone_attribution_comparison.png`, `fig_4_1_spatial_complementarity.png` — ConvNeXt vs Swin attend to different evidence.
- **Per-class IG + method comparison:** `ig_8class_grid.png`, `xai_comparison_grid.png`, `fig_4_2_xai_summary.png`, `fig_4_3_xai_benchmark.png`.
- **Faithfulness (quantitative XAI):** `faithfulness_deletion_3methods.png`, `deletion_auc_boxplot.png`; localisation `iou_histogram.png`, `heatmap_iou_histogram.png`.
- **Curated case-study panels** (`results/xai/{hirescam_convnext, swin_attention, swin_gradcam}/`) — filenames encode the case: `*_conv_wins_*` (0033, 0088 DC), `*_swin_wins_*` (0053 DC, 0214 PC), `*_both_wrong_*` (0172/0173 **lobular** — the hard class), `*_high_disagree_*` (0203 MC, 0227 PT), `*_easy_*` (0007, 0094, 0222, 0243). Use a **conv_wins + swin_wins pair** to prove complementarity visually, and a **both_wrong lobular** case to discuss the hardest subtype.
- ⚠️ **Gap:** the per-sample XAI panels are per-backbone (ConvNeXt HiResCAM, Swin attention/GradCAM). Verify whether a **v3.6 / fusion-level** attribution map per sample exists; if not, that's the one explainability figure worth generating ("Exp A").

