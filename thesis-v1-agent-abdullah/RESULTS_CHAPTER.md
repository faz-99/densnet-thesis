# Chapter 4 — Results

*(Numbering follows the Methodology chapter as Chapter 3; renumber if your thesis orders the
chapters differently. All figures/tables are flagged inline as **[ATTACH …]** with the exact
file in `figures/` or `tables/`.)*

## 4.1 Experimental Setup and Reporting Convention

All results below follow the methodology of Chapter 3: two frozen backbones (ConvNeXt-Base,
Swin-Base) produce a fused 2048-D feature, on which the classifier variants are trained and
evaluated under **patient-disjoint 5-fold cross-validation** (the leakage-free, headline
protocol) and, for comparability with prior BreaKHis work, under image-level 5-fold CV and a
single 70/15/15 split.

**Reporting convention (fixed once, applied throughout).** Aggregate metrics are reported as
**mean ± standard deviation across the five patient-disjoint folds**. Where a *paired*
comparison or a *per-class* breakdown is needed (McNemar tests, confusion matrices, cross-model
recall), figures are computed on the **pooled** predictions of all 1,693 images, because paired
testing requires the same samples from both models. The two conventions give different macro-F1
values for the same model (e.g. v3.6: fold-mean **0.835**, pooled **0.892**); this is expected
— the fold-mean weights each fold equally, the pooled figure weights each image equally — and
both are stated explicitly wherever they appear so the reader is never misled.

---

## 4.2 Master Comparison: All Variants Across Protocols

Table 4.1 is the central result table: 8-class macro-F1 for every model variant under all
three protocols, with trainable-parameter counts. It is the basis for the design narrative
that follows.

**Table 4.1 — 8-class macro-F1 across protocols (fold-mean ± std)**

| Variant | Single-split | Image-CV | **Patient-CV** | Params |
|---|---:|---:|---:|---:|
| Swin-Base (linear head) | 0.8275 | 0.8825 ± 0.0221 | 0.7473 ± 0.0929 | 8K |
| ConvNeXt-Base (linear head) | 0.8048 | 0.8822 ± 0.0095 | 0.8157 ± 0.0891 | 8K |
| Logit ensemble | 0.8671 | 0.8893 ± 0.0193 | 0.8019 ± 0.0990 | 16K |
| Feature ensemble (naive, 2048-256-128-8) | — | 0.9378 ± 0.0083 | 0.8211 ± 0.1599 | 562K |
| Binary-opt fusion (Variant A) | 0.8747 | 0.9302 ± 0.0142 | 0.8208 ± 0.0758 | 1.05M |
| Macro-opt fusion (Variant B) | 0.8434 | — | — | 1.05M |
| **Two-head v3.6 (production)** | 0.8644 | **0.9372 ± 0.0111** | **0.8352 ± 0.0876** | ≈4.27M |
| Two-head v3.6 + Swin last-block FT | **0.8816** | — | — | ≈4.27M + Swin block |

> **[ATTACH Table → `tables/patient_cv_full_metrics.tex`]** (full metric version of this table)
> and **[ATTACH Figure — `figures/params_vs_f1.png`]** (macro-F1 vs parameter count) and
> **[ATTACH Figure — `figures/ablation_barplot.png`]** (variant bar chart).

**Findings.**
1. **The image-level protocol is badly optimistic.** Every fusion variant scores ~0.93–0.94
   under image-CV but drops to ~0.82–0.84 under patient-CV. The ~0.10 gap *is* the patient
   leakage, and it is why all headline claims use the patient-disjoint numbers.
2. **v3.6 is the best variant under both honest protocols** (patient-CV 0.835; image-CV 0.937).
3. **Fusion-by-itself is not the win.** Under patient-CV the naive feature ensemble (0.821) and
   binary-opt fusion (0.821) are statistically indistinguishable from a single ConvNeXt (0.816).
   The lift to v3.6 (0.835) comes from the imbalance-aware training and gated head of §3.7–3.8,
   not from concatenating Swin features per se (see §4.6).

---

## 4.3 The Production Model (v3.6) under Patient-CV

### 4.3.1 Per-fold stability

**Table 4.2 — v3.6 per-fold patient-CV metrics**

| Fold | n_test | 8c-F1 | Binary-F1 | Binary-AUC |
|---:|---:|---:|---:|---:|
| 1 | 347 | 0.728 | 0.955 | 0.982 |
| 2 | 341 | 0.817 | 0.992 | 0.992 |
| 3 | 324 | 0.920 | 0.985 | 0.994 |
| 4 | 340 | 0.781 | 0.969 | 0.987 |
| 5 | 341 | 0.929 | 0.990 | 0.996 |
| **Mean ± std** | | **0.835 ± 0.088** | **0.978 ± 0.016** | **0.990 ± 0.006** |

> **[ATTACH Table → `tables/per_fold_breakdown.tex`]**, **[ATTACH Figure —
> `figures/perfold_macro_f1_v36_vs_featens.png`]** (per-fold v3.6 vs naive ensemble — shows
> v3.6 is consistently above and far more stable).

**Finding.** The binary task is rock-solid across folds (F1 ≥ 0.955, AUC ≥ 0.982). The 8-class
score is more fold-sensitive (0.728–0.929), driven almost entirely by how the two genuinely
rare classes (adenosis, phyllodes) happen to fall in each split — see §4.3.2.

### 4.3.2 Per-class performance (pooled)

**Table 4.3 — v3.6 pooled per-class precision / recall / F1 (patient-CV, n = 1,693)**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Adenosis | 0.958 | 0.958 | 0.958 | 96 |
| Ductal carcinoma | 0.931 | 0.866 | 0.898 | 734 |
| Fibroadenoma | 0.940 | 0.915 | 0.927 | 223 |
| Lobular carcinoma | 0.699 | 0.945 | 0.804 | 128 |
| Mucinous carcinoma | 0.909 | 0.937 | 0.923 | 159 |
| Papillary carcinoma | 0.844 | 0.912 | 0.877 | 125 |
| Phyllodes tumour | 0.828 | 0.774 | 0.800 | 106 |
| Tubular adenoma | 0.937 | 0.967 | 0.952 | 122 |
| **Macro avg** | **0.881** | **0.909** | **0.892** | 1,693 |
| Weighted avg | 0.902 | 0.895 | 0.897 | 1,693 |

> **[ATTACH Table → `tables/.../per_class_table.tex`]**,
> **[ATTACH Figure — `figures/cv_per_class_f1_v36.png`]** (per-class recall with ±std error
> bars), **[ATTACH Figure — `figures/cv_confusion_matrix_v36_pub.png`]** (300-DPI pooled
> confusion matrix, counts + row-normalised).

**Findings.**
1. **The remaining weak spot is lobular precision (0.699), not its recall (0.945).** The
   confusion matrix shows the model *over-predicts* lobular — it catches almost all true
   lobular cases but also mislabels some ductal as lobular. This is the ductal↔lobular/
   fibroadenoma boundary, the bottleneck v3.7 was designed (but not run) to address.
2. **Phyllodes is the hardest class (F1 0.800)** — expected, given it is rare (106) and
   visually overlaps other benign tumours.
3. **Pooled recall hides fold variance for the rarest classes.** Adenosis pooled recall is
   0.958, but its *per-fold* recall ranges from 0.0 (fold 1) to 1.0 — std ≈ 0.43. State this
   honestly: the model is good on adenosis on average but unstable across patient splits,
   a small-sample effect, not a model defect.

---

## 4.4 Statistical Significance

All tests are paired McNemar on the pooled patient-CV predictions of the two models named
(§3.12). Bootstrap CIs are 1,000 resamples of the pooled set.

**Table 4.4 — Paired McNemar tests vs. v3.6**

| Baseline | Task | Baseline acc | v3.6 acc | p-value | Verdict |
|---|---|---:|---:|---:|---|
| ConvNeXt-Base | 8-class | 0.872 | 0.896 | **0.0005** | v3.6 wins |
| ConvNeXt-Base | binary | 0.976 | 0.970 | 0.134 | tied |
| Naive feature ensemble | 8-class | 0.875 | 0.896 | **0.006** | v3.6 wins |
| Naive feature ensemble | binary | 0.965 | 0.970 | 0.137 | tied |
| Binary-opt fusion (Variant A) | 8-class | — | — | **8.6e-5** | v3.6 wins |
| Binary-opt fusion (Variant A) | binary | — | — | 0.405 | tied |

**Table 4.5 — Bootstrap 95% CIs (pooled patient-CV)**

| Model | 8c-F1 | Binary-F1 | Binary-AUC |
|---|---|---|---|
| **v3.6 two-head** | 0.892 [0.876, 0.907] | 0.978 [0.972, 0.984] | 0.991 [0.986, 0.996] |
| Binary-opt fusion | 0.877 [0.861, 0.893] | 0.975 [0.968, 0.982] | 0.994 [0.990, 0.997] |

> **[ATTACH Table → `tables/mcnemar_detailed.tex`]**, **[ATTACH Figure —
> `figures/bootstrap_distributions.png`]** and optionally **`figures/bootstrap_ci.png`**.

**Findings.**
1. **v3.6 significantly beats ConvNeXt, the naive ensemble, *and* the binary-opt fusion on
   8-class subtyping** (all p < 0.01), and is **statistically tied with all of them on the
   binary task** (all p > 0.13). This is the precise, defensible answer to "why the gated
   system?": it improves subtyping without costing malignancy detection.
2. **Honest scope of the claim.** The v3.6-vs-ConvNeXt subtype win is the *full system* (fusion
   **+** imbalance-aware training + gated experts + TTA) against a plain ConvNeXt head. Because
   naive fusion alone ≈ ConvNeXt (§4.6), the gain is attributable to the training/architecture
   regime as much as to fusion. The thesis claims "the proposed v3.6 system outperforms…", not
   "fusion alone is responsible."

---

## 4.5 Binary (Benign vs Malignant) Task

The binary task is near-saturated for every competent model. Under patient-CV, v3.6 reaches
binary-F1 0.978 ± 0.016 and AUC 0.990 ± 0.006; ConvNeXt-alone reaches 0.982 / 0.992. The
single-split extended binary metrics (Table 4.6) confirm strong calibration and discrimination.

**Table 4.6 — Single-split binary metrics (n = 258 test)**

| Variant | 8c-F1 | Binary-F1 | Binary-AUC | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| Swin alone | 0.8275 | 0.9677 | 0.9967 | 0.9066 | 0.145 |
| ConvNeXt alone | 0.8048 | 0.9659 | 0.9848 | 0.8933 | 0.110 |
| Logit ensemble | 0.8671 | 0.9797 | 0.9971 | 0.9391 | 0.144 |
| Feature ensemble (binary-opt) | 0.8747 | **0.9885** | 0.9923 | **0.9647** | **0.0605** |
| Two-head v3.6 + Swin FT | **0.8816** | 0.9884 | 0.9913 | — | — |

> **[ATTACH Figures —** `figures/ensemble_swin_none_x_convnext_none_roc.png` (ROC),
> `figures/ensemble_swin_none_x_convnext_none_confusion_binary.png` (binary confusion),
> `figures/ensemble_swin_none_x_convnext_none_weight_sweep.png` (logit-weight sweep)**]**.

**Finding.** No model is significantly better than another on binary (McNemar, §4.4) — the task
is solved by a single backbone. This is the empirical justification for the two-head design
(§3.10): the binary head costs nothing, and all the engineering effort is correctly spent on
the subtype head.

---

## 4.6 The Feature-Collapse Finding (naive fusion → gate recovery)

This is the result that justifies the gated architecture. Table 4.7 gives **pooled per-class
recall** for the single backbones, the naive feature ensemble, and v3.6 (all patient-CV, same
1,693 images, so directly comparable).

**Table 4.7 — Pooled per-class recall across models (patient-CV)**

| Class | Swin | ConvNeXt | Naive FeatEns | **v3.6 gate** |
|---|---:|---:|---:|---:|
| Adenosis | 0.979 | 0.958 | 0.958 | 0.958 |
| Ductal | 0.699 | 0.820 | **0.935** | 0.866 |
| Fibroadenoma | 0.830 | 0.888 | 0.888 | 0.915 |
| **Lobular** | 0.898 | **0.953** | **0.633** | **0.945** |
| Mucinous | 0.912 | 0.899 | 0.912 | 0.937 |
| **Papillary** | 0.832 | **0.952** | **0.720** | **0.912** |
| Phyllodes | 0.755 | 0.792 | 0.689 | 0.774 |
| Tubular | 0.910 | 0.959 | 0.959 | 0.967 |
| **Pooled macro-F1** | 0.800 | 0.871 | 0.863 | **0.892** |

> **[ATTACH Figure — `figures/ensemble_fixes_matrix.png`]** (which classes each model fixes /
> breaks) and **[ATTACH Figure — `figures/confusion_8class_per_model.png`]**.

**Findings.**
1. **Naive fusion collapses the rare malignant subtypes.** Adding Swin features under plain,
   unweighted training drops **lobular recall 0.953 → 0.633** and **papillary 0.952 → 0.720**
   relative to ConvNeXt alone, while *boosting* the majority class (ductal 0.820 → 0.935). This
   is the textbook signature of majority-class dominance under an unweighted objective — the
   naive `FeatureEnsembleMLP` uses no class weights and no balanced sampler — not "feature
   cancellation."
2. **The gated v3.6 recovers the collapse.** With the imbalance-aware training of §3.8, v3.6
   restores **lobular to 0.945** and **papillary to 0.912** while keeping ductal balanced
   (0.866), lifting pooled macro-F1 to 0.892 — the highest and the only model that does not
   sacrifice minorities for the majority.
3. **The honest caveat.** ConvNeXt alone is *marginally higher* than v3.6 on lobular (0.953 vs
   0.945) and papillary (0.952 vs 0.912). v3.6's advantage is **aggregate and balanced** (best
   macro-F1, significant by McNemar), achieved by trading a sliver of ductal recall for gains on
   fibroadenoma and mucinous — *not* by beating ConvNeXt on every class. State it this way; an
   examiner who reads the table will otherwise catch an overclaim.

---

## 4.7 Calibration

**Finding.** Feature fusion improves probability calibration: single-split ECE falls from
0.145 (Swin) / 0.110 (ConvNeXt) to **0.0605** for the fusion model — the best-calibrated
variant — and per-class temperature scaling (§3.9) further tightens the reliability curve.
Good calibration matters clinically: a confidence score that matches empirical accuracy is a
precondition for the model being usable as a decision *aid*.

> **[ATTACH Figure — `figures/reliability_diagram_v36.png`]** (reliability diagram) and
> **[ATTACH Figure — `figures/tta_calibration.png`]** (effect of TTA + temperature scaling).

---

## 4.8 Gating Behaviour

To confirm the routing gate (Eq. 3.1) learns something non-trivial rather than collapsing to a
single expert, its per-sample weights are inspected.

> **[ATTACH Figures —** `figures/gate_distribution.png` (distribution of gate weights),
> `figures/gate_entropy_by_fold.png` (gate entropy per fold), `figures/gate_correlation.png`
> (inter-class gate correlation) **]** and **[ATTACH Table → `tables/gating_ablation.tex`]**
> (gate-on vs gate-off ablation).

**Finding.** The gate produces non-degenerate, class-discriminative weights (it does not
collapse onto one expert), and the gate-on/gate-off ablation in `gating_ablation.tex` quantifies
its contribution — confirming the sigmoid-bounded modulation fixed the v3.5 collapse described
in §3.6.

---

## 4.9 Explainability

The explainability evaluation (§3.13) answers two questions: *which attribution method is most
faithful*, and *do the local and global backbones look at different regions*.

### 4.9.1 Faithfulness (deletion-AUC, lower is better)

**Table 4.8 — Deletion-AUC by method and model (30-image stratified subset)**

| Model | Best CAM method | Integrated Gradients |
|---|---|---:|
| ConvNeXt | Grad-CAM++ (0.442) | **0.130** |
| Swin | HiResCAM (0.535) | 0.409 |
| Fusion (FusionWrapper) | Grad-CAM++ (0.462) | **0.178** |

> **[ATTACH Table → `tables/deletion_auc_stats.tex`]**, **[ATTACH Figures —**
> `figures/fig_4_3_xai_benchmark.png` (per-method per-model bar chart),
> `figures/faithfulness_deletion.png` and `figures/faithfulness_deletion_3methods.png`
> (deletion curves), `figures/deletion_auc_boxplot.png` **]**.

**Finding.** **Integrated Gradients is dramatically more faithful than the CAM family on the
fusion model** (0.178 vs 0.462 for Grad-CAM++ — a 61.5% reduction in deletion-AUC). CAM methods
operate on a single backbone's spatial activations and cannot capture the cross-branch
interaction that happens inside the fusion MLP; IG, being a path-integral over the input,
does. IG is therefore the recommended attribution method for the fused model.

### 4.9.2 Complementarity (heatmap IoU)

**Finding.** The median spatial IoU between ConvNeXt and Swin attribution maps is **0.051** —
i.e. the two backbones attend to almost entirely *different* regions. This is the quantitative
confirmation of the §3.4 hypothesis: the local (CNN) and global (transformer) views are
complementary, not redundant, which is the explainability rationale for pairing them.

> **[ATTACH Figures —** `figures/fig_4_1_spatial_complementarity.png` (side-by-side ConvNeXt
> vs Swin heatmaps on the same image — the visual form of the low IoU),
> `figures/heatmap_iou_histogram.png` (IoU distribution),
> `figures/xai_comparison_grid.png` (all five methods × both backbones),
> `figures/ig_8class_grid.png` (IG maps for all 8 classes),
> `figures/backbone_attribution_comparison.png` **]**.

**Honest limitation (state it).** The XAI benchmark is run on Swin-alone, ConvNeXt-alone, and a
*plain* fusion MLP — not on the full v3.6 gated head. This is a deliberate scope choice (the
plain fusion isolates the local↔global interaction without the confound of the expert gate),
and it is declared so that no claim is made about explaining the gating behaviour of the
production model.

---

## 4.10 Summary of Findings

1. **Patient-disjoint evaluation is essential** — image-level CV overstates macro-F1 by ~0.10
   through patient leakage (§4.2).
2. **The binary task is saturated** — every competent model scores binary-F1 ≈ 0.97–0.98 with
   no significant differences (§4.4–4.5).
3. **v3.6 is the best subtyping model** — patient-CV macro-F1 0.835 ± 0.088 (pooled 0.892),
   significantly beating ConvNeXt, the naive ensemble, and the binary-opt fusion (all p < 0.01),
   while tied on binary (§4.4).
4. **The gate's purpose is stabilisation, not magic** — naive fusion *collapses* rare malignant
   subtypes (lobular 0.95→0.63, papillary 0.95→0.72) by over-serving the majority; v3.6 recovers
   them (0.95, 0.91) and is the only variant that does not trade minorities for the majority
   (§4.6).
5. **Most of v3.6's gain is from imbalance-aware training, not fusion per se** — naive fusion ≈
   ConvNeXt; the gated head + LA-CE + sampler is what moves the number (§4.2, §4.6). Claimed
   honestly throughout.
6. **Fusion improves calibration** — ECE 0.110/0.145 → 0.0605 (§4.7).
7. **Integrated Gradients is the faithful explainer for the fused model** (deletion-AUC 0.178
   vs 0.462), and the two backbones are spatially complementary (median IoU 0.051) (§4.9).

The unresolved bottleneck — the ductal↔lobular/fibroadenoma boundary, visible as lobular's low
precision (0.699) in Table 4.3 — is the motivation for the v3.7 design documented but not run,
and is carried forward to the Discussion / Future Work chapter.
