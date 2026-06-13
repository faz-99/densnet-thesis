# Chapter 3 — Methodology

## 3.1 Problem Statement and Overview

### 3.1.1 Problem statement

Breast cancer is among the most common cancers worldwide, and its definitive diagnosis still
rests on the microscopic examination of stained tissue by a pathologist. Beyond the first,
coarse question — *is the tissue benign or malignant?* — clinical management depends on the
finer question of *which histological subtype* is present, since subtypes differ in prognosis
and treatment. This second task is difficult and time-consuming even for experts, which
motivates computer-aided support.

Framed as a machine-learning problem on the BreaKHis 400× dataset, subtype classification
presents three coupled difficulties that any credible method must confront:

1. **Severe class imbalance.** The eight subtypes are highly unequal in frequency — the
   majority class (ductal carcinoma) is roughly 7.6× larger than the rarest (adenosis) — so a
   model can attain high overall accuracy while failing the clinically important rare classes.
2. **Patient-level data scarcity and leakage.** The images come from a small cohort, and
   multiple images originate from the same patient. Evaluations that split by *image* rather
   than by *patient* leak patient-specific cues into the test set and report optimistic numbers
   that do not reflect performance on unseen patients.
3. **Two decisions of unequal difficulty.** The binary benign/malignant decision and the
   eight-class subtype decision are not equally hard. A single modern backbone already
   near-solves the binary task, whereas subtype discrimination — especially among rare classes
   — remains unsolved, so a method optimised for one task is not automatically right for the
   other.

**Objective.** This thesis investigates whether *fusing complementary features* from a
convolutional backbone (ConvNeXt, sensitive to local cellular morphology) and a transformer
backbone (Swin, sensitive to global tissue architecture), combined with explicit
imbalance-aware training, can improve eight-class subtype discrimination **without** sacrificing
the binary decision or relying on patient leakage, while keeping the resulting model
interpretable. The remainder of this chapter develops the methodology designed to answer that
question.

### 3.1.2 Overview and design rationale

This chapter describes the methodology developed for the automated classification of
breast-cancer histopathology images at two clinically distinct levels of decision: the
**binary** distinction between benign and malignant tissue, and the finer **eight-class**
distinction between histological subtypes. The two tasks differ sharply in difficulty.
Separating malignant from benign tissue is, on this dataset, an almost-solved problem for
a modern pretrained backbone, whereas distinguishing eight subtypes — several of which are
represented by very few patients — is the genuinely hard problem and the one on which this
work concentrates.

The methodology was not arrived at in a single step. It is the product of an iterative
engineering process in which each design decision was a direct response to a specific,
observed failure mode of the preceding model. For this reason the chapter is written as a
*problem journey*: each component is introduced together with the difficulty that motivated
it. Section 3.2–3.5 establish the data, preprocessing, and the two feature-extracting
backbones. Section 3.6 narrates the architectural evolution from a naive fusion classifier
to the final production model. Sections 3.7–3.10 specify that final model — its two-head
gated architecture, loss, optimisation, calibration, and routing logic — in full. Sections
3.11–3.13 describe the patient-disjoint cross-validation protocol, the evaluation metrics,
and the explainability methodology.

> **[ATTACH Figure 3.1 — `figures/methodology_overview.png`]**
> Place at the end of §3.1 as the high-level pipeline diagram (data → two backbones →
> feature fusion → two-head gated classifier → routed binary / subtype outputs →
> explainability layer). This figure orients the reader before the per-component detail.

---

## 3.2 Dataset

All experiments use the **BreaKHis** breast-cancer histopathology dataset at **400×
magnification**, comprising **1,693 images** of native resolution 700×460 pixels. Each
image belongs to one of **eight histological subtypes**, which group into a benign/malignant
binary label as follows:

- **Benign (4):** adenosis, fibroadenoma, phyllodes tumour, tubular adenoma
- **Malignant (4):** ductal carcinoma, lobular carcinoma, mucinous carcinoma, papillary carcinoma

A single magnification was chosen deliberately. Restricting to 400× isolates the
classification problem from the confound of multi-resolution fusion, so that any
performance difference can be attributed to model design rather than to the availability of
additional magnifications.

The central methodological challenge of the dataset is its **class imbalance**. The
majority class (ductal carcinoma, 734 images) is roughly **7.6×** larger than the rarest
(adenosis, 96 images). Two classes in particular — adenosis and phyllodes tumour — are so
scarce that they dominate the design decisions described later in this chapter. Table 3.1
gives the full distribution.

**Table 3.1 — BreaKHis 400× class distribution (n = 1,693)**

| Class | Label | Count | % of dataset |
|---|---|---:|---:|
| Ductal carcinoma | Malignant | 734 | 43.4 |
| Fibroadenoma | Benign | 223 | 13.2 |
| Mucinous carcinoma | Malignant | 159 | 9.4 |
| Lobular carcinoma | Malignant | 128 | 7.6 |
| Papillary carcinoma | Malignant | 125 | 7.4 |
| Tubular adenoma | Benign | 122 | 7.2 |
| Phyllodes tumour | Benign | 106 | 6.3 |
| Adenosis | Benign | 96 | 5.7 |
| **Total** | | **1,693** | **100** |

*Imbalance ratio (max/min) ≈ 7.6×.*

> **[ATTACH Figure 3.2 — `figures/class_distribution.png`]** next to Table 3.1.
> **[ATTACH Figure 3.3 — `figures/sample_images.png`]** a representative-image grid (one or
> more exemplars per subtype) so the reader can see the visual heterogeneity that motivates
> the local+global backbone choice in §3.4.

---

## 3.3 Preprocessing and Data Augmentation

### 3.3.1 Resizing and normalisation

Images are resized to **224×224 pixels** to match the input resolution of the ImageNet-pretrained
backbones. During training a `RandomResizedCrop(224, scale=(0.8, 1.0))` is used, which
introduces scale invariance; at validation and test time the deterministic
`Resize(256) → CenterCrop(224)` pipeline is applied instead. Pixel values are normalised with
the standard ImageNet statistics (μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225]),
consistent with the pretraining distribution of the backbones.

### 3.3.2 Augmentation

Training images pass through the augmentation pipeline in Table 3.2. Histopathology images
have no canonical orientation, which justifies the aggressive use of flips and 90° rotations;
the colour jitter provides robustness to staining variation between slides; and random erasing
discourages reliance on any single local region.

**Table 3.2 — Training-time augmentation pipeline**

| Transform | Parameters | Purpose |
|---|---|---|
| RandomResizedCrop | size 224, scale (0.8, 1.0) | scale invariance |
| RandomHorizontalFlip | p = 0.5 | orientation invariance |
| RandomVerticalFlip | p = 0.5 | orientation invariance |
| RandomRotation | 90° | orientation invariance |
| ColorJitter | brightness 0.15, contrast 0.15, saturation 0.1, hue 0.05 | staining robustness |
| RandomErasing | p = 0.1, scale (0.02, 0.1) | occlusion robustness |
| Normalize | ImageNet μ, σ | match pretraining distribution |

> **[ATTACH Figure 3.4 — `figures/augmentation_examples.png`]** in §3.3.2 to show the
> visual effect of the pipeline (original vs augmented panels).

---

## 3.4 Backbone Selection: a Local and a Global View

Two backbones were chosen, **not at random but as a complementary pair**, and this choice is
the conceptual spine that connects classification accuracy to explainability:

- **ConvNeXt-Base** (`convnext_base.fb_in22k_ft_in1k`, 87.6 M parameters) is a modern
  convolutional network. Its hierarchy of local receptive fields makes it a **local-feature**
  model, sensitive to fine cellular morphology and texture — nuclei, glandular detail.

- **Swin-Base** (`swin_base_patch4_window7_224.ms_in22k_ft_in1k`, 86.8 M parameters) is a
  hierarchical vision transformer whose shifted-window self-attention captures longer-range
  spatial relationships. It is the **global-context** model, sensitive to tissue
  architecture and the arrangement of structures.

The motivation is that a pathologist reasons at *both* scales — nuclear detail and overall
architecture — and a model pair spanning both scales should (i) make **decorrelated errors**,
which is the precondition for fusion to help, and (ii) yield **complementary explanations**,
which is the precondition for the explainability contribution in §3.13. Both backbones are
initialised from ImageNet-22K→1K pretrained weights, transfer being essential given the
small dataset.

> **[ATTACH Figure 3.5 — `figures/architecture_comparison.png`]** here: a side-by-side of
> the ConvNeXt and Swin stage stacks, annotated with the local-vs-global contrast.

### 3.4.1 Backbone training

Each backbone is first fine-tuned end-to-end as an eight-class classifier, after which it is
**frozen** and used purely as a feature extractor (§3.5). Both are trained with AdamW
(weight decay 1e-4), batch size 16, gradient-norm clipping at 1.0, label smoothing 0.1,
class-weighted cross-entropy, and a `WeightedRandomSampler` to counter imbalance, with early
stopping (patience 15) on validation macro-F1 over a 100-epoch budget. The two differ in
their fine-tuning schedule, reflecting their different architectures:

- **ConvNeXt** uses discriminative learning rates (backbone 1e-5, head 1e-3) with a cosine
  schedule.
- **Swin** uses **three-phase progressive unfreezing** (20 / 40 / 40 epochs) with a
  layer-wise learning-rate decay (γ = 0.7) in the final phase, which we found necessary for
  stable transformer fine-tuning on this small dataset.

> **[ATTACH Figure 3.6 — `figures/swin_progressive_unfreezing.png`]** to illustrate the
> three-phase schedule, and optionally **`figures/lr_schedule.png`** for the cosine/warmup
> curve.

---

## 3.5 Feature Extraction and the Case for Fusion

After fine-tuning (§3.4.1), each backbone is frozen and used as a deterministic feature
extractor. The two penultimate-layer vectors — local $z_{\text{loc}}\in\mathbb{R}^{1024}$ from
ConvNeXt and global $z_{\text{glob}}\in\mathbb{R}^{1024}$ from Swin — are concatenated into the
**fused representation** $x_f = [\,z_{\text{loc}} \Vert z_{\text{glob}}\,] \in \mathbb{R}^{2048}$.
Because the backbones are frozen, $x_f$ is computed once and cached for all 1,693 images, which
makes the classifier experiments cheap enough to support the long design journey of §3.6.

The hypothesis under test is that a classifier on $x_f$ outperforms one on either view alone,
*because* the local and global features are complementary (their predictions are imperfectly
correlated; inter-backbone agreement $\kappa = 0.68$). As the journey shows, this holds for
**subtype** discrimination but is essentially neutral for the **binary** task — which directly
motivates the single-model two-head design of §3.10.

> **[ATTACH Figure 3.7 — `figures/model_parameter_comparison.png`]** to contrast the heavy
> frozen backbones with the lightweight trainable fusion head, motivating why iteration was
> cheap.

---

## 3.6 The Problem Journey: from Naive Fusion to the v3.6 Model

The final classifier was reached through a sequence of versions, each fixing a defect of the
last. Documenting this trajectory is itself part of the methodology, because it justifies
*why* the final architecture has the unusual features it does (a group-normalised subtype
trunk, per-expert heads, a sigmoid-bounded gate, logit-adjusted loss). The trajectory is
summarised in Table 3.3 and narrated below.

**The driving tension throughout is a trade-off between rare and common classes.** Almost
every intervention that lifted the rare subtypes (papillary, lobular) initially *cost*
recall on the common ones (ductal, fibroadenoma), and vice-versa. The journey is largely the
search for a configuration that improves the tail without sacrificing the head.

**v1 — Single-head baseline.** A plain MLP (2048→512→8) on the fused features, trained with
class-weighted cross-entropy. This established that fusion lifts subtype accuracy over single
backbones, but its selection criterion optimised the binary objective and it left rare
subtypes weak. *Lesson: class weighting alone does not deliver the tail.*

**v2 — Two heads + focal loss.** The network was split into a binary head and a subtype head
sharing a trunk, and focal loss was introduced to emphasise hard/rare examples. The focal
hyper-parameters were, in retrospect, set backwards, and the majority class (ductal) recall
collapsed. *Lesson: heuristic focal weighting is fragile and easy to mis-tune.*

**v3 / v3.1 — Capacity and decoupling.** The shared trunk was widened (2048→1024→512) and
then the two heads were **fully decoupled** into separate sub-trunks, a Squeeze-and-Excitation
block and a scaled residual connection were added to the subtype branch, and — critically —
focal loss was replaced by **Logit-Adjusted Cross-Entropy** (Menon et al., 2020), a
principled long-tail loss. Two learning-rate groups were introduced. *Lesson: a principled
long-tail loss is more stable than focal; the binary and subtype tasks want different
capacities.*

**v3.2 — Aggressive per-class adjustment (collapse).** Per-class temperature and α values
were pushed hard to rescue the rarest classes. This *did* lift papillary and lobular
substantially, but the aggressive common-class setting destroyed ductal, fibroadenoma and
mucinous recall. *Lesson: the rare/common trade-off is real and punishing; do not buy the
tail by taxing the head.*

**v3.3 / v3.3.1 — Recovery and post-hoc calibration.** The per-class loss settings were
relaxed back to a uniform, gentle adjustment (α = 1.5 rare / 1.0 common), the majority class
was *exempted* from logit adjustment (τ[ductal] = 0), and **post-hoc per-class temperature
scaling** (Guo et al., 2017 style, fitted by LBFGS on validation NLL) was added to fix
calibration without retraining. *Lesson: separate the problems — train for discrimination,
calibrate afterwards.*

**v3.4 — Stable sampling (the 0.91 push).** The batch sampler was changed to a
`WeightedRandomSampler` with **inverse-square-root frequency** weights, which softens the
rebalancing (ductal:papillary effective ratio 1:2.4 rather than the raw 1:5.9) and reduces
fold-to-fold variance. Test-time augmentation (10 passes with small feature-space Gaussian
noise) and a temperature cap on the majority class were added. *Lesson: gentle, variance-aware
rebalancing beats hard balancing.*

**v3.5 — Over-engineered experts (collapse).** A full multi-expert head with cross-attention,
a raw residual and split batch-normalisation was introduced. It collapsed: **batch-norm
produced NaNs on the one-sample rare-class batches**, and lobular recall fell to zero.
*Lesson: batch-statistic-dependent normalisation is incompatible with extreme class
imbalance; complexity for its own sake hurts.*

**v3.6 — The production model.** v3.5 was stripped back to its working core. The multi-expert
head was **kept** but simplified (per-class expert hidden width 128; ductal widened to 256),
the gate was made a **sigmoid-bounded modulation** that cannot collapse, and — the key fix —
**every batch-norm in the subtype branch was replaced by GroupNorm(32, 512)**, which is
independent of batch composition and therefore immune to the rare-class batch problem that
killed v3.5. With the v3.4 loss, sampler, calibration and TTA retained, this configuration
recovered the rare classes (papillary and lobular both restored) **without** sacrificing the
common ones. **v3.6 is the production model for this thesis.**

**Table 3.3 — Architectural evolution and the problem each version addressed**

| Ver. | Problem targeted | Key change | Outcome (qualitative) |
|---|---|---|---|
| v1 | Baseline | Single-head MLP on fused features | Fusion helps subtype; tail weak |
| v2 | Rare-class recall | Two heads + focal loss | Majority recall collapsed (mis-tuned focal) |
| v3 / v3.1 | Capacity, stability | Widened + decoupled trunks; SE block; **Logit-Adjusted CE** | More stable; two LR groups |
| v3.2 | Rescue rarest | Aggressive per-class τ/α | Rare up, **common collapsed** |
| v3.3 / v3.3.1 | Recover common + calibrate | Relaxed α; exempt ductal (τ=0); **post-hoc temp scaling** | Balance restored |
| v3.4 | Variance + rarity | `WeightedRandomSampler(1/√freq)`; TTA; temp cap | Lower variance; macro-F1 push |
| v3.5 | More capacity | Cross-attn, raw residual, **split BN** | **Collapse** (BN NaN on rare batches) |
| **v3.6** | **Fix over-engineering** | Simplify experts; **GroupNorm**; sigmoid-bounded gate | **Worked — production model** |
| v3.7 | Ductal↔Fibro confusion | Wider ductal expert; τ[Fibro]=0.5; hard-sample focal | Designed, not run end-to-end |

*Detailed per-version metrics are reported in the Results chapter; this table records design
intent only.*

> **[ATTACH Figure 3.8 — `figures/fusion_architecture_diagram.png`]** immediately after
> Table 3.3 — the v3.6 two-head block diagram with the expert gate. This is the central
> methodology figure.

---

## 3.7 The v3.6 Two-Head Gated Architecture

The production classifier maps the fused feature $x_f \in \mathbb{R}^{2048}$ to a binary logit
$\ell_{\text{bin}}$ and eight subtype logits $\ell_1,\dots,\ell_8$. It is a **conditional, gated
mixture-of-experts** with two task-specific heads on a shared trunk; every component is the
direct remedy to a failure documented in §3.6.

**Shared trunk and two heads.** A single linear projection compresses $x_f$ to a 1024-D shared
representation $h_s$ (`Linear → BatchNorm → ReLU → Dropout 0.5`). From $h_s$, a shallow MLP
yields the binary logit $\ell_{\text{bin}}$ (BatchNorm is fine here — the binary labels are
near-balanced), while a **decoupled** subtype branch projects to a 512-D representation $h_u$.
The subtype branch uses **GroupNorm(32) instead of BatchNorm** — the single most important
architectural choice, because GroupNorm is independent of batch composition and so immune to
the one-sample rare-class batches that produced NaNs and collapsed v3.5. $h_u$ also carries a
Squeeze-and-Excitation channel recalibration ($r=16$) and a scaled residual
$0.3\cdot\mathrm{LN}(W_r x_f)$ that re-injects the raw fused feature.

**Gated mixture-of-experts.** Each class $c$ receives its own *view* of $h_u$, scaled
channel-wise by a learned, **sigmoid-bounded** modulation, and is scored by a dedicated expert
MLP $\mathcal{E}_c$ (ductal widened to $512\!\to\!256\!\to\!1$, the other seven
$512\!\to\!128\!\to\!1$). A softmax routing gate $g=\mathrm{softmax}(\mathrm{MLP}(h_u))$ then
biases the per-class logits:

$$
m_c = h_u \odot \big(0.5 + 0.5\,\sigma(\theta_c)\big), \qquad
\ell_c = \mathcal{E}_c(m_c) + 0.1\,\log g_c. \tag{3.1}
$$

The bound $[0.5,1]$ on the modulation is critical: it lets the model emphasise channels per
class but **never zero them out** (the v3.5 collapse). And because the gate enters as an
*additive bias* (Eq. 3.1) rather than a convex combination of expert outputs, every class's
evidence survives in the decision even when the gate is uncertain. The whole classifier has
$\approx 4.27\,\text{M}$ trainable parameters — a fraction of the $\sim$174 M frozen in the
backbones.

> **[ATTACH Figure 3.9 — `figures/gate_distribution.png`]** (and optionally
> `gate_entropy_by_fold.png`) in §3.7 to show that the routing gate $g$ learns non-degenerate,
> class-discriminative weights rather than collapsing to a single expert.

---

## 3.8 Loss Function, Sampling, and Optimisation

**Joint objective.** The two heads are trained jointly: binary cross-entropy on
$\ell_{\text{bin}}$ plus **Logit-Adjusted Cross-Entropy** (LA-CE; Menon et al., 2020) on the
subtype logits, a principled long-tailed objective that replaced the fragile focal loss of v2.
LA-CE shifts each logit by a temperature-scaled log-prior *before* the softmax, enlarging the
margin demanded of frequent classes without destructively reweighting gradients:

$$
\mathcal{L} = \mathrm{BCE}\big(\sigma(\ell_{\text{bin}}),\, b\big)
\;-\;\alpha_y \log \frac{\exp(\ell_y + \tau_y \log \pi_y)}
{\sum_{c} \exp(\ell_c + \tau_c \log \pi_c)}. \tag{3.2}
$$

Two journey-driven settings appear here: ductal is **exempted** from the adjustment
($\tau_{\text{ductal}}=0$, others $\tau_c=1$) because the log-prior otherwise subtracts
$\approx0.83$ from its logit and suppressed its recall (v3.3.1); and a mild class weight
($\alpha_c=1.5$ rare, $1.0$ common) adds gradient emphasis on the tail, kept at 1.5 rather than
2.0 because the aggressive setting collapsed the common classes (v3.2). Label smoothing
$\varepsilon=0.05$ is applied throughout.

**Variance-aware resampling.** Mini-batches are drawn with a `WeightedRandomSampler` whose
weight is the *inverse square root* of class frequency, $w_i \propto 1/\sqrt{n_{y_i}}$. The
square root deliberately *softens* the rebalancing relative to plain inverse-frequency — it
lifts the effective ductal:papillary ratio from the raw $\approx5.9{:}1$ to $\approx2.4{:}1$
rather than over-correcting, which lowered fold-to-fold variance (v3.4). Sampler and LA-CE are
complementary: the sampler changes *how often* a class is seen, LA-CE the *margin* demanded of
it.

**Optimisation and selection.** AdamW (weight decay $5\times10^{-4}$) with two learning-rate
groups ($10^{-4}$ binary, $3\times10^{-4}$ shared+subtype), 60 epochs, 3-epoch warm-up then
cosine decay, and an EMA of the weights ($\rho=0.999$). The checkpoint maximises
$0.3\,F_1^{\text{bin}} + 0.7\,F_1^{\text{macro}}$ on validation **subject to**
$F_1^{\text{bin}}>0.970$ — encoding the clinical priority *never trade away malignancy
detection for subtype accuracy* while still optimising primarily for the harder subtype task.

**Table 3.4 — v3.6 training configuration**

| Hyper-parameter | Value |
|---|---|
| Epochs | 60 (per fold) |
| Optimiser | AdamW, weight decay 5e-4 |
| Learning rates | binary 1e-4 / shared+subtype 3e-4 |
| Schedule | 3-epoch warm-up → cosine decay |
| EMA decay | 0.999 |
| Loss | BCE + LogitAdjustedCE (τ=1.0, ductal τ=0; α=1.5/1.0; smoothing 0.05) |
| Sampler | WeightedRandomSampler, weights ∝ 1/√(class freq) |
| Dropout / label smoothing | 0.5 / 0.05 |
| Selection | max(0.3·bin-F1 + 0.7·macro-F1), gated by bin-F1 > 0.970 |
| Hardware | AMD Radeon RX 6800 XT (16 GB, ROCm 6.2); PyTorch 2.5.1; timm 1.0.26 |

> **[ATTACH Figure 3.10 — `figures/sampler_distribution.png`]** in §3.8 to visualise the
> 1/√freq effective sampling probabilities, and **`figures/selection_metric.png`** to
> illustrate the gated selection criterion.

---

## 3.9 Calibration and Test-Time Augmentation

Two post-hoc procedures, applied after training, improve the reliability of the predicted
probabilities without altering the learned representation:

**Per-class temperature scaling.** Following Guo et al. (2017), each class logit is divided by
a temperature $T_c$ fitted on the validation set by L-BFGS minimisation of the negative
log-likelihood. The majority class is constrained to $T_{\text{ductal}}\ge1$ to avoid
over-sharpening its already-confident predictions, and the whole step is discarded if it fails
to improve validation macro-F1. Being monotonic, it improves calibration without re-ordering
predictions within a class.

**Test-time augmentation (TTA).** At inference the softmax outputs of $K=10$ forward passes are
averaged — the first on the clean feature $x_f$, the rest on small feature-space Gaussian
perturbations $x_f + \epsilon,\ \epsilon\sim\mathcal{N}(0,0.01^2 I)$. Averaging reduces
prediction variance, which dominates the error budget on the high-variance rare classes.

> **[ATTACH Figure 3.11 — `figures/reliability_diagram_v36.png`]** and
> **`figures/tta_calibration.png`** in §3.9.

---

## 3.10 Two-Head Design: One Model, Two Decisions

A key empirical observation from the journey (§3.6) is that the benefit of fusion is
**task-dependent**: it is substantial for eight-class subtype discrimination but negligible
for the binary benign/malignant decision, on which even a single backbone is already
near-ceiling. Rather than deploy two separate networks and route between them, this is
handled *within a single model* by the two-head design of §3.7 — one shared trunk feeding a
binary head and a gated subtype head. Each head is trained for its own task (the binary head
on BCE, the subtype head on the long-tail logit-adjusted objective), so neither task is
compromised to serve the other.

The justification for this single-model choice is that the binary head loses nothing relative
to the strongest single backbone while the subtype head gains substantially. Under
patient-disjoint cross-validation the binary head (F1 = 0.9781 ± 0.0157) matches the best
single backbone, ConvNeXt-Base (0.9822 ± 0.0089), to within fold-level variance, while on the
subtype task the two-head model (0.8352 ± 0.0876) clearly exceeds the single backbones and the
naive fusion baselines. Keeping both capabilities in one two-head network — rather than an
external router over two models — also pre-empts the obvious objection, *"why not just ensemble
ConvNeXt and v3.6?"*, since the binary capability already lives inside the same network.

> The supporting statistics belong in the Results chapter. The relevant paired McNemar tests
> (§3.12) are: v3.6 vs ConvNeXt-Base — subtype $p\approx5\times10^{-4}$ (v3.6 wins), binary
> $p\approx0.13$ (tied); v3.6 vs the naive feature ensemble — subtype $p\approx6\times10^{-3}$
> (v3.6 wins), binary $p\approx0.14$ (tied). Report these against the *specific* baseline named,
> not against the binary-opt fusion. **Caveat to state honestly:** the v3.6-vs-ConvNeXt subtype
> win reflects the full system (fusion *and* the imbalance-aware training of §3.8); since
> naive fusion alone ≈ ConvNeXt, the gain is attributable to the training regime as much as to
> fusion, and the chapter should claim "the proposed system outperforms…", not "fusion is why".

---

## 3.11 Cross-Validation Protocol

A naive image-level split leaks information: multiple images originate from the same patient,
so a patient appearing in both training and test inflates apparent performance. The headline
evaluation therefore uses **patient-disjoint cross-validation**.

- **Patient-level 5-fold `StratifiedGroupKFold`.** The patient identifier is parsed from the
  BreaKHis filename and used as the grouping key, while folds remain stratified by class.
  Every image from a given patient falls entirely within one of train/validation or test, so
  there is **no patient leakage**. The architecture and all hyper-parameters are pinned to the
  v3.6 configuration across folds (no per-fold tuning), and predictions are pooled across all
  five folds (n = 1,693) for metric computation.

- An **image-level 5-fold `StratifiedKFold`** is also reported for comparison with prior work
  that uses the standard (leaky) BreaKHis protocol, but it is treated as secondary.

> **[ATTACH Figure 3.12 — `figures/patient_cv_diagram.png`]** to show the patient-disjoint
> fold construction, and **`figures/filename_schema.png`** to show how the patient ID is
> extracted from the filename.

---

## 3.12 Evaluation Metrics and Statistical Testing

Because the dataset is imbalanced and the two tasks differ, a single accuracy figure would be
misleading. The metrics below are computed; their values appear in the Results chapter.

- **Subtype task:** **macro-F1** (the primary metric — the unweighted mean of per-class F1, so
  a rare class counts as much as the majority), plus per-class recall/precision and balanced
  accuracy.
- **Binary task:** F1, ROC-AUC, sensitivity, specificity, and the Matthews correlation
  coefficient (MCC), which is informative under imbalance because it accounts for all four
  cells of the confusion matrix.
- **Calibration:** Expected Calibration Error (ECE) over 15 equal-width confidence bins — the
  occupancy-weighted gap between confidence and accuracy, $\mathrm{ECE}=\sum_m \tfrac{|B_m|}{N}\,|\mathrm{acc}(B_m)-\mathrm{conf}(B_m)|$ — and reliability diagrams.
- **Statistical testing:** **McNemar's paired test** on the discordant predictions of two
  models over the *same* pooled patient-CV set (exact binomial form when the discordant count
  $<25$). Running it separately on the binary and 8-class predictions is what reveals that the
  models differ on subtyping but not on malignancy. **Bootstrap 95% CIs** (1,000 resamples)
  accompany the macro-F1, binary-F1 and AUC point estimates.

All cross-validated metrics in this thesis are reported as the **mean ± standard deviation
across the five patient-disjoint folds**, unless explicitly stated otherwise. This convention
captures cross-patient generalisation variance and follows standard medical-imaging reporting
practice. Under this convention the production model's eight-class macro-F1 is
**0.8352 ± 0.0876**. A pooled macro-F1 (obtained by concatenating all 1,693 predictions and
scoring once, 0.892 with bootstrap 95% CI [0.876, 0.907]) is reported in Appendix A for
reference only; because the pooled estimate does not reflect fold-to-fold variance, it is not
used for any significance test or headline claim. The two figures answer different questions
and are not interchangeable, which is why the convention is fixed here at the outset.

> **[ATTACH Figure 3.13 — `figures/bootstrap_distributions.png`]** as an illustration of the
> bootstrap procedure (the resulting CIs go in Results).

---

## 3.13 Explainability Methodology

Explainability is a first-class objective of this thesis, not an afterthought, and the
local+global backbone pairing of §3.4 was chosen partly to make the explanations richer. The
methodology has three components.

**Attribution methods.** Five attribution techniques are applied: **Grad-CAM, Grad-CAM++,
HiResCAM, LayerCAM**, and **Integrated Gradients (IG)**. For ConvNeXt, attributions are taken
at the final convolutional block (`stages[-1].blocks[-1]`); for Swin, at the final
post-attention normalisation layer (`layers[-1].blocks[-1].norm2`) with the appropriate
reshape, and Swin's native attention rollout is used for qualitative inspection.

**Faithfulness metric — deletion AUC.** Attribution maps are compared objectively, not by eye:
pixels are removed in order of decreasing attribution and the predicted probability of the true
class is tracked. The **deletion AUC** is the area under that curve and is **lower for more
faithful** maps — a faithful attribution identifies pixels whose removal collapses the
prediction quickly. This ranks Grad-CAM, Grad-CAM++, HiResCAM, LayerCAM and IG on one
quantitative scale.

**Complementarity metric — heatmap IoU.** To test the §3.4 hypothesis that the two backbones
attend to *different* regions, each heatmap is thresholded (at its 75th percentile) to a binary
salient set, and the spatial Intersection-over-Union between the ConvNeXt and Swin sets is
measured per image. A *low* IoU is the desired outcome — quantitative evidence that the local
and global models contribute complementary, non-redundant explanations rather than redundantly
highlighting the same pixels.

**Scope and an honest limitation.** The XAI benchmarks are run on three models — Swin alone,
ConvNeXt alone, and a *plain* fusion MLP — rather than on the full v3.6 gated head. This is a
deliberate scoping decision (the plain fusion isolates the local+global interaction without
the confound of the expert gate), and it is stated openly as a limitation so that no claim is
made about explaining the production model's gating behaviour.

> **[ATTACH in §3.13, in order:]**
> - **Figure 3.14 — `figures/fig_4_1_spatial_complementarity.png`** (side-by-side ConvNeXt vs
>   Swin heatmaps showing different attended regions — the visual form of the low-IoU claim);
> - **Figure 3.15 — `figures/xai_comparison_grid.png`** (the five attribution methods per
>   backbone);
> - **Figure 3.16 — `figures/faithfulness_deletion.png`** (deletion-AUC curves illustrating
>   the metric);
> - **Figure 3.17 — `figures/heatmap_iou_histogram.png`** (distribution of ConvNeXt–Swin IoU).
>
> *If your thesis numbers these as Chapter 4 figures, keep the filenames and renumber; the
> placement logic is what matters.*

---

## 3.14 Summary

The methodology couples two complementary ImageNet-pretrained backbones — a local-feature
ConvNeXt and a global-context Swin transformer — into a fused 2048-dimensional representation,
on top of which a compact two-head gated classifier (v3.6) was developed through an iterative,
failure-driven design process. The architecture's distinctive choices — group normalisation in
the subtype trunk, bounded per-class expert gating, a logit-adjusted long-tail loss, softened
inverse-square-root sampling, and post-hoc per-class calibration — are each the direct remedy
to a concrete failure observed in an earlier version. The system is evaluated under a
leakage-free patient-disjoint cross-validation protocol with imbalance-aware metrics and
paired statistical testing, and is accompanied by a quantitative explainability methodology
that measures both the faithfulness and the complementarity of the local and global views.
The numerical results of this methodology are presented in the following chapter.
```
