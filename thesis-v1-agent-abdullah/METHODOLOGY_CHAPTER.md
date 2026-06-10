# Chapter 3 — Methodology

## 3.1 Overview and Design Rationale

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

### 3.3.2 The stain-normalisation decision

Stain normalisation (Macenko and Reinhard) was evaluated as a candidate preprocessing step,
on the standard assumption that reducing colour variation between slides should help. In
practice **both methods destabilised backbone training** — training collapsed to
near-zero macro-F1 rather than converging. We therefore made the deliberate methodological
choice to use **no stain normalisation**, relying instead on colour-space augmentation
(below) to provide robustness to staining variation. This is an important decision to state
explicitly because it runs against a common default in the literature; the justification is
empirical, and the collapsed runs are retained as negative evidence.

### 3.3.3 Augmentation

Training images pass through the augmentation pipeline in Table 3.2. Histopathology images
have no canonical orientation, which justifies the aggressive use of flips and 90° rotations;
the colour jitter substitutes for the rejected stain normalisation; and random erasing
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

> **[ATTACH Figure 3.4 — `figures/augmentation_examples.png`]** in §3.3.3 to show the
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

### 3.5.1 Notation

Let an input image be $x \in \mathbb{R}^{3\times 224\times 224}$ with subtype label
$y \in \{1,\dots,8\}$ and binary label $b \in \{0,1\}$ (1 = malignant). The eight subtypes are
indexed by $\mathcal{C}=\{1,\dots,8\}$, of which the malignant subset is
$\mathcal{M}=\{\text{ductal},\text{lobular},\text{mucinous},\text{papillary}\}$, so that
$b = \mathbb{1}[y \in \mathcal{M}]$. The class frequencies are $n_c$ (Table 3.1) with priors
$\pi_c = n_c / \sum_{c'} n_{c'}$. Table 3.5 summarises the symbols used throughout this
chapter.

**Table 3.5 — Notation**

| Symbol | Meaning |
|---|---|
| $z_{\text{loc}},\, z_{\text{glob}} \in \mathbb{R}^{1024}$ | ConvNeXt (local) and Swin (global) feature vectors |
| $x_f = [z_{\text{loc}} \,\Vert\, z_{\text{glob}}] \in \mathbb{R}^{2048}$ | fused feature (concatenation) |
| $h_s \in \mathbb{R}^{1024}$ | shared-trunk representation |
| $h_b \in \mathbb{R}^{256},\; h_u \in \mathbb{R}^{512}$ | binary-branch and subtype-branch representations |
| $\ell_{\text{bin}} \in \mathbb{R}$ | binary logit |
| $\ell_c,\; c\in\mathcal{C}$ | per-class subtype logits |
| $g_c$ | gate weight for class $c$ |
| $\pi_c,\, \tau_c,\, \alpha_c$ | class prior, logit-adjustment temperature, loss weight |

### 3.5.2 Feature extraction

After fine-tuning (§3.4.1), each backbone is frozen and used as a deterministic feature map.
Writing $\phi_{\text{cnx}}$ and $\phi_{\text{swin}}$ for the two backbones truncated at their
penultimate (post-global-pool, pre-classifier) layer,

$$
z_{\text{loc}} = \phi_{\text{cnx}}(x) \in \mathbb{R}^{1024}, \qquad
z_{\text{glob}} = \phi_{\text{swin}}(x) \in \mathbb{R}^{1024},
$$

and the **fused representation** is their concatenation,

$$
x_f = [\,z_{\text{loc}} \,\Vert\, z_{\text{glob}}\,] \in \mathbb{R}^{2048}.
$$

Because the backbones are frozen, $x_f$ is computed once and cached for all 1,693 images,
making the classifier experiments cheap enough to support the long design journey of §3.6.

The hypothesis under test is that a classifier on $x_f$ outperforms one on either
$z_{\text{loc}}$ or $z_{\text{glob}}$ alone, *because* the local and global views are
complementary (their predictions are imperfectly correlated; the inter-backbone agreement is
$\kappa = 0.68$). As the journey shows, this hypothesis holds for **subtype** discrimination
but is essentially neutral for the **binary** task, which directly motivates the single-model
two-head design of §3.10.

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
$\ell_{\text{bin}}$ and an eight-class logit vector $\ell = (\ell_1,\dots,\ell_8)$. It is a
**conditional, gated mixture-of-experts** with two task-specific heads sharing a common trunk.
Every component below is the direct remedy to a failure documented in §3.6. The full forward
pass is given in Equations (3.1)–(3.9); $\mathrm{LN}$ denotes LayerNorm, $\mathrm{GN}_{32}$
GroupNorm with 32 groups, $\sigma$ the logistic function, and $\odot$ elementwise product.

**Shared trunk.** A single linear projection compresses the fused feature and is the only
component both tasks share:

$$
h_s = \mathrm{Drop}_{0.5}\!\Big(\mathrm{ReLU}\big(\mathrm{BN}(W_s x_f + b_s)\big)\Big),
\qquad W_s \in \mathbb{R}^{1024\times 2048}. \tag{3.1}
$$

**Binary head.** A shallow MLP produces the malignancy logit. BatchNorm is retained here
because the binary labels are near-balanced, so batch statistics are stable:

$$
h_b = \mathrm{Drop}_{0.5}\!\Big(\mathrm{ReLU}\big(\mathrm{BN}(W_b h_s + b_b)\big)\Big),
\qquad
\ell_{\text{bin}} = w_b^\top h_b + \beta_b. \tag{3.2}
$$

**Subtype trunk.** The subtype branch is deliberately decoupled from the binary branch and
uses **GroupNorm instead of BatchNorm** — the single most important architectural choice,
because $\mathrm{GN}$ is independent of batch composition and therefore immune to the
one-sample rare-class batches that produced NaNs and collapsed v3.5:

$$
u_0 = \mathrm{Drop}_{0.5}\!\Big(\mathrm{ReLU}\big(\mathrm{GN}_{32}(W_u h_s + b_u)\big)\Big),
\qquad W_u \in \mathbb{R}^{512\times 1024}. \tag{3.3}
$$

A **Squeeze-and-Excitation** block ($r=16$) then recalibrates the 512 channels, gated by a
**scaled residual** that re-injects the raw fused feature so that information compressed away
by the shared trunk is recoverable:

$$
s = \sigma\!\big(W_2\,\mathrm{ReLU}(W_1 u_0)\big),\quad W_1\in\mathbb{R}^{32\times512},\ W_2\in\mathbb{R}^{512\times32}, \tag{3.4}
$$
$$
h_u = \mathrm{Drop}_{0.5}\!\Big(\mathrm{ReLU}\big(u_0 \odot s \;+\; 0.3\cdot \mathrm{LN}(W_r x_f)\big)\Big),
\qquad W_r \in \mathbb{R}^{512\times 2048}. \tag{3.5}
$$

**Per-class feature modulation.** Rather than feed one shared vector to all classes, each
class $c$ receives its own *view* of $h_u$, scaled channel-wise by a learned, **sigmoid-bounded**
gate. The bound $[0.5,1]$ is critical: it lets the model emphasise or de-emphasise channels per
class but **never zero them out**, which is what caused the v3.5 gate collapse:

$$
m_c = h_u \odot \big(0.5 + 0.5\,\sigma(\theta_c)\big), \qquad \theta_c \in \mathbb{R}^{512},\ c \in \mathcal{C}. \tag{3.6}
$$

**Experts and routing gate.** Each class has a dedicated expert MLP $\mathcal{E}_c$ producing a
scalar; the ductal expert is widened ($512\!\to\!256\!\to\!1$) to match its data abundance, the
other seven are $512\!\to\!128\!\to\!1$. A separate softmax **routing gate** computes a
distribution over the eight experts from $h_u$:

$$
g = \mathrm{softmax}\big(W_{g2}\,\mathrm{ReLU}(W_{g1} h_u)\big) \in \Delta^{7},
\quad W_{g1}\in\mathbb{R}^{64\times512},\ W_{g2}\in\mathbb{R}^{8\times64}. \tag{3.7}
$$

The final subtype logit for class $c$ adds a small log-gate bias to the expert output (so the
gate sharpens or softens a class without overriding the expert evidence):

$$
\ell_c = \mathcal{E}_c(m_c) \;+\; 0.1\,\log g_c. \tag{3.8}
$$

The predicted distributions are then

$$
p_{\text{bin}} = \sigma(\ell_{\text{bin}}/T_{\text{bin}}), \qquad
p_c = \frac{\exp(\ell_c / T_c)}{\sum_{c'}\exp(\ell_{c'}/T_{c'})}, \tag{3.9}
$$

where the per-class temperatures $T_c$ are the post-hoc calibration parameters of §3.9 (all
$T_c = 1$ during training).

Note that this differs from a *standard* mixture-of-experts in two ways that matter for the
imbalanced setting: (i) the experts are **class-indexed**, not anonymous, so each carries a
fixed semantic role; and (ii) the gate **biases** the per-class logits additively (Eq. 3.8)
rather than forming a convex combination of expert *outputs*, which keeps every class's
evidence in the final decision even when the gate is uncertain. The whole classifier has
$\approx 4.27\,\text{M}$ trainable parameters — a tiny fraction of the $\sim$174 M frozen in
the two backbones.

> **[ATTACH Figure 3.9 — `figures/gate_distribution.png`]** (and optionally
> `gate_entropy_by_fold.png`) in §3.7 to show that the routing gate $g$ learns non-degenerate,
> class-discriminative weights rather than collapsing to a single expert.

---

## 3.8 Loss Function, Sampling, and Optimisation

### 3.8.1 Joint objective

The two heads are trained jointly. The binary head uses binary cross-entropy on
$\ell_{\text{bin}}$; the subtype head uses **Logit-Adjusted Cross-Entropy** (LA-CE, Menon et
al., 2020), a principled long-tailed objective that replaced the fragile focal loss of v2
(§3.6). The combined per-sample loss is

$$
\mathcal{L} = \underbrace{\mathrm{BCE}\big(\sigma(\ell_{\text{bin}}),\, b\big)}_{\text{malignancy}}
\;+\; \underbrace{\mathcal{L}_{\text{LA}}(\ell, y)}_{\text{subtype}}. \tag{3.10}
$$

LA-CE shifts each logit by a temperature-scaled log-prior *before* the softmax, which enlarges
the decision margin required of frequent classes and so protects the tail without reweighting
gradients destructively:

$$
\mathcal{L}_{\text{LA}}(\ell, y) \;=\; -\,\alpha_y \,
\log \frac{\exp\!\big(\ell_y + \tau_y \log \pi_y\big)}
{\sum_{c\in\mathcal{C}} \exp\!\big(\ell_c + \tau_c \log \pi_c\big)}, \tag{3.11}
$$

with label smoothing $\varepsilon = 0.05$ applied to the target. The per-class hyper-parameters
encode two journey-driven decisions:

$$
\tau_c = \begin{cases} 0 & c=\text{ductal}\\ 1 & \text{otherwise}\end{cases}
\qquad
\alpha_c = \begin{cases} 1.5 & c \text{ rare}\\ 1.0 & c \text{ common}.\end{cases} \tag{3.12}
$$

Exempting ductal ($\tau_{\text{ductal}}=0$) removes the $\log\pi$ penalty from the majority
class — without it, the log-prior alone subtracts $\approx 0.83$ from the ductal logit and
suppressed its recall (§3.6, v3.3.1). The mild $\alpha$ adds gradient emphasis on rare classes
*on top of* the sampler below, but is kept at 1.5 rather than 2.0 because the aggressive
setting collapsed the common classes (v3.2).

### 3.8.2 Variance-aware resampling

Mini-batches are drawn with a `WeightedRandomSampler` whose per-sample weight is the
**inverse square root** of the class frequency,

$$
w_i \;=\; \frac{1}{\sqrt{n_{y_i}}}, \qquad
\Pr(\text{draw } i) \;=\; \frac{w_i}{\sum_{j} w_j}. \tag{3.13}
$$

The square root deliberately *softens* the rebalancing relative to plain inverse-frequency
($1/n_{y_i}$): it lifts the effective ductal:papillary ratio from the raw $\approx\!5.9{:}1$ to
$\approx\!2.4{:}1$ rather than over-correcting to $\approx\!0.4{:}1$, which lowered fold-to-fold
variance (§3.6, v3.4). Sampler and LA-CE are complementary: the sampler changes *how often* a
class is seen, LA-CE changes the *margin* demanded of it.

### 3.8.3 Optimisation and selection

Optimisation uses AdamW (weight decay $5\times10^{-4}$) with **two learning-rate groups** —
$10^{-4}$ for the binary branch and $3\times10^{-4}$ for the shared trunk and subtype branch —
for 60 epochs under a 3-epoch linear warm-up followed by cosine decay. An exponential moving
average of the weights ($\rho=0.999$) is tracked and evaluated alongside the raw weights each
epoch.

The checkpoint is selected to maximise a **clinically weighted validation criterion** subject
to a hard floor on malignancy detection:

$$
\text{select } \arg\max_{\text{epoch}}\;\big(0.3\,F_1^{\text{bin}} + 0.7\,F_1^{\text{macro}}\big)
\quad \text{s.t.}\quad F_1^{\text{bin}} > 0.970. \tag{3.14}
$$

The 0.7 weight and the floor together encode the priority *never trade away malignancy
detection for subtype accuracy*, while still optimising primarily for the harder subtype task.

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

**Per-class temperature scaling.** Following Guo et al. (2017), the logits are divided by
per-class temperatures $T_c$ (Eq. 3.9) fitted on the validation set by minimising the
negative log-likelihood,

$$
\{T_c\}^\star = \arg\min_{\{T_c\}}\; -\sum_{i\in\text{val}} \log
\frac{\exp(\ell_{i,y_i}/T_{y_i})}{\sum_{c}\exp(\ell_{i,c}/T_c)},
\quad \text{subject to}\quad T_{\text{ductal}} \ge 1, \tag{3.15}
$$

solved by L-BFGS. The constraint $T_{\text{ductal}}\ge 1$ prevents over-sharpening the
already-confident majority class, and the whole step is discarded if it fails to improve
validation macro-F1. Temperature scaling is monotonic, so it changes calibration and the
argmax under ties but not the ranking within a class.

**Test-time augmentation (TTA).** At inference the prediction is averaged over $K=10$ forward
passes in probability space, the first on the clean feature and the rest on small
feature-space Gaussian perturbations:

$$
\bar{p}_c = \frac{1}{K}\sum_{k=1}^{K} p_c\big(x_f + \epsilon_k\big),
\qquad \epsilon_1 = 0,\quad \epsilon_{k>1}\sim\mathcal{N}(0,\,0.01^2 I). \tag{3.16}
$$

Averaging reduces prediction variance, which dominates the error budget on the high-variance
rare classes.

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

**Subtype task — macro-F1 (primary).** Per-class F1 is averaged with equal weight, so that a
rare class counts as much as the majority:

$$
F_1^{\text{macro}} = \frac{1}{8}\sum_{c\in\mathcal{C}} \frac{2\,\mathrm{TP}_c}{2\,\mathrm{TP}_c + \mathrm{FP}_c + \mathrm{FN}_c}. \tag{3.17}
$$

Per-class recall/precision and balanced accuracy are also reported.

**Binary task.** F1, ROC-AUC, sensitivity, specificity, and the Matthews correlation
coefficient,

$$
\mathrm{MCC} = \frac{\mathrm{TP}\cdot\mathrm{TN} - \mathrm{FP}\cdot\mathrm{FN}}
{\sqrt{(\mathrm{TP}+\mathrm{FP})(\mathrm{TP}+\mathrm{FN})(\mathrm{TN}+\mathrm{FP})(\mathrm{TN}+\mathrm{FN})}}, \tag{3.18}
$$

which is informative under class imbalance because it accounts for all four cells of the
confusion matrix.

**Calibration — Expected Calibration Error.** Predictions are partitioned into $M=15$
equal-width confidence bins $B_1,\dots,B_M$, and ECE is the gap between confidence and accuracy
weighted by bin occupancy:

$$
\mathrm{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N}\,\big|\,\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\,\big|. \tag{3.19}
$$

**Statistical testing.** Two models are compared on the *same* pooled patient-CV predictions
with **McNemar's paired test** on the discordant counts $b$ (only model A correct) and $c$
(only model B correct):

$$
\chi^2_{\text{McNemar}} = \frac{(b-c)^2}{b+c}, \tag{3.20}
$$

using the exact binomial form when $b+c<25$. Running the test separately on the binary and the
8-class predictions is what reveals that the models differ on subtyping but not on malignancy.
**Bootstrap 95% confidence intervals** (1,000 resamples of the pooled predictions) accompany the
macro-F1, binary-F1 and AUC point estimates.

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

**Faithfulness metric — deletion AUC.** Attribution maps are compared objectively, not by eye,
using a perturbation test. Pixels are removed in order of decreasing attribution; let
$f_y(x^{(k)})$ be the predicted probability of the true class after the top $k$ fraction has
been removed. The **deletion AUC** is the area under this curve,

$$
\mathrm{AUC}_{\text{del}} = \int_0^1 f_y\big(x^{(k)}\big)\,dk \;\approx\; \frac{1}{K}\sum_{k} f_y\big(x^{(k)}\big), \tag{3.21}
$$

and is **lower for more faithful** attributions — a faithful map identifies pixels whose
removal collapses the prediction quickly. This ranks Grad-CAM, Grad-CAM++, HiResCAM, LayerCAM,
and IG on a common, quantitative scale.

**Complementarity metric — heatmap IoU.** To test the §3.4 hypothesis that the two backbones
attend to *different* regions, each heatmap is thresholded at its $75^{\text{th}}$ percentile to
a binary salient set $A_{\text{cnx}}, A_{\text{swin}}$, and their spatial
**Intersection-over-Union** is measured per image:

$$
\mathrm{IoU} = \frac{|A_{\text{cnx}} \cap A_{\text{swin}}|}{|A_{\text{cnx}} \cup A_{\text{swin}}|}. \tag{3.22}
$$

A *low* IoU is the desired outcome: it is the quantitative evidence that the local and global
models contribute complementary, non-redundant explanations rather than redundantly
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
