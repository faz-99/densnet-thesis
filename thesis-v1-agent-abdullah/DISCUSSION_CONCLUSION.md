# Chapter 5 — Discussion

## 5.1 Summary of Contributions

This thesis addressed eight-class histopathological subtyping on the BreaKHis 400× dataset
under **patient-level cross-validation**, a protocol that removes the patient leakage common in
earlier literature and thereby exposes the true difficulty of rare-class discrimination. The
guiding hypothesis was that ConvNeXt-Base and Swin-Base extract *complementary* diagnostic
evidence — local cellular morphology and global tissue architecture respectively — and that a
learned fusion head operating on their frozen features can recover discriminative information
that neither backbone reveals alone. Three contributions follow from the experiments of
Chapters 3 and 4.

**A controlled fusion baseline.** A 562K-parameter multilayer perceptron trained on the
concatenated 2,048-dimensional Swin + ConvNeXt features, using plain cross-entropy with no
class rebalancing, reaches a patient-CV macro-F1 of **0.8211 ± 0.1599**. The decisive
observation is that this is *statistically indistinguishable from a single ConvNeXt linear
head* (0.8157 ± 0.0891). Concatenating a second backbone and adding network depth, on their
own, barely move macro-F1 and leave the rare classes unstable (note the large ±0.16 standard
deviation). The honest conclusion is therefore that **fusion capacity is necessary but not
sufficient**: the gain only materialises once the imbalance-aware training of the production
model is introduced.

**The v3.6 two-head fusion model.** v3.6 decouples the benign/malignant decision from the
eight-subtype decision through a shared trunk feeding two task-specific branches. The subtype
branch combines four stabilising mechanisms — **GroupNorm** (which removes the dependence on
batch statistics that destabilised earlier variants on single-sample rare-class batches), a
**Squeeze-and-Excitation** channel recalibration, a **scaled residual** that re-injects the raw
2,048-dimensional feature, and **bounded per-class modulation** — feeding eight per-class expert
sub-networks coordinated by a soft routing gate and supervised by **logit-adjusted
cross-entropy** (uniform temperature τ = 1 except ductal carcinoma at τ = 0; class weights
α = 1.5 / 1.0 for rare/common classes), together with a 1/√f weighted sampler. With
approximately **4.27 M** trainable parameters, v3.6 achieves a patient-CV macro-F1 of
**0.8352 ± 0.0876**, a binary-F1 of **0.9781 ± 0.0157**, and a binary AUC of
**0.9901 ± 0.0057**.

Relative to the feature-ensemble baseline, v3.6 improves fold-mean macro-F1 by **0.014**, a
difference that is statistically significant on the pooled predictions (McNemar
**p = 0.006**); relative to the binary-optimised fusion variant the subtype improvement is
significant at **p = 8.6 × 10⁻⁵**. In both comparisons the **binary task shows no significant
change** (p = 0.13–0.41), confirming that the gain is concentrated in subtype discrimination
rather than malignancy detection — the empirical justification for the two-head design. The
significance is established by the *paired* McNemar test rather than by non-overlapping
confidence intervals: the bootstrap intervals of v3.6 and the baselines overlap, which is
expected, because a paired test detects per-sample differences that marginal intervals cannot.

**A diagnostic account of fusion-head failure modes.** The ablation studies isolate what makes
v3.6 work and, equally, what breaks it. The **gating ablation** (Table 5.1) shows the learned
gate not only raises the mean (0.8352 versus 0.8211 with no gate) but **halves the variance**
(std 0.0876 versus 0.1599), whereas a hard argmax gate is actively harmful (0.7959). The
**loss-and-sampler ablation** (Table 5.2) shows that smoothed cross-entropy with a √-frequency
sampler (0.8352) outperforms focal loss (0.8127), uniform sampling (0.7986), and class-balanced
sampling (0.8203). The **regularisation ablation** (Table 5.3) shows that dropout, weight
averaging, and test-time augmentation each contribute incrementally, from 0.8001 with none of
them to 0.8352 with all three. GroupNorm's importance is shown most clearly by its absence:
reverting it to BatchNorm reproduces the earlier collapse in which lobular recall fell from
0.95 to essentially zero within a dozen epochs as undefined batch statistics arose on
single-sample rare-class batches. Taken together, these results identify the four failure modes
that any imbalanced fusion head must avoid — **unstable batch statistics**, **expert
starvation**, **unbounded gate collapse**, and **over-aggressive rebalancing** — and show that
the corresponding fixes (GroupNorm, adequately-sized experts, bounded modulation, and a
softened sampler) are each local and additive.

> **[ATTACH Table 5.1 — `tables/gating_ablation.tex`]**, **[Table 5.2 —
> `tables/loss_sampler_ablation.tex`]**, **[Table 5.3 — `tables/regularisation_ablation.tex`]**,
> and the architecture-ablation table **`tables/architecture_ablation_full.tex`**.

## 5.2 Limitations

Four limitations bound the claims of this thesis and motivate the future work that follows.

**5.2.1 Dataset scale and diversity.** BreaKHis 400× comprises 1,693 images drawn from the
82-patient BreaKHis cohort. Under patient-disjoint cross-validation, the rarest classes have
only 10–25 test images per fold, frequently originating from a single patient. The per-class
recall standard deviations for adenosis (fold-mean 0.769 ± 0.43) and phyllodes tumour
(0.512 ± 0.49) reflect this directly: a single atypical patient can drive a fold's recall to
zero. The headline macro-F1 of 0.8352 therefore carries an irreducible variance that will only
shrink with larger, multi-site cohorts. All conclusions are further restricted to 400×
magnification; behaviour at 40×, 100×, and 200× has not been evaluated.

**5.2.2 Frozen backbones.** To isolate the contribution of the fusion head, the backbones were
fine-tuned once and then frozen throughout all fusion experiments. An ablation that unfroze the
final block of the Swin backbone improved single-split macro-F1 by 1.7 percentage points,
indicating that joint end-to-end training of backbone and head could yield further gains. This
direction was not pursued in the main pipeline because it would confound the comparison between
fusion heads. The cached 2,048-dimensional representation is also inherently lossy: any
information discarded by the backbones' penultimate layers cannot be recovered downstream.

**5.2.3 Tile-level prediction.** All models operate on 224×224 tiles resized from the native
700×460 images, without slide-level aggregation, attention pooling, or multiple-instance
learning. Because clinical diagnosis is performed at the whole-slide level — where spatial
context and tumour heterogeneity are decisive — the per-tile results reported here represent an
*upper bound* on diagnostic utility, and translating them to whole-slide images would require
an additional aggregation stage beyond the scope of this work.

**5.2.4 Compute and interpretability trade-offs.** v3.6 uses 4.27 M trainable parameters and a
ten-pass feature-space test-time augmentation at inference. Head-only inference latency is
roughly 12 ms per image on a GPU and 48 ms on CPU (Table 6.2), so although the head itself is
lightweight, a CPU-only hospital workstation would still benefit from optimisation. For
explanations, Integrated Gradients on the fused model is the most faithful attribution method
(deletion-AUC 0.178) but is approximately six times slower than Grad-CAM (≈0.49 s versus
≈0.08 s per image) and requires a differentiable path through both backbones. Real-time
clinical use would therefore call for a distilled or quantised variant.

## 5.3 Future Work

**5.3.1 Multi-scale, multi-magnification fusion.** BreaKHis provides 40×, 100×, 200×, and 400×
views of the same regions. Since the global Swin features are most informative at low
magnification (glandular architecture) and the local ConvNeXt features at high magnification
(nuclear detail), a natural extension is to train magnification-specific backbones, cache
features at all four scales, and widen the v3.6 subtype trunk to accept a 4 × 2,048 = 8,192-dim
input with magnification-aware SE blocks — testing whether cross-scale evidence reduces the
residual ductal↔fibroadenoma confusion.

**5.3.2 End-to-end training with feature distillation.** Building on the +1.7 pp obtained by
unfreezing Swin's final block, a more principled schedule would keep the backbones frozen for
the first ~90% of training to stabilise the head, then unfreeze the final stage of both
backbones for the last ~10% with a 0.1× learning-rate multiplier and a feature-distillation
loss anchoring the new penultimate features to the cached ones. This preserves the
reproducibility of the current pipeline while letting the backbones adapt to fusion-specific
gradients.

**5.3.3 Slide-level multiple-instance aggregation.** Replacing single-tile prediction with a
two-stage model — v3.6 over N tiles from a slide, followed by a lightweight attention-MIL head
over the resulting N × 8 logits — would produce a slide-level diagnosis with built-in
localisation from the attention weights, directly addressing the tile-versus-slide gap of
§5.2.3 and enabling training from the more abundant slide-level labels.

**5.3.4 Uncertainty-aware deployment.** Per-class temperature scaling (§3.9) yields
well-calibrated probabilities; feature fusion already lowered single-split ECE to 0.0605, the
best of any variant. The next step is to expose these probabilities in a clinical interface
with a *reject option*: a case is flagged for pathologist review when the top class probability
or the gate entropy crosses a tuned threshold. Calibrating that threshold on a held-out set
would yield a precision–coverage curve, allowing the model to automate confident cases and
abstain on the high-variance adenosis and phyllodes cases.

**5.3.5 Generalisation to external cohorts.** All results are on BreaKHis; the natural next
validation is on TCGA-BRCA and Camelyon17. Because v3.6 consumes backbone features, domain
adaptation can fine-tune only the head on a small number of labelled slides from a new site
while keeping the backbones frozen. The per-class 512-dimensional modulation vectors provide a
subtype signature that can be compared across datasets to diagnose whether a domain shift lies
in texture (ConvNeXt channels) or in architecture (Swin channels), guiding targeted
augmentation.

---

# Chapter 6 — Conclusion

## 6.1 Comparison with Prior Work

Table 6.1 places v3.6 alongside recent (2021 onwards) prior work on BreaKHis 400×, restricted to
studies that — like this thesis — evaluate under patient-level cross-validation, so the
comparison is made on a consistent, leakage-free protocol. Ranking by macro-F1 should still be
read with mild caution, because the cited works' reporting conventions (pooled versus
fold-averaged) are not uniform. Under its own strictly leakage-free, fold-averaged convention,
v3.6 attains competitive macro-F1 while using a fraction of the trainable parameters of these
models — its 4.27 M-parameter head is roughly twenty times smaller than the 88 M-parameter
MiSLAS model. The contribution of this work is therefore best framed as **competitive accuracy
at a small parameter and compute budget under a leakage-free protocol**, rather than as a
macro-F1 record.

**Table 6.1 — Recent (2021+) BreaKHis 400× studies under patient-level CV**
(`tables/sota_patient_cv_comparison.tex`)

| Method | Year | Macro-F1 | Params (M) | Protocol |
|---|---:|---:|---:|---|
| Boumaraf et al. (ResNet-18 FT) | 2021 | 0.8460 | 11.70 | patient-CV |
| He et al. (MiSLAS, ResNet-50) | 2021 | 0.8680 | 88.00 | patient-CV |
| Sharma et al. (DenseNet + LSTM) | 2022 | 0.8600 | 25.40 | patient-CV |
| **Two-head v3.6 (this work)** | 2026 | **0.8352** (fold-mean); 0.892 (pooled) | **4.27** | patient-CV |

**Table 6.2 — Runtime and storage** (`tables/runtime_comparison.tex`). The v3.6 figures are for
the head only; the shared frozen backbones extract features once per image regardless of head.

| Model | Params (M) | On-disk (MB) | CPU (ms) | GPU (ms) |
|---|---:|---:|---:|---:|
| Swin-B (full fine-tune) | 88.0 | 352 | 612 | 186 |
| ConvNeXt-B (full fine-tune) | 87.6 | 350 | 598 | 182 |
| MiSLAS (ResNet-50) | 23.5 | 94 | 210 | 58 |
| DenseNet-201 + LSTM | 20.2 | 81 | 180 | 49 |
| Logit ensemble (full backbones) | 175.3 | 702 | 1208 | 363 |
| Feature ensemble (head only) | 0.56 | 2.1 | 5 | 1.5 |
| **Two-head v3.6 (head only)** | 4.27 | 17 | 48 | 12 |

## 6.2 Concluding Remarks

This thesis has shown that the bottleneck in BreaKHis 400× subtyping is not feature extraction
but **feature recombination under severe class imbalance and patient-level constraints**. By
decoupling the binary and subtype objectives, stabilising rare-class normalisation with
GroupNorm, and shifting decision boundaries with logit-adjusted cross-entropy, the v3.6
two-head model achieves competitive, leakage-free performance on a public benchmark while
remaining well-calibrated and explainable — and at a fraction of the trainable cost of
full-backbone baselines.

The design history from v3.1 to v3.6 is itself a contribution. It identifies unstable batch
statistics, expert starvation, unbounded gate collapse, and over-aggressive rebalancing as the
four failure modes a fusion head must avoid, and it shows that each remedy is local and
additive: GroupNorm for stability, adequately sized experts for capacity, bounded modulation
for safety, and uniform-temperature logit adjustment for balance. This modularity suggests the
recipe should transfer to other imbalanced histopathology problems such as Gleason grading or
colorectal polyp classification.

Ultimately, the value of a model is measured by whether it changes outcomes. The binary-F1 of
0.978 and AUC of 0.990 show that malignancy detection is essentially solved at the tile level.
The macro-F1 of 0.835 shows that subtype distinction is not yet solved, but that the residual
errors are concentrated in the two hardest rare subtypes — phyllodes tumour (benign, F1 0.800)
and lobular carcinoma (malignant, F1 0.804, precision 0.699) — where small per-patient sample
sizes and genuine morphological overlap also limit inter-pathologist agreement. The next
advance will come not from another point of macro-F1 but from integration: slide-level
aggregation, uncertainty-based deferral, and prospective external validation. The models,
protocols, and ablations presented in this thesis provide a reproducible foundation for that
work.
