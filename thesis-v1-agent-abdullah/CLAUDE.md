# CLAUDE.md — Thesis Project Log

Living changelog of decisions, code edits, and experimental artefacts for the *Multimodal Explainable AI for Breast Histopathology* thesis. Single source of truth for "what state is the project actually in?".

---

## Project at a glance

- **Goal:** Compare ConvNeXt-Base and Swin Transformer-Base on BreaKHis 400×, build a feature-level fusion model, and benchmark explainability methods (Grad-CAM family, Integrated Gradients on the FusionWrapper).
- **Working directory:** `thesis-v1-agent-abdullah/`
- **Hardware:** AMD Radeon RX 6800 XT (16 GB VRAM, ROCm 6.2), Ryzen 5 5600, 32 GB RAM.
- **Software:** Python 3.12, PyTorch 2.5.1+rocm6.2, timm 1.0.26, captum, pytorch-grad-cam, netcal, statsmodels.
- **Random seed:** 42 throughout.

---

## Dataset and split (frozen)

- **BreaKHis 400×:** 1,693 images, native 700×460, 8 subtypes.
- **Stratified 70/15/15** with seed 42 → 1,182 / 253 / 258. **Image-level** (not patient-level); follows the standard BreaKHis benchmark of Spanhol et al. (2016).
- **Imbalance:** Ductal Carcinoma 43.4% vs Adenosis 5.7% (max/min ≈ 7.6×).
- **Augmentation:** RandomResizedCrop, H/V flips, RandomRotation 90°, ColorJitter (0.15/0.15/0.1/0.05), RandomErasing (p=0.1). Validation/test deterministic.
- **ImageNet normalisation** (μ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225]).
- **Stain ablation skipped** in the final thesis (Macenko/Reinhard variants exist on disk but are out of scope).
- Metadata: `data/dataset_metadata.json`.

---

## Backbones (frozen — do not retrain)

| Model | timm checkpoint | Params | Recipe | Best val macro F1 |
|---|---|---:|---|---:|
| ConvNeXt-Base | `convnext_base.fb_in22k_ft_in1k` | 87.57 M | discriminative LR (backbone 1e-5, head 1e-3) + cosine, 100 epochs | **0.8480** (epoch 55) |
| Swin-Base | `swin_base_patch4_window7_224.ms_in22k_ft_in1k` | 86.75 M | 3-phase progressive (20/40/40 epochs), layer-wise decay γ=0.7 in phase 3 | **0.8121** (epoch 26) |

Common: AdamW(wd=1e-4), batch 16, gradient-clip ‖g‖₂ ≤ 1.0, label smoothing 0.1, class-weighted CE, WeightedRandomSampler, early-stop patience 15 on val macro F1.

These are now **feature extractors only**. All fusion variants below operate on cached penultimate-layer features (`results/features/`).

---

## Test-set classification — single backbones + ensembles (n=258)

From `results/table_4_1.json` and `results/feature_ensemble_swin_none_x_convnext_none/test_summary.json`.

| Variant | 8c-F1 | Bin-F1 | Bin-AUC | Bin-MCC | ECE |
|---|---:|---:|---:|---:|---:|
| Swin alone | 0.8275 | 0.9677 | 0.9967 | 0.9066 | 0.145 |
| ConvNeXt alone | 0.8048 | 0.9659 | 0.9848 | 0.8933 | 0.110 |
| Logit ensemble (w_swin=0.56) | 0.8671 | 0.9797 | 0.9971 | 0.9391 | 0.144 |
| Feature ens. (binary-opt MLP) | **0.8747** | **0.9885** | 0.9923 | **0.9647** | **0.0605** |

McNemar (paired binary): Fusion vs Swin **p=0.039**; vs ConvNeXt **p=0.008**; vs Logit Ens. p=0.375.
Cohen's κ on val 8-class predictions (Swin vs ConvNeXt) = **0.679**.
Median cross-model heatmap IoU = **0.0513**.

### Per-class recall (binary-opt fusion vs single backbones)

| Class | n_test | Swin | ConvNeXt | Fusion | Δ best→fus |
|---|---:|---:|---:|---:|---:|
| Adenosis | 15 | 93.3% | 86.7% | 86.7% | −6.7 pp |
| Ductal Carcinoma | 111 | 81.1% | 89.2% | **97.3%** | +8.1 pp |
| Fibroadenoma | 34 | 82.4% | 73.5% | **94.1%** | +11.8 pp |
| Lobular Carcinoma | 20 | 70.0% | 70.0% | 60.0% | −10.0 pp |
| Mucinous Carcinoma | 24 | 91.7% | 87.5% | 91.7% | 0.0 pp |
| Papillary Carcinoma | 19 | 84.2% | 68.4% | 68.4% | −15.8 pp |
| Phyllodes Tumour | 16 | 93.8% | 68.8% | 87.5% | −6.2 pp |
| Tubular Adenoma | 19 | 94.7% | 89.5% | 89.5% | −5.3 pp |

**Honest reading:** binary-opt fusion lifts the malignancy decision (driven by Ductal/Fibroadenoma). It does **not** uniformly lift rare-class recall. The macro-opt and two-head variants below address that trade-off.

---

## Fusion-MLP variants (operate on cached features)

### Variant A — `fusion_mlp/` — DONE
- Cell **2.5.2**.
- Single-head `Linear(2048→512)→ReLU→Dropout(0.3)→Linear(512→8)`.
- Loss: weighted CE + smoothing 0.1. Selection: max **val binary F1**.
- Numbers above.

### Variant B — `fusion_mlp_macro/` — DONE
- Cell **2.5.4**.
- Same single-head architecture, **dropout 0.5**, weight_decay 5e-4.
- Loss: Focal (γ=2.0, **flat α=0.25**) + smoothing 0.1.
- Sampler: WeightedRandomSampler.
- Selection: max **val macro F1** gated by val binary F1 > 0.975.
- Idempotent — skips if checkpoint exists.

### Variant C — `fusion_mlp_twohead/` — IN PROGRESS, currently **v3.3**
- Cell **2.5.5**.
- Multi-iteration distinction-tier model. Full evolution in [Two-head evolution](#two-head-evolution).
- Idempotent **with architecture-mismatch fall-through** — saved state_dict is probed against the current class def; if it doesn't match (e.g. after editing the class), retrain triggers automatically.

---

## Two-head evolution

| Version | Date | Architecture | Loss | Selection | Outcome |
|---|---|---|---|---|---|
| v1 | 2026-05-03 | single-head MLP, CE + class weight | CE + smoothing 0.1 | max val bin F1 | Baseline binary-opt (Variant A). |
| v2 | 2026-05-03 | shared 2048→512 (BN+ReLU+Drop), binary head (512→1), subtype head (512→8). EMA(0.999), 3-epoch warmup + cosine. | BCE + 0.3·Focal(γ=2, α=0.25, smoothing rare 0.20 / common 0.05) | 0.3·binF1+0.7·macroF1 gate>0.975 | First two-head; α/smoothing backwards → Ductal lost 9 pp. |
| v3 | 2026-05-03 | wider trunk 2048→1024→512 (BN ×2), skip 2048→512 added pre-final-ReLU, subtype head 512→256→8. | BCE + 0.1·Focal, α=0.50/0.25, smoothing 0.05/0.10 | gate>0.975 | First trunk-widening; still single pathway. |
| v3.1 | 2026-05-04 | **Decoupled trunks** post-shared (binary 256-D / subtype 512-D), **SE block** (r=16) on subtype, **scaled-LN skip** (×0.3), deeper subtype head 512→384→256→8 with progressive dropout 0.2/0.1. | BCE + 1.0·**LogitAdjustedCE** (Menon et al. 2020, τ=1.0, smoothing 0.05). Focal dropped. | gate>**0.970** | Two LR groups (binary 1e-4, other 3e-4). **BalancedBatchSampler** (≥2 rare per batch). |
| v3.2 | 2026-05-04 | v3.1 + **ductal_head** Linear(512→64)→ReLU→Linear(64→1) added to subtype_logits[:, DUCTAL_IDX]. | v3.1 loss + per-class **τ=2.0/0.5 rare/common** + per-class **α=2.0/1.0 rare/common**. | gate>0.970 | Papillary 68→84%, Lobular 60→85%; **but** Ductal 87→79%, Fibro 94→68%, Mucinous 96→83%. tau_common=0.5 was eating common classes. |
| **v3.3 (current)** | 2026-05-04 | v3.1 architecture (ductal_head **removed**). Plus **post-hoc per-class temperature scaling** (Guo et al. 2017 style), 8 temperatures fit on val NLL via LBFGS, applied to test subtype logits before argmax. | BCE + 1.0·LogitAdjustedCE(**τ=1.0 uniform**, **α=1.5/1.0 rare/common**, smoothing 0.05). | gate>0.970 | Targets: Ductal 94-96%, Fibro 91-94%, Mucinous 92-95%, Papillary 82-85%, Macro F1 0.92+. |

### v3.3 hyperparameters (current)

```
Architecture
  shared:   Linear(2048→1024) → BN → ReLU → Dropout(0.5)
  binary:   Linear(1024→256)  → BN → ReLU → Dropout(0.5) → Linear(256→1)
  subtype:  Linear(1024→512)  → BN → ReLU → Dropout(0.5)
            → SE(512, r=16)
            → + 0.3 * LayerNorm(Linear(2048→512)(x))      [scaled residual skip]
            → ReLU → Dropout(0.5)
            → Linear(512→384) → ReLU → Dropout(0.2)
            → Linear(384→256) → ReLU → Dropout(0.1)
            → Linear(256→8)

Loss      BCE(binary) + 1.0 * LogitAdjustedCE(subtype, τ=1.0 uniform,
                                              α=1.5/1.0 rare/common,
                                              smoothing=0.05)
Sampler   BalancedBatchSampler(min_rare_per_batch=2, rare_threshold=100)
Optim     AdamW(weight_decay=5e-4)
          binary branch lr = 1e-4
          shared+subtype lr = 3e-4
Sched     LambdaLR: 3-epoch warmup → cosine decay over 60 epochs
EMA       decay 0.999, evaluated alongside raw each epoch
Sel.      0.3*val_bin_F1 + 0.7*val_macro_F1, gated by val_bin_F1 > 0.970
Post-hoc  Per-class temperature scaling (one T per class, fit on val NLL via
          LBFGS, applied to test subtype logits before argmax).
```

### v3.2 was abandoned because

- τ_common = 0.5 added only −0.41 to Ductal logits, effectively no penalty.
- α_rare = 2.0 + α_common = 1.0 meant common classes got 4× less gradient than rare (compounded with BalancedBatchSampler's 2× frequency boost).
- ductal_head routed Ductal gradient AWAY from the main subtype head — main head learned to ignore Ductal because ductal_head was carrying it.
- Net effect: Papillary/Lobular recovered nicely but Ductal/Fibro/Mucinous tanked.

### v3.3 fixes

1. **Uniform τ=1.0** — log-prior alone gives Ductal log_pi = −0.83 vs Papillary log_pi = −2.61 → 1.78 natural margin. No need to widen further.
2. **α=1.5/1.0** — BalancedBatchSampler already gives rare classes ~2× batch frequency; α only adds gradient-magnitude bias on top. 1.5 is enough, 2.0 was too much.
3. **Removed ductal_head** entirely. Decoupled trunk + logit adjustment is sufficient for the long tail.
4. **Post-hoc per-class temperature scaling** — free recovery of common classes when training-time loss skewed margins. Standard SOTA practice (Guo 2017, MDCA, CaPE).

### Diagnostic prints in v3.3 (cell 2.5.5)

Added 2026-05-04. Active during the run:

1. **Per-class recall row** every 10 epochs (4-letter class names so the row is narrow).
2. **Worst class on every epoch** (one line, even non-milestone) — exposes a stuck rare class immediately.
3. **Logit-adjustment margins** printed once before the training loop — sanity check.
4. **Best-epoch per-class F1 + recall breakdown** at the end.
5. **Post-hoc temperature scan** — prints per-class T values + val macro F1 before/after scaling.

---

## Macenko stain-norm cell (`2.2`) — fixed but unused

`03_training.ipynb` cell 2.2 retrains `swin_macenko` after a stain-vector bug was fixed. Idempotent:
- Skips data regeneration if `data/processed/macenko/.split_complete` exists.
- Skips training if `weights/swin_macenko/best_model.pth` exists.
- Destructive `shutil.rmtree(macenko_dir)` removed on 2026-05-03.

Stain ablation is **out of scope** for the final thesis. Cell preserved for reference only.

---

## Explainability framework

Methodology Section 3.5 / Section 4.3.

- **Per-method × per-model AUC-DEL:** `results/table_4_2.json`, 5 methods × 3 models on a 30-image stratified subset (`results/xai/case_indices.json`).
- **Headline:** Integrated Gradients on `FusionWrapper` AUC-DEL = **0.178** vs Grad-CAM++ on fusion = **0.462** (61.5% relative reduction). CAM-family methods cannot capture cross-branch interactions through the MLP.
- **Per-backbone winners (lowest AUC-DEL):**
  - ConvNeXt: Grad-CAM++ on `stages[-1].blocks[-1]`.
  - Swin: HiResCAM on `layers[-1].blocks[-1].norm2` with reshape transform. Plus native Attention Rollout.
- **Cross-model IoU** at 75th-percentile, median **0.051**.

---

## Figures generated

In `figures/`. Diagrammatic figures regenerable from `scripts/draw_*.py`.

### Dataset/EDA chapter
`class_distribution.png`, `sample_images.png`, `image_statistics.png`, `augmentation_examples.png`, `stain_normalization_comparison.png` (dropped from final thesis).

### Methodology chapter
- `methodology_overview.png` — Figure 3.1 pipeline overview.
- `architecture_comparison.png` — Figure 3.2 ConvNeXt vs Swin block stacks.
- `swin_progressive_unfreezing.png` — Figure 3.3 three-phase schedule.
- `fusion_architecture_diagram.png` — Figure 3.4 late-fusion block diagram.
- `model_parameter_comparison.png`, `lr_schedule.png` — supplementary.

### Results chapter
`learning_curves_loss.png`, `learning_curves_accuracy.png`, `confusion_8class_per_model.png`, `ensemble_swin_none_x_convnext_none_weight_sweep.png`, `ensemble_swin_none_x_convnext_none_confusion_binary.png`, `ensemble_swin_none_x_convnext_none_roc.png`, `ensemble_fixes_matrix.png`. (`ablation_study_comparison.png` dropped — stain section removed.)

### XAI chapter
`xai_comparison_grid.png`, `faithfulness_deletion.png`, `faithfulness_deletion_3methods.png`, `fig_4_1_spatial_complementarity.png`, `fig_4_2_xai_summary.png`, `fig_4_3_xai_benchmark.png`, `heatmap_iou_histogram.png`.

Diagram-rendering scripts: `scripts/draw_methodology_overview.py`, `scripts/draw_architecture_comparison.py`, `scripts/draw_swin_progressive.py`, `scripts/draw_fusion_diagram.py`.

---

## Artefacts on disk

```
weights/
  convnext_none/best_model.pth          ← fine-tuned ConvNeXt (frozen)
  swin_none/best_model.pth              ← fine-tuned Swin (frozen)
  convnext_macenko/, swin_macenko/, …   ← stain ablation (out of scope)
  fusion_mlp/best_model.pth             ← Variant A (binary-opt)
  fusion_mlp_macro/best_model.pth       ← Variant B (macro-opt)
  fusion_mlp_twohead/                   ← Variant C — currently empty (cleared for v3.3 retrain)

results/
  features/                             ← cached frozen-backbone features
  ensemble_swin_none_x_convnext_none/   ← logit ensemble outputs
  feature_ensemble_swin_none_x_convnext_none/ ← Variant A test outputs
  fusion_mlp_macro/                     ← Variant B test outputs
  fusion_mlp_twohead/                   ← Variant C test outputs (will refresh on next run)
  xai/                                  ← deletion-AUC arrays, IoU arrays, case indices
  table_4_1.json, table_4_1_extended.csv, table_4_2.json
```

---

## Outstanding work (priority order)

| # | Task | Backbone retrain? | Cost | Status |
|---:|---|:---:|:---:|---|
| 1 | Run cell 2.5.5 (v3.3) and capture results | ❌ | ~8 h | pending |
| 2 | Bootstrap CIs on existing test predictions | ❌ | 30 min | not started |
| 3 | Patient-stratified split sensitivity analysis | ❌ | ~8 h | not started |
| 4 | 3-seed CV of fusion head | ❌ | ~6 h | not started |
| 5 | Reliability diagram for ECE visualisation | ❌ | 30 min | not started |
| 6 | Comparison table vs prior BreaKHis literature | ❌ | 2 h | not started |
| 7 | IG sanity checks (model + label randomisation) | ❌ | half day | not started |
| 8 | Insertion-AUC alongside Deletion-AUC | ❌ | 2 h | not started |
| — | Optional: backbone fine-tune (LDAM, last 2 blocks, +re-extract features) | ✓ | ~1 day | optional |

**Backbones never get retrained for items 1–8.** All work operates on cached frozen-backbone features.

---

## Conventions

- **Idempotency:** every training cell guards on `os.path.exists(checkpoint_path)`. Two-head cell additionally probes saved `state_dict` against the current class def and falls through to retraining on architecture mismatch.
- **No backbone retraining.** Cell 2.4 produced the only Swin/ConvNeXt checkpoints we'll ever use.
- **Stain ablation skipped.** Cell 2.2 preserved but not referenced in the thesis.
- **Selection criteria are task-specific:**
  - Variant A (binary-opt): max val binary F1.
  - Variant B (macro-opt): max val macro F1, gated by val binary F1 > 0.975.
  - Variant C (v3.3 two-head): 0.3·val binary F1 + 0.7·val macro F1, gated by > 0.970.
- **Test set is touched once per variant.** All hyperparameter and selection decisions are made on validation.

---

## Update protocol

When making future changes, append to:

1. **Changelog** (date-stamped one-line summary).
2. The relevant table (training results, fusion variants, outstanding work) — keep in sync with JSON in `results/`.
3. **Two-head evolution** table — for any future v3.4, v4, … add a row.

Keep this file under 600 lines so it stays readable.

---

## Changelog

### 2026-05-03
- Removed destructive `shutil.rmtree` from cell 2.2 (Macenko regen) and added skip-if-exists guard.
- Added cell **2.5.4** — macro-opt fusion variant, separate save path.
- Added cell **2.5.5 v1/v2** — first two-head distinction variants.
- Generated 4 methodology diagrams: `methodology_overview.png`, `architecture_comparison.png`, `swin_progressive_unfreezing.png`, `fusion_architecture_diagram.png`.
- Skipped stain-normalisation chapter from the final thesis. Methodology and conclusion drafts updated.
- Honest correction: rare-class-recall narrative in conclusion template not supported by data; rewrote to highlight Fibroadenoma (+11.8 pp) and Ductal Carcinoma (+8.1 pp) as the actual winners.
- Drafted full Methodology chapter and Section 4.3 Explainability Framework.

### 2026-05-04
- Cell 2.5.5 promoted to **v3** — wider trunk, residual skip, deeper subtype head.
- Cell 2.5.5 promoted to **v3.1** — decoupled trunks, SE block, scaled-LN skip, two LR groups, BalancedBatchSampler, Logit-Adjusted CE replaces Focal.
- Cell 2.5.5 promoted to **v3.2** — per-class τ, per-class α reweight, ductal-specific head branch.
- Added 4 diagnostic prints to v3.2: per-class recall every 10 epochs, worst-class tracker every epoch, logit-adjustment margin sanity check, best-epoch per-class F1 breakdown.
- Hardened cell 2.5.5 idempotency guard with architecture-mismatch fall-through.
- v3.2 results showed Papillary +16 pp / Lobular +25 pp gains came at the cost of Ductal −8 pp / Fibroadenoma −26 pp / Mucinous −12 pp regressions.
- Cell 2.5.5 promoted to **v3.3** — uniform τ=1.0, α=1.5/1.0 (relaxed), **ductal_head removed**, **post-hoc per-class temperature scaling** added (Guo et al. 2017 style, LBFGS-fit on val NLL).
- Created this `CLAUDE.md`.
