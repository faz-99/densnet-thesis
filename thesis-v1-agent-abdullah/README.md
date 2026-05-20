# Multimodal Explainable AI for Breast Histopathology

Thesis implementation: ConvNeXt-Base × Swin-Base feature-fusion for the BreaKHis 400× benchmark, with a Grad-CAM-family + Integrated Gradients explainability study on the fusion model.

For day-to-day project state, decisions, and the full changelog, see [`CLAUDE.md`](CLAUDE.md) — it is the single source of truth.

## Finalized models

The thesis reports two fusion models, both operating on cached frozen-backbone features (`results/features/`):

| Model | Where | Headline metric (patient-CV, n=1693) |
|---|---|---|
| **Two-head v3.6** | `03_training.ipynb` cell 2.5.5 (pinned in 2.5.6 as `TwoHeadFusionMLP_v36`) | 8c-F1 **0.835 ± 0.088**, bin-F1 0.978, AUC 0.990 |
| **Feature ensemble** (2048→256→128→8 MLP) | `03_training.ipynb` cell 2.5.17 (`FeatureEnsembleMLP`) | image-CV 8c-F1 **0.9378 ± 0.0083**, patient-CV 0.8211 ± 0.160 |

Pooled bootstrap 95% CI for v3.6 8c-F1: [0.876, 0.907]. McNemar 8-class vs binary-opt fusion: p = 8.6e−5 (v3.6 wins on subtype distinction).

## Repository layout

```
thesis-v1-agent-abdullah/
├── 01_data_preparation.ipynb   ← BreaKHis 400× split + augmentation
├── 02_model_architecture.ipynb ← backbone definitions, fusion architectures
├── 03_training.ipynb           ← all training cells (2.4 backbones, 2.5.* fusion variants + CV)
├── CLAUDE.md                   ← living project log (read this first)
├── src/                        ← reusable modules (config, dataset, models, trainer)
├── scripts/                    ← figure/table builders + diagram-rendering scripts
├── figures/                    ← generated figures for the thesis
└── tables/                     ← LaTeX tabular fragments for the thesis
```

## Environment

- Python 3.12, PyTorch 2.5.1+rocm6.2, timm 1.0.26, captum, pytorch-grad-cam, netcal, statsmodels.
- AMD Radeon RX 6800 XT (16 GB VRAM, ROCm 6.2), Ryzen 5 5600, 32 GB RAM.
- Random seed 42 throughout.

## Dataset

- **BreaKHis 400×:** 1,693 images, 8 subtypes, native 700×460.
- Stratified 70/15/15 image-level split (1,182 / 253 / 258) following Spanhol et al. (2016) for benchmark comparability.
- Cross-validation reported in two protocols: image-level 5-fold (`StratifiedKFold`) and patient-level 5-fold (`StratifiedGroupKFold`, leakage-free).

## Running

Training cells are idempotent — each guards on `os.path.exists(checkpoint_path)`, and the two-head cell additionally probes the saved `state_dict` against the current class definition and retrains on architecture mismatch. Re-running a notebook will skip already-trained variants.

Backbones (cell 2.4) are not retrained; the only sanctioned exception is the Swin last-block fine-tune ablation in cell 2.5.9.

## Outputs

Artefacts live under (paths relative to project root, gitignored):

- `weights/` — checkpoints for each variant.
- `results/` — cached features, per-variant test summaries, CV summaries, pooled prediction caches, statistical tests, XAI deletion/IoU arrays.
- `figures/`, `tables/` — thesis-ready figures and LaTeX fragments.

See the **Artefacts on disk** section of `CLAUDE.md` for the full tree.
