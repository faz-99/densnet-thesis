"""
End-to-end research-grade pipeline for breast cancer histopathology
classification and explanation generation.

Stages:
  1. Data loading (BreaKHis)
  2. Model training / loading (ConvNeXt, Swin Transformer)
  3. Classification evaluation
  4. XAI explanation generation (all methods)
  5. Fidelity evaluation (Insertion/Deletion AUC per method)
  6. Stability, Cross-method consistency, Sparsity metrics
  7. Clinical report generation (MedGemma / Llama-3-Medical / rule-based)
  8. Report quality evaluation
  9. Comparative summary (ConvNeXt vs Swin)
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ── project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.breakhis_dataset import setup_breakhis_dataset, BINARY_CLASSES, MULTICLASS_NAMES
from model.convnext_model import create_convnext
from model.swin_classifier import create_swin
from model.denlsnet_corrected import create_denlsnet

from explainability.xai_pipeline import XAIPipeline
from xai.metrics.fidelity import FidelityEvaluator
from xai.metrics.stability import StabilityMetric
from xai.metrics.cross_method_consistency import CrossMethodConsistency
from xai.metrics.sparsity import compute_sparsity

from evaluation.classification_metrics import evaluate_classification
from report_generation.llm_report_generator import ClinicalReportGenerator
from evaluation.report_metrics import evaluate_reports


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_or_create_model(model_name: str, num_classes: int,
                          checkpoint: Optional[str],
                          device: torch.device) -> torch.nn.Module:
    """Load model from checkpoint or create fresh."""
    if model_name == 'convnext':
        model = create_convnext(num_classes=num_classes, pretrained=(checkpoint is None))
    elif model_name == 'swin':
        model = create_swin(num_classes=num_classes, variant='tiny')
    elif model_name == 'denlsnet':
        model = create_denlsnet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if checkpoint and Path(checkpoint).exists():
        state = torch.load(checkpoint, map_location=device)
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'], strict=False)
        else:
            model.load_state_dict(state, strict=False)
        print(f"[Pipeline] Loaded checkpoint: {checkpoint}")

    model.to(device)
    return model


def get_target_layer(model_name: str, model) -> str:
    """Return the Grad-CAM target layer name for each model."""
    if model_name == 'convnext':
        return 'backbone.stages.3.blocks.2.norm'
    elif model_name == 'swin':
        return 'backbone.layers.3.blocks.1.norm1'
    else:  # denlsnet
        return 'densenet.features.norm5'


def collect_test_samples(dataloader, model, device,
                          max_samples: int = 50) -> Tuple[List, List, List, List]:
    """Collect images, labels, predictions, probabilities from test loader."""
    images, labels, preds, probs = [], [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            out = model(imgs)
            p = F.softmax(out, dim=1)
            pred = p.argmax(dim=1)
            for i in range(imgs.size(0)):
                images.append(imgs[i:i+1].cpu())
                labels.append(lbls[i].item())
                preds.append(pred[i].item())
                probs.append(p[i].cpu().numpy())
                if len(images) >= max_samples:
                    return images, labels, preds, probs
    return images, labels, preds, probs


def build_xai_summary(heatmap: np.ndarray, method: str,
                       fidelity: Optional[Dict] = None) -> Dict:
    """Build XAI summary dict for report generation."""
    sparsity = compute_sparsity(heatmap)
    # Describe dominant region heuristically
    h, w = heatmap.shape
    cy, cx = np.unravel_index(heatmap.argmax(), heatmap.shape)
    vert = 'upper' if cy < h // 3 else ('lower' if cy > 2 * h // 3 else 'central')
    horiz = 'left' if cx < w // 3 else ('right' if cx > 2 * w // 3 else 'central')
    region_desc = f"{vert}-{horiz} tissue region"

    summary = {
        'method': method,
        'region_description': region_desc,
        **sparsity,
    }
    if fidelity:
        summary['insertion_auc'] = round(fidelity.get('insertion_auc', 0.0), 3)
        summary['deletion_auc']  = round(fidelity.get('deletion_auc', 0.0), 3)
    return summary


# ------------------------------------------------------------------ #
# Main pipeline                                                        #
# ------------------------------------------------------------------ #

class BreastCancerPipeline:

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.device = get_device()
        self.results_dir = Path(cfg.get('results_dir', 'pipeline_results')) / \
                           datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Pipeline] Results → {self.results_dir}")
        print(f"[Pipeline] Device  → {self.device}")

        self.task = cfg.get('task', 'binary')
        self.class_names = BINARY_CLASSES if self.task == 'binary' else MULTICLASS_NAMES
        self.all_results: Dict = {}

    # ---------------------------------------------------------------- #
    # Stage 1: Data                                                     #
    # ---------------------------------------------------------------- #
    def setup_data(self):
        print("\n[Stage 1] Setting up BreaKHis dataset...")
        dm = setup_breakhis_dataset(
            root_dir=self.cfg['data_root'],
            magnification=self.cfg.get('magnification', '400X')
        )
        if dm is None:
            raise RuntimeError("Dataset not found. Check cfg['data_root'].")

        self.dataloaders, self.datasets = dm.create_dataloaders(
            task=self.task,
            batch_size=self.cfg.get('batch_size', 16),
            num_workers=self.cfg.get('num_workers', 0),
        )
        print(f"  Train: {len(self.datasets['train'])}  "
              f"Val: {len(self.datasets['val'])}  "
              f"Test: {len(self.datasets['test'])}")

    # ---------------------------------------------------------------- #
    # Stage 2: Models                                                   #
    # ---------------------------------------------------------------- #
    def setup_models(self):
        print("\n[Stage 2] Loading models...")
        n = len(self.class_names)
        self.models: Dict[str, torch.nn.Module] = {}

        for model_name in self.cfg.get('models', ['convnext', 'swin']):
            ckpt = self.cfg.get(f'{model_name}_checkpoint')
            self.models[model_name] = load_or_create_model(
                model_name, n, ckpt, self.device)
            print(f"  {model_name}: ready")

    # ---------------------------------------------------------------- #
    # Stage 3: Classification evaluation                               #
    # ---------------------------------------------------------------- #
    def evaluate_classification(self):
        print("\n[Stage 3] Classification evaluation...")
        self.all_results['classification'] = {}

        for model_name, model in self.models.items():
            print(f"  Evaluating {model_name}...")
            images, labels, preds, probs = collect_test_samples(
                self.dataloaders['test'], model, self.device,
                max_samples=self.cfg.get('eval_samples', 200)
            )
            y_true = np.array(labels)
            y_pred = np.array(preds)
            y_prob = np.array(probs)

            save_sub = self.results_dir / 'classification' / model_name
            metrics = evaluate_classification(
                y_true, y_pred, y_prob, self.class_names, str(save_sub))

            self.all_results['classification'][model_name] = metrics
            self._store_test_data(model_name, images, labels, preds, probs)

            print(f"    Acc={metrics['accuracy']:.4f}  "
                  f"F1={metrics['f1']:.4f}  "
                  f"ROC-AUC={metrics['roc_auc']:.4f}")

    def _store_test_data(self, model_name, images, labels, preds, probs):
        if not hasattr(self, '_test_data'):
            self._test_data = {}
        self._test_data[model_name] = {
            'images': images, 'labels': labels,
            'preds': preds, 'probs': probs
        }

    # ---------------------------------------------------------------- #
    # Stage 4: XAI generation                                          #
    # ---------------------------------------------------------------- #
    def generate_explanations(self):
        print("\n[Stage 4] Generating XAI explanations...")
        self._xai_heatmaps: Dict[str, Dict[str, List]] = {}

        # Collect background data for SHAP (small subset of train)
        bg_images = []
        for imgs, _ in self.dataloaders['train']:
            bg_images.append(imgs)
            if sum(b.shape[0] for b in bg_images) >= 50:
                break
        background = torch.cat(bg_images, dim=0)[:50].to(self.device)

        for model_name, model in self.models.items():
            print(f"  XAI for {model_name}...")
            target_layer = get_target_layer(model_name, model)
            model_type = 'swin_classifier' if model_name == 'swin' else 'denlsnet'

            pipeline = XAIPipeline(
                model, self.device,
                target_layer=target_layer,
                model_type=model_type,
                background_data=background,
            )
            available = pipeline.available_methods()
            print(f"    Available methods: {available}")

            test_data = self._test_data[model_name]
            images = test_data['images']
            preds  = test_data['preds']

            n_xai = min(self.cfg.get('xai_samples', 20), len(images))
            heatmaps_by_method: Dict[str, List] = {m: [] for m in available}

            for i in range(n_xai):
                img = images[i].to(self.device)
                cls = preds[i]
                for method in available:
                    hm = pipeline.explain(method, img, cls)
                    heatmaps_by_method[method].append(hm)

            self._xai_heatmaps[model_name] = heatmaps_by_method
            print(f"    Generated heatmaps for {n_xai} images × {len(available)} methods")

    # ---------------------------------------------------------------- #
    # Stage 5: Fidelity (Insertion / Deletion AUC)                    #
    # ---------------------------------------------------------------- #
    def evaluate_fidelity(self):
        print("\n[Stage 5] Fidelity evaluation (Insertion/Deletion AUC)...")
        self.all_results['fidelity'] = {}

        for model_name, model in self.models.items():
            test_data = self._test_data[model_name]
            images = test_data['images']
            preds  = test_data['preds']
            heatmaps_by_method = self._xai_heatmaps[model_name]

            n = min(self.cfg.get('fidelity_samples', 10), len(images))
            imgs_sub = [images[i].to(self.device) for i in range(n)]
            cls_sub  = preds[:n]
            hm_sub   = {m: heatmaps_by_method[m][:n]
                        for m in heatmaps_by_method}

            evaluator = FidelityEvaluator(model, self.device,
                                          num_steps=self.cfg.get('fidelity_steps', 30))
            results = evaluator.evaluate(imgs_sub, hm_sub, cls_sub)

            save_sub = str(self.results_dir / 'fidelity' / model_name)
            evaluator.plot_curves(results, save_sub)
            df = evaluator.summary_table(results)
            df.to_csv(f"{save_sub}/fidelity_summary.csv", index=False)

            best = evaluator.best_method(results)
            self.all_results['fidelity'][model_name] = {
                'per_method': {m: {k: v for k, v in r.items()
                                   if not k.endswith('_curves')}
                               for m, r in results.items()},
                'best_method': best,
            }
            print(f"  {model_name}: best XAI method = {best}")
            print(df[['Method', 'Insertion AUC', 'Deletion AUC', 'Fidelity Score']].to_string(index=False))

    # ---------------------------------------------------------------- #
    # Stage 6: Stability, Consistency, Sparsity                       #
    # ---------------------------------------------------------------- #
    def evaluate_xai_quality(self):
        print("\n[Stage 6] XAI quality metrics (stability, consistency, sparsity)...")
        self.all_results['xai_quality'] = {}

        consistency_metric = CrossMethodConsistency()

        for model_name, model in self.models.items():
            test_data = self._test_data[model_name]
            images = test_data['images']
            preds  = test_data['preds']
            heatmaps_by_method = self._xai_heatmaps[model_name]

            n = min(self.cfg.get('quality_samples', 10), len(images))
            stability_metric = StabilityMetric(self.device)

            # Stability per method
            stability_results = {}
            for method, heatmaps in heatmaps_by_method.items():
                scores = []
                for i in range(n):
                    if heatmaps[i] is None:
                        continue
                    img = images[i].to(self.device)
                    cls = preds[i]

                    from explainability.xai_pipeline import XAIPipeline
                    target_layer = get_target_layer(model_name, model)
                    xai_pipe = XAIPipeline(model, self.device, target_layer)

                    def gen_fn(perturbed_img, target_cls):
                        return xai_pipe.explain(method, perturbed_img, target_cls)

                    stab, _ = stability_metric.evaluate_stability(
                        img, gen_fn, cls, similarity_metric='ssim')
                    scores.append(stab)
                stability_results[method] = float(np.mean(scores)) if scores else 0.0

            # Cross-method consistency per image
            per_image_consistency = []
            for i in range(n):
                hm_dict = {m: heatmaps_by_method[m][i]
                           for m in heatmaps_by_method
                           if heatmaps_by_method[m][i] is not None}
                if len(hm_dict) >= 2:
                    per_image_consistency.append(
                        consistency_metric.compute(hm_dict))

            consistency_df = consistency_metric.aggregate(per_image_consistency)

            # Sparsity per method
            sparsity_results = {}
            for method, heatmaps in heatmaps_by_method.items():
                valid = [h for h in heatmaps[:n] if h is not None]
                if valid:
                    ginis = [compute_sparsity(h)['gini'] for h in valid]
                    sparsity_results[method] = float(np.mean(ginis))

            self.all_results['xai_quality'][model_name] = {
                'stability': stability_results,
                'sparsity':  sparsity_results,
            }

            # Save consistency table
            save_sub = self.results_dir / 'xai_quality' / model_name
            save_sub.mkdir(parents=True, exist_ok=True)
            if not consistency_df.empty:
                consistency_df.to_csv(save_sub / 'cross_method_consistency.csv', index=False)

            print(f"  {model_name} stability: {stability_results}")
            print(f"  {model_name} sparsity:  {sparsity_results}")

    # ---------------------------------------------------------------- #
    # Stage 7: Report generation                                       #
    # ---------------------------------------------------------------- #
    def generate_reports(self):
        print("\n[Stage 7] Generating clinical reports...")
        self.all_results['reports'] = {}

        report_gen = ClinicalReportGenerator(
            device=str(self.device),
            preferred_model=self.cfg.get('llm_model', 'rule_based'),
        )

        for model_name in self.models:
            test_data = self._test_data[model_name]
            images = test_data['images']
            labels = test_data['labels']
            preds  = test_data['preds']
            probs  = test_data['probs']
            heatmaps_by_method = self._xai_heatmaps[model_name]

            # Pick best XAI method for report grounding
            best_method = self.all_results.get('fidelity', {}).get(
                model_name, {}).get('best_method', 'gradcam_plus')
            best_heatmaps = heatmaps_by_method.get(
                best_method, heatmaps_by_method.get('gradcam_plus', []))

            n = min(self.cfg.get('report_samples', 10), len(images))
            samples = []
            for i in range(n):
                hm = best_heatmaps[i] if i < len(best_heatmaps) else None
                xai_summary = build_xai_summary(
                    hm if hm is not None else np.zeros((224, 224)),
                    best_method,
                    self.all_results.get('fidelity', {}).get(
                        model_name, {}).get('per_method', {}).get(best_method)
                )
                samples.append({
                    'sample_id': f"{model_name}_sample_{i:03d}",
                    'classification': {
                        'predicted_class': preds[i],
                        'confidence': float(probs[i][preds[i]]),
                        'model_name': model_name,
                    },
                    'xai_summary': xai_summary,
                })

            save_sub = str(self.results_dir / 'reports' / model_name)
            report_results = report_gen.batch_generate(
                samples, self.class_names, save_dir=save_sub)

            self.all_results['reports'][model_name] = {
                'count': len(report_results),
                'sources': list({r['source'] for r in report_results}),
            }
            print(f"  {model_name}: {len(report_results)} reports generated "
                  f"(source: {self.all_results['reports'][model_name]['sources']})")

            # Store for metric evaluation
            self._report_results = getattr(self, '_report_results', {})
            self._report_results[model_name] = {
                'reports': [r['report'] for r in report_results],
                'true_classes': labels[:n],
                'xai_summaries': [s['xai_summary'] for s in samples],
            }

    # ---------------------------------------------------------------- #
    # Stage 8: Report quality metrics                                  #
    # ---------------------------------------------------------------- #
    def evaluate_report_quality(self):
        print("\n[Stage 8] Report quality evaluation...")
        self.all_results['report_quality'] = {}

        for model_name, data in getattr(self, '_report_results', {}).items():
            metrics = evaluate_reports(
                hypotheses=data['reports'],
                references=None,          # No reference reports available
                true_classes=data['true_classes'],
                class_names=self.class_names,
                xai_summaries=data['xai_summaries'],
            )
            self.all_results['report_quality'][model_name] = metrics
            print(f"  {model_name}: hallucination_rate={metrics.get('hallucination_rate', 0):.3f}  "
                  f"grounding_score={metrics.get('grounding_score', 0):.3f}")

    # ---------------------------------------------------------------- #
    # Stage 9: Comparative summary                                     #
    # ---------------------------------------------------------------- #
    def generate_summary(self):
        print("\n[Stage 9] Generating comparative summary...")
        summary = {
            'timestamp': datetime.now().isoformat(),
            'task': self.task,
            'class_names': self.class_names,
            'models_evaluated': list(self.models.keys()),
            'results': self.all_results,
        }

        # Model comparison table
        rows = []
        for model_name in self.models:
            cls_m = self.all_results.get('classification', {}).get(model_name, {})
            fid_m = self.all_results.get('fidelity', {}).get(model_name, {})
            rq_m  = self.all_results.get('report_quality', {}).get(model_name, {})
            rows.append({
                'Model':            model_name,
                'Accuracy':         round(cls_m.get('accuracy', 0), 4),
                'F1':               round(cls_m.get('f1', 0), 4),
                'ROC-AUC':          round(cls_m.get('roc_auc', 0), 4),
                'Best XAI Method':  fid_m.get('best_method', 'N/A'),
                'Hallucination %':  round(rq_m.get('hallucination_rate', 0) * 100, 1),
                'Grounding Score':  round(rq_m.get('grounding_score', 0), 3),
            })

        df = pd.DataFrame(rows)
        print("\n" + "=" * 70)
        print("COMPARATIVE SUMMARY")
        print("=" * 70)
        print(df.to_string(index=False))
        print("=" * 70)

        # Save
        df.to_csv(self.results_dir / 'comparative_summary.csv', index=False)
        with open(self.results_dir / 'full_results.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n[Pipeline] All results saved to: {self.results_dir}")

    # ---------------------------------------------------------------- #
    # Run all stages                                                    #
    # ---------------------------------------------------------------- #
    def run(self):
        t0 = time.time()
        self.setup_data()
        self.setup_models()
        self.evaluate_classification()
        self.generate_explanations()
        self.evaluate_fidelity()
        self.evaluate_xai_quality()
        self.generate_reports()
        self.evaluate_report_quality()
        self.generate_summary()
        elapsed = time.time() - t0
        print(f"\n[Pipeline] Completed in {elapsed/60:.1f} minutes.")


# ------------------------------------------------------------------ #
# CLI entry point                                                      #
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(
        description='Breast Cancer Histopathology XAI Pipeline')
    p.add_argument('--data_root', type=str, default='datasets/BreaKHis 400X',
                   help='Path to BreaKHis dataset root')
    p.add_argument('--task', choices=['binary', 'multiclass'], default='binary')
    p.add_argument('--models', nargs='+', default=['convnext', 'swin'],
                   choices=['convnext', 'swin', 'denlsnet'])
    p.add_argument('--convnext_checkpoint', type=str, default=None)
    p.add_argument('--swin_checkpoint',     type=str, default=None)
    p.add_argument('--denlsnet_checkpoint', type=str, default=None)
    p.add_argument('--magnification', type=str, default='400X')
    p.add_argument('--batch_size',    type=int, default=16)
    p.add_argument('--eval_samples',  type=int, default=200,
                   help='Max test samples for classification eval')
    p.add_argument('--xai_samples',   type=int, default=20,
                   help='Samples for XAI heatmap generation')
    p.add_argument('--fidelity_samples', type=int, default=10,
                   help='Samples for Insertion/Deletion AUC')
    p.add_argument('--fidelity_steps',   type=int, default=30,
                   help='Steps for Insertion/Deletion curves')
    p.add_argument('--quality_samples',  type=int, default=10)
    p.add_argument('--report_samples',   type=int, default=10)
    p.add_argument('--llm_model', choices=['medgemma', 'llama3', 'rule_based'],
                   default='rule_based',
                   help='LLM for report generation')
    p.add_argument('--results_dir', type=str, default='pipeline_results')
    p.add_argument('--num_workers', type=int, default=0)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = vars(args)
    pipeline = BreastCancerPipeline(cfg)
    pipeline.run()
