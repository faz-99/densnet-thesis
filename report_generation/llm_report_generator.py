"""
Clinical-style report generation using LLMs.

Supports:
  - MedGemma (google/medgemma-4b-it) via HuggingFace
  - Llama-3-Medical (johnsnowlabs/JSL-MedLlama-3-8B-v2.0 or similar)
  - Fallback: rule-based template report

The generator takes:
  - Classification result (class, confidence)
  - XAI heatmap summary (dominant region, sparsity, method used)
  - Patient/image metadata (optional)

And produces a structured clinical report.
"""
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


# ------------------------------------------------------------------ #
# Prompt builder                                                       #
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = (
    "You are an expert pathologist AI assistant. "
    "Generate a concise, structured clinical report for a breast histopathology image. "
    "Base your report strictly on the provided classification result and XAI findings. "
    "Do not hallucinate clinical details not supported by the input."
)

def build_prompt(classification: Dict, xai_summary: Dict,
                 class_names: List[str]) -> str:
    """Build the user prompt for the LLM."""
    pred_class = class_names[classification['predicted_class']]
    confidence = classification['confidence']
    xai_method = xai_summary.get('method', 'Grad-CAM++')
    region_desc = xai_summary.get('region_description', 'focal tissue region')
    sparsity = xai_summary.get('gini', 0.0)

    prompt = f"""
Histopathology Classification Report Request

Classification Result:
- Predicted class: {pred_class}
- Confidence: {confidence:.1%}
- Model: {classification.get('model_name', 'ConvNeXt/Swin')}

XAI Explanation Summary ({xai_method}):
- Highlighted region: {region_desc}
- Explanation focus (Gini): {sparsity:.3f} (higher = more focused)
- Insertion AUC: {xai_summary.get('insertion_auc', 'N/A')}
- Deletion AUC: {xai_summary.get('deletion_auc', 'N/A')}

Please generate a structured clinical report with the following sections:
1. Diagnosis
2. Histological Findings
3. XAI Interpretation
4. Confidence Assessment
5. Recommendation
"""
    return prompt.strip()


# ------------------------------------------------------------------ #
# LLM backends                                                         #
# ------------------------------------------------------------------ #

class MedGemmaReportGenerator:
    """Uses google/medgemma-4b-it via HuggingFace transformers."""

    MODEL_ID = "google/medgemma-4b-it"

    def __init__(self, device: str = 'cpu', max_new_tokens: int = 512):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-generation",
                model=self.MODEL_ID,
                device=0 if 'cuda' in self.device else -1,
                max_new_tokens=self.max_new_tokens,
                trust_remote_code=True,
            )
            print(f"[Report] MedGemma loaded on {self.device}")
        except Exception as e:
            print(f"[Report] MedGemma load failed: {e}")
            self._pipeline = None

    def generate(self, prompt: str) -> str:
        self._load()
        if self._pipeline is None:
            return ""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        try:
            out = self._pipeline(messages)
            return out[0]['generated_text'][-1]['content']
        except Exception as e:
            print(f"[Report] MedGemma generation failed: {e}")
            return ""


class Llama3MedicalReportGenerator:
    """Uses a Llama-3-Medical model via HuggingFace transformers."""

    MODEL_ID = "johnsnowlabs/JSL-MedLlama-3-8B-v2.0"

    def __init__(self, device: str = 'cpu', max_new_tokens: int = 512):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-generation",
                model=self.MODEL_ID,
                device=0 if 'cuda' in self.device else -1,
                max_new_tokens=self.max_new_tokens,
                trust_remote_code=True,
            )
            print(f"[Report] Llama-3-Medical loaded on {self.device}")
        except Exception as e:
            print(f"[Report] Llama-3-Medical load failed: {e}")
            self._pipeline = None

    def generate(self, prompt: str) -> str:
        self._load()
        if self._pipeline is None:
            return ""
        full_prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{prompt}\n<|assistant|>\n"
        try:
            out = self._pipeline(full_prompt, do_sample=False)
            text = out[0]['generated_text']
            # Strip the prompt prefix
            if '<|assistant|>' in text:
                text = text.split('<|assistant|>')[-1].strip()
            return text
        except Exception as e:
            print(f"[Report] Llama-3-Medical generation failed: {e}")
            return ""


# ------------------------------------------------------------------ #
# Rule-based fallback                                                  #
# ------------------------------------------------------------------ #

def _rule_based_report(classification: Dict, xai_summary: Dict,
                       class_names: List[str]) -> str:
    pred_class = class_names[classification['predicted_class']]
    confidence = classification['confidence']
    is_malignant = 'malignant' in pred_class.lower() or classification['predicted_class'] == 1

    report = f"""BREAST HISTOPATHOLOGY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Model: {classification.get('model_name', 'DL Model')}

1. DIAGNOSIS
   Predicted: {pred_class}
   Confidence: {confidence:.1%}
   {'⚠ Malignant tissue detected.' if is_malignant else '✓ Benign tissue detected.'}

2. HISTOLOGICAL FINDINGS
   The model identified features consistent with {pred_class}.
   {'Irregular nuclear morphology and high mitotic activity may be present.' if is_malignant
    else 'Regular glandular architecture and uniform nuclear morphology observed.'}

3. XAI INTERPRETATION ({xai_summary.get('method', 'Grad-CAM++')})
   - Highlighted region: {xai_summary.get('region_description', 'focal tissue area')}
   - Explanation focus (Gini): {xai_summary.get('gini', 0.0):.3f}
   - Insertion AUC: {xai_summary.get('insertion_auc', 'N/A')}
   - Deletion AUC: {xai_summary.get('deletion_auc', 'N/A')}
   The explanation is {'highly focused' if xai_summary.get('gini', 0) > 0.6 else 'moderately distributed'}.

4. CONFIDENCE ASSESSMENT
   {'High confidence (>80%) — reliable classification.' if confidence > 0.8
    else 'Moderate confidence (60-80%) — expert review recommended.' if confidence > 0.6
    else 'Low confidence (<60%) — manual pathologist review required.'}

5. RECOMMENDATION
   {'Immediate oncology referral recommended.' if is_malignant and confidence > 0.8
    else 'Follow-up biopsy and expert pathologist review recommended.' if is_malignant
    else 'Routine follow-up as per clinical protocol.'}

NOTE: This report is AI-generated and must be reviewed by a qualified pathologist.
"""
    return report


# ------------------------------------------------------------------ #
# Unified generator                                                    #
# ------------------------------------------------------------------ #

class ClinicalReportGenerator:
    """
    Unified clinical report generator.
    Tries MedGemma → Llama-3-Medical → rule-based fallback.
    """

    def __init__(self, device: str = 'cpu',
                 preferred_model: str = 'medgemma',
                 max_new_tokens: int = 512):
        self.device = device
        self.preferred_model = preferred_model
        self._medgemma = None
        self._llama3 = None

        if preferred_model == 'medgemma':
            self._medgemma = MedGemmaReportGenerator(device, max_new_tokens)
        elif preferred_model == 'llama3':
            self._llama3 = Llama3MedicalReportGenerator(device, max_new_tokens)

    def generate(self, classification: Dict, xai_summary: Dict,
                 class_names: List[str]) -> Tuple[str, str]:
        """
        Generate a clinical report.

        Returns:
            (report_text, source) where source is 'medgemma'|'llama3'|'rule_based'
        """
        prompt = build_prompt(classification, xai_summary, class_names)

        # Try preferred LLM
        if self._medgemma:
            text = self._medgemma.generate(prompt)
            if text:
                return text, 'medgemma'

        if self._llama3:
            text = self._llama3.generate(prompt)
            if text:
                return text, 'llama3'

        # Fallback
        return _rule_based_report(classification, xai_summary, class_names), 'rule_based'

    def batch_generate(self, samples: List[Dict],
                       class_names: List[str],
                       save_dir: Optional[str] = None) -> List[Dict]:
        """
        Generate reports for a list of samples.

        Each sample dict must have 'classification' and 'xai_summary' keys.
        """
        results = []
        for i, sample in enumerate(samples):
            report, source = self.generate(
                sample['classification'], sample['xai_summary'], class_names)
            result = {
                'sample_id': sample.get('sample_id', f'sample_{i}'),
                'report': report,
                'source': source,
                'classification': sample['classification'],
            }
            results.append(result)

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            for r in results:
                fname = Path(save_dir) / f"{r['sample_id']}_report.txt"
                fname.write_text(r['report'])

        return results
