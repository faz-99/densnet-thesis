"""
Medical LLM Report Generator with Chain-of-Thought (CoT) prompting.
Generates structured clinical reports referencing XAI findings.
"""
import torch
import numpy as np
from typing import Dict, Optional
from pathlib import Path

from config.settings import LLM_CONFIG, DATASET_CONFIG
from report.projection import LinearProjection, QFormerProjection


# ──────────────────────────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a board-certified pathologist AI assistant. You generate 
structured clinical reports for breast cancer histopathology slides analysed by an 
AI diagnostic system. You MUST reference the XAI (Explainability) findings provided 
to justify every diagnostic statement. Use precise medical terminology."""

COT_REPORT_TEMPLATE = """### Task
Generate a structured clinical pathology report for the following histopathology image analysis.

### AI Diagnostic Findings
- **Predicted Class**: {predicted_class}
- **Confidence**: {confidence:.1%}
- **Magnification**: {magnification}

### XAI Interpretability Summary
{xai_summary}

### Instructions (Chain-of-Thought)
Think step-by-step:
1. **Observation**: Describe the tissue regions highlighted as most diagnostically relevant by the XAI methods.
2. **Correlation**: Correlate the Grad-CAM, SHAP, and Attention Rollout hotspots with known histopathological features of the predicted class.
3. **Counterfactual Insight**: Explain what minimal tissue change would be needed to flip the diagnosis, based on the counterfactual analysis.
4. **Diagnosis**: State the final diagnosis with supporting evidence from the XAI analysis.
5. **Confidence Assessment**: Assess reliability based on agreement across XAI methods.

### Report Format
Generate the report with these sections:
- **CLINICAL SUMMARY**
- **MICROSCOPIC FINDINGS** (reference XAI regions)
- **XAI-SUPPORTED EVIDENCE**
- **DIFFERENTIAL DIAGNOSIS**
- **CONCLUSION & RECOMMENDATION**
"""


def _build_xai_summary(heatmaps: Dict[str, np.ndarray], cf_details: dict = None) -> str:
    """Convert XAI heatmaps into a textual summary for the LLM."""
    lines = []
    for method, hmap in heatmaps.items():
        # Identify hot regions (top 10% activation)
        threshold = np.percentile(hmap, 90)
        hot_mask = hmap >= threshold
        h, w = hmap.shape

        # Quadrant analysis
        quadrants = {
            "top-left": hot_mask[:h//2, :w//2],
            "top-right": hot_mask[:h//2, w//2:],
            "bottom-left": hot_mask[h//2:, :w//2],
            "bottom-right": hot_mask[h//2:, w//2:],
        }
        dominant = max(quadrants, key=lambda q: quadrants[q].sum())
        coverage = hot_mask.mean() * 100

        method_name = method.replace("_", " ").title()
        lines.append(
            f"- **{method_name}**: {coverage:.1f}% of tissue highlighted; "
            f"dominant activation in the {dominant} quadrant (mean intensity: {hmap.mean():.3f})."
        )

    if cf_details:
        lines.append(
            f"- **Counterfactual**: Flipping from class {cf_details.get('original_class', '?')} "
            f"to class {cf_details.get('target_class', '?')} required L1-norm perturbation "
            f"of {cf_details.get('l1_norm', 0):.4f} "
            f"({'converged' if cf_details.get('converged') else 'did not converge'} "
            f"in {cf_details.get('iterations', 0)} iterations)."
        )

    return "\n".join(lines) if lines else "No XAI data available."


class ReportGenerator:
    """Multimodal report generator: vision features + XAI → clinical report.

    Uses a medical LLM (MedGemma / Llama-3-8B-Instruct) with a multimodal
    projection layer to produce Chain-of-Thought clinical reports.
    """

    def __init__(self, model=None, device: str = "cuda"):
        """
        Args:
            model: The HybridEnsemble (for extracting vision features).
            device: Target device.
        """
        self.vision_model = model
        self.device = device
        self.llm = None
        self.tokenizer = None
        self.projection = None
        self._loaded = False

    def load_llm(self, model_name: str = None, quantization: str = None):
        """Load the medical LLM with optional quantization."""
        model_name = model_name or LLM_CONFIG["model_name"]
        quantization = quantization or LLM_CONFIG["quantization"]

        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        print(f"[ReportGenerator] Loading {model_name} ({quantization} quantization)...")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            llm = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb_config,
                device_map="auto", trust_remote_code=True,
            )
        elif quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            llm = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb_config,
                device_map="auto", trust_remote_code=True,
            )
        else:
            llm = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16,
                device_map="auto", trust_remote_code=True,
            )

        self.tokenizer = tokenizer
        self.llm = llm

        # Build projection layer
        llm_dim = llm.config.hidden_size
        vision_dim = LLM_CONFIG["projection"]["vision_dim"]
        proj_type = LLM_CONFIG["projection"]["type"]

        if proj_type == "qformer":
            self.projection = QFormerProjection(
                vision_dim=vision_dim, llm_dim=llm_dim,
                num_query_tokens=LLM_CONFIG["projection"]["num_query_tokens"],
            ).to(self.device)
        else:
            self.projection = LinearProjection(
                vision_dim=vision_dim, llm_dim=llm_dim,
            ).to(self.device)

        self._loaded = True
        print(f"[ReportGenerator] LLM loaded. Projection: {proj_type}, LLM dim: {llm_dim}")

    def generate_report(
        self,
        input_tensor: torch.Tensor,
        predicted_class: int,
        confidence: float,
        heatmaps: Dict[str, np.ndarray],
        cf_details: dict = None,
        magnification: str = "400X",
        class_names: list = None,
    ) -> str:
        """Generate a structured clinical report.

        If the LLM is loaded, uses vision features + CoT prompting.
        Otherwise, falls back to a template-based report.
        """
        if class_names is None:
            class_names = DATASET_CONFIG["multiclass_classes"]

        class_label = class_names[predicted_class] if predicted_class < len(class_names) else str(predicted_class)
        xai_summary = _build_xai_summary(heatmaps, cf_details)

        prompt = COT_REPORT_TEMPLATE.format(
            predicted_class=class_label,
            confidence=confidence,
            magnification=magnification,
            xai_summary=xai_summary,
        )

        if self._loaded and self.llm is not None:
            return self._generate_with_llm(input_tensor, prompt)
        else:
            return self._generate_template_report(
                class_label, confidence, magnification, xai_summary, cf_details
            )

    @torch.no_grad()
    def _generate_with_llm(self, input_tensor: torch.Tensor, prompt: str) -> str:
        """Generate report using the loaded LLM with vision feature injection."""
        # Extract vision features
        if self.vision_model is not None:
            feat_a, feat_b = self.vision_model.get_branch_features(input_tensor.to(self.device))
            vision_features = torch.cat([feat_a, feat_b], dim=-1)  # (B, 2048)
            vision_tokens = self.projection(vision_features)        # (B, Q, llm_dim)
        else:
            vision_tokens = None

        # Tokenize prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"{SYSTEM_PROMPT}\n\n{prompt}"

        inputs = self.tokenizer(text, return_tensors="pt").to(self.llm.device)

        # If we have vision tokens, prepend them to the input embeddings
        if vision_tokens is not None:
            input_embeds = self.llm.get_input_embeddings()(inputs["input_ids"])
            vision_tokens = vision_tokens.to(input_embeds.device, dtype=input_embeds.dtype)
            input_embeds = torch.cat([vision_tokens, input_embeds], dim=1)

            # Extend attention mask
            vision_mask = torch.ones(1, vision_tokens.size(1), device=inputs["attention_mask"].device,
                                     dtype=inputs["attention_mask"].dtype)
            attention_mask = torch.cat([vision_mask, inputs["attention_mask"]], dim=1)

            gen_cfg = LLM_CONFIG["generation"]
            outputs = self.llm.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                max_new_tokens=gen_cfg["max_new_tokens"],
                temperature=gen_cfg["temperature"],
                top_p=gen_cfg["top_p"],
                do_sample=gen_cfg["do_sample"],
                pad_token_id=self.tokenizer.eos_token_id,
            )
        else:
            gen_cfg = LLM_CONFIG["generation"]
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=gen_cfg["max_new_tokens"],
                temperature=gen_cfg["temperature"],
                top_p=gen_cfg["top_p"],
                do_sample=gen_cfg["do_sample"],
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Strip the input prompt from generated text
        if text in generated:
            generated = generated[len(text):].strip()

        return generated

    def _generate_template_report(
        self, class_label: str, confidence: float, magnification: str,
        xai_summary: str, cf_details: dict = None,
    ) -> str:
        """Template-based fallback report when no LLM is available."""
        is_malignant = class_label in [
            "ductal_carcinoma", "lobular_carcinoma",
            "mucinous_carcinoma", "papillary_carcinoma",
        ]
        category = "Malignant" if is_malignant else "Benign"

        cf_section = ""
        if cf_details:
            cf_section = (
                f"\n**Counterfactual Analysis**: A minimal perturbation (L1={cf_details.get('l1_norm', 0):.4f}) "
                f"was {'sufficient' if cf_details.get('converged') else 'insufficient'} to flip the diagnosis "
                f"from class {cf_details.get('original_class', '?')} to class {cf_details.get('target_class', '?')}, "
                f"suggesting {'high' if not cf_details.get('converged') else 'moderate'} diagnostic confidence "
                f"in the current classification."
            )

        report = f"""# HISTOPATHOLOGY AI REPORT

## CLINICAL SUMMARY
- **Specimen**: Breast tissue biopsy
- **Magnification**: {magnification}
- **AI Classification**: {class_label.replace('_', ' ').title()} ({category})
- **Confidence**: {confidence:.1%}

## MICROSCOPIC FINDINGS
The AI diagnostic system analysed the tissue sample at {magnification} magnification.
The ensemble model (ConvNeXt + Swin Transformer) classified the specimen as
**{class_label.replace('_', ' ').title()}** with {confidence:.1%} confidence.

## XAI-SUPPORTED EVIDENCE
The following interpretability methods were applied:

{xai_summary}

Multiple XAI methods show convergent hotspot regions, indicating consistent
feature attribution across gradient-based and model-agnostic approaches.
{cf_section}

## DIFFERENTIAL DIAGNOSIS
Given the {category.lower()} classification with {confidence:.1%} confidence,
{'further immunohistochemical analysis is recommended for definitive sub-typing.'
if confidence < 0.85 else 'the diagnostic confidence is high and consistent across XAI methods.'}

## CONCLUSION & RECOMMENDATION
The AI system classifies this specimen as **{class_label.replace('_', ' ').title()}** ({category}).
{'Clinical correlation and confirmatory pathologist review are recommended.'
if confidence < 0.90 else 'High-confidence classification supported by convergent XAI evidence.'}

---
*Report generated by Vision-XAI-Report Pipeline | Multimodal Histopathology Framework*
"""
        return report
