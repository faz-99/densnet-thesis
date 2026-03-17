"""
Report quality evaluation metrics.

Computes:
  - BLEU (1-4 gram)
  - ROUGE (1, 2, L)
  - BERTScore
  - Hallucination rate (unsupported clinical claims)
  - Grounding score (text-XAI alignment)
"""
import re
import numpy as np
from typing import Dict, List, Optional, Tuple


# ------------------------------------------------------------------ #
# BLEU                                                                 #
# ------------------------------------------------------------------ #

def compute_bleu(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """Compute corpus-level BLEU-1 through BLEU-4."""
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        smooth = SmoothingFunction().method1
        ref_tokens  = [[r.lower().split()] for r in references]
        hyp_tokens  = [h.lower().split() for h in hypotheses]

        return {
            'bleu_1': corpus_bleu(ref_tokens, hyp_tokens, weights=(1,0,0,0), smoothing_function=smooth),
            'bleu_2': corpus_bleu(ref_tokens, hyp_tokens, weights=(.5,.5,0,0), smoothing_function=smooth),
            'bleu_3': corpus_bleu(ref_tokens, hyp_tokens, weights=(.33,.33,.33,0), smoothing_function=smooth),
            'bleu_4': corpus_bleu(ref_tokens, hyp_tokens, weights=(.25,.25,.25,.25), smoothing_function=smooth),
        }
    except ImportError:
        print("[Metrics] nltk not available; BLEU skipped.")
        return {'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}


# ------------------------------------------------------------------ #
# ROUGE                                                                #
# ------------------------------------------------------------------ #

def compute_rouge(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        r1, r2, rl = [], [], []
        for ref, hyp in zip(references, hypotheses):
            scores = scorer.score(ref, hyp)
            r1.append(scores['rouge1'].fmeasure)
            r2.append(scores['rouge2'].fmeasure)
            rl.append(scores['rougeL'].fmeasure)
        return {
            'rouge_1': float(np.mean(r1)),
            'rouge_2': float(np.mean(r2)),
            'rouge_L': float(np.mean(rl)),
        }
    except ImportError:
        print("[Metrics] rouge_score not available; ROUGE skipped.")
        return {'rouge_1': 0.0, 'rouge_2': 0.0, 'rouge_L': 0.0}


# ------------------------------------------------------------------ #
# BERTScore                                                            #
# ------------------------------------------------------------------ #

def compute_bertscore(references: List[str], hypotheses: List[str],
                      lang: str = 'en') -> Dict[str, float]:
    """Compute BERTScore P/R/F1."""
    try:
        from bert_score import score as bert_score
        P, R, F1 = bert_score(hypotheses, references, lang=lang, verbose=False)
        return {
            'bertscore_precision': float(P.mean()),
            'bertscore_recall':    float(R.mean()),
            'bertscore_f1':        float(F1.mean()),
        }
    except ImportError:
        print("[Metrics] bert_score not available; BERTScore skipped.")
        return {'bertscore_precision': 0.0, 'bertscore_recall': 0.0, 'bertscore_f1': 0.0}


# ------------------------------------------------------------------ #
# Hallucination rate                                                   #
# ------------------------------------------------------------------ #

# Clinical claims that must be grounded in the classification result
_MALIGNANT_CLAIMS = [
    r'malignant', r'carcinoma', r'cancer', r'tumor', r'invasive',
    r'metastasis', r'oncology referral', r'biopsy recommended',
]
_BENIGN_CLAIMS = [
    r'benign', r'adenosis', r'fibroadenoma', r'normal tissue',
    r'no malignancy', r'routine follow-up',
]


def compute_hallucination_rate(reports: List[str],
                               true_classes: List[int],
                               class_names: List[str]) -> float:
    """
    Estimate hallucination rate as fraction of reports containing
    claims inconsistent with the ground-truth class.

    Simplified heuristic: checks for malignant/benign keyword mismatches.
    """
    hallucinated = 0
    for report, true_cls in zip(reports, true_classes):
        report_lower = report.lower()
        is_malignant_gt = 'malignant' in class_names[true_cls].lower() or true_cls == 1

        if is_malignant_gt:
            # Report should NOT contain strong benign-only claims
            for pattern in _BENIGN_CLAIMS:
                if re.search(pattern, report_lower):
                    hallucinated += 1
                    break
        else:
            # Report should NOT contain strong malignant-only claims
            for pattern in _MALIGNANT_CLAIMS:
                if re.search(pattern, report_lower):
                    hallucinated += 1
                    break

    return hallucinated / len(reports) if reports else 0.0


# ------------------------------------------------------------------ #
# Grounding score                                                      #
# ------------------------------------------------------------------ #

def compute_grounding_score(reports: List[str],
                             xai_summaries: List[Dict]) -> float:
    """
    Measure alignment between report text and XAI-highlighted regions.

    Heuristic: checks whether the report mentions the dominant tissue
    region identified by the XAI method.
    """
    grounded = 0
    for report, xai in zip(reports, xai_summaries):
        region = xai.get('region_description', '').lower()
        if not region:
            grounded += 1  # No XAI info → can't penalise
            continue
        # Check if any keyword from region description appears in report
        keywords = [w for w in region.split() if len(w) > 3]
        if any(kw in report.lower() for kw in keywords):
            grounded += 1
    return grounded / len(reports) if reports else 0.0


# ------------------------------------------------------------------ #
# Unified evaluator                                                    #
# ------------------------------------------------------------------ #

def evaluate_reports(hypotheses: List[str],
                     references: Optional[List[str]],
                     true_classes: List[int],
                     class_names: List[str],
                     xai_summaries: Optional[List[Dict]] = None) -> Dict:
    """
    Full report quality evaluation.

    Args:
        hypotheses: generated reports
        references: reference reports (optional; needed for BLEU/ROUGE/BERTScore)
        true_classes: ground-truth class indices
        class_names: list of class name strings
        xai_summaries: list of XAI summary dicts (for grounding score)

    Returns:
        metrics dict
    """
    metrics = {}

    if references:
        metrics.update(compute_bleu(references, hypotheses))
        metrics.update(compute_rouge(references, hypotheses))
        metrics.update(compute_bertscore(references, hypotheses))

    metrics['hallucination_rate'] = compute_hallucination_rate(
        hypotheses, true_classes, class_names)

    if xai_summaries:
        metrics['grounding_score'] = compute_grounding_score(
            hypotheses, xai_summaries)

    return metrics
