# Quantitative Explainability Benchmark Report

## Analysis Overview
- **Date**: 2025-12-24 04:06:53
- **Methods Evaluated**: gradcam, gradcam_plus
- **Metrics Computed**: Insertion AUC, Deletion AUC, Stability, IoU, Processing Time

## Summary Results

### Performance Ranking

#### Insertion AUC (Higher is Better)

#### Deletion AUC (Lower is Better)

#### Stability (Higher is Better)

## Detailed Results

| Method | Insertion AUC | Deletion AUC | Stability | Processing Time (s) | Success Rate |
|--------|---------------|--------------|-----------|-------------------|--------------|
| gradcam | N/A | N/A | N/A | 0.457 | 0.0% |
| gradcam_plus | N/A | N/A | N/A | 0.463 | 0.0% |

## Metric Interpretations

### Insertion AUC
- **Range**: 0.0 - 1.0
- **Interpretation**: Measures how quickly model confidence increases when adding important pixels
- **Good Score**: > 0.6 indicates faithful explanations

### Deletion AUC
- **Range**: 0.0 - 1.0  
- **Interpretation**: Measures how quickly model confidence decreases when removing important pixels
- **Good Score**: < 0.4 indicates explanations identify truly important regions

### Stability Score
- **Range**: -1.0 - 1.0
- **Interpretation**: Correlation between explanations under input perturbations
- **Good Score**: > 0.7 indicates robust explanations

### Processing Time
- **Interpretation**: Computational efficiency of explanation generation
- **Consideration**: Balance between quality and speed for practical applications

## Recommendations

Based on the quantitative evaluation:

- Use methods with high insertion AUC and low deletion AUC for faithful explanations
- Prioritize methods with high stability for consistent clinical interpretation
- Consider processing time constraints for real-time applications
- Validate explanations with domain experts regardless of quantitative scores

## Files Generated
- `quantitative_benchmark_results.json`: Complete numerical results
- `explainability_summary.csv`: Summary table in CSV format
- `quantitative_comparison_chart.png`: Performance comparison visualization
- `metric_distributions.png`: Distribution analysis plots

## Usage Notes
- Results are averaged across multiple samples for statistical reliability
- Error bars represent standard deviation across samples
- Success rate indicates percentage of samples successfully processed
- Consider both quantitative metrics and qualitative clinical relevance
