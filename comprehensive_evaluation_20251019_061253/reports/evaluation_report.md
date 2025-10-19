# Model Evaluation Report

## Analysis Overview
- **Timestamp**: 2025-10-19T06:13:56.839638
- **Classes**: Benign, Malignant
- **Total Samples**: 545

## Performance Metrics

### Overall Performance
- **Accuracy**: 0.8404
- **Precision**: 0.8398
- **Recall**: 0.8404
- **F1-Score**: 0.8337

### Binary Classification Metrics
- **Sensitivity (True Positive Rate)**: 0.9404
- **Specificity (True Negative Rate)**: 0.6307
- **ROC-AUC**: 0.8474
- **Positive Predictive Value**: 0.8422
- **Negative Predictive Value**: 0.8346

### Per-Class Performance

#### Benign
- **Precision**: 0.8346
- **Recall**: 0.6307
- **F1-Score**: 0.7184

#### Malignant
- **Precision**: 0.8422
- **Recall**: 0.9404
- **F1-Score**: 0.8886

## Data Distribution

### True Class Distribution
- **Benign**: 176 samples (32.3%)
- **Malignant**: 369 samples (67.7%)

### Predicted Class Distribution
- **Benign**: 133 predictions (24.4%)
- **Malignant**: 412 predictions (75.6%)

## Confidence Analysis
- **Mean Confidence**: 0.6820
- **Std Confidence**: 0.0735
- **Min Confidence**: 0.5005
- **Max Confidence**: 0.7979
- **Median Confidence**: 0.6943

## Files Generated
- `evaluation_results.json`: Complete results in JSON format
- `metrics_summary.csv`: Summary metrics in CSV format
- `confusion_matrix.png`: Confusion matrix visualization
- `class_distribution.png`: Class distribution comparison
- `confidence_analysis.png`: Confidence distribution analysis
- `metrics_summary.png`: Performance metrics summary
- `roc_curve.png`: ROC curve
- `precision_recall_curve.png`: Precision-Recall curve
- `interactive_roc_curve.html`: Interactive ROC curve
- `interactive_confusion_matrix.html`: Interactive confusion matrix

## Recommendations
- Good model performance. Consider additional training or data augmentation.
- Low specificity detected. Model may have false positives.
- Low average confidence. Model may be uncertain about predictions.
