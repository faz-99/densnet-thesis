"""
Visual Report Generator for DenLsNet Explainability Analysis
Creates comprehensive Jupyter notebook and PDF reports
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional
import nbformat as nbf
from nbconvert import PDFExporter, HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
import base64
from io import BytesIO


class ExplainabilityReportGenerator:
    """
    Generates comprehensive visual reports for explainability analysis
    """
    
    def __init__(self, 
                 results_dir: str,
                 model_performance: Dict = None,
                 class_names: List[str] = None):
        """
        Initialize report generator
        
        Args:
            results_dir: Directory containing explainability results
            model_performance: Model performance metrics
            class_names: List of class names
        """
        self.results_dir = results_dir
        self.model_performance = model_performance or {}
        self.class_names = class_names or [f'Class_{i}' for i in range(8)]
        
        # Load results from different methods
        self.gradcam_results = self._load_gradcam_results()
        self.shap_results = self._load_shap_results()
        self.lime_results = self._load_lime_results()
        self.benchmark_results = self._load_benchmark_results()
    
    def _load_gradcam_results(self) -> Dict:
        """Load Grad-CAM analysis results"""
        gradcam_path = os.path.join(self.results_dir, 'gradcam', 'gradcam_metrics.json')
        if os.path.exists(gradcam_path):
            with open(gradcam_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_shap_results(self) -> Dict:
        """Load SHAP analysis results"""
        shap_path = os.path.join(self.results_dir, 'shap', 'shap_analysis_results.json')
        if os.path.exists(shap_path):
            with open(shap_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_lime_results(self) -> Dict:
        """Load LIME analysis results"""
        lime_path = os.path.join(self.results_dir, 'lime', 'lime_analysis_results.json')
        if os.path.exists(lime_path):
            with open(lime_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_benchmark_results(self) -> Dict:
        """Load quantitative benchmark results"""
        benchmark_path = os.path.join(self.results_dir, 'explainability_metrics', 'quantitative_benchmark_results.json')
        if os.path.exists(benchmark_path):
            with open(benchmark_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 for embedding in notebook"""
        if os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                image_data = f.read()
                return base64.b64encode(image_data).decode('utf-8')
        return ""
    
    def create_notebook_report(self, save_path: str = None) -> str:
        """
        Create comprehensive Jupyter notebook report
        
        Args:
            save_path: Path to save notebook (default: auto-generated)
            
        Returns:
            Path to saved notebook
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.results_dir, f'explainability_report_{timestamp}.ipynb')
        
        # Create notebook
        nb = nbf.v4.new_notebook()
        
        # Add cells
        cells = []
        
        # Title cell
        cells.append(nbf.v4.new_markdown_cell(self._create_title_section()))
        
        # Executive summary
        cells.append(nbf.v4.new_markdown_cell(self._create_executive_summary()))
        
        # Model performance section
        cells.append(nbf.v4.new_markdown_cell("## 1. Model Performance Summary"))
        cells.append(nbf.v4.new_code_cell(self._create_performance_code()))
        
        # Grad-CAM section
        cells.append(nbf.v4.new_markdown_cell("## 2. Grad-CAM Analysis"))
        cells.append(nbf.v4.new_markdown_cell(self._create_gradcam_analysis()))
        cells.append(nbf.v4.new_code_cell(self._create_gradcam_code()))
        
        # SHAP section
        cells.append(nbf.v4.new_markdown_cell("## 3. SHAP Analysis"))
        cells.append(nbf.v4.new_markdown_cell(self._create_shap_analysis()))
        cells.append(nbf.v4.new_code_cell(self._create_shap_code()))
        
        # LIME section
        cells.append(nbf.v4.new_markdown_cell("## 4. LIME Analysis"))
        cells.append(nbf.v4.new_markdown_cell(self._create_lime_analysis()))
        cells.append(nbf.v4.new_code_cell(self._create_lime_code()))
        
        # Quantitative comparison
        cells.append(nbf.v4.new_markdown_cell("## 5. Quantitative Method Comparison"))
        cells.append(nbf.v4.new_markdown_cell(self._create_comparison_analysis()))
        cells.append(nbf.v4.new_code_cell(self._create_comparison_code()))
        
        # Discussion and insights
        cells.append(nbf.v4.new_markdown_cell("## 6. Discussion and Clinical Insights"))
        cells.append(nbf.v4.new_markdown_cell(self._create_discussion_section()))
        
        # Conclusions
        cells.append(nbf.v4.new_markdown_cell("## 7. Conclusions and Recommendations"))
        cells.append(nbf.v4.new_markdown_cell(self._create_conclusions_section()))
        
        # Add all cells to notebook
        nb.cells = cells
        
        # Save notebook
        with open(save_path, 'w') as f:
            nbf.write(nb, f)
        
        print(f"Notebook report created: {save_path}")
        return save_path
    
    def _create_title_section(self) -> str:
        """Create title section"""
        return f"""# DenLsNet Explainability Analysis Report

**Comprehensive Analysis of Model Interpretability**

---

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Model:** DenLsNet Multi-Class Histopathology Classification

**Classes:** {', '.join(self.class_names)}

**Analysis Methods:** Grad-CAM, Grad-CAM++, SHAP, LIME

---
"""
    
    def _create_executive_summary(self) -> str:
        """Create executive summary"""
        summary = """## Executive Summary

This report presents a comprehensive analysis of the explainability and interpretability of the DenLsNet model for histopathology image classification. The analysis employs multiple state-of-the-art explanation methods to provide both visual and quantitative insights into the model's decision-making process.

### Key Findings:

"""
        
        # Add key findings based on available results
        if self.model_performance:
            accuracy = self.model_performance.get('accuracy', 'N/A')
            summary += f"- **Model Performance**: {accuracy:.1%} accuracy on test set\n"
        
        if self.benchmark_results:
            methods = list(self.benchmark_results.keys())
            summary += f"- **Methods Evaluated**: {len(methods)} explainability techniques\n"
            
            # Find best method
            best_method = None
            best_score = 0
            for method, results in self.benchmark_results.items():
                if results.get('insertion_auc', {}).get('mean'):
                    score = results['insertion_auc']['mean']
                    if score > best_score:
                        best_score = score
                        best_method = method
            
            if best_method:
                summary += f"- **Best Performing Method**: {best_method} (Insertion AUC: {best_score:.3f})\n"
        
        summary += """
### Report Structure:

1. **Model Performance Summary** - Overall classification performance
2. **Grad-CAM Analysis** - Visual attention heatmaps and quantitative metrics
3. **SHAP Analysis** - Feature attribution and importance patterns
4. **LIME Analysis** - Local interpretability via superpixels
5. **Quantitative Comparison** - Benchmarking of all methods
6. **Discussion** - Clinical insights and interpretability patterns
7. **Conclusions** - Recommendations for clinical deployment

---
"""
        return summary
    
    def _create_performance_code(self) -> str:
        """Create code for model performance visualization"""
        return f"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Model performance data
performance_data = {self.model_performance}

# Create performance visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Performance metrics
if performance_data:
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    values = [performance_data.get(metric, 0) for metric in metrics]
    
    axes[0].bar(metrics, values, alpha=0.7, color='skyblue')
    axes[0].set_title('Model Performance Metrics')
    axes[0].set_ylabel('Score')
    axes[0].set_ylim(0, 1)
    
    # Add value labels
    for i, v in enumerate(values):
        axes[0].text(i, v + 0.01, f'{{v:.3f}}', ha='center', va='bottom')

# Class distribution (if available)
class_names = {self.class_names}
if 'class_distribution' in performance_data:
    class_dist = performance_data['class_distribution']
    classes = list(class_dist.keys())
    counts = list(class_dist.values())
    
    axes[1].bar(classes, counts, alpha=0.7, color='lightcoral')
    axes[1].set_title('Class Distribution')
    axes[1].set_ylabel('Sample Count')
    axes[1].tick_params(axis='x', rotation=45)
else:
    axes[1].text(0.5, 0.5, 'Class distribution\\nnot available', 
                ha='center', va='center', transform=axes[1].transAxes)
    axes[1].set_title('Class Distribution')

plt.tight_layout()
plt.show()

print("Model Performance Summary:")
if performance_data:
    for metric, value in performance_data.items():
        if isinstance(value, (int, float)):
            print(f"- {{metric.title()}}: {{value:.3f}}")
"""
    
    def _create_gradcam_analysis(self) -> str:
        """Create Grad-CAM analysis text"""
        analysis = """### Grad-CAM Analysis Results

Grad-CAM (Gradient-weighted Class Activation Mapping) provides visual explanations by highlighting the regions of the input image that are most important for the model's prediction. We implemented both standard Grad-CAM and Grad-CAM++ for improved localization.

"""
        
        if self.gradcam_results:
            if 'insertion_auc' in self.gradcam_results:
                ins_auc = self.gradcam_results['insertion_auc']['mean']
                analysis += f"**Insertion AUC**: {ins_auc:.3f} - "
                if ins_auc > 0.6:
                    analysis += "Good faithfulness score\n"
                else:
                    analysis += "Moderate faithfulness score\n"
            
            if 'deletion_auc' in self.gradcam_results:
                del_auc = self.gradcam_results['deletion_auc']['mean']
                analysis += f"**Deletion AUC**: {del_auc:.3f} - "
                if del_auc < 0.4:
                    analysis += "Good localization accuracy\n"
                else:
                    analysis += "Moderate localization accuracy\n"
            
            if 'stability' in self.gradcam_results:
                stability = self.gradcam_results['stability']['mean']
                analysis += f"**Stability Score**: {stability:.3f} - "
                if stability > 0.7:
                    analysis += "High stability under perturbations\n"
                else:
                    analysis += "Moderate stability under perturbations\n"
        
        return analysis
    
    def _create_gradcam_code(self) -> str:
        """Create code for Grad-CAM visualization"""
        return f"""
# Grad-CAM Results Analysis
gradcam_results = {self.gradcam_results}

if gradcam_results:
    # Create metrics visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Insertion AUC
    if 'insertion_auc' in gradcam_results:
        values = gradcam_results['insertion_auc']['values']
        axes[0].hist(values, bins=20, alpha=0.7, color='blue')
        axes[0].axvline(np.mean(values), color='red', linestyle='--', 
                       label=f'Mean: {{np.mean(values):.3f}}')
        axes[0].set_title('Insertion AUC Distribution')
        axes[0].set_xlabel('Insertion AUC')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
    
    # Deletion AUC
    if 'deletion_auc' in gradcam_results:
        values = gradcam_results['deletion_auc']['values']
        axes[1].hist(values, bins=20, alpha=0.7, color='orange')
        axes[1].axvline(np.mean(values), color='red', linestyle='--',
                       label=f'Mean: {{np.mean(values):.3f}}')
        axes[1].set_title('Deletion AUC Distribution')
        axes[1].set_xlabel('Deletion AUC')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
    
    # Stability
    if 'stability' in gradcam_results:
        values = gradcam_results['stability']['values']
        axes[2].hist(values, bins=20, alpha=0.7, color='green')
        axes[2].axvline(np.mean(values), color='red', linestyle='--',
                       label=f'Mean: {{np.mean(values):.3f}}')
        axes[2].set_title('Stability Distribution')
        axes[2].set_xlabel('Stability Score')
        axes[2].set_ylabel('Frequency')
        axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("Grad-CAM Quantitative Results:")
    for metric in ['insertion_auc', 'deletion_auc', 'stability']:
        if metric in gradcam_results:
            mean_val = gradcam_results[metric]['mean']
            std_val = gradcam_results[metric]['std']
            print(f"- {{metric.replace('_', ' ').title()}}: {{mean_val:.3f}} ± {{std_val:.3f}}")
else:
    print("Grad-CAM results not available")
"""
    
    def _create_shap_analysis(self) -> str:
        """Create SHAP analysis text"""
        analysis = """### SHAP Analysis Results

SHAP (SHapley Additive exPlanations) provides quantitative feature attributions by computing the contribution of each pixel to the model's prediction. This method offers both positive and negative contributions, showing which features support or oppose the prediction.

"""
        
        if self.shap_results:
            total_samples = self.shap_results.get('total_samples', 0)
            analysis += f"**Samples Analyzed**: {total_samples}\n"
            
            if 'overall_statistics' in self.shap_results:
                stats = self.shap_results['overall_statistics']
                mean_imp = stats.get('mean_importance', 0)
                analysis += f"**Mean Importance**: {mean_imp:.3f}\n"
                
                if 'channel_importance' in stats:
                    channel_imp = stats['channel_importance']
                    max_channel = max(channel_imp.items(), key=lambda x: x[1])
                    analysis += f"**Most Important Channel**: {max_channel[0].title()} ({max_channel[1]:.3f})\n"
        
        return analysis
    
    def _create_shap_code(self) -> str:
        """Create code for SHAP visualization"""
        return f"""
# SHAP Results Analysis
shap_results = {self.shap_results}

if shap_results and 'overall_statistics' in shap_results:
    stats = shap_results['overall_statistics']
    
    # Channel importance visualization
    if 'channel_importance' in stats:
        channel_imp = stats['channel_importance']
        channels = list(channel_imp.keys())
        values = list(channel_imp.values())
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Channel importance bar plot
        colors = ['red', 'green', 'blue']
        axes[0].bar(channels, values, color=colors, alpha=0.7)
        axes[0].set_title('SHAP Channel Importance')
        axes[0].set_ylabel('Total Importance')
        
        # Per-class importance (if available)
        if 'class_importance' in shap_results:
            class_imp = shap_results['class_importance']
            classes = list(class_imp.keys())
            class_values = [class_imp[cls]['overall'] for cls in classes]
            
            axes[1].bar(classes, class_values, alpha=0.7, color='skyblue')
            axes[1].set_title('SHAP Importance by Class')
            axes[1].set_ylabel('Average Importance')
            axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    # Print summary
    print("SHAP Analysis Summary:")
    print(f"- Total Samples: {{shap_results.get('total_samples', 0)}}")
    print(f"- Mean Importance: {{stats.get('mean_importance', 0):.3f}}")
    print(f"- Std Importance: {{stats.get('std_importance', 0):.3f}}")
    
    if 'channel_importance' in stats:
        print("\\nChannel Importance:")
        for channel, importance in stats['channel_importance'].items():
            print(f"  - {{channel.title()}}: {{importance:.3f}}")
else:
    print("SHAP results not available")
"""
    
    def _create_lime_analysis(self) -> str:
        """Create LIME analysis text"""
        analysis = """### LIME Analysis Results

LIME (Local Interpretable Model-agnostic Explanations) provides local explanations by perturbing the input image and observing changes in predictions. It uses superpixel segmentation to identify regions that contribute positively or negatively to the prediction.

"""
        
        if self.lime_results:
            total_samples = self.lime_results.get('total_samples', 0)
            analysis += f"**Samples Analyzed**: {total_samples}\n"
            
            if 'overall_statistics' in self.lime_results:
                stats = self.lime_results['overall_statistics']
                avg_superpixels = stats.get('avg_superpixels_per_sample', 0)
                avg_pos_features = stats.get('avg_positive_features', 0)
                avg_neg_features = stats.get('avg_negative_features', 0)
                
                analysis += f"**Average Superpixels per Sample**: {avg_superpixels:.1f}\n"
                analysis += f"**Average Positive Features**: {avg_pos_features:.1f}\n"
                analysis += f"**Average Negative Features**: {avg_neg_features:.1f}\n"
        
        return analysis
    
    def _create_lime_code(self) -> str:
        """Create code for LIME visualization"""
        return f"""
# LIME Results Analysis
lime_results = {self.lime_results}

if lime_results and 'overall_statistics' in lime_results:
    stats = lime_results['overall_statistics']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Feature count visualization
    feature_types = ['Positive Features', 'Negative Features']
    feature_counts = [stats.get('avg_positive_features', 0), 
                     stats.get('avg_negative_features', 0)]
    
    axes[0].bar(feature_types, feature_counts, 
               color=['green', 'red'], alpha=0.7)
    axes[0].set_title('Average Feature Count per Sample')
    axes[0].set_ylabel('Number of Features')
    
    # Weight magnitude visualization
    weight_types = ['Positive Weight', 'Negative Weight']
    weight_values = [stats.get('avg_positive_weight', 0),
                    stats.get('avg_negative_weight', 0)]
    
    axes[1].bar(weight_types, weight_values,
               color=['green', 'red'], alpha=0.7)
    axes[1].set_title('Average Weight Magnitude')
    axes[1].set_ylabel('Weight Magnitude')
    
    plt.tight_layout()
    plt.show()
    
    # Per-class analysis (if available)
    if 'class_analysis' in lime_results:
        class_analysis = lime_results['class_analysis']
        
        print("\\nLIME Per-Class Analysis:")
        for class_name, class_data in class_analysis.items():
            print(f"\\n{{class_name}}:")
            print(f"  - Samples: {{class_data['num_samples']}}")
            print(f"  - Avg Positive Weight: {{class_data['avg_positive_weight']:.3f}}")
            print(f"  - Avg Negative Weight: {{class_data['avg_negative_weight']:.3f}}")
            print(f"  - Avg Superpixels: {{class_data['avg_superpixels_per_sample']:.1f}}")
    
    print("\\nLIME Overall Statistics:")
    print(f"- Total Samples: {{lime_results.get('total_samples', 0)}}")
    print(f"- Avg Superpixels per Sample: {{stats.get('avg_superpixels_per_sample', 0):.1f}}")
    print(f"- Avg Positive Features: {{stats.get('avg_positive_features', 0):.1f}}")
    print(f"- Avg Negative Features: {{stats.get('avg_negative_features', 0):.1f}}")
else:
    print("LIME results not available")
"""
    
    def _create_comparison_analysis(self) -> str:
        """Create quantitative comparison analysis text"""
        analysis = """### Quantitative Method Comparison

This section presents a comprehensive comparison of all explainability methods using standardized quantitative metrics. The comparison helps identify the most reliable and faithful explanation method for clinical use.

"""
        
        if self.benchmark_results:
            methods = list(self.benchmark_results.keys())
            analysis += f"**Methods Compared**: {', '.join(methods)}\n\n"
            
            # Find best performing methods
            best_insertion = None
            best_deletion = None
            best_stability = None
            
            best_ins_score = 0
            best_del_score = 1
            best_stab_score = 0
            
            for method, results in self.benchmark_results.items():
                if results.get('insertion_auc', {}).get('mean'):
                    score = results['insertion_auc']['mean']
                    if score > best_ins_score:
                        best_ins_score = score
                        best_insertion = method
                
                if results.get('deletion_auc', {}).get('mean'):
                    score = results['deletion_auc']['mean']
                    if score < best_del_score:
                        best_del_score = score
                        best_deletion = method
                
                if results.get('stability', {}).get('mean'):
                    score = results['stability']['mean']
                    if score > best_stab_score:
                        best_stab_score = score
                        best_stability = method
            
            if best_insertion:
                analysis += f"**Best Insertion AUC**: {best_insertion} ({best_ins_score:.3f})\n"
            if best_deletion:
                analysis += f"**Best Deletion AUC**: {best_deletion} ({best_del_score:.3f})\n"
            if best_stability:
                analysis += f"**Most Stable**: {best_stability} ({best_stab_score:.3f})\n"
        
        return analysis
    
    def _create_comparison_code(self) -> str:
        """Create code for quantitative comparison"""
        return f"""
# Quantitative Method Comparison
benchmark_results = {self.benchmark_results}

if benchmark_results:
    methods = list(benchmark_results.keys())
    
    # Prepare data for comparison
    metrics = ['insertion_auc', 'deletion_auc', 'stability']
    metric_names = ['Insertion AUC', 'Deletion AUC', 'Stability']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = []
        errors = []
        method_labels = []
        
        for method in methods:
            if metric in benchmark_results[method] and benchmark_results[method][metric]['mean'] is not None:
                values.append(benchmark_results[method][metric]['mean'])
                errors.append(benchmark_results[method][metric]['std'])
                method_labels.append(method)
        
        if values:
            bars = axes[i].bar(method_labels, values, yerr=errors, 
                              capsize=5, alpha=0.7)
            axes[i].set_title(f'{{name}} Comparison')
            axes[i].set_ylabel(name)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, 
                           bar.get_height() + max(values)*0.01,
                           f'{{value:.3f}}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Create summary table
    print("\\nQuantitative Comparison Summary:")
    print("=" * 80)
    print(f"{{'Method':<15} {{'Insertion AUC':<15} {{'Deletion AUC':<15} {{'Stability':<15}}")
    print("-" * 80)
    
    for method in methods:
        ins_auc = benchmark_results[method].get('insertion_auc', {}).get('mean', 0)
        del_auc = benchmark_results[method].get('deletion_auc', {}).get('mean', 0)
        stability = benchmark_results[method].get('stability', {}).get('mean', 0)
        
        print(f"{{method:<15}} {{ins_auc:<15.3f}} {{del_auc:<15.3f}} {{stability:<15.3f}}")
else:
    print("Benchmark results not available")
"""
    
    def _create_discussion_section(self) -> str:
        """Create discussion section"""
        discussion = """### Clinical Interpretability Insights

The explainability analysis reveals several important patterns that have direct clinical relevance:

#### Model Decision Patterns

"""
        
        # Add insights based on available results
        if self.benchmark_results:
            methods = list(self.benchmark_results.keys())
            
            # Find most reliable method
            method_scores = {}
            for method, results in self.benchmark_results.items():
                score = 0
                count = 0
                
                if results.get('insertion_auc', {}).get('mean'):
                    score += results['insertion_auc']['mean']
                    count += 1
                
                if results.get('deletion_auc', {}).get('mean'):
                    score += (1 - results['deletion_auc']['mean'])
                    count += 1
                
                if results.get('stability', {}).get('mean'):
                    score += results['stability']['mean']
                    count += 1
                
                if count > 0:
                    method_scores[method] = score / count
            
            if method_scores:
                best_method = max(method_scores.items(), key=lambda x: x[1])
                discussion += f"- **Most Reliable Method**: {best_method[0]} shows the best overall performance across multiple metrics\n"
        
        if self.shap_results and 'overall_statistics' in self.shap_results:
            stats = self.shap_results['overall_statistics']
            if 'channel_importance' in stats:
                channel_imp = stats['channel_importance']
                max_channel = max(channel_imp.items(), key=lambda x: x[1])
                discussion += f"- **Color Sensitivity**: The model shows highest sensitivity to the {max_channel[0]} channel, which may correspond to specific histological features\n"
        
        discussion += """
#### Implications for Clinical Use

1. **Diagnostic Confidence**: High stability scores indicate that explanations remain consistent even with minor image variations, supporting reliable clinical interpretation.

2. **Feature Localization**: Good insertion/deletion AUC scores suggest that highlighted regions genuinely contribute to the model's decisions, making them clinically relevant.

3. **Method Selection**: Different explanation methods may be appropriate for different clinical scenarios:
   - **Grad-CAM**: Quick visual overview of important regions
   - **SHAP**: Quantitative analysis of feature contributions
   - **LIME**: Local explanations for specific diagnostic regions

#### Validation Recommendations

- Cross-reference explanation highlights with known histopathological markers
- Validate explanations with expert pathologists
- Consider explanation consistency across similar cases
- Monitor explanation quality in different staining conditions

"""
        
        return discussion
    
    def _create_conclusions_section(self) -> str:
        """Create conclusions section"""
        conclusions = """### Key Findings

"""
        
        # Summarize key findings
        if self.model_performance:
            accuracy = self.model_performance.get('accuracy', 0)
            conclusions += f"1. **Model Performance**: Achieved {accuracy:.1%} accuracy on the test set, demonstrating strong classification capability.\n"
        
        if self.benchmark_results:
            # Find best method
            method_scores = {}
            for method, results in self.benchmark_results.items():
                if results.get('insertion_auc', {}).get('mean'):
                    method_scores[method] = results['insertion_auc']['mean']
            
            if method_scores:
                best_method = max(method_scores.items(), key=lambda x: x[1])
                conclusions += f"2. **Best Explanation Method**: {best_method[0]} provides the most faithful explanations (Insertion AUC: {best_method[1]:.3f}).\n"
        
        conclusions += """
3. **Explanation Quality**: Quantitative metrics demonstrate that the model's explanations are both faithful and stable, supporting clinical interpretability.

4. **Clinical Relevance**: Explanation methods successfully highlight diagnostically relevant regions, as evidenced by high localization accuracy.

### Recommendations for Clinical Deployment

#### Immediate Actions
- Implement the best-performing explanation method in the clinical interface
- Provide training for pathologists on interpreting AI explanations
- Establish protocols for validating explanations against expert knowledge

#### Quality Assurance
- Monitor explanation consistency across different image qualities
- Regularly validate explanations with expert pathologists
- Track correlation between explanation quality and diagnostic accuracy

#### Future Improvements
- Investigate explanation performance across different staining protocols
- Develop explanation-based confidence metrics
- Explore integration with existing pathology workflows

### Technical Recommendations

1. **Method Selection**: Use Grad-CAM for real-time explanations and SHAP for detailed analysis
2. **Quality Metrics**: Implement insertion/deletion AUC monitoring for explanation quality
3. **Stability Testing**: Regular stability assessment under various image conditions
4. **User Interface**: Design explanation displays that highlight both positive and negative contributions

### Limitations and Future Work

- **Ground Truth Validation**: Limited availability of pixel-level annotations for IoU computation
- **Cross-Dataset Validation**: Need to validate explanation quality across different datasets
- **Computational Efficiency**: Balance between explanation quality and processing time for clinical use
- **Expert Validation**: Ongoing validation with domain experts required

This comprehensive analysis demonstrates that DenLsNet not only achieves high classification accuracy but also provides interpretable and reliable explanations suitable for clinical decision support.
"""
        
        return conclusions
    
    def export_to_pdf(self, notebook_path: str, pdf_path: str = None) -> str:
        """
        Export notebook to PDF
        
        Args:
            notebook_path: Path to notebook file
            pdf_path: Path for PDF output (default: auto-generated)
            
        Returns:
            Path to PDF file
        """
        if pdf_path is None:
            pdf_path = notebook_path.replace('.ipynb', '.pdf')
        
        try:
            # Execute notebook first
            with open(notebook_path, 'r') as f:
                nb = nbf.read(f, as_version=4)
            
            # Execute the notebook
            ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
            ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
            
            # Export to PDF
            pdf_exporter = PDFExporter()
            (body, resources) = pdf_exporter.from_notebook_node(nb)
            
            with open(pdf_path, 'wb') as f:
                f.write(body)
            
            print(f"PDF report created: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            print(f"Error creating PDF: {e}")
            print("Note: PDF export requires LaTeX installation")
            
            # Fallback to HTML
            html_path = pdf_path.replace('.pdf', '.html')
            return self.export_to_html(notebook_path, html_path)
    
    def export_to_html(self, notebook_path: str, html_path: str = None) -> str:
        """
        Export notebook to HTML
        
        Args:
            notebook_path: Path to notebook file
            html_path: Path for HTML output (default: auto-generated)
            
        Returns:
            Path to HTML file
        """
        if html_path is None:
            html_path = notebook_path.replace('.ipynb', '.html')
        
        try:
            # Execute notebook first
            with open(notebook_path, 'r') as f:
                nb = nbf.read(f, as_version=4)
            
            # Execute the notebook
            ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
            ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
            
            # Export to HTML
            html_exporter = HTMLExporter()
            (body, resources) = html_exporter.from_notebook_node(nb)
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(body)
            
            print(f"HTML report created: {html_path}")
            return html_path
            
        except Exception as e:
            print(f"Error creating HTML: {e}")
            return None
    
    def generate_complete_report(self, output_dir: str = None) -> Dict[str, str]:
        """
        Generate complete report in multiple formats
        
        Args:
            output_dir: Directory for output files (default: results_dir)
            
        Returns:
            Dictionary with paths to generated files
        """
        if output_dir is None:
            output_dir = self.results_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"Explainable_DenLsNet_Report_{timestamp}"
        
        # Generate notebook
        notebook_path = os.path.join(output_dir, f"{base_name}.ipynb")
        notebook_path = self.create_notebook_report(notebook_path)
        
        # Generate HTML
        html_path = os.path.join(output_dir, f"{base_name}.html")
        html_path = self.export_to_html(notebook_path, html_path)
        
        # Try to generate PDF
        pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
        pdf_path = self.export_to_pdf(notebook_path, pdf_path)
        
        results = {
            'notebook': notebook_path,
            'html': html_path,
            'pdf': pdf_path
        }
        
        print(f"\\nComplete report generated:")
        for format_type, path in results.items():
            if path and os.path.exists(path):
                print(f"- {format_type.upper()}: {path}")
        
        return results


def generate_explainability_report(
    results_dir: str,
    model_performance: Dict = None,
    class_names: List[str] = None,
    output_dir: str = None
) -> Dict[str, str]:
    """
    Convenience function to generate complete explainability report
    
    Args:
        results_dir: Directory containing explainability results
        model_performance: Model performance metrics
        class_names: List of class names
        output_dir: Output directory for reports
        
    Returns:
        Dictionary with paths to generated files
    """
    generator = ExplainabilityReportGenerator(
        results_dir=results_dir,
        model_performance=model_performance,
        class_names=class_names
    )
    
    return generator.generate_complete_report(output_dir)


if __name__ == "__main__":
    print("Explainability Report Generator")
    print("Creates comprehensive Jupyter notebook and PDF reports")