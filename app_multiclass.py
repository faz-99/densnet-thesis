"""
Enhanced Streamlit UI for Multi-class DenLsNet with Stain Normalization
Supports 8-class BreakHis classification and comprehensive explainability
"""
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import io
import os
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Import project modules
import config_multiclass as config
from model.multiclass_model import MultiClassDenLsNet
from explainability.grad_cam import GradCAM, GradCAMPlusPlus, overlay_heatmap
from explainability.shap_explainer import SHAPExplainer
from explainability.lime_explainer import LIMEExplainer
from explainability.interpretability_framework import InterpretabilityFramework
from stain_normalization import StainNormalizer, compare_stain_methods
from evaluation.metrics import ModelEvaluator

# Page configuration
st.set_page_config(
    page_title="DenLsNet Multi-Class Interpretability",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        text-align: center;
    }
    .benign-prediction {
        background-color: #d4edda;
        border-color: #28a745;
    }
    .malignant-prediction {
        background-color: #f8d7da;
        border-color: #dc3545;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .stain-comparison {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path):
    """Load the trained multi-class model with caching"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device)
        model = checkpoint['model']
        model.to(device)
        model.eval()
        
        # Get model info
        best_acc = checkpoint.get('best_metrics', {}).get('accuracy', 'N/A')
        best_f1 = checkpoint.get('best_metrics', {}).get('f1_macro', 'N/A')
        epoch = checkpoint.get('best_metrics', {}).get('epoch', 'N/A')
        stain_method = checkpoint.get('config', {}).get('stain_method', 'unknown')
        
        return model, device, best_acc, best_f1, epoch, stain_method
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None, None, None


@st.cache_resource
def initialize_explainers(_model, _device):
    """Initialize explainability tools with caching"""
    try:
        framework = InterpretabilityFramework(_model, str(_device), config.class_names)
        return framework
    except Exception as e:
        st.error(f"Error initializing explainers: {str(e)}")
        return None


@st.cache_resource
def initialize_stain_normalizers():
    """Initialize stain normalizers"""
    normalizers = {}
    for method in ['macenko', 'reinhard']:
        try:
            normalizers[method] = StainNormalizer(method=method)
        except Exception as e:
            st.warning(f"Could not initialize {method} normalizer: {e}")
    return normalizers


def preprocess_image(image, target_size=(224, 224), stain_method='none', stain_normalizers=None):
    """Preprocess uploaded image for model inference with optional stain normalization"""
    # Convert PIL to numpy
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        pass  # Already RGB
    elif len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]  # Remove alpha channel
    else:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Apply stain normalization if requested
    normalized_image = image.copy()
    if stain_method != 'none' and stain_normalizers and stain_method in stain_normalizers:
        try:
            normalized_image = stain_normalizers[stain_method].normalize(image)
        except Exception as e:
            st.warning(f"Stain normalization failed: {e}")
            normalized_image = image
    
    # Resize
    image_resized = cv2.resize(normalized_image, target_size)
    
    # Normalize to [0, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Apply dataset normalization
    mean = np.array(config.dataset_mean)
    std = np.array(config.dataset_std)
    image_normalized = (image_normalized - mean) / std
    
    # Convert to tensor (C, H, W)
    image_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).unsqueeze(0)
    
    return image_tensor, image_resized, normalized_image


def get_confidence_color(confidence):
    """Get color class based on confidence level"""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    else:
        return "confidence-low"


def get_prediction_class(predicted_class):
    """Get CSS class based on prediction (benign vs malignant)"""
    # Benign classes: 0-3, Malignant classes: 4-7
    if predicted_class < 4:
        return "benign-prediction"
    else:
        return "malignant-prediction"


def create_stain_comparison_plot(original, normalized_images):
    """Create comparison of original and stain-normalized images"""
    num_methods = len(normalized_images) + 1
    fig, axes = plt.subplots(1, num_methods, figsize=(5 * num_methods, 5))
    
    if num_methods == 1:
        axes = [axes]
    
    # Original image
    axes[0].imshow(original)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Normalized images
    for i, (method, norm_img) in enumerate(normalized_images.items(), 1):
        axes[i].imshow(norm_img)
        axes[i].set_title(f'{method.title()}')
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def create_multiclass_probability_plot(probabilities, class_names):
    """Create interactive probability plot for multi-class predictions"""
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Class': class_names,
        'Probability': probabilities,
        'Category': ['Benign' if i < 4 else 'Malignant' for i in range(len(class_names))]
    })
    
    # Create bar plot with color coding
    fig = px.bar(
        df, 
        x='Class', 
        y='Probability',
        color='Category',
        title='Class Probability Distribution',
        color_discrete_map={'Benign': '#28a745', 'Malignant': '#dc3545'},
        hover_data=['Probability']
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        showlegend=True
    )
    
    return fig


def create_explainability_comparison(original_img, explanations):
    """Create comprehensive explainability visualization"""
    num_methods = len(explanations)
    if num_methods == 0:
        return None
    
    # Create subplot grid
    cols = min(3, num_methods + 1)  # +1 for original
    rows = (num_methods + 1 + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1) if num_methods > 0 else [axes]
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Original image
    axes_flat[0].imshow(original_img)
    axes_flat[0].set_title('Original Image')
    axes_flat[0].axis('off')
    
    # Explanation methods
    for i, (method, explanation) in enumerate(explanations.items(), 1):
        if i < len(axes_flat):
            if explanation is not None:
                # Create overlay
                overlay = overlay_heatmap(original_img, explanation, alpha=0.4)
                axes_flat[i].imshow(overlay)
                axes_flat[i].set_title(f'{method.upper()}')
            else:
                axes_flat[i].text(0.5, 0.5, f'{method}\nFailed', 
                                ha='center', va='center', transform=axes_flat[i].transAxes)
            axes_flat[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(explanations) + 1, len(axes_flat)):
        axes_flat[i].axis('off')
    
    plt.tight_layout()
    return fig


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔬 DenLsNet Multi-Class Interpretability</h1>', unsafe_allow_html=True)
    st.markdown("**Interactive demonstration for 8-class histopathology classification with explainable AI**")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    model_variants = {
        'Baseline (No Stain Norm)': 'weight/multiclass/none/denlsnet_mc_none_best.pth',
        'Macenko Normalized': 'weight/multiclass/macenko/denlsnet_mc_macenko_best.pth',
        'Reinhard Normalized': 'weight/multiclass/reinhard/denlsnet_mc_reinhard_best.pth'
    }
    
    selected_variant = st.sidebar.selectbox("Select Model Variant", list(model_variants.keys()))
    model_path = model_variants[selected_variant]
    
    # Load model
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.info("Please train the multi-class models first.")
        return
    
    with st.spinner("Loading model..."):
        model, device, best_acc, best_f1, epoch, stain_method = load_model(model_path)
    
    if model is None:
        return
    
    # Model info
    st.sidebar.success("✅ Model loaded successfully!")
    st.sidebar.info(f"**Variant:** {selected_variant}")
    st.sidebar.info(f"**Best Accuracy:** {best_acc}")
    st.sidebar.info(f"**Best F1-Score:** {best_f1}")
    st.sidebar.info(f"**Epoch:** {epoch}")
    st.sidebar.info(f"**Device:** {device}")
    
    # Stain normalization options
    st.sidebar.header("🎨 Stain Normalization")
    apply_stain_norm = st.sidebar.checkbox("Apply Stain Normalization", value=False)
    stain_norm_method = st.sidebar.selectbox(
        "Normalization Method", 
        ['macenko', 'reinhard'],
        disabled=not apply_stain_norm
    )
    show_stain_comparison = st.sidebar.checkbox("Show Stain Comparison", value=True)
    
    # Initialize stain normalizers
    stain_normalizers = initialize_stain_normalizers() if apply_stain_norm else {}
    
    # Explainability options
    st.sidebar.header("🧠 Explainability Options")
    use_gradcam = st.sidebar.checkbox("Grad-CAM", value=True)
    use_gradcam_plus = st.sidebar.checkbox("Grad-CAM++", value=True)
    use_shap = st.sidebar.checkbox("SHAP", value=False, help="Computationally intensive")
    use_lime = st.sidebar.checkbox("LIME", value=False, help="Computationally intensive")
    
    # Initialize explainers
    explainability_framework = None
    if any([use_gradcam, use_gradcam_plus, use_shap, use_lime]):
        with st.spinner("Initializing explainability tools..."):
            explainability_framework = initialize_explainers(model, device)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a histopathology image",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            help="Upload a histopathology image for 8-class classification"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.info(f"**Image Size:** {image.size[0]} x {image.size[1]} pixels")
            st.info(f"**File Size:** {uploaded_file.size / 1024:.1f} KB")
    
    with col2:
        st.header("🔍 Analysis Results")
        
        if uploaded_file is not None:
            with st.spinner("Processing image..."):
                # Preprocess image
                stain_method_to_use = stain_norm_method if apply_stain_norm else 'none'
                image_tensor, image_resized, normalized_image = preprocess_image(
                    image, stain_method=stain_method_to_use, stain_normalizers=stain_normalizers
                )
                image_tensor = image_tensor.to(device)
                
                # Run inference
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0, predicted_class].item()
                
                # Get predictions
                predicted_label = config.class_names[predicted_class]
                predicted_category = "Benign" if predicted_class < 4 else "Malignant"
                
                # Display prediction
                confidence_class = get_confidence_color(confidence)
                prediction_class = get_prediction_class(predicted_class)
                
                st.markdown(f"""
                <div class="prediction-box {prediction_class}">
                    <h3>Prediction Results</h3>
                    <h2>{predicted_label}</h2>
                    <h4>({predicted_category})</h4>
                    <p class="{confidence_class}">
                        Confidence: {confidence:.1%}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability distribution
                prob_array = probabilities[0].cpu().numpy()
                fig_prob = create_multiclass_probability_plot(prob_array, config.class_names)
                st.plotly_chart(fig_prob, use_container_width=True)
    
    # Stain normalization comparison
    if uploaded_file is not None and show_stain_comparison:
        st.header("🎨 Stain Normalization Comparison")
        
        # Apply all normalization methods for comparison
        original_img = np.array(image.resize((224, 224)))
        normalized_images = {}
        
        for method in ['macenko', 'reinhard']:
            if method in stain_normalizers:
                try:
                    norm_img = stain_normalizers[method].normalize(original_img)
                    normalized_images[method] = norm_img
                except Exception as e:
                    st.warning(f"Failed to apply {method} normalization: {e}")
        
        if normalized_images:
            fig_stain = create_stain_comparison_plot(original_img, normalized_images)
            st.pyplot(fig_stain)
            
            # Stain normalization metrics
            st.subheader("Normalization Effects")
            cols = st.columns(len(normalized_images) + 1)
            
            # Original statistics
            with cols[0]:
                st.metric("Original", "Mean RGB", f"{np.mean(original_img):.1f}")
                st.metric("", "Std RGB", f"{np.std(original_img):.1f}")
            
            # Normalized statistics
            for i, (method, norm_img) in enumerate(normalized_images.items(), 1):
                with cols[i]:
                    st.metric(f"{method.title()}", "Mean RGB", f"{np.mean(norm_img):.1f}")
                    st.metric("", "Std RGB", f"{np.std(norm_img):.1f}")
    
    # Explainability section
    if uploaded_file is not None and explainability_framework is not None:
        st.header("🧠 Model Interpretability")
        
        with st.spinner("Generating explanations..."):
            # Determine which methods to use
            methods_to_use = []
            if use_gradcam:
                methods_to_use.append('gradcam')
            if use_gradcam_plus:
                methods_to_use.append('gradcam_plus')
            if use_shap:
                methods_to_use.append('shap')
            if use_lime:
                methods_to_use.append('lime')
            
            # Generate explanations
            explanations = explainability_framework.generate_explanations(
                image_tensor, predicted_class, methods_to_use
            )
            
            # Create visualization
            if explanations:
                original_img_display = np.array(image_resized) / 255.0
                fig_explain = create_explainability_comparison(original_img_display, explanations)
                
                if fig_explain:
                    st.pyplot(fig_explain)
                
                # Explanation details
                st.subheader("Explanation Details")
                
                explanation_cols = st.columns(len(explanations))
                for i, (method, explanation) in enumerate(explanations.items()):
                    with explanation_cols[i]:
                        if explanation is not None:
                            # Calculate explanation statistics
                            mean_importance = np.mean(explanation)
                            max_importance = np.max(explanation)
                            coverage = np.sum(explanation > np.percentile(explanation, 80)) / explanation.size
                            
                            st.metric(f"{method.upper()}", "Mean Importance", f"{mean_importance:.3f}")
                            st.metric("", "Max Importance", f"{max_importance:.3f}")
                            st.metric("", "Coverage (Top 20%)", f"{coverage:.1%}")
                        else:
                            st.error(f"{method} failed")
    
    # Model comparison section
    if uploaded_file is not None:
        st.header("📊 Model Variant Comparison")
        
        if st.button("Compare All Model Variants"):
            comparison_results = {}
            
            with st.spinner("Running comparison across all model variants..."):
                for variant_name, variant_path in model_variants.items():
                    if os.path.exists(variant_path):
                        try:
                            # Load variant model
                            variant_model, _, _, _, _, _ = load_model(variant_path)
                            if variant_model is not None:
                                # Run inference
                                with torch.no_grad():
                                    variant_outputs = variant_model(image_tensor)
                                    variant_probs = F.softmax(variant_outputs, dim=1)
                                    variant_pred = torch.argmax(variant_probs, dim=1).item()
                                    variant_conf = variant_probs[0, variant_pred].item()
                                
                                comparison_results[variant_name] = {
                                    'prediction': config.class_names[variant_pred],
                                    'confidence': variant_conf,
                                    'probabilities': variant_probs[0].cpu().numpy()
                                }
                        except Exception as e:
                            st.warning(f"Failed to load {variant_name}: {e}")
            
            # Display comparison results
            if comparison_results:
                st.subheader("Prediction Comparison")
                
                comparison_df = pd.DataFrame([
                    {
                        'Model Variant': name,
                        'Prediction': results['prediction'],
                        'Confidence': f"{results['confidence']:.1%}",
                        'Category': 'Benign' if config.class_names.index(results['prediction']) < 4 else 'Malignant'
                    }
                    for name, results in comparison_results.items()
                ])
                
                st.dataframe(comparison_df, use_container_width=True)
                
                # Confidence comparison chart
                fig_comparison = px.bar(
                    comparison_df,
                    x='Model Variant',
                    y=[float(c.strip('%'))/100 for c in comparison_df['Confidence']],
                    color='Category',
                    title='Confidence Comparison Across Model Variants',
                    color_discrete_map={'Benign': '#28a745', 'Malignant': '#dc3545'}
                )
                fig_comparison.update_layout(yaxis_title='Confidence')
                st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Performance metrics section
    st.header("📈 Model Performance")
    
    # Create tabs for different metrics
    perf_tab1, perf_tab2, perf_tab3 = st.tabs(["Overall Metrics", "Per-Class Performance", "Training History"])
    
    with perf_tab1:
        if model is not None:
            # Display model performance metrics
            metrics_cols = st.columns(4)
            
            with metrics_cols[0]:
                st.metric("Best Accuracy", f"{best_acc:.1%}" if isinstance(best_acc, float) else str(best_acc))
            
            with metrics_cols[1]:
                st.metric("Best F1-Score", f"{best_f1:.3f}" if isinstance(best_f1, float) else str(best_f1))
            
            with metrics_cols[2]:
                st.metric("Training Epochs", str(epoch))
            
            with metrics_cols[3]:
                st.metric("Model Variant", selected_variant.split()[0])
    
    with perf_tab2:
        st.info("Per-class performance metrics would be displayed here from saved evaluation results.")
    
    with perf_tab3:
        st.info("Training history plots would be displayed here from saved training logs.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this Multi-Class Application:**
    - **DenLsNet-MC**: Multi-class extension supporting 8 BreakHis subclasses
    - **Stain Normalization**: Macenko and Reinhard methods for domain adaptation
    - **Comprehensive XAI**: Multiple interpretability techniques with quantitative evaluation
    - **Academic Framework**: Designed for thesis demonstration and research validation
    
    **Class Categories:**
    - **Benign**: Adenosis, Fibroadenoma, Phyllodes Tumor, Tubular Adenoma
    - **Malignant**: Ductal Carcinoma, Lobular Carcinoma, Mucinous Carcinoma, Papillary Carcinoma
    """)
    
    # Save analysis results
    if uploaded_file is not None:
        if st.button("💾 Save Analysis Results"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results = {
                'timestamp': timestamp,
                'filename': uploaded_file.name,
                'model_variant': selected_variant,
                'stain_normalization': {
                    'applied': apply_stain_norm,
                    'method': stain_norm_method if apply_stain_norm else 'none'
                },
                'prediction': {
                    'class': predicted_label,
                    'category': predicted_category,
                    'confidence': float(confidence),
                    'class_probabilities': {
                        config.class_names[i]: float(prob_array[i])
                        for i in range(len(config.class_names))
                    }
                },
                'model_info': {
                    'path': model_path,
                    'best_accuracy': str(best_acc),
                    'best_f1': str(best_f1),
                    'epoch': str(epoch),
                    'stain_method': stain_method
                },
                'explainability': {
                    'methods_used': methods_to_use if 'methods_to_use' in locals() else [],
                    'explanations_generated': list(explanations.keys()) if 'explanations' in locals() else []
                }
            }
            
            # Save to file
            os.makedirs('analysis_results/multiclass', exist_ok=True)
            result_path = f'analysis_results/multiclass/analysis_{timestamp}.json'
            
            with open(result_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            st.success(f"Analysis results saved to: {result_path}")


if __name__ == "__main__":
    main()