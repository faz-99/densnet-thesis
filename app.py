"""
Interactive Streamlit UI for DenseNet Model Interpretability
Purpose: Interactive demonstration for thesis with comprehensive explainability features
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

# Import project modules
import config
from model.model import class_model
from explainability.grad_cam import GradCAM, GradCAMPlusPlus, overlay_heatmap
from explainability.shap_explainer import SHAPExplainer
from explainability.lime_explainer import LIMEExplainer
from evaluation.metrics import ModelEvaluator


# Page configuration
st.set_page_config(
    page_title="DenseNet Medical Image Interpretability",
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
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path):
    """Load the trained model with caching"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Fix for PyTorch 2.6 compatibility
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model = checkpoint['model']
        model.to(device)
        model.float()  # Ensure model uses float32
        model.eval()
        
        # Get model info
        best_acc = checkpoint.get('best_acc', 'N/A')
        epoch = checkpoint.get('epoch', 'N/A')
        
        return model, device, best_acc, epoch
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None


@st.cache_resource
def initialize_explainers(_model, _device):
    """Initialize explainability tools with caching"""
    try:
        # Initialize Grad-CAM
        gradcam = GradCAM(_model, target_layer_name='densenet.features.norm5')
        gradcam_plus = GradCAMPlusPlus(_model, target_layer_name='densenet.features.norm5')
        
        # Create dummy background data for SHAP (in real app, use proper background)
        background_data = torch.randn(10, 3, 224, 224).to(_device)
        shap_explainer = SHAPExplainer(_model, background_data, str(_device))
        
        # Initialize LIME
        lime_explainer = LIMEExplainer(_model, str(_device), num_samples=100)
        
        return gradcam, gradcam_plus, shap_explainer, lime_explainer
    except Exception as e:
        st.error(f"Error initializing explainers: {str(e)}")
        return None, None, None, None


def preprocess_image(image, target_size=(224, 224)):
    """Preprocess uploaded image for model inference"""
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
    
    # Resize
    image_resized = cv2.resize(image, target_size)
    
    # Normalize to [0, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Apply dataset normalization (from config)
    mean = np.array(config.dataset_mean)
    std = np.array(config.dataset_std)
    image_normalized = (image_normalized - mean) / std
    
    # Convert to tensor (C, H, W) with correct dtype
    image_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).unsqueeze(0).float()
    
    return image_tensor, image_resized


def get_confidence_color(confidence):
    """Get color class based on confidence level"""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    else:
        return "confidence-low"


def create_comparison_plot(original, normalized):
    """Create side-by-side comparison of original and normalized images"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.imshow(original)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Denormalize for display
    mean = np.array(config.dataset_mean)
    std = np.array(config.dataset_std)
    normalized_display = normalized * std + mean
    normalized_display = np.clip(normalized_display, 0, 1)
    
    ax2.imshow(normalized_display)
    ax2.set_title('Normalized Image')
    ax2.axis('off')
    
    plt.tight_layout()
    return fig


def create_explainability_plot(original_img, gradcam_heatmap, gradcam_plus_heatmap, shap_values=None):
    """Create comprehensive explainability visualization"""
    if shap_values is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = np.pad(axes, ((0, 0), (0, 1)), mode='constant', constant_values=None)
    
    # Original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Grad-CAM
    axes[0, 1].imshow(gradcam_heatmap, cmap='jet')
    axes[0, 1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Grad-CAM++
    axes[0, 2].imshow(gradcam_plus_heatmap, cmap='jet')
    axes[0, 2].set_title('Grad-CAM++ Heatmap', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Overlays
    gradcam_overlay = overlay_heatmap(original_img, gradcam_heatmap)
    gradcam_plus_overlay = overlay_heatmap(original_img, gradcam_plus_heatmap)
    
    axes[1, 0].imshow(gradcam_overlay)
    axes[1, 0].set_title('Grad-CAM Overlay', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(gradcam_plus_overlay)
    axes[1, 1].set_title('Grad-CAM++ Overlay', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # SHAP if available
    if shap_values is not None:
        shap_combined = np.sum(np.abs(shap_values), axis=0)
        shap_combined = (shap_combined - shap_combined.min()) / (shap_combined.max() - shap_combined.min() + 1e-8)
        
        axes[1, 2].imshow(shap_combined, cmap='hot')
        axes[1, 2].set_title('SHAP Importance', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
    else:
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    return fig


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔬 DenseNet Medical Image Interpretability</h1>', unsafe_allow_html=True)
    st.markdown("**Interactive demonstration for histopathology image classification with explainable AI**")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    model_path = st.sidebar.text_input(
        "Model Path", 
        value="weight/save/40/iaff40_5.pth",
        help="Path to the trained model file"
    )
    
    # Load model
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.info("Please train a model first or update the model path.")
        return
    
    with st.spinner("Loading model..."):
        model, device, best_acc, epoch = load_model(model_path)
    
    if model is None:
        return
    
    # Model info
    st.sidebar.success("✅ Model loaded successfully!")
    st.sidebar.info(f"**Best Accuracy:** {best_acc}")
    st.sidebar.info(f"**Epoch:** {epoch}")
    st.sidebar.info(f"**Device:** {device}")
    
    # Explainability options
    st.sidebar.header("🧠 Explainability Options")
    use_gradcam = st.sidebar.checkbox("Grad-CAM", value=True)
    use_gradcam_plus = st.sidebar.checkbox("Grad-CAM++", value=True)
    use_shap = st.sidebar.checkbox("SHAP", value=False, help="Computationally intensive")
    use_lime = st.sidebar.checkbox("LIME", value=False, help="Computationally intensive")
    show_comparison = st.sidebar.checkbox("Show Original vs Normalized", value=True)
    
    # Initialize explainers
    if use_gradcam or use_gradcam_plus or use_shap or use_lime:
        with st.spinner("Initializing explainability tools..."):
            gradcam, gradcam_plus, shap_explainer, lime_explainer = initialize_explainers(model, device)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a histopathology image",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            help="Upload a histopathology image for classification"
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
                image_tensor, image_resized = preprocess_image(image)
                image_tensor = image_tensor.to(device)
                
                # Run inference
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0, predicted_class].item()
                
                # Class names
                class_names = ['Benign', 'Malignant']
                predicted_label = class_names[predicted_class]
                
                # Display prediction
                confidence_class = get_confidence_color(confidence)
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>Prediction Results</h3>
                    <h2 style="color: {'#28a745' if predicted_class == 0 else '#dc3545'};">
                        {predicted_label}
                    </h2>
                    <p class="{confidence_class}">
                        Confidence: {confidence:.1%}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability distribution
                prob_data = {
                    'Class': class_names,
                    'Probability': probabilities[0].cpu().numpy()
                }
                
                fig_prob = px.bar(
                    prob_data, 
                    x='Class', 
                    y='Probability',
                    title='Class Probabilities',
                    color='Probability',
                    color_continuous_scale='viridis'
                )
                fig_prob.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig_prob, use_container_width=True)
    
    # Explainability section
    if uploaded_file is not None and (use_gradcam or use_gradcam_plus or use_shap or use_lime):
        st.header("🧠 Model Interpretability")
        
        with st.spinner("Generating explanations..."):
            # Prepare image for visualization
            original_img = np.array(image_resized) / 255.0
            
            explanations = {}
            
            # Grad-CAM
            if use_gradcam and gradcam is not None:
                gradcam_heatmap = gradcam.generate_cam(image_tensor, predicted_class)
                explanations['gradcam'] = gradcam_heatmap
            
            # Grad-CAM++
            if use_gradcam_plus and gradcam_plus is not None:
                gradcam_plus_heatmap = gradcam_plus.generate_cam(image_tensor, predicted_class)
                explanations['gradcam_plus'] = gradcam_plus_heatmap
            
            # SHAP
            shap_values = None
            if use_shap and shap_explainer is not None:
                try:
                    shap_values = shap_explainer.explain_image(image_tensor, predicted_class)
                    explanations['shap'] = shap_values
                except Exception as e:
                    st.warning(f"SHAP analysis failed: {str(e)}")
            
            # Create visualization
            if 'gradcam' in explanations and 'gradcam_plus' in explanations:
                fig_explain = create_explainability_plot(
                    original_img,
                    explanations['gradcam'],
                    explanations['gradcam_plus'],
                    shap_values
                )
                st.pyplot(fig_explain)
            
            # LIME (separate visualization)
            if use_lime and lime_explainer is not None:
                try:
                    with st.spinner("Generating LIME explanation..."):
                        explanation, segments = lime_explainer.explain_image(original_img)
                        
                        # Get LIME visualization
                        temp, mask = explanation.get_image_and_mask(
                            predicted_class, positive_only=False, num_features=10, hide_rest=True
                        )
                        
                        fig_lime, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                        
                        ax1.imshow(original_img)
                        ax1.set_title('Original Image')
                        ax1.axis('off')
                        
                        from skimage.segmentation import mark_boundaries
                        ax2.imshow(mark_boundaries(temp, mask))
                        ax2.set_title('LIME Explanation')
                        ax2.axis('off')
                        
                        plt.tight_layout()
                        st.pyplot(fig_lime)
                        
                except Exception as e:
                    st.warning(f"LIME analysis failed: {str(e)}")
    
    # Image comparison
    if uploaded_file is not None and show_comparison:
        st.header("📊 Image Preprocessing Comparison")
        
        # Get normalized image for display
        normalized_img = image_tensor[0].cpu().numpy().transpose(1, 2, 0)
        
        fig_comparison = create_comparison_plot(
            np.array(image_resized) / 255.0,
            normalized_img
        )
        st.pyplot(fig_comparison)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this application:**
    - Built with Streamlit for interactive model interpretability
    - Uses DenseNet architecture with attention mechanisms
    - Implements multiple explainability techniques (Grad-CAM, SHAP, LIME)
    - Designed for histopathology image classification
    """)
    
    # Save analysis results
    if uploaded_file is not None:
        if st.button("💾 Save Analysis Results"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results = {
                'timestamp': timestamp,
                'filename': uploaded_file.name,
                'prediction': predicted_label,
                'confidence': float(confidence),
                'probabilities': {
                    class_names[i]: float(probabilities[0, i]) 
                    for i in range(len(class_names))
                },
                'model_info': {
                    'path': model_path,
                    'best_accuracy': str(best_acc),
                    'epoch': str(epoch)
                }
            }
            
            # Save to file
            os.makedirs('analysis_results', exist_ok=True)
            result_path = f'analysis_results/analysis_{timestamp}.json'
            
            with open(result_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            st.success(f"Analysis results saved to: {result_path}")


if __name__ == "__main__":
    main()