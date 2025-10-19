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
import base64

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
        gradcam = None
        gradcam_plus = None
        shap_explainer = None
        lime_explainer = None
        
        # Initialize Grad-CAM - try different layer names for custom model
        try:
            gradcam = GradCAM(_model, target_layer_name='densenet.features.norm5')
        except Exception as e1:
            try:
                gradcam = GradCAM(_model, target_layer_name='features.norm5')
            except Exception as e2:
                try:
                    gradcam = GradCAM(_model, target_layer_name='norm5')
                except Exception as e3:
                    st.warning(f"Could not initialize Grad-CAM: {str(e3)}")
        
        try:
            gradcam_plus = GradCAMPlusPlus(_model, target_layer_name='densenet.features.norm5')
        except Exception as e1:
            try:
                gradcam_plus = GradCAMPlusPlus(_model, target_layer_name='features.norm5')
            except Exception as e2:
                try:
                    gradcam_plus = GradCAMPlusPlus(_model, target_layer_name='norm5')
                except Exception as e3:
                    st.warning(f"Could not initialize Grad-CAM++: {str(e3)}")
        
        # Initialize SHAP with proper PyTorch model wrapper
        try:
            import shap
            
            # Create a proper PyTorch model wrapper for SHAP
            class SHAPModelWrapper(torch.nn.Module):
                def __init__(self, model, device):
                    super(SHAPModelWrapper, self).__init__()
                    self.model = model
                    self.device = device
                
                def forward(self, x):
                    # Ensure input is on correct device and has correct dtype
                    if not isinstance(x, torch.Tensor):
                        x = torch.tensor(x, dtype=torch.float32, device=self.device)
                    else:
                        x = x.to(self.device).float()
                    
                    # Get model output
                    outputs = self.model(x)
                    return F.softmax(outputs, dim=1)
            
            # Create wrapped model
            wrapped_model = SHAPModelWrapper(_model, _device)
            wrapped_model.eval()
            
            # Create dummy background data for SHAP
            background_data = torch.randn(3, 3, 224, 224).to(_device).float()
            
            # Try DeepExplainer first, fallback to GradientExplainer
            try:
                shap_explainer = shap.DeepExplainer(wrapped_model, background_data)
            except Exception as e1:
                try:
                    st.info("Trying GradientExplainer as fallback...")
                    # Fallback to GradientExplainer
                    shap_explainer = shap.GradientExplainer(wrapped_model, background_data)
                except Exception as e2:
                    # If both fail, create a simple attribution method
                    st.info("Using simple gradient-based attribution as final fallback...")
                    shap_explainer = "simple_gradients"
            
        except Exception as e:
            st.warning(f"Could not initialize SHAP: {str(e)}")
            shap_explainer = None
        
        # Initialize LIME with better error handling
        try:
            lime_explainer = LIMEExplainer(_model, str(_device), num_samples=50)
        except Exception as e:
            st.warning(f"Could not initialize LIME: {str(e)}")
            lime_explainer = None
        
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
    # Always create a 2x3 grid for consistency
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Grad-CAM
    if gradcam_heatmap.max() > 0:
        axes[0, 1].imshow(gradcam_heatmap, cmap='jet')
        axes[0, 1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    else:
        axes[0, 1].text(0.5, 0.5, 'Grad-CAM\nNot Available', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Grad-CAM++
    if gradcam_plus_heatmap.max() > 0:
        axes[0, 2].imshow(gradcam_plus_heatmap, cmap='jet')
        axes[0, 2].set_title('Grad-CAM++ Heatmap', fontsize=12, fontweight='bold')
    else:
        axes[0, 2].text(0.5, 0.5, 'Grad-CAM++\nNot Available', ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Grad-CAM++ Heatmap', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Overlays
    if gradcam_heatmap.max() > 0:
        gradcam_overlay = overlay_heatmap(original_img, gradcam_heatmap)
        axes[1, 0].imshow(gradcam_overlay)
        axes[1, 0].set_title('Grad-CAM Overlay', fontsize=12, fontweight='bold')
    else:
        axes[1, 0].imshow(original_img)
        axes[1, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    if gradcam_plus_heatmap.max() > 0:
        gradcam_plus_overlay = overlay_heatmap(original_img, gradcam_plus_heatmap)
        axes[1, 1].imshow(gradcam_plus_overlay)
        axes[1, 1].set_title('Grad-CAM++ Overlay', fontsize=12, fontweight='bold')
    else:
        axes[1, 1].imshow(original_img)
        axes[1, 1].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # SHAP if available
    if shap_values is not None:
        shap_combined = np.sum(np.abs(shap_values), axis=0)
        shap_combined = (shap_combined - shap_combined.min()) / (shap_combined.max() - shap_combined.min() + 1e-8)
        
        axes[1, 2].imshow(shap_combined, cmap='hot')
        axes[1, 2].set_title('SHAP Importance', fontsize=12, fontweight='bold')
    else:
        axes[1, 2].text(0.5, 0.5, 'SHAP\nNot Available', ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('SHAP Importance', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    return fig


def generate_html_report(results, uploaded_filename):
    """Generate comprehensive HTML report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Convert matplotlib figures to base64 images
    def fig_to_base64(fig):
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    
    # Create explainability plot
    explanations = results['explanations']
    original_img = results['original_img']
    
    gradcam_hm = explanations.get('gradcam', np.zeros((224, 224)))
    gradcam_plus_hm = explanations.get('gradcam_plus', np.zeros((224, 224)))
    shap_values = explanations.get('shap', None)
    
    fig_explain = create_explainability_plot(original_img, gradcam_hm, gradcam_plus_hm, shap_values)
    explain_img_b64 = fig_to_base64(fig_explain)
    plt.close(fig_explain)
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>DenseNet Interpretability Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .header {{ text-align: center; color: #1f77b4; margin-bottom: 30px; }}
            .section {{ margin: 30px 0; }}
            .prediction-box {{ 
                background-color: #e8f4fd; 
                padding: 20px; 
                border-radius: 10px; 
                border: 2px solid #1f77b4; 
                text-align: center; 
                margin: 20px 0;
            }}
            .confidence-high {{ color: #28a745; font-weight: bold; }}
            .confidence-medium {{ color: #ffc107; font-weight: bold; }}
            .confidence-low {{ color: #dc3545; font-weight: bold; }}
            .explanation-item {{ margin: 15px 0; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }}
            .image-container {{ text-align: center; margin: 20px 0; }}
            .metadata {{ background-color: #f0f2f6; padding: 15px; border-radius: 5px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #1f77b4; color: white; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔬 DenseNet Medical Image Interpretability Report</h1>
            <p>Generated on {timestamp}</p>
        </div>
        
        <div class="section">
            <h2>📋 Analysis Summary</h2>
            <div class="metadata">
                <p><strong>Image File:</strong> {uploaded_filename}</p>
                <p><strong>Analysis Date:</strong> {timestamp}</p>
                <p><strong>Model:</strong> DenseNet with Attention Mechanisms</p>
                <p><strong>Task:</strong> Histopathology Image Classification</p>
            </div>
        </div>
        
        <div class="section">
            <h2>🎯 Prediction Results</h2>
            <div class="prediction-box">
                <h3>Model Prediction</h3>
                <h2 style="color: {'#28a745' if results['predicted_class'] == 0 else '#dc3545'};">
                    {results['predicted_label']}
                </h2>
                <p class="{'confidence-high' if results['confidence'] >= 0.8 else 'confidence-medium' if results['confidence'] >= 0.6 else 'confidence-low'}">
                    Confidence: {results['confidence']:.1%}
                </p>
            </div>
            
            <table>
                <tr>
                    <th>Class</th>
                    <th>Probability</th>
                    <th>Confidence Level</th>
                </tr>
    """
    
    # Add probability table
    for i, class_name in enumerate(results['class_names']):
        prob = results['confidence'] if i == results['predicted_class'] else (1 - results['confidence'])
        confidence_level = "High" if prob >= 0.8 else "Medium" if prob >= 0.6 else "Low"
        html_content += f"""
                <tr>
                    <td>{class_name}</td>
                    <td>{prob:.3f}</td>
                    <td>{confidence_level}</td>
                </tr>
        """
    
    html_content += f"""
            </table>
        </div>
        
        <div class="section">
            <h2>🧠 Explainability Analysis</h2>
            <div class="image-container">
                <img src="{explain_img_b64}" alt="Explainability Analysis" style="max-width: 100%; height: auto;">
            </div>
            
            <h3>📊 Analysis Results</h3>
    """
    
    # Add explanation results
    for text in results['explanation_text']:
        html_content += f'<div class="explanation-item">{text}</div>'
    
    # Add morphological analysis if available
    morphological_section = ""
    if 'morphological_features' in results:
        features = results['morphological_features']
        clinical_desc = results.get('clinical_description', '')
        analysis_method = results.get('analysis_method', 'Unknown')
        
        morphological_section = f"""
        </div>
        
        <div class="section">
            <h2>🔬 Morphological Analysis</h2>
            <p><strong>Analysis Method:</strong> {analysis_method}</p>
            
            <h3>Quantitative Features</h3>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Value</th>
                    <th>Clinical Significance</th>
                </tr>
                <tr>
                    <td>Tissue Area Highlighted</td>
                    <td>{features['tissue_area_percent']:.1f}%</td>
                    <td>{'Extensive' if features['tissue_area_percent'] > 40 else 'Moderate' if features['tissue_area_percent'] > 20 else 'Focal'} model attention</td>
                </tr>
                <tr>
                    <td>Dominant Stain</td>
                    <td>{features['stain_analysis']['dominant_stain'].title()}</td>
                    <td>{'Nuclear focus' if features['stain_analysis']['dominant_stain'] == 'hematoxylin' else 'Cytoplasmic focus' if features['stain_analysis']['dominant_stain'] == 'eosin' else 'Balanced staining'}</td>
                </tr>
                <tr>
                    <td>Cellular Entropy</td>
                    <td>{features['texture_features']['entropy']:.2f}</td>
                    <td>{'High heterogeneity' if features['texture_features']['entropy'] > 6.0 else 'Moderate heterogeneity' if features['texture_features']['entropy'] > 4.0 else 'Low heterogeneity'}</td>
                </tr>
                <tr>
                    <td>Edge Density</td>
                    <td>{features['texture_features']['edge_density']:.3f}</td>
                    <td>{'Sharp boundaries' if features['texture_features']['edge_density'] > 0.3 else 'Smooth boundaries'}</td>
                </tr>
                <tr>
                    <td>Number of Regions</td>
                    <td>{features['morphological_features']['num_regions']}</td>
                    <td>{'Fragmented pattern' if features['morphological_features']['num_regions'] > 10 else 'Cohesive pattern'}</td>
                </tr>
            </table>
            
            <h3>Clinical Interpretation</h3>
            <div class="explanation-item">
                <strong>Automated Analysis:</strong> {clinical_desc}
            </div>
        """
    
    html_content += morphological_section + f"""
        </div>
        
        <div class="section">
            <h2>📖 Interpretation Guide</h2>
            <h3>Grad-CAM Analysis</h3>
            <ul>
                <li><strong>Heatmap Colors:</strong> Red/yellow regions indicate areas most important for the model's decision</li>
                <li><strong>Interpretation:</strong> Shows where the model 'looks' when making predictions</li>
                <li><strong>Usage:</strong> Helps identify if the model focuses on clinically relevant regions</li>
            </ul>
            
            <h3>Grad-CAM++ Analysis</h3>
            <ul>
                <li><strong>Improvement:</strong> Better localization compared to standard Grad-CAM</li>
                <li><strong>Multiple Objects:</strong> Better handling of multiple instances of the same class</li>
                <li><strong>Precision:</strong> More precise attribution of importance</li>
            </ul>
            
            <h3>Clinical Relevance</h3>
            <ul>
                <li><strong>Validation:</strong> Check if highlighted regions correspond to known pathological features</li>
                <li><strong>Trust:</strong> High confidence predictions with relevant focus areas increase model trustworthiness</li>
                <li><strong>Bias Detection:</strong> Unusual focus patterns may indicate model bias or artifacts</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>⚠️ Important Notes</h2>
            <div class="metadata">
                <p><strong>Model Limitations:</strong> This model is for research purposes and should not be used for clinical diagnosis without proper validation.</p>
                <p><strong>Explainability:</strong> Visualization techniques provide insights into model behavior but should be interpreted by domain experts.</p>
                <p><strong>Validation:</strong> Always validate model predictions with ground truth and clinical expertise.</p>
            </div>
        </div>
        
        <div class="section">
            <h2>📞 Contact Information</h2>
            <p>For questions about this analysis or the underlying model, please contact the research team.</p>
            <p><em>Generated by DenseNet Medical Image Interpretability System</em></p>
        </div>
    </body>
    </html>
    """
    
    return html_content


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
            
            # Debug: Show available layers
            if st.sidebar.checkbox("🔧 Debug: Show Model Layers", value=False):
                st.sidebar.write("**Available model layers:**")
                layer_names = []
                for name, module in model.named_modules():
                    if 'norm' in name.lower() or 'conv' in name.lower():
                        layer_names.append(name)
                
                for name in layer_names[:10]:  # Show first 10 relevant layers
                    st.sidebar.text(name)
    
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
            explanation_text = []
            
            # Grad-CAM
            if use_gradcam:
                if gradcam is not None:
                    try:
                        gradcam_heatmap = gradcam.generate_cam(image_tensor, predicted_class)
                        explanations['gradcam'] = gradcam_heatmap
                        explanation_text.append("✅ Grad-CAM analysis completed")
                        st.success("Grad-CAM heatmap generated successfully")
                    except Exception as e:
                        st.error(f"Grad-CAM failed: {str(e)}")
                        explanation_text.append(f"❌ Grad-CAM failed: {str(e)}")
                else:
                    st.warning("Grad-CAM not available - initialization failed")
                    explanation_text.append("⚠️ Grad-CAM not available")
            
            # Grad-CAM++
            if use_gradcam_plus:
                if gradcam_plus is not None:
                    try:
                        gradcam_plus_heatmap = gradcam_plus.generate_cam(image_tensor, predicted_class)
                        explanations['gradcam_plus'] = gradcam_plus_heatmap
                        explanation_text.append("✅ Grad-CAM++ analysis completed")
                        st.success("Grad-CAM++ heatmap generated successfully")
                    except Exception as e:
                        st.error(f"Grad-CAM++ failed: {str(e)}")
                        explanation_text.append(f"❌ Grad-CAM++ failed: {str(e)}")
                else:
                    st.warning("Grad-CAM++ not available - initialization failed")
                    explanation_text.append("⚠️ Grad-CAM++ not available")
            
            # SHAP
            shap_values = None
            if use_shap:
                if shap_explainer is not None:
                    try:
                        if shap_explainer == "simple_gradients":
                            # Simple gradient-based attribution fallback
                            with torch.enable_grad():
                                # Ensure we have a tensor for gradient computation
                                if isinstance(image_tensor, torch.Tensor):
                                    input_tensor = image_tensor.clone().detach().requires_grad_(True)
                                else:
                                    input_tensor = torch.tensor(image_tensor, dtype=torch.float32, 
                                                              device=device, requires_grad=True)
                                
                                output = model(input_tensor)
                                class_score = output[0, predicted_class]
                                class_score.backward()
                                
                                # Use gradients as attribution
                                if input_tensor.grad is not None:
                                    gradients = input_tensor.grad.detach().cpu().numpy()[0]
                                    input_values = input_tensor.detach().cpu().numpy()[0]
                                    shap_values = gradients * input_values
                                    
                                    explanation_text.append("✅ Simple gradient attribution completed")
                                    st.success("Gradient-based attribution generated successfully")
                                else:
                                    raise Exception("No gradients computed")
                        else:
                            # Use SHAP explainer with proper tensor handling
                            if isinstance(image_tensor, torch.Tensor):
                                input_array = image_tensor.detach().cpu().numpy()
                            else:
                                input_array = np.array(image_tensor)
                            
                            shap_values_raw = shap_explainer.shap_values(input_array)
                            
                            # Extract SHAP values for the predicted class
                            if isinstance(shap_values_raw, list):
                                # Multi-class case - get values for predicted class
                                if len(shap_values_raw) > predicted_class:
                                    shap_values = shap_values_raw[predicted_class][0]  # [0] for first sample
                                else:
                                    shap_values = shap_values_raw[0][0]  # Fallback to first class
                            else:
                                # Single output case
                                shap_values = shap_values_raw[0]  # [0] for first sample
                            
                            explanation_text.append("✅ SHAP analysis completed")
                            st.success("SHAP explanation generated successfully")
                        
                        # Ensure shap_values is properly formatted
                        if shap_values is not None:
                            # Convert to single channel if needed
                            if len(shap_values.shape) == 3:
                                shap_values = np.sum(np.abs(shap_values), axis=0)
                            
                            # Normalize to [0, 1]
                            if shap_values.max() > shap_values.min():
                                shap_values = (shap_values - shap_values.min()) / (shap_values.max() - shap_values.min())
                            
                            explanations['shap'] = shap_values
                        
                    except Exception as e:
                        st.warning(f"SHAP analysis failed: {str(e)}")
                        explanation_text.append(f"⚠️ SHAP failed: {str(e)}")
                        # Try one more fallback - simple saliency
                        try:
                            st.info("Trying simple saliency as final fallback...")
                            with torch.enable_grad():
                                # Ensure we have a tensor for gradient computation
                                if isinstance(image_tensor, torch.Tensor):
                                    input_tensor = image_tensor.clone().detach().requires_grad_(True)
                                else:
                                    input_tensor = torch.tensor(image_tensor, dtype=torch.float32, 
                                                              device=device, requires_grad=True)
                                
                                output = model(input_tensor)
                                class_score = output[0, predicted_class]
                                saliency = torch.autograd.grad(class_score, input_tensor)[0]
                                
                                # Convert to numpy and take absolute values
                                saliency_map = torch.abs(saliency).detach().cpu().numpy()[0]
                                saliency_map = np.sum(saliency_map, axis=0)  # Sum across channels
                                
                                # Normalize
                                if saliency_map.max() > saliency_map.min():
                                    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
                                
                                explanations['shap'] = saliency_map
                                explanation_text.append("✅ Saliency map generated as fallback")
                                st.success("Saliency-based attribution generated successfully")
                        except Exception as e2:
                            st.error(f"All SHAP methods failed: {str(e2)}")
                else:
                    st.warning("SHAP not available - initialization failed")
                    explanation_text.append("⚠️ SHAP not available")
            
            # Create main visualization
            if 'gradcam' in explanations or 'gradcam_plus' in explanations:
                gradcam_hm = explanations.get('gradcam', np.zeros((224, 224)))
                gradcam_plus_hm = explanations.get('gradcam_plus', np.zeros((224, 224)))
                
                fig_explain = create_explainability_plot(
                    original_img,
                    gradcam_hm,
                    gradcam_plus_hm,
                    shap_values
                )
                st.pyplot(fig_explain)
                
                # Add interpretation text
                st.subheader("🔍 Interpretation")
                
                if 'gradcam' in explanations:
                    st.write("**Grad-CAM Analysis:**")
                    st.write("- Red/yellow regions indicate areas most important for the model's decision")
                    st.write("- Grad-CAM shows where the model 'looks' when making predictions")
                    
                if 'gradcam_plus' in explanations:
                    st.write("**Grad-CAM++ Analysis:**")
                    st.write("- Improved localization compared to standard Grad-CAM")
                    st.write("- Better handling of multiple instances of the same class")
                
                if shap_values is not None:
                    st.write("**SHAP Analysis:**")
                    st.write("- Shows pixel-level feature importance")
                    st.write("- Quantifies each pixel's contribution to the prediction")
            
            # LIME (separate visualization)
            if use_lime:
                if lime_explainer is not None:
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
                            
                            st.write("**LIME Analysis:**")
                            st.write("- Green regions contribute positively to the prediction")
                            st.write("- Red regions contribute negatively to the prediction")
                            explanation_text.append("✅ LIME analysis completed")
                            
                    except Exception as e:
                        st.warning(f"LIME analysis failed: {str(e)}")
                        explanation_text.append(f"⚠️ LIME failed: {str(e)}")
                else:
                    st.warning("LIME not available - initialization failed")
                    explanation_text.append("⚠️ LIME not available")
            
            # Store explanation results for report
            if 'explanation_results' not in st.session_state:
                st.session_state.explanation_results = {}
            
            st.session_state.explanation_results = {
                'explanations': explanations,
                'explanation_text': explanation_text,
                'original_img': original_img,
                'predicted_class': predicted_class,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'class_names': class_names
            }
            
            # Generate comprehensive morphological analysis
            if explanations:
                st.subheader("🔬 Morphological Analysis")
                
                try:
                    from explainability.morphological_analyzer import MorphologicalAnalyzer, ClinicalDescriptorGenerator
                    
                    morphological_analyzer = MorphologicalAnalyzer()
                    descriptor_generator = ClinicalDescriptorGenerator()
                    
                    # Analyze the best available explanation map
                    best_map = None
                    best_method = None
                    
                    if 'gradcam' in explanations:
                        best_map = explanations['gradcam']
                        best_method = 'Grad-CAM'
                    elif 'gradcam_plus' in explanations:
                        best_map = explanations['gradcam_plus']
                        best_method = 'Grad-CAM++'
                    elif 'shap' in explanations:
                        best_map = explanations['shap']
                        if len(best_map.shape) == 3:
                            best_map = np.sum(np.abs(best_map), axis=0)
                        best_map = (best_map - best_map.min()) / (best_map.max() - best_map.min() + 1e-8)
                        best_method = 'SHAP'
                    
                    if best_map is not None:
                        # Extract morphological features
                        try:
                            features = morphological_analyzer.analyze_activation_map(original_img, best_map)
                        except Exception as morph_error:
                            st.error(f"Morphological analysis failed: {str(morph_error)}")
                            features = None
                        
                        if features is not None:
                            # Generate clinical description
                            clinical_description = descriptor_generator.generate_description(
                                features, predicted_label, confidence)
                            
                            # Display morphological analysis
                            col_morph1, col_morph2 = st.columns(2)
                            
                            with col_morph1:
                                st.write("**Quantitative Features:**")
                                st.write(f"• Tissue area highlighted: {features['tissue_area_percent']:.1f}%")
                                st.write(f"• Dominant stain: {features['stain_analysis']['dominant_stain']}")
                                st.write(f"• Cellular entropy: {features['texture_features']['entropy']:.2f}")
                                st.write(f"• Edge density: {features['texture_features']['edge_density']:.3f}")
                                st.write(f"• Number of regions: {features['morphological_features']['num_regions']}")
                            
                            with col_morph2:
                                st.write("**Color Analysis:**")
                                mean_rgb = features['color_features']['mean_rgb']
                                st.write(f"• Mean RGB: ({mean_rgb[0]:.2f}, {mean_rgb[1]:.2f}, {mean_rgb[2]:.2f})")
                                st.write(f"• Brightness: {features['color_features']['brightness']:.2f}")
                                st.write(f"• Contrast: {features['color_features']['contrast']:.2f}")
                                
                                # H&E stain analysis
                                stain = features['stain_analysis']
                                st.write(f"• Hematoxylin intensity: {stain['hematoxylin_intensity']:.3f}")
                                st.write(f"• Eosin intensity: {stain['eosin_intensity']:.3f}")
                            
                            # Clinical interpretation
                            st.write("**Clinical Interpretation:**")
                            st.info(clinical_description)
                            
                            # Store morphological results
                            st.session_state.explanation_results['morphological_features'] = features
                            st.session_state.explanation_results['clinical_description'] = clinical_description
                            st.session_state.explanation_results['analysis_method'] = best_method
                        
                except Exception as e:
                    st.warning(f"Morphological analysis failed: {str(e)}")
    
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
    
    # Save analysis results and generate report
    if uploaded_file is not None:
        col_save1, col_save2 = st.columns(2)
        
        with col_save1:
            if st.button("💾 Save Analysis Results (JSON)"):
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
        
        with col_save2:
            # Generate comprehensive report if explainability was run
            if hasattr(st.session_state, 'explanation_results') and st.session_state.explanation_results:
                if st.button("📄 Download Comprehensive Report (HTML)"):
                    try:
                        html_report = generate_html_report(st.session_state.explanation_results, uploaded_file.name)
                        
                        # Create download
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"interpretability_report_{timestamp}.html"
                        
                        st.download_button(
                            label="📥 Download HTML Report",
                            data=html_report,
                            file_name=filename,
                            mime="text/html",
                            help="Download comprehensive interpretability report"
                        )
                        
                        st.success("Report generated successfully! Click the download button above.")
                        
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")
            else:
                st.info("💡 Run explainability analysis first to generate comprehensive report")
    
    # Batch processing section
    st.markdown("---")
    st.header("📁 Batch Processing")
    st.markdown("Process multiple images for comprehensive explainability analysis")
    
    uploaded_files = st.file_uploader(
        "Choose multiple histopathology images",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        accept_multiple_files=True,
        help="Upload multiple images for batch analysis"
    )
    
    if uploaded_files:
        st.info(f"Selected {len(uploaded_files)} images for batch processing")
        
        if st.button("🚀 Start Batch Analysis"):
            try:
                from explainability.comprehensive_explainer import ComprehensiveExplainer
                
                # Initialize comprehensive explainer
                comprehensive_explainer = ComprehensiveExplainer(model, device, class_names)
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                batch_results = []
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
                    
                    try:
                        # Load and preprocess image
                        image = Image.open(uploaded_file)
                        image_tensor, image_resized = preprocess_image(image)
                        image_tensor = image_tensor.to(device)
                        
                        # Prepare original image
                        original_img = np.array(image_resized) / 255.0
                        
                        # Generate image ID
                        image_id = os.path.splitext(uploaded_file.name)[0]
                        
                        # Generate comprehensive explanation
                        result = comprehensive_explainer.generate_comprehensive_explanation(
                            image_tensor, original_img, image_id, "explainability_reports")
                        
                        batch_results.append(result)
                        
                    except Exception as e:
                        st.warning(f"Failed to process {uploaded_file.name}: {str(e)}")
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("Batch processing complete!")
                
                # Display batch summary
                st.success(f"Successfully processed {len(batch_results)} out of {len(uploaded_files)} images")
                
                if batch_results:
                    # Create batch summary
                    st.subheader("📊 Batch Summary")
                    
                    # Aggregate statistics
                    predictions = [r['metadata']['predicted_class'] for r in batch_results]
                    confidences = [r['metadata']['confidence'] for r in batch_results]
                    
                    # Class distribution
                    class_counts = {}
                    for pred in predictions:
                        class_counts[pred] = class_counts.get(pred, 0) + 1
                    
                    col_batch1, col_batch2 = st.columns(2)
                    
                    with col_batch1:
                        st.write("**Class Distribution:**")
                        for class_name, count in class_counts.items():
                            percentage = (count / len(predictions)) * 100
                            st.write(f"• {class_name}: {count} ({percentage:.1f}%)")
                    
                    with col_batch2:
                        st.write("**Confidence Statistics:**")
                        st.write(f"• Mean confidence: {np.mean(confidences):.1%}")
                        st.write(f"• Min confidence: {np.min(confidences):.1%}")
                        st.write(f"• Max confidence: {np.max(confidences):.1%}")
                    
                    # Download batch results
                    if st.button("📥 Download Batch Results"):
                        # Create batch summary JSON
                        batch_summary = {
                            'batch_metadata': {
                                'total_images': len(uploaded_files),
                                'successful_analyses': len(batch_results),
                                'analysis_timestamp': datetime.now().isoformat()
                            },
                            'class_distribution': class_counts,
                            'confidence_statistics': {
                                'mean': float(np.mean(confidences)),
                                'std': float(np.std(confidences)),
                                'min': float(np.min(confidences)),
                                'max': float(np.max(confidences))
                            },
                            'individual_results': [r['metadata'] for r in batch_results]
                        }
                        
                        batch_json = json.dumps(batch_summary, indent=2)
                        
                        st.download_button(
                            label="📥 Download Batch Summary (JSON)",
                            data=batch_json,
                            file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                
            except Exception as e:
                st.error(f"Batch processing failed: {str(e)}")


if __name__ == "__main__":
    main()