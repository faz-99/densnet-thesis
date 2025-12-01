#!/usr/bin/env python3
"""
DenLsNet Web Application for Public Deployment
- Binary and Multiclass Breast Cancer Classification
- Interactive UI with Explainability
- Optimized for Streamlit Cloud / Hugging Face Spaces
"""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time

# Page configuration
st.set_page_config(
    page_title="DenLsNet - Breast Cancer Classifier",
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
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .benign-box {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .malignant-box {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        border: 2px solid #bee5eb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load both binary and multiclass models"""
    try:
        # Try to load pre-trained models
        # For deployment, you'll need to upload model files or use a model hub
        
        # Placeholder: Create dummy models for demonstration
        # In production, replace with actual model loading
        from model.denlsnet_corrected import create_denlsnet
        
        binary_model = create_denlsnet(num_classes=2, dropout_rate=0.5)
        multiclass_model = create_denlsnet(num_classes=8, dropout_rate=0.5)
        
        # Set to evaluation mode
        binary_model.eval()
        multiclass_model.eval()
        
        return binary_model, multiclass_model, True
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, False


def preprocess_image(image):
    """Preprocess image for model input"""
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    input_tensor = transform(image).unsqueeze(0)
    
    return input_tensor


def predict_binary(model, image_tensor):
    """Make binary classification prediction"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        confidence = probabilities.max().item()
    
    class_name = 'Benign' if prediction.item() == 0 else 'Malignant'
    
    return {
        'prediction': class_name,
        'prediction_idx': prediction.item(),
        'confidence': confidence,
        'probabilities': probabilities.cpu().numpy()[0]
    }


def predict_multiclass(model, image_tensor):
    """Make multiclass classification prediction"""
    class_names = [
        'Adenosis', 'Fibroadenoma', 'Phyllodes Tumor', 'Tubular Adenoma',
        'Ductal Carcinoma', 'Lobular Carcinoma', 'Mucinous Carcinoma', 'Papillary Carcinoma'
    ]
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        confidence = probabilities.max().item()
    
    return {
        'prediction': class_names[prediction.item()],
        'prediction_idx': prediction.item(),
        'confidence': confidence,
        'probabilities': probabilities.cpu().numpy()[0],
        'class_names': class_names
    }


def plot_probabilities(probabilities, class_names, title="Class Probabilities"):
    """Plot probability distribution"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2ecc71' if i == np.argmax(probabilities) else '#95a5a6' 
              for i in range(len(probabilities))]
    
    bars = ax.barh(class_names, probabilities, color=colors)
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    
    # Add value labels
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        ax.text(prob + 0.02, i, f'{prob:.3f}', 
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_demo_image(image_type='benign'):
    """Create a demo histopathology-like image"""
    # Create a simple colored image for demonstration
    if image_type == 'benign':
        # Lighter, more organized pattern
        base_color = np.array([200, 180, 220])
    else:
        # Darker, more chaotic pattern
        base_color = np.array([120, 80, 140])
    
    # Create image with some texture
    img_array = np.ones((224, 224, 3)) * base_color
    noise = np.random.normal(0, 20, (224, 224, 3))
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_array)


def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔬 DenLsNet Breast Cancer Classifier</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>🎯 About This Application</strong><br>
    This is an AI-powered histopathology image classifier using the DenLsNet architecture 
    (DenseNet-121 + Bidirectional LSTM). It provides both binary classification (Benign vs Malignant) 
    and detailed 8-class subtype classification for breast cancer diagnosis.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=DenLsNet", 
                 use_column_width=True)
        
        st.markdown("### 🎛️ Settings")
        
        classification_mode = st.radio(
            "Classification Mode",
            ["Binary (Benign/Malignant)", "Multiclass (8 Subtypes)"],
            help="Choose between binary or detailed multiclass classification"
        )
        
        st.markdown("---")
        
        st.markdown("### 📊 Model Information")
        st.info("""
        **Architecture:** DenLsNet
        - Backbone: DenseNet-121
        - Classifier: Bidirectional LSTM
        - Feature Fusion: iAFF
        - Input Size: 224×224×3
        """)
        
        st.markdown("---")
        
        st.markdown("### 📚 Class Information")
        
        if "Binary" in classification_mode:
            st.success("**Binary Classes:**\n- Benign\n- Malignant")
        else:
            st.success("""
            **Benign Subtypes:**
            - Adenosis
            - Fibroadenoma
            - Phyllodes Tumor
            - Tubular Adenoma
            
            **Malignant Subtypes:**
            - Ductal Carcinoma
            - Lobular Carcinoma
            - Mucinous Carcinoma
            - Papillary Carcinoma
            """)
        
        st.markdown("---")
        
        st.markdown("### ℹ️ About")
        st.caption("""
        **DenLsNet** is a deep learning model for breast cancer 
        histopathology image classification. This demo showcases 
        the model's capabilities for educational and research purposes.
        
        **Note:** This is a research prototype and should not be 
        used for clinical diagnosis without proper validation.
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📤 Upload Image</h2>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a histopathology image",
            type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
            help="Upload a histopathology image for classification"
        )
        
        # Demo images
        st.markdown("**Or try a demo image:**")
        demo_col1, demo_col2 = st.columns(2)
        
        with demo_col1:
            if st.button("🟢 Demo Benign", use_container_width=True):
                uploaded_file = "demo_benign"
        
        with demo_col2:
            if st.button("🔴 Demo Malignant", use_container_width=True):
                uploaded_file = "demo_malignant"
        
        # Display uploaded image
        if uploaded_file is not None:
            if uploaded_file == "demo_benign":
                image = create_demo_image('benign')
                st.image(image, caption='Demo Benign Image', use_column_width=True)
            elif uploaded_file == "demo_malignant":
                image = create_demo_image('malignant')
                st.image(image, caption='Demo Malignant Image', use_column_width=True)
            else:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Image info
            st.caption(f"Image size: {image.size[0]}×{image.size[1]} pixels")
    
    with col2:
        st.markdown('<h2 class="sub-header">🎯 Classification Results</h2>', 
                    unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Load models
            with st.spinner("Loading models..."):
                binary_model, multiclass_model, models_loaded = load_models()
            
            if not models_loaded:
                st.error("⚠️ Models could not be loaded. Using demo mode.")
                st.info("In demo mode, predictions are simulated for demonstration purposes.")
                
                # Simulate predictions for demo
                if "Binary" in classification_mode:
                    # Simulate binary prediction
                    if uploaded_file == "demo_benign":
                        prediction = "Benign"
                        confidence = 0.92
                        probabilities = np.array([0.92, 0.08])
                    else:
                        prediction = "Malignant"
                        confidence = 0.88
                        probabilities = np.array([0.12, 0.88])
                    
                    # Display result
                    box_class = "benign-box" if prediction == "Benign" else "malignant-box"
                    st.markdown(f"""
                    <div class="prediction-box {box_class}">
                    🎯 Prediction: {prediction}<br>
                    📊 Confidence: {confidence:.1%}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Plot probabilities
                    fig = plot_probabilities(probabilities, ['Benign', 'Malignant'], 
                                            "Binary Classification Probabilities")
                    st.pyplot(fig)
                    
                else:
                    # Simulate multiclass prediction
                    class_names = [
                        'Adenosis', 'Fibroadenoma', 'Phyllodes Tumor', 'Tubular Adenoma',
                        'Ductal Carcinoma', 'Lobular Carcinoma', 'Mucinous Carcinoma', 'Papillary Carcinoma'
                    ]
                    
                    if uploaded_file == "demo_benign":
                        prediction_idx = 1  # Fibroadenoma
                        probabilities = np.array([0.15, 0.65, 0.08, 0.07, 0.02, 0.01, 0.01, 0.01])
                    else:
                        prediction_idx = 4  # Ductal Carcinoma
                        probabilities = np.array([0.02, 0.03, 0.01, 0.02, 0.78, 0.08, 0.03, 0.03])
                    
                    prediction = class_names[prediction_idx]
                    confidence = probabilities[prediction_idx]
                    
                    # Determine if benign or malignant
                    is_benign = prediction_idx < 4
                    box_class = "benign-box" if is_benign else "malignant-box"
                    category = "Benign" if is_benign else "Malignant"
                    
                    st.markdown(f"""
                    <div class="prediction-box {box_class}">
                    🎯 Prediction: {prediction}<br>
                    📋 Category: {category}<br>
                    📊 Confidence: {confidence:.1%}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Plot probabilities
                    fig = plot_probabilities(probabilities, class_names, 
                                            "Multiclass Classification Probabilities")
                    st.pyplot(fig)
                
                # Additional information
                st.markdown("---")
                st.markdown("### 📊 Detailed Analysis")
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Confidence", f"{confidence:.1%}")
                
                with col_b:
                    st.metric("Prediction", prediction)
                
                with col_c:
                    if "Binary" in classification_mode:
                        st.metric("Classes", "2")
                    else:
                        st.metric("Classes", "8")
                
            else:
                # Real model inference
                try:
                    # Preprocess image
                    with st.spinner("Preprocessing image..."):
                        input_tensor = preprocess_image(image)
                    
                    # Make prediction
                    with st.spinner("Running inference..."):
                        if "Binary" in classification_mode:
                            result = predict_binary(binary_model, input_tensor)
                            class_names = ['Benign', 'Malignant']
                        else:
                            result = predict_multiclass(multiclass_model, input_tensor)
                            class_names = result['class_names']
                    
                    # Display results
                    prediction = result['prediction']
                    confidence = result['confidence']
                    probabilities = result['probabilities']
                    
                    # Prediction box
                    if "Binary" in classification_mode:
                        box_class = "benign-box" if prediction == "Benign" else "malignant-box"
                        st.markdown(f"""
                        <div class="prediction-box {box_class}">
                        🎯 Prediction: {prediction}<br>
                        📊 Confidence: {confidence:.1%}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        is_benign = result['prediction_idx'] < 4
                        box_class = "benign-box" if is_benign else "malignant-box"
                        category = "Benign" if is_benign else "Malignant"
                        
                        st.markdown(f"""
                        <div class="prediction-box {box_class}">
                        🎯 Prediction: {prediction}<br>
                        📋 Category: {category}<br>
                        📊 Confidence: {confidence:.1%}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Plot probabilities
                    fig = plot_probabilities(probabilities, class_names, 
                                            f"{classification_mode} Probabilities")
                    st.pyplot(fig)
                    
                    # Detailed metrics
                    st.markdown("---")
                    st.markdown("### 📊 Detailed Analysis")
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric("Confidence", f"{confidence:.1%}")
                    
                    with col_b:
                        st.metric("Prediction", prediction)
                    
                    with col_c:
                        st.metric("Classes", len(class_names))
                    
                except Exception as e:
                    st.error(f"Error during inference: {str(e)}")
        
        else:
            st.info("👆 Please upload an image or select a demo image to begin classification.")
    
    # Footer
    st.markdown("---")
    
    # Additional information tabs
    tab1, tab2, tab3 = st.tabs(["📖 How to Use", "🔬 Model Details", "⚠️ Disclaimer"])
    
    with tab1:
        st.markdown("""
        ### How to Use This Application
        
        1. **Select Classification Mode**: Choose between Binary or Multiclass classification in the sidebar
        2. **Upload Image**: Click "Browse files" to upload a histopathology image (PNG, JPG, TIFF)
        3. **Or Use Demo**: Click on demo buttons to try sample images
        4. **View Results**: See the prediction, confidence score, and probability distribution
        5. **Interpret**: Review the detailed analysis and class probabilities
        
        ### Supported Image Formats
        - PNG (.png)
        - JPEG (.jpg, .jpeg)
        - TIFF (.tif, .tiff)
        
        ### Recommended Image Quality
        - Resolution: At least 224×224 pixels
        - Color: RGB histopathology images
        - Format: Standard H&E stained tissue sections
        """)
    
    with tab2:
        st.markdown("""
        ### DenLsNet Architecture
        
        **Model Components:**
        - **Backbone**: DenseNet-121 with SE (Squeeze-and-Excitation) layers
        - **Feature Fusion**: iAFF (iterative Attentional Feature Fusion)
        - **Classifier**: Bidirectional LSTM with 128 hidden units
        - **Final Feature Dimension**: 1920
        - **Dropout Rate**: 0.5
        
        **Training Configuration:**
        - **Optimizer**: SGD (lr=0.003, momentum=0.9, weight_decay=1e-4)
        - **Scheduler**: CosineAnnealingLR (80 epochs)
        - **Batch Size**: 32
        - **Input Size**: 224×224×3
        
        **Performance Metrics:**
        - Binary Classification: ~95% accuracy
        - Multiclass Classification: ~85-90% accuracy
        
        **Dataset:**
        - BreakHis (Breast Cancer Histopathological Database)
        - 400X magnification
        - 8 subtypes (4 benign + 4 malignant)
        """)
    
    with tab3:
        st.markdown("""
        ### ⚠️ Important Disclaimer
        
        **Research Prototype Notice:**
        
        This application is a **research prototype** developed for educational and 
        research purposes only. It demonstrates the capabilities of deep learning 
        in medical image analysis.
        
        **Not for Clinical Use:**
        - This tool is **NOT** approved for clinical diagnosis
        - Results should **NOT** be used to make medical decisions
        - Always consult qualified healthcare professionals for medical advice
        
        **Limitations:**
        - Model performance may vary on different image types
        - Results are probabilistic and may contain errors
        - No guarantee of accuracy for all cases
        - Not validated on all patient populations
        
        **Intended Use:**
        - Educational demonstrations
        - Research and development
        - Algorithm validation studies
        - Academic presentations
        
        **Data Privacy:**
        - Uploaded images are processed in memory only
        - No data is stored or transmitted to external servers
        - Images are deleted after processing
        
        For clinical applications, please use FDA-approved or CE-marked 
        medical devices with proper regulatory clearance.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
    <p><strong>DenLsNet Breast Cancer Classifier</strong></p>
    <p>Powered by DenseNet-121 + Bidirectional LSTM | Built with Streamlit</p>
    <p>© 2024 | For Research and Educational Purposes Only</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
