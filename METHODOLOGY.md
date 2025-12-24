# METHODOLOGY - DenLsNet: Multi-Class Medical Image Classification with Explainable AI

## 3. METHODOLOGY

### 3.1 Overview

This research proposes **DenLsNet** (DenseNet with LSTM and SE attention), a deep learning framework for histopathology image classification with comprehensive explainability analysis. The methodology encompasses three main contributions: (1) multi-class extension for fine-grained breast cancer subtype classification, (2) stain normalization ablation study, and (3) quantitative interpretability framework.

### 3.2 Dataset

#### 3.2.1 BreakHis Dataset
- **Source**: Breast Cancer Histopathological Database (BreakHis)
- **Magnification**: 400X microscopy images
- **Image Resolution**: 700×460 pixels
- **Total Samples**: 7,909 histopathology images
- **Classes**: 8 subtypes (4 benign + 4 malignant)

**Benign Subtypes (Classes 0-3):**
- Adenosis: Benign proliferative breast disease
- Fibroadenoma: Common benign breast tumor
- Phyllodes Tumor: Rare benign breast tumor
- Tubular Adenoma: Benign epithelial tumor

**Malignant Subtypes (Classes 4-7):**
- Ductal Carcinoma: Most common breast cancer type
- Lobular Carcinoma: Second most common type
- Mucinous Carcinoma: Rare mucin-producing cancer
- Papillary Carcinoma: Rare papillary growth pattern

#### 3.2.2 Data Preprocessing Pipeline

**Step 1: Stain Normalization (Ablation Study)**
Three experimental variants were implemented:

1. **Macenko Normalization**
   - Optical density-based stain separation
   - H&E stain matrix extraction using PCA
   - Concentration normalization to target image
   - Mathematical formulation:
     ```
     OD = -log(RGB/255)
     StainMatrix = PCA(OD, n_components=2)
     Concentrations = OD × StainMatrix^(-1)
     Normalized = exp(-TargetStains × NormalizedConcentrations) × 255
     ```

2. **Reinhard Normalization**
   - LAB color space statistics matching
   - Mean and standard deviation alignment
   - Mathematical formulation:
     ```
     LAB_normalized = (LAB_source - μ_source) × (σ_target/σ_source) + μ_target
     ```

3. **Baseline (No Normalization)**
   - Direct image processing without stain normalization
   - Control condition for ablation study

**Step 2: Image Enhancement**
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Clip limit: 2.0
  - Tile grid size: 8×8
- Tissue region detection and background removal
- Gaussian blur for noise reduction (kernel: 3×3)

**Step 3: Data Augmentation**

*Geometric Augmentations:*
- Random rotation: [0°, 90°, 180°, 270°]
- Horizontal flip: p=0.5
- Vertical flip: p=0.5

*Color Augmentations:*
- Brightness adjustment: ±20%
- Contrast adjustment: ±20%
- Saturation adjustment: ±20%
- Hue shift: ±15%

*Advanced Augmentations:*
- Random erasing: p=0.15, area=2-15%
- Cutout augmentation for robustness

**Step 4: Normalization**
- Image resize: 224×224 pixels
- Pixel normalization:
  - Mean: (0.5613, 0.5778, 0.6032)
  - Std: (0.2114, 0.1957, 0.1590)
  - Computed from BreakHis training set

**Step 5: Class Balancing**
- Weighted sampling for imbalanced classes
- Focal loss implementation (α=1.0, γ=2.0)
- Label smoothing (ε=0.1) for regularization

### 3.3 Model Architecture

#### 3.3.1 DenLsNet Architecture Components

**Base Architecture: DenseNet-201**
- Pre-trained on ImageNet (transfer learning)
- 201 layers with dense connectivity
- Feature reuse through concatenation
- Parameters: ~20M trainable parameters

**Component 1: SE (Squeeze-and-Excitation) Layers**
- Channel-wise attention mechanism
- Integrated into all dense blocks and transitions
- Architecture:
  ```
  SE(x) = x ⊗ σ(W₂ · ReLU(W₁ · GAP(x)))
  where GAP = Global Average Pooling
        σ = Sigmoid activation
        ⊗ = Channel-wise multiplication
  ```
- Reduction ratio: 16
- Placement: After each dense block and transition layer

**Component 2: iAFF (iterative Attentional Feature Fusion)**
- Multi-scale feature fusion mechanism
- Two fusion points in the architecture:
  - Fusion 1: After DenseBlock2 (512→1792 channels)
  - Fusion 2: After DenseBlock3 (1792→1920 channels)
- Architecture:
  ```
  iAFF(x, y) = x ⊗ M(x,y) + y ⊗ (1-M(x,y))
  where M(x,y) = σ(Conv(x+y))
  ```

**Component 3: LSTM Classification Head**
- Temporal feature processing
- Architecture:
  ```
  Input: (batch, 1920, 7, 7) → AdaptiveAvgPool → (batch, 1920, 1, 1)
  Reshape: (batch, 1, 1920)
  LSTM: hidden_size=256, dropout=0.5
  Output: (batch, 256)
  ```

**Component 4: Multi-Class Classifier**
- Fully connected layers:
  ```
  FC1: 256 → 128 (ReLU, Dropout=0.5)
  FC2: 128 → 8 (Softmax for 8 classes)
  ```

#### 3.3.2 Complete Forward Pass

```
Input Image (3×224×224)
    ↓
DenseNet Conv0 + Norm0 + Pool0
    ↓
DenseBlock1 (with SE) → Transition1 (with SE)
    ↓
DenseBlock2 (with SE) → Transition2 (with SE)
    ↓ (512 channels)
Conv2D_1 (512→1792) ──┐
    ↓                  │
DenseBlock3 (with SE)  │
    ↓                  │
iAFF Fusion 1 ←────────┘
    ↓ (1792 channels)
Conv2D_2 (1792→1920) ──┐
    ↓                   │
Transition3 (with SE)   │
    ↓                   │
DenseBlock4 (with SE)   │
    ↓                   │
iAFF Fusion 2 ←─────────┘
    ↓
Norm5 (Final Features: 1920 channels)
    ↓
AdaptiveAvgPool (1920×1×1)
    ↓
LSTM (256 hidden units)
    ↓
FC Classifier (256→128→8)
    ↓
Softmax Output (8 classes)
```

### 3.4 Training Strategy

#### 3.4.1 Loss Functions

**Primary Loss: Categorical Cross-Entropy**
```
L_CE = -∑(y_i × log(ŷ_i))
```

**Alternative Loss: Focal Loss (for class imbalance)**
```
L_Focal = -α(1-p_t)^γ × log(p_t)
where α=1.0, γ=2.0
```

**Regularization: Label Smoothing**
```
y_smooth = y(1-ε) + ε/K
where ε=0.1, K=8 classes
```

#### 3.4.2 Optimization Configuration

**Optimizer: AdamW**
- Learning rate: 0.003
- Weight decay: 0.05
- Betas: (0.9, 0.999)
- Epsilon: 1e-8

**Learning Rate Schedule:**
- Warmup epochs: 5 (linear warmup)
- Scheduler: MultiStepLR
- Milestones: [20, 40, 60, 80]
- Gamma: 0.5
- Minimum LR: 1e-6

**Training Hyperparameters:**
- Batch size: 32
- Maximum epochs: 100
- Early stopping: patience=15
- Dropout rate: 0.5
- Drop path rate: 0.8

#### 3.4.3 Training Protocol

1. **Initialization**: Load ImageNet pre-trained DenseNet-201
2. **Freeze-Unfreeze Strategy**:
   - Epochs 1-10: Freeze DenseNet backbone, train classifier only
   - Epochs 11-100: Unfreeze all layers, fine-tune end-to-end
3. **Gradient Clipping**: Max norm = 1.0
4. **Mixed Precision Training**: FP16 for efficiency
5. **Checkpoint Strategy**: Save best model based on validation F1-score

### 3.5 Experimental Design: Ablation Study

#### 3.5.1 Model Variants

**Variant 1: DenLsNet-MC-None (Baseline)**
- No stain normalization
- Standard preprocessing only
- Control condition

**Variant 2: DenLsNet-MC-Macenko**
- Macenko stain normalization
- Optical density-based approach
- H&E stain separation

**Variant 3: DenLsNet-MC-Reinhard**
- Reinhard color normalization
- LAB color space statistics
- Mean/std alignment

#### 3.5.2 Evaluation Protocol

**Training Phase:**
- 5-fold cross-validation for robust evaluation
- Stratified splits to maintain class distribution
- Independent training for each variant

**Testing Phase:**
- Hold-out test set (20% of data)
- Same test set for all variants
- Consistent evaluation metrics

### 3.6 Explainability Framework (DenLsNet-XAI)

#### 3.6.1 Visual Explanation Methods

**Method 1: Grad-CAM (Gradient-weighted Class Activation Mapping)**
- Target layer: densenet.features.norm5
- Mathematical formulation:
  ```
  α_k^c = (1/Z) ∑∑ ∂y^c/∂A_ij^k
  L_Grad-CAM^c = ReLU(∑ α_k^c × A^k)
  ```
- Output: Class-discriminative localization map

**Method 2: Grad-CAM++ (Improved Grad-CAM)**
- Weighted gradient computation
- Better localization for multiple instances
- Mathematical formulation:
  ```
  α_ij^kc = ReLU(∂y^c/∂A_ij^k) / (∑∑ ReLU(∂y^c/∂A_ij^k))
  L_Grad-CAM++^c = ReLU(∑∑∑ α_ij^kc × ReLU(∂y^c/∂A_ij^k) × A_ij^k)
  ```

**Method 3: SHAP (SHapley Additive exPlanations)**
- Game-theoretic approach
- Pixel-level attribution values
- Background samples: 50 images
- Perturbation samples: 500
- Output: Shapley values for each pixel

**Method 4: LIME (Local Interpretable Model-agnostic Explanations)**
- Superpixel-based segmentation (SLIC algorithm)
- Local linear approximation
- Number of superpixels: 50-100
- Perturbation samples: 1000
- Output: Superpixel importance weights

**Method 5: Integrated Gradients (Extended Analysis)**
- Multi-baseline approach
- Baselines: [zeros, ones, random noise]
- Integration steps: 50
- Mathematical formulation:
  ```
  IG_i(x) = (x_i - x'_i) × ∫₀¹ ∂F(x' + α(x-x'))/∂x_i dα
  ```

**Method 6: Occlusion Sensitivity**
- Sliding window approach
- Patch size: 30×30 pixels
- Stride: 15 pixels
- Baseline value: 0.0 (black)

#### 3.6.2 Quantitative Evaluation Metrics

**Metric 1: Insertion AUC**
- Measures explanation quality by pixel addition
- Protocol:
  1. Start with black image
  2. Progressively add pixels in order of importance
  3. Record prediction confidence at each step
  4. Compute AUC of insertion curve
- Higher is better (closer to 1.0)
- Interpretation: Good explanations achieve high confidence with fewer pixels

**Metric 2: Deletion AUC**
- Measures explanation quality by pixel removal
- Protocol:
  1. Start with original image
  2. Progressively remove pixels in order of importance
  3. Record prediction confidence at each step
  4. Compute AUC of deletion curve
- Lower is better (closer to 0.0)
- Interpretation: Good explanations cause rapid confidence drop

**Metric 3: Stability Analysis**
- Measures consistency under perturbations
- Protocol:
  1. Generate baseline explanation
  2. Add Gaussian noise (σ=0.1) to input
  3. Generate perturbed explanation
  4. Compute Pearson correlation
  5. Repeat 10 times, average correlations
- Higher is better (closer to 1.0)
- Interpretation: Stable explanations are robust to input changes

**Metric 4: Localization Accuracy (IoU)**
- Requires ground truth ROI annotations
- Metrics computed:
  - Intersection over Union (IoU)
  - Dice coefficient
  - Precision and Recall
- Threshold: Top 20% of explanation pixels

**Metric 5: Processing Time**
- Computational efficiency measurement
- Average time per explanation
- Hardware: CPU/GPU specifications

#### 3.6.3 Validation Techniques

**Model Randomization Check**
- Compare explanations from trained vs. randomized model
- Expectation: Low similarity for good explanations
- Metric: Spearman correlation

**Model Degradation Test**
- Zero out last N parameters
- Compare explanations before/after degradation
- Expectation: Explanations should change significantly

**Random Baseline Comparison**
- Generate random attribution maps
- Compare with actual explanations
- Expectation: Actual explanations should outperform random

### 3.7 Evaluation Metrics

#### 3.7.1 Classification Performance Metrics

**Per-Class Metrics:**
- Precision: TP/(TP+FP)
- Recall (Sensitivity): TP/(TP+FN)
- F1-Score: 2×(Precision×Recall)/(Precision+Recall)
- Specificity: TN/(TN+FP)

**Overall Metrics:**
- Accuracy: (TP+TN)/(TP+TN+FP+FN)
- Macro-averaged F1: Average of per-class F1 scores
- Weighted F1: Class-weighted average
- Cohen's Kappa: Inter-rater agreement

**Confusion Matrix Analysis:**
- 8×8 confusion matrix for multi-class
- Per-class error analysis
- Misclassification patterns

#### 3.7.2 Statistical Analysis

**Significance Testing:**
- Paired t-test for model comparison
- Wilcoxon signed-rank test (non-parametric)
- Bonferroni correction for multiple comparisons
- Significance level: α=0.05

**Confidence Intervals:**
- 95% confidence intervals for all metrics
- Bootstrap resampling (n=1000)

### 3.8 Implementation Details

#### 3.8.1 Software and Libraries

**Deep Learning Framework:**
- PyTorch 2.0+
- torchvision for data augmentation
- timm (PyTorch Image Models) for DenseNet-201

**Explainability Libraries:**
- Custom Grad-CAM implementation
- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)
- Captum for Integrated Gradients

**Data Processing:**
- NumPy, OpenCV for image processing
- scikit-image for stain normalization
- Pillow for image I/O

**Visualization:**
- Matplotlib, Seaborn for plotting
- Plotly for interactive visualizations

**Evaluation:**
- scikit-learn for metrics
- pandas for data management

#### 3.8.2 Hardware Configuration

**Training Environment:**
- GPU: NVIDIA GPU (CUDA-enabled) or CPU fallback
- RAM: 16GB minimum
- Storage: SSD for fast data loading

**Training Time:**
- Per epoch: ~15-20 minutes (GPU)
- Total training: ~25-30 hours per variant
- Complete ablation study: ~75-90 hours

#### 3.8.3 Reproducibility

**Random Seed Control:**
- PyTorch seed: 42
- NumPy seed: 42
- Python random seed: 42
- CUDA deterministic mode: enabled

**Version Control:**
- Git repository for code versioning
- Model checkpoints saved with metadata
- Configuration files for all experiments

**Documentation:**
- Detailed README with setup instructions
- Jupyter notebooks for analysis
- Automated logging of all experiments

### 3.9 Experimental Workflow

#### Phase 1: Data Preparation (Week 1-2)
1. Download and organize BreakHis dataset
2. Create multi-class directory structure
3. Generate stain-normalized variants
4. Split data (train/validation/test)
5. Compute dataset statistics

#### Phase 2: Model Development (Week 3-4)
1. Implement DenLsNet architecture
2. Integrate SE layers and iAFF fusion
3. Implement LSTM classification head
4. Validate forward pass and gradients
5. Unit testing of components

#### Phase 3: Training and Ablation (Week 5-8)
1. Train DenLsNet-MC-None (baseline)
2. Train DenLsNet-MC-Macenko
3. Train DenLsNet-MC-Reinhard
4. Hyperparameter tuning
5. Cross-validation experiments

#### Phase 4: Evaluation (Week 9-10)
1. Test set evaluation for all variants
2. Compute classification metrics
3. Statistical significance testing
4. Generate performance visualizations
5. Confusion matrix analysis

#### Phase 5: Explainability Analysis (Week 11-13)
1. Implement XAI methods
2. Generate visual explanations
3. Quantitative evaluation (Insertion/Deletion AUC)
4. Stability analysis
5. Method comparison and ranking

#### Phase 6: Analysis and Documentation (Week 14-16)
1. Comprehensive results analysis
2. Clinical interpretation of findings
3. Thesis writing and documentation
4. Interactive UI development
5. Final presentation preparation

### 3.10 Validation Strategy

#### 3.10.1 Internal Validation
- 5-fold stratified cross-validation
- Consistent splits across all variants
- Per-fold performance reporting

#### 3.10.2 External Validation
- Hold-out test set (never seen during training)
- Independent evaluation
- Generalization assessment

#### 3.10.3 Explainability Validation
- Visual inspection by domain experts
- Quantitative metrics (Insertion/Deletion AUC)
- Stability under perturbations
- Comparison with baseline methods

### 3.11 Ethical Considerations

- Dataset usage complies with BreakHis terms
- No patient identifiable information
- Model intended for research purposes
- Clinical validation required before deployment
- Transparent reporting of limitations

### 3.12 Expected Contributions

1. **Multi-Class Extension**: Fine-grained 8-class breast cancer classification
2. **Stain Normalization Study**: Quantitative impact analysis on performance
3. **Quantitative XAI Framework**: Novel metrics for interpretability evaluation
4. **Comprehensive Comparison**: Benchmarking of multiple XAI methods
5. **Reproducible Pipeline**: Complete open-source implementation

---

**Summary**: This methodology presents a comprehensive approach to developing and evaluating DenLsNet for multi-class histopathology classification with explainable AI. The systematic ablation study, quantitative interpretability framework, and rigorous evaluation protocol ensure robust and clinically relevant results.
