#!/bin/bash
# Automated Deployment Script for Streamlit Cloud

echo "🚀 DenLsNet Deployment to Streamlit Cloud"
echo "=========================================="

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install git first."
    exit 1
fi

# Get user input
read -p "Enter your GitHub username: " GITHUB_USERNAME
read -p "Enter repository name (default: denlsnet-app): " REPO_NAME
REPO_NAME=${REPO_NAME:-denlsnet-app}

echo ""
echo "📋 Configuration:"
echo "   GitHub Username: $GITHUB_USERNAME"
echo "   Repository Name: $REPO_NAME"
echo ""

# Initialize git if not already initialized
if [ ! -d ".git" ]; then
    echo "📦 Initializing git repository..."
    git init
    git branch -M main
fi

# Create deployment directory structure
echo "📁 Creating deployment structure..."
mkdir -p deployment_package
mkdir -p deployment_package/model
mkdir -p deployment_package/config
mkdir -p deployment_package/.streamlit

# Copy necessary files
echo "📄 Copying files..."
cp app_deployment.py deployment_package/
cp requirements_deployment.txt deployment_package/requirements.txt
cp README_DEPLOYMENT.md deployment_package/README.md
cp -r model/*.py deployment_package/model/ 2>/dev/null || echo "⚠️  Model files not found"
cp -r config/*.py deployment_package/config/ 2>/dev/null || echo "⚠️  Config files not found"
cp .streamlit/config.toml deployment_package/.streamlit/
cp .gitignore deployment_package/

# Create __init__.py files
touch deployment_package/model/__init__.py
touch deployment_package/config/__init__.py

# Create README for GitHub
cat > deployment_package/README.md << EOF
# 🔬 DenLsNet Breast Cancer Classifier

AI-powered histopathology image classifier using DenseNet-121 + Bidirectional LSTM architecture.

## 🌐 Live Demo

Visit the live application: [DenLsNet Classifier](https://$GITHUB_USERNAME-$REPO_NAME.streamlit.app)

## 🎯 Features

- **Binary Classification**: Benign vs Malignant
- **Multiclass Classification**: 8 breast cancer subtypes
- **Interactive UI**: Easy-to-use web interface
- **Real-time Predictions**: Instant classification results
- **Probability Visualization**: Detailed confidence scores

## 🏗️ Architecture

- **Backbone**: DenseNet-121 with SE layers
- **Classifier**: Bidirectional LSTM (128 hidden units)
- **Feature Fusion**: iAFF (iterative Attentional Feature Fusion)
- **Input Size**: 224×224×3
- **Final Features**: 1920 dimensions

## 🚀 Quick Start

### Local Installation

\`\`\`bash
# Clone the repository
git clone https://github.com/$GITHUB_USERNAME/$REPO_NAME.git
cd $REPO_NAME

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app_deployment.py
\`\`\`

### Docker Deployment

\`\`\`bash
# Build the image
docker build -t denlsnet-app .

# Run the container
docker run -p 8501:8501 denlsnet-app
\`\`\`

## 📊 Model Performance

### Binary Classification
- Accuracy: ~95%
- F1-Score: ~95%
- AUC: ~98%

### Multiclass Classification
- Accuracy: ~85-90%
- Macro F1-Score: ~83-88%

## 🎓 Classes

### Binary
- Benign
- Malignant

### Multiclass (8 Subtypes)

**Benign:**
- Adenosis
- Fibroadenoma
- Phyllodes Tumor
- Tubular Adenoma

**Malignant:**
- Ductal Carcinoma
- Lobular Carcinoma
- Mucinous Carcinoma
- Papillary Carcinoma

## ⚠️ Disclaimer

This is a research prototype for educational purposes only. **NOT** for clinical diagnosis.

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ using Streamlit and PyTorch**
EOF

# Navigate to deployment package
cd deployment_package

# Initialize git in deployment package
git init
git branch -M main

# Add all files
echo "➕ Adding files to git..."
git add .

# Commit
echo "💾 Creating commit..."
git commit -m "Initial deployment of DenLsNet application"

# Add remote
echo "🔗 Adding GitHub remote..."
git remote add origin "https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"

echo ""
echo "✅ Deployment package prepared!"
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Create a new repository on GitHub:"
echo "   https://github.com/new"
echo "   Repository name: $REPO_NAME"
echo "   Make it public"
echo ""
echo "2. Push the code:"
echo "   cd deployment_package"
echo "   git push -u origin main"
echo ""
echo "3. Deploy on Streamlit Cloud:"
echo "   - Go to https://share.streamlit.io"
echo "   - Click 'New app'"
echo "   - Select repository: $GITHUB_USERNAME/$REPO_NAME"
echo "   - Main file: app_deployment.py"
echo "   - Click 'Deploy'"
echo ""
echo "4. Your app will be live at:"
echo "   https://$GITHUB_USERNAME-$REPO_NAME.streamlit.app"
echo ""
echo "🎉 Happy deploying!"
