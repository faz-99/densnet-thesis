# DenLsNet Deployment Guide

## 🚀 Quick Deployment Options

This guide provides instructions for deploying the DenLsNet web application on free hosting platforms.

---

## Option 1: Streamlit Cloud (Recommended)

### Prerequisites
- GitHub account
- Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

### Steps

1. **Push Code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - DenLsNet deployment"
   git remote add origin https://github.com/YOUR_USERNAME/denlsnet-app.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your GitHub repository
   - Set main file path: `app_deployment.py`
   - Click "Deploy"

3. **Your app will be live at:**
   ```
   https://YOUR_USERNAME-denlsnet-app.streamlit.app
   ```

### Configuration

Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 10
enableCORS = false
enableXsrfProtection = true
```

---

## Option 2: Hugging Face Spaces

### Prerequisites
- Hugging Face account (free at [huggingface.co](https://huggingface.co))

### Steps

1. **Create a New Space**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Streamlit" as SDK
   - Name your space (e.g., "denlsnet-classifier")

2. **Upload Files**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/denlsnet-classifier
   cd denlsnet-classifier
   
   # Copy your files
   cp app_deployment.py app.py
   cp requirements_deployment.txt requirements.txt
   cp -r model/ .
   cp -r config/ .
   
   # Commit and push
   git add .
   git commit -m "Deploy DenLsNet"
   git push
   ```

3. **Your app will be live at:**
   ```
   https://huggingface.co/spaces/YOUR_USERNAME/denlsnet-classifier
   ```

### Configuration

Create `README.md` in your Space:
```markdown
---
title: DenLsNet Breast Cancer Classifier
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# DenLsNet Breast Cancer Classifier

AI-powered histopathology image classifier for breast cancer diagnosis.
```

---

## Option 3: Render.com

### Prerequisites
- Render account (free at [render.com](https://render.com))

### Steps

1. **Create `render.yaml`**
   ```yaml
   services:
     - type: web
       name: denlsnet-app
       env: python
       buildCommand: pip install -r requirements_deployment.txt
       startCommand: streamlit run app_deployment.py --server.port $PORT --server.address 0.0.0.0
       envVars:
         - key: PYTHON_VERSION
           value: 3.9.0
   ```

2. **Deploy**
   - Connect your GitHub repository to Render
   - Select "New Web Service"
   - Choose your repository
   - Render will auto-detect and deploy

3. **Your app will be live at:**
   ```
   https://denlsnet-app.onrender.com
   ```

---

## Option 4: Railway.app

### Prerequisites
- Railway account (free at [railway.app](https://railway.app))

### Steps

1. **Create `railway.json`**
   ```json
   {
     "build": {
       "builder": "NIXPACKS"
     },
     "deploy": {
       "startCommand": "streamlit run app_deployment.py --server.port $PORT --server.address 0.0.0.0",
       "restartPolicyType": "ON_FAILURE",
       "restartPolicyMaxRetries": 10
     }
   }
   ```

2. **Deploy**
   - Go to [railway.app](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository
   - Railway will auto-deploy

---

## 📦 File Structure for Deployment

```
denlsnet-deployment/
├── app_deployment.py              # Main Streamlit app
├── requirements_deployment.txt    # Python dependencies
├── README_DEPLOYMENT.md          # This file
├── model/
│   ├── __init__.py
│   ├── denlsnet_corrected.py    # Model architecture
│   └── SENet.py                  # SE layer implementation
├── config/
│   ├── __init__.py
│   └── training_config.py        # Configuration
├── .streamlit/
│   └── config.toml               # Streamlit config
└── .gitignore                    # Git ignore file
```

---

## 🔧 Environment Variables

For production deployment, set these environment variables:

```bash
# Optional: Model paths (if using external storage)
MODEL_BINARY_PATH=path/to/binary_model.pth
MODEL_MULTICLASS_PATH=path/to/multiclass_model.pth

# Optional: API keys for model hosting services
HUGGINGFACE_TOKEN=your_token_here
```

---

## 📊 Resource Requirements

### Minimum Requirements
- **RAM**: 2GB
- **CPU**: 1 core
- **Storage**: 500MB
- **Bandwidth**: Unlimited (for free tiers)

### Recommended for Better Performance
- **RAM**: 4GB
- **CPU**: 2 cores
- **Storage**: 1GB

---

## 🎯 Free Tier Limitations

### Streamlit Cloud
- ✅ Unlimited public apps
- ✅ 1GB RAM per app
- ✅ Unlimited bandwidth
- ❌ No custom domain on free tier
- ❌ Apps sleep after inactivity

### Hugging Face Spaces
- ✅ Unlimited public spaces
- ✅ 16GB RAM (CPU)
- ✅ Persistent storage
- ✅ Custom domain possible
- ❌ Limited GPU access on free tier

### Render.com
- ✅ 750 hours/month free
- ✅ 512MB RAM
- ❌ Apps sleep after 15 min inactivity
- ❌ Slower cold starts

### Railway.app
- ✅ $5 free credit/month
- ✅ 512MB RAM
- ✅ Fast deployment
- ❌ Limited free hours

---

## 🚀 Quick Start Commands

### Local Testing
```bash
# Install dependencies
pip install -r requirements_deployment.txt

# Run locally
streamlit run app_deployment.py

# Access at http://localhost:8501
```

### Docker Deployment (Optional)
```bash
# Build image
docker build -t denlsnet-app .

# Run container
docker run -p 8501:8501 denlsnet-app
```

---

## 🔒 Security Considerations

1. **No Sensitive Data**: Don't include API keys or credentials in code
2. **Environment Variables**: Use platform-specific env var management
3. **HTTPS**: All platforms provide HTTPS by default
4. **Rate Limiting**: Implement if needed for production
5. **Input Validation**: Already included in app

---

## 📈 Monitoring and Analytics

### Streamlit Cloud
- Built-in analytics dashboard
- View app usage and performance
- Monitor errors and logs

### Hugging Face Spaces
- View space analytics
- Monitor resource usage
- Check visitor statistics

---

## 🐛 Troubleshooting

### App Won't Start
```bash
# Check logs
streamlit run app_deployment.py --logger.level=debug

# Verify dependencies
pip list | grep -E "torch|streamlit"
```

### Memory Issues
- Reduce model size
- Use model quantization
- Implement lazy loading

### Slow Performance
- Enable caching with @st.cache_resource
- Optimize image preprocessing
- Use smaller batch sizes

---

## 📞 Support

For deployment issues:
1. Check platform-specific documentation
2. Review error logs
3. Test locally first
4. Check GitHub Issues

---

## 🎉 Success Checklist

- [ ] Code pushed to GitHub
- [ ] Requirements file updated
- [ ] App tested locally
- [ ] Platform account created
- [ ] Repository connected
- [ ] App deployed successfully
- [ ] Public URL accessible
- [ ] Functionality verified

---

## 📝 Example Deployment URLs

After deployment, your app will be accessible at:

- **Streamlit Cloud**: `https://username-denlsnet.streamlit.app`
- **Hugging Face**: `https://huggingface.co/spaces/username/denlsnet`
- **Render**: `https://denlsnet.onrender.com`
- **Railway**: `https://denlsnet.up.railway.app`

---

## 🌟 Next Steps

1. **Share Your App**: Get a public URL and share with colleagues
2. **Add Analytics**: Track usage and performance
3. **Collect Feedback**: Improve based on user input
4. **Scale Up**: Upgrade to paid tiers if needed
5. **Add Features**: Implement additional functionality

---

**Happy Deploying! 🚀**

For questions or issues, please open a GitHub issue or contact the development team.
