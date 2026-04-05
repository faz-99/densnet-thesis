@echo off
echo ============================================
echo  Installing dependencies for XAI Framework
echo ============================================
echo.

REM Step 1: Install PyTorch with CUDA 12.4
echo [1/4] Installing PyTorch with CUDA 12.4...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if %errorlevel% neq 0 (
    echo WARNING: CUDA 12.4 install failed, trying CUDA 12.1...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
)
echo.

REM Step 2: Install core ML libraries
echo [2/4] Installing core ML libraries (timm, captum, scikit-learn, matplotlib)...
pip install timm captum scikit-learn matplotlib seaborn tqdm numpy Pillow
echo.

REM Step 3: Install XAI libraries
echo [3/4] Installing XAI libraries (shap, lime)...
pip install shap lime
echo.

REM Step 4: Install LLM and logging libraries
echo [4/4] Installing LLM, quantisation, and logging libraries...
pip install transformers accelerate bitsandbytes wandb rouge-score
echo.

echo ============================================
echo  Installation complete!
echo ============================================
echo.
echo Verifying key packages...
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import timm; print(f'timm: {timm.__version__}')"
python -c "import captum; print(f'captum: {captum.__version__}')"
python -c "import shap; print(f'shap: {shap.__version__}')"
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
echo.
echo Done! You can now run: python run_train.py
pause
