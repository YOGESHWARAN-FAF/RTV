#!/usr/bin/env bash
# ==============================================================================
# RTV Real-Time Virtual Try-On - Google Colab Automated Setup Script
# ==============================================================================
set -e

echo "========================================================================"
echo " [RTV] Starting Automated Google Colab Environment Setup"
echo "========================================================================"

# 1. Verify NVIDIA GPU
echo "[1/5] Checking GPU acceleration..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not detected. Please ensure your Colab runtime is set to GPU (T4 / L4 / A100)."
else
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# 2. System Packages (Headless OpenGL, EGL, Mesa, Xvfb, FFmpeg)
echo "[2/5] Installing headless system graphics libraries..."
apt-get update -qq
apt-get install -y -qq \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libgl1-mesa-glx \
    libosmesa6-dev \
    xvfb \
    ffmpeg \
    libglfw3-dev \
    libglfw3 > /dev/null 2>&1
echo "System graphics packages installed successfully."

# 3. Python Core Dependencies
echo "[3/5] Installing Python virtual try-on dependencies..."
pip install -q --upgrade pip
pip install -q pyopengl pyopengl-accelerate trimesh simple-romp gradio==6.30.0 opencv-python scipy gdown

if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt || true
fi

# 4. Install Detectron2 (Linux wheel / git source)
echo "[4/5] Installing Detectron2 for DensePose..."
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VER=$(python -c "import torch; print('cu' + torch.version.cuda.replace('.', '') if torch.cuda.is_available() else 'cpu')")
echo "Detected PyTorch: $TORCH_VER ($CUDA_VER)"

pip install -q 'git+https://github.com/facebookresearch/detectron2.git' || \
    python -m pip install -q detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/${CUDA_VER}/torch${TORCH_VER}/index.html" || true

# 5. Configure Headless Environment Variables
echo "[5/5] Configuring Headless EGL environment..."
export PYOPENGL_PLATFORM=egl
export QT_QPA_PLATFORM=offscreen

echo "========================================================================"
echo " [RTV] Setup Complete! Ready to launch interactive try-on studio."
echo " Run: python colab_gradio_app.py --share"
echo "========================================================================"
