# RTV - Real-Time Virtual Try-On

Real-Time Per-Garment 3D Virtual Try-On system with temporal consistency, 60 FPS motion tracking, hardware-accelerated OpenGL offscreen rendering, and interactive Gradio Web Studio.

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOGESHWARAN-FAF/RTV.git
cd RTV
```

### 2. Environment Setup

#### Windows (1-Click Automated Setup)
Simply double-click **`setup_new_system.bat`**.  
This automatically detects Python 3.10, creates a `venv`, installs PyTorch with CUDA 12.1 acceleration, and configures all dependencies.

#### Manual / Linux / macOS
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate       # On Windows: .\venv\Scripts\activate

# Install PyTorch (CUDA 12.1 recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r RTV/requirements.txt

# Install ROMP 3D pose submodule
cd ROMP_repo/simple_romp
pip install -e .
cd ../../
```

---

## 🧥 Acquiring Pretrained Garment Models
Model weights (`*.pth`, ~700 MB each) are hosted on Hugging Face. To download them in 1 click:

- **Windows**: Double-click **`download_models.bat`**
- **CLI / Linux / Mac**:
  ```bash
  python download_models.py --mode recommended   # Core garments (~2.1 GB)
  python download_models.py --mode all           # All 8 garments (~5.6 GB)
  ```

---

## 🚀 Running the Virtual Try-On

### Interactive Gradio Web Studio (Browser UI)
- **Windows**: Double-click **`run_web_ui.bat`**
- **CLI**:
  ```bash
  cd RTV
  python colab_gradio_app.py --share
  ```

### Desktop Camera & Video Demo (PyQt5 GUI)
- **Windows**: Double-click **`run_demo.bat`** (or **`run_fullbody_demo.bat`**)
- **CLI**:
  ```bash
  cd RTV
  python demo.py                     # Live full-body webcam try-on
  python demo.py assets/sample_video.mp4  # Offline sample video try-on
  python rtl_demo.py                 # Live upper-body wardrobe try-on
  ```

---

## 📁 Repository Structure
```
├── RTV/                     # Core virtual try-on engine
│   ├── demo.py              # Full-body sequence demo
│   ├── rtl_demo.py          # Upper-body wardrobe demo
│   ├── colab_gradio_app.py  # Interactive Gradio Web Studio
│   ├── rtv_ckpts/           # Pretrained garment models directory
│   ├── SMPL/                # 3D body models (model.pkl)
│   ├── assets/              # Demo video, texture templates, icons
│   └── requirements.txt     # Python package requirements
├── ROMP_repo/               # Monocular 3D SMPL pose recovery module
├── run_demo.bat             # 1-click desktop app launcher
├── run_web_ui.bat           # 1-click browser studio launcher
├── setup_new_system.bat     # 1-click environment installer
├── download_models.bat      # 1-click model weight downloader
├── download_models.py       # Hugging Face model downloader script
├── package_project.bat      # Portable zip packager
└── PROJECT_DOCUMENTATION.md # Full technical documentation
```

