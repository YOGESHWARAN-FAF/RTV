# Real-Time Virtual Try-On (RTV) - Portable Distribution Guide

This package contains the complete, portable **Real-Time Virtual Try-On (RTV)** project optimized for transfer between systems.

Massive machine-specific files (`venv` directory with Windows DLLs, 5.6 GB Git LFS cache, and raw video dataset training folders) have been excluded to keep the package lightweight and fast to copy.

---

## 🚀 Quick Setup on Destination System

### Windows
1. Make sure **Python 3.10 (64-bit)** is installed and added to PATH.
2. Double-click **`setup_new_system.bat`** to automatically create the virtual environment and install all dependencies.
3. Start using RTV:
   - **`run_demo.bat`**: Interactive desktop application with live webcam / sample video.
   - **`run_web_ui.bat`**: Gradio browser-based virtual try-on studio.

---

### Linux / macOS
1. Open terminal inside this extracted directory:
   ```bash
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate

   # Install PyTorch
   pip install torch torchvision

   # Install requirements
   pip install -r RTV/requirements.txt

   # Install ROMP module
   cd ROMP_repo/simple_romp
   pip install -e .
   cd ../../
   ```
2. Run the application:
   ```bash
   cd RTV
   python colab_gradio_app.py --share
   # Or desktop demo:
   python demo.py assets/sample_video.mp4
   ```

---

## 📁 Directory Structure
```
├── RTV/                      # Core virtual try-on engine
│   ├── demo.py               # Full-body sequence demo
│   ├── rtl_demo.py           # Upper-body demo
│   ├── colab_gradio_app.py   # Gradio Web Studio
│   ├── rtv_ckpts/            # Pretrained garment neural models
│   ├── SMPL/                 # 3D body models (model.pkl)
│   └── assets/               # Demo videos, SMPL geometries, templates
├── ROMP_repo/                # Monocular 3D human pose recovery module
├── run_demo.bat              # 1-click launcher (Desktop UI)
├── run_web_ui.bat            # 1-click launcher (Web Studio)
├── setup_new_system.bat      # 1-click installer on new machines
└── PROJECT_DOCUMENTATION.md  # Complete architecture reference
```

