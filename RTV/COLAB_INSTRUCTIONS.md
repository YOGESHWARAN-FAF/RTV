# RTV Real-Time Virtual Try-On — Google Colab Guide

This guide explains how to run the complete **RTV Real-Time Virtual Try-On** system with an interactive web interface on **Google Colab** (using a cloud GPU such as NVIDIA T4, L4, or A100).

---

## 🌟 Why Google Colab & How the Interface Works

On a local PC, RTV uses PyQt5 to open a desktop window. However, **Google Colab runs on a headless remote Linux container in the cloud** with no monitor or desktop window manager.

To give you a seamless experience:
- We built an **interactive Gradio Web Studio** (`colab_gradio_app.py`) that runs directly inside Google Colab.
- When launched, Gradio generates a **public shareable link** (e.g., `https://xxxx.gradio.live`).
- You can open this link on **any browser, laptop, or mobile phone**!
- It connects your device's webcam directly to the cloud GPU in real time, allows video uploads, lets you switch garments dynamically, and displays live telemetry HUD metrics.

---

## 🚀 Quick Start (Step-by-Step)

### Step 1: Open Google Colab & Select GPU
1. Go to [Google Colab](https://colab.research.google.com).
2. Click **File** -> **Upload notebook** -> Select `RTV_Google_Colab.ipynb` from this repository.
3. In Colab, make sure GPU acceleration is turned on:
   - Click **Runtime** in the top menu -> **Change runtime type**.
   - Under *Hardware accelerator*, select **T4 GPU** (free tier) or **L4 / A100 GPU** (Colab Pro).
   - Click **Save**.

---

### Step 2: Upload Your Code & Checkpoints to Google Drive

Because the model checkpoints (`rtv_ckpts/`) and assets are large, the best way is to keep them in your Google Drive:

1. In your local machine, your project is packed as `RTV.zip` in `D:\projects\RTV.zip`.
2. Upload `RTV.zip` to your Google Drive (for example, into the root folder `MyDrive/RTV.zip`).
3. In the Colab notebook, run **Cell 2** to mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Unzip `RTV.zip` into the Colab environment:
   ```python
   !unzip -q /content/drive/MyDrive/RTV.zip -d /content/
   %cd /content/RTV
   ```

---

### Step 3: Run Setup & Dependencies

Run **Cell 3** and **Cell 4** in the notebook (or run `!bash setup_colab.sh` in a cell):
```bash
!bash setup_colab.sh
```
This automatically installs:
- Headless graphics packages: `libegl1-mesa-dev`, `libgles2-mesa-dev`, `libosmesa6-dev`, `xvfb`, `ffmpeg`
- Python libraries: `pyopengl`, `trimesh`, `simple-romp`, `gradio`, `detectron2`
- Configures headless EGL rendering environment variables (`PYOPENGL_PLATFORM=egl`, `QT_QPA_PLATFORM=offscreen`)

---

### Step 4: Launch the Interactive Studio

Run **Cell 6**:
```python
!python colab_gradio_app.py --share
```

You will see output similar to:
```
======================================================================
  RTV Real-Time Virtual Try-On : Interactive Colab Web Interface
======================================================================
  Host: 0.0.0.0:7860
  Public Sharing: Enabled (gradio.live link will be generated)
  PyTorch CUDA: True (Tesla T4)
======================================================================
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://38c5b96a98711ef2.gradio.live
```

Click the **public URL (`https://xxxx.gradio.live`)**!

---

## 🎛️ Using the Web Interface

The web studio features 4 dedicated tabs:

### 1. 🎥 Live Webcam Try-On
- Allows you to use your laptop or mobile phone webcam directly!
- **Garment Wardrobe Dropdown**:
  - `Winter Puffer Coat (Full-Body)` — `coat_seq_vmssdp2ta_576`
  - `Hanfu Silk Dress (Full-Body)` — `han_seq_vmssdp2ta_576`
  - `Lab Coat 03 / 04 / 07 (Upper-Body)`
  - `Jin Apparel 17 / 18 / 22 (Upper-Body)`
  - `Raw Camera / AR Skeleton` (Pose calibration mode)
- **Engine Mode**:
  - **LIVE** (Recommended): 25–30 FPS real-time optical flow tracking with < 30ms latency.
  - **SYNC**: Lookahead buffer matching exact neural inference delay.
- **HUD Overlays**:
  - Telemetry HUD displaying Cam FPS, Tracking FPS, ML Latency (ms), and body orientation angle.
  - AR Skeleton showing anatomical landmarks on shoulders, hips, and hands.

### 2. 🎞️ Video File Try-On
- Upload any video file (`.mp4`, `.mov`, `.avi`) or click **"Load Bundled Sample Video"**.
- Pick your garment and adjust the **Max Frames** slider (e.g. 120 frames for a quick 5-second test).
- Click **"⚡ Render Virtual Try-On Video"**.
- When rendering completes, the fitted video plays directly in your browser with a download button.

### 3. 📸 Single Photo Studio
- Upload any portrait or full-body photo.
- Apply virtual try-on instantly.
- Expand the **Neural Diagnostic Inspection** drawer to view:
  - Garment Alpha Mask (Opacity heatmap)
  - Synthesized Garment ROI patch

### 4. 📊 Hardware & Architecture
- Live telemetry diagnostics, GPU name, VRAM usage, and pipeline latency breakdowns.

---

## ⏱️ Expected Performance on Google Colab

| GPU Runtime | VRAM | ML Inference (BEV+DensePose+Pix2PixHD) | Tracking & Display FPS | Latency (LIVE mode) |
| :--- | :--- | :--- | :--- | :--- |
| **Tesla T4 (Free Colab)** | 16 GB | ~180–220 ms | **24–28 FPS** | ~30 ms |
| **NVIDIA L4 (Colab Pro)** | 24 GB | ~90–120 ms | **28–30 FPS** | ~28 ms |
| **NVIDIA A100 (Colab Pro+)**| 40 GB | ~60–80 ms | **30 FPS sustained** | ~25 ms |

---

## 💡 Troubleshooting & Tips

1. **"Cannot initialize headless OpenGL / EGL display"**:
   - Ensure you ran `export PYOPENGL_PLATFORM=egl` or let `setup_colab.sh` configure it.
2. **"Detectron2 installation failed"**:
   - Use the pre-built command included in the notebook: `!pip install -q 'git+https://github.com/facebookresearch/detectron2.git'`.
3. **Webcam permission denied in browser**:
   - If using Chrome/Brave/Edge, make sure you allow camera permissions when prompted on the `gradio.live` tab.
