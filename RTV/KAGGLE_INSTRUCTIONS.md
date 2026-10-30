# 🧥 RTV Virtual Try-On — Kaggle Notebooks Guide

This guide provides complete step-by-step instructions to run the **RTV (Real-Time Virtual Try-On)** system on **Kaggle Notebooks** using free cloud GPUs (**NVIDIA Tesla T4 x 2** or **NVIDIA Tesla P100 16GB VRAM**).

---

## 🌟 Why Kaggle Notebooks?

- **Free High-End GPU Access**: Kaggle gives you **30 hours per week** of free GPU computing (Dual Tesla T4 or Tesla P100 with 16GB VRAM).
- **Public Gradio Web Link**: Generates a shareable URL (`https://xxxx.gradio.live`) so you can try on dresses from your phone or PC browser using your live camera or video.
- **Fast 30 FPS Video Rendering**: Cloud GPUs process dense neural keyframes and synthesize missing intermediate frames with zero lag.

---

## 🚀 Step-by-Step Instructions to Run in Kaggle

### Step 1: Create a Kaggle Account & New Notebook
1. Go to [kaggle.com](https://www.kaggle.com) and log in (or sign up for free).
2. Click the **"+ Create"** button in the top left, then select **"New Notebook"**.
3. In the new notebook menu, click **File** -> **Import Notebook**.
4. Select and upload the file [`RTV_Kaggle.ipynb`](./RTV_Kaggle.ipynb) from this repository.

---

### Step 2: Configure Kaggle Settings (CRITICAL ⚠️)
In the right-hand **Notebook Settings** panel:
1. **Accelerator**: Click the dropdown and select **GPU T4 x 2** (or **GPU P100**).
2. **Internet**: Toggle the switch to **ON** *(Essential for installing graphics packages and generating the public web link!)*.
3. **Environment**: Leave as *Always use latest environment*.

---

### Step 3: Upload Code & Model Checkpoints
Because Kaggle starts in an empty workspace, you need to provide the RTV project code and checkpoints (`rtv_ckpts`):

#### Method A: Upload as a Kaggle Dataset (Recommended)
1. In your local project folder, zip the `RTV` directory (or use `RTV.zip`).
2. On Kaggle, click **+ Add Data** (in the top right of the notebook).
3. Click **Upload Dataset** -> Select your `RTV.zip` file.
4. Give it a title like `rtv-dataset` and click **Create**.
5. Once uploaded, the notebook's **Step 2** cell will automatically detect and extract it into `/kaggle/working/RTV`.

#### Method B: Direct Git Clone (If hosted online)
If your repository is on GitHub or Hugging Face, run in a code cell:
```bash
!git clone <YOUR_REPO_URL> /kaggle/working/RTV
%cd /kaggle/working/RTV
```

---

### Step 4: Run the Notebook Cells in Sequence

1. **Step 1 (Cell 1)**: Verify GPU
   - Checks that CUDA is active and displays your Tesla T4 or P100 GPU with 16GB VRAM.
2. **Step 2 (Cell 2)**: Set Up Workspace
   - Automatically unzips and switches directory to `/kaggle/working/RTV`.
3. **Step 3 (Cell 3)**: Install System Graphics & Headless EGL
   - Installs `libegl1-mesa-dev`, `libgl1-mesa-glx`, `xvfb`, `ffmpeg`.
4. **Step 4 (Cell 4)**: Install Python Packages & Detectron2
   - Installs `simple-romp`, `trimesh`, `pyopengl`, `gradio`, and `detectron2`.
   - Sets `PYOPENGL_PLATFORM="egl"` for headless offscreen 3D rendering.

---

### Step 5: Launch the Interactive Web Studio
Run **Step 5 (Cell 5)**:
```python
!python colab_gradio_app.py --share
```
Output will display:
```
======================================================================
  RTV Real-Time Virtual Try-On : Interactive Web Studio
======================================================================
  Host: 0.0.0.0:7860
  Public Sharing: Enabled
  PyTorch CUDA: True (Tesla T4 / P100)
======================================================================
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://38c5b96a98711ef2.gradio.live
```
👉 **Click the `https://xxxx.gradio.live` link** to open the full interactive virtual try-on studio in your browser or on your smartphone!

---

### Step 6: Render Accurate 30 FPS Try-On Videos (CLI Mode)
If you want to render an entire video with dense neural keyframing, 100% solid cloth, and missing frame morphing:
Run **Step 6 (Cell 6)**:
```python
from render_accurate_video import render_accurate_video
from IPython.display import Video, display

final_out = render_accurate_video(
    input_video_path="assets/sample_video.mp4",
    garment_name="coat_seq_vmssdp2ta_576",
    output_video_path="/kaggle/working/output/accurate_tryon_30fps.mp4",
    keyframe_stride=2,
    output_fps=30.0,
    max_frames=0
)
display(Video(final_out, embed=True, width=640))
```

---

### Step 7: Download Output Videos
Run **Step 7 (Cell 7)**:
- Generates a direct clickable download link inside the notebook to save the rendered 30 FPS MP4 video to your local computer.
- You can also find your videos in the right sidebar under **Output** -> `/kaggle/working/output/`.

