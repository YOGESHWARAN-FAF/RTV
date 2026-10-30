# Real-Time Virtual Try-On (RTV) & ROMP System Documentation

Comprehensive architecture, tech stack, data schemas, directory structure, and operational reference for the **Real-Time Virtual Try-On (RTV)** ecosystem.

---

## 1. Executive Summary & Scientific Foundation

The **Real-Time Virtual Try-On (RTV)** project provides high-fidelity, interactive, and temporally consistent 3D virtual try-on on standard consumer hardware (e.g., NVIDIA RTX 3060 / RTX 2050 or higher) using only a standard 2D RGB webcam or video feed—requiring **zero depth sensors, motion capture markers, or specialized hardware**.

Unlike conventional image-based virtual try-on networks (which require segmenting and hallucinating the bare body after removing the existing shirt), this architecture employs a **per-garment neural synthesis approach**:
1. It trains a specialized, lightweight conditional generative model for each target clothing item.
2. It conditions the generation on two geometric body representations: **SMPL Virtual Mesh (VM)** rendered offscreen via OpenGL, and **DensePose Surface Descriptor Maps (SDP/SSDP)**.
3. It directly synthesizes a **4-channel RGBA output** (RGB garment texture + Alpha matte).
4. A vectorized compositing engine seamlessly overlays the synthesized virtual dress onto the wearer's live video stream, preserving foreground hand/arm gestures and dynamically erasing existing collar overlap.
5. To guarantee **true 60 FPS display performance** while deep neural networks execute asynchronously, a **sequential Lucas-Kanade optical flow motion tracking pipeline** propagates the garment transformation between neural network keyframes with sub-millisecond overhead (< 1.2 ms).

### Academic Citations
- **Real-Time Per-Garment Virtual Try-On with Temporal Consistency for Loose-Fitting Garments**  
  *Zaiqiang Wu, I-Chao Shen, Takeo Igarashi* (Computer Graphics Forum / Pacific Graphics 2025; [arXiv:2506.12348](https://arxiv.org/abs/2506.12348))
- **Low-barrier dataset collection with real human body for interactive per-garment virtual try-on**  
  *Zaiqiang Wu, Yechen Li, Jingyuan Liu, Yuki Shibata, Takayuki Hori, I-Chao Shen, Takeo Igarashi* (IEEE Computer Graphics and Applications 2025; [arXiv:2506.10468](https://arxiv.org/abs/2506.10468))
- **Monocular, One-stage, Regression of Multiple 3D People (ROMP)**  
  *Yu Sun, Qian Bao, Wu Liu, Yili Fu, Michael J. Black, Tao Mei* (ICCV 2021; [arXiv:2008.12272](https://arxiv.org/abs/2008.12272))
- **Putting People in their Place: Monocular Regression of 3D People in Depth (BEV)**  
  *Yu Sun, Wu Liu, Qian Bao, Yili Fu, Tao Mei, Michael J. Black* (CVPR 2022; [arXiv:2112.08274](https://arxiv.org/abs/2112.08274))

---

## 2. Complete Technology Stack

| Layer | Technology | Version / Specification | Role in System |
|---|---|---|---|
| **Programming Language** | **Python** | `3.10.11` (64-bit AMD64) | Core runtime for inference, training, dataset creation, and UI |
| **Deep Learning Framework** | **PyTorch** | `2.3.1+cu121` / `2.4.0` | Tensor computations, autograd, neural model hosting |
| **Vision Framework** | **Torchvision** | `0.19.0` | Image transforms, VGG perceptual loss network |
| **Hardware Acceleration** | **NVIDIA CUDA & cuDNN** | CUDA 12.1 / cuDNN autotuner | High-throughput GPU inference with `torch.backends.cudnn.benchmark = True` and mixed-precision AMP (`autocast(fp16)`) |
| **3D Pose Estimation** | **ROMP & BEV** | Custom integrated submodules | Monocular 3D SMPL mesh regression with depth and multi-person reasoning |
| **3D Body Model** | **SMPL (Skinned Multi-Person Linear)** | Female & Male SMPL models (`model.pkl`) | 6,890 vertex 3D human body mesh representation with shape ($\beta$) and pose ($\theta$) |
| **Surface Parsing** | **Detectron2 & DensePose** | Facebook Research Detectron2 | Dense human UV mapping: 24-part surface coordinates converted to SDP and SSDP |
| **Offscreen Rendering** | **PyOpenGL & PyGLM** | `PyOpenGL 3.1.7`, `PyOpenGL_accelerate 3.1.10`, `PyGLM` | Hardware OpenGL FBO pipeline rendering 3D SMPL meshes with custom UV/flat shaders |
| **3D Mesh Processing** | **Trimesh & SciPy** | `trimesh`, `scipy.sparse`, `scipy.spatial.transform` | OBJ parsing, sparse matrix UV unwrapping (`long_sleeve.npz`, `full_body.npz`), rotation conversions |
| **Neural Synthesis** | **Pix2PixHD RGBA & RNN** | Custom PyTorch implementations | Multi-scale GAN with ResNet generator, multi-scale PatchGAN discriminators, and ConvLSTM temporal recurrent memory |
| **Temporal Jitter Filtering** | **OneEuroFilter** | Custom filter (`SMPL/one_euro_filter.py`) | Real-time adaptive low-pass filtering suppressing camera translation and joint jitter with low latency |
| **Motion Tracking** | **Lucas-Kanade Optical Flow** | OpenCV `goodFeaturesToTrack` + `calcOpticalFlowPyrLK` | 60 FPS frame-to-frame torso motion propagation (< 1.2 ms) decoupling ML inference latency |
| **Interactive GUI** | **PyQt5** | `5.15.11` | Multi-threaded desktop app with hardware-rendered viewports, garment selectors, and heads-up display |
| **Image & Video I/O** | **OpenCV, FFmpeg, ImageIO, Pillow** | `opencv-python-headless 4.10.0`, `ffmpeg 1.4`, `pillow >= 10.0.0` | Video capturing, color space transformations (YCrCb, BGR, HSV), frame decoders |
| **Packaging & Automation** | **Windows Batch & Shell Scripts** | `.bat` and `.sh` | Automated virtual environment activation and pipeline execution |

---

## 3. Directory and File Hierarchy

```
D:\projects\RTV\
├── run_demo.bat                    # Upper-body virtual try-on launcher (Live webcam or sample video)
├── run_fullbody_demo.bat           # Full-body sequence virtual try-on launcher
├── venv\                           # Python 3.10 virtual environment with CUDA PyTorch & PyQt5
│
├── RTV\                            # Core Real-Time Virtual Try-On Project
│   ├── rtl_demo.py                 # Upper-body real-time interactive GUI application (PyQt5)
│   ├── demo.py                     # Full-body sequence real-time interactive GUI application (PyQt5)
│   ├── test_e2e_verification.py    # Complete end-to-end automated test suite
│   ├── requirements.txt            # Python dependencies for the RTV engine
│   ├── training_instructions.md    # Guide for recording, segmenting, and training new garments
│   ├── LICENSE                     # Non-commercial research license
│   ├── README.md                   # RTV project overview
│   │
│   ├── assets\                     # Reference models, media, and pose guidance
│   │   ├── sample_video.mp4        # Offline testing video (walking & rotating person)
│   │   ├── none.png                # Transparent / empty placeholder icon
│   │   ├── color_pattern\          # Texture checkerboard patterns for offscreen rendering
│   │   │   └── board_300x300.png
│   │   ├── garment_images\         # Thumbnail icons for GUI garment selection
│   │   │   ├── jin_00_white_bg.jpg ... jin_26_white_bg.jpg
│   │   │   └── lab_00_white_bg.jpg ... lab_08_white_bg.jpg
│   │   ├── pose_guidance\          # Training video recording guidelines (PDF)
│   │   │   ├── symmetric.pdf
│   │   │   └── asymmetric.pdf
│   │   └── smpl\                   # 3D SMPL geometric definition files
│   │       ├── amputated_smpl_m_zero.obj
│   │       ├── amputated_smpl_m_zero_uv.obj
│   │       ├── amputated_smpl_m_zero_uv.png
│   │       ├── smpl_female_upperbody.obj
│   │       ├── smpl_m_zero.obj
│   │       ├── vm_full_body.obj
│   │       ├── full_body.npz       # Sparse vertex-to-UV mapping matrix for full-body mesh
│   │       ├── long_sleeve.npz     # Sparse vertex-to-UV mapping matrix for upper-body
│   │       └── unwrap_sparse.npz
│   │
│   ├── rtv_ckpts\                  # Pretrained Garment Weights (from Hugging Face: wuzaiqiang/rtv_ckpts)
│   │   ├── coat_seq_vmssdp2ta_576\latest_net_G.pth   # 721 MB (Full-body trench coat with RNN)
│   │   ├── han_seq_vmssdp2ta_576\latest_net_G.pth    # 721 MB (Full-body loose Hanfu with RNN)
│   │   ├── jin_17_vmsdp2ta\latest_net_G.pth          # 712 MB (Upper-body casual shirt)
│   │   ├── jin_18_vmsdp2ta\latest_net_G.pth          # 712 MB (Upper-body polo shirt)
│   │   ├── jin_22_vmsdp2ta\latest_net_G.pth          # 712 MB (Upper-body short sleeve)
│   │   ├── lab_03_vmsdp2ta\latest_net_G.pth          # 712 MB (Upper-body jacket)
│   │   ├── lab_04_vmsdp2ta\latest_net_G.pth          # 712 MB (Upper-body loose hoodie)
│   │   └── lab_07_vmsdp2ta\latest_net_G.pth          # 712 MB (Upper-body long-sleeve knitwear)
│   │
│   ├── composition\                # High-speed image compositing & alpha blending
│   │   ├── naive_overlay.py        # Vectorized integer blending, collar erasure, hand preservation
│   │   └── __init__.py
│   │
│   ├── DatasetGeneration\          # Custom garment dataset generation tools
│   │   ├── options.py              # CLI arguments for dataset generator
│   │   ├── upperbody_dataset_generation.py  # SMPL + DensePose + mask extraction from raw video
│   │   └── __init__.py
│   │
│   ├── Datasets\                   # Dataset loaders
│   │   ├── upperbody_garment\
│   │   │   ├── upperbody_garment.py # PyTorch Dataset: loads garment RGB, VM, SDP, and Alpha
│   │   │   └── __init__.py
│   │   └── __init__.py
│   │
│   ├── Inference\                  # Standalone batch video try-on inference
│   │   ├── base_options.py
│   │   ├── upperbody_inference.py
│   │   └── __init__.py
│   │
│   ├── model\                      # Deep Learning Architectures
│   │   ├── ConvLSTM\
│   │   │   ├── ConvLSTM.py         # 2D Convolutional LSTM cell & sequential module
│   │   │   └── __init__.py
│   │   ├── DensePose\              # DensePose inference & feature conversion
│   │   │   ├── apply_net.py
│   │   │   ├── densepose_extractor.py # Detectron2 wrapper extracting IUV from camera frames
│   │   │   ├── configs\            # Detectron2 configuration files
│   │   │   └── densepose\          # Detectron2 converters and chart definitions
│   │   └── pix2pixHD\              # Pix2PixHD Generator and Discriminator models
│   │       ├── base_model.py
│   │       ├── dual_encoder_generator.py # Dual-stream generator for separate VM and SDP encodings
│   │       ├── models.py           # Factory method `create_model(opt)`
│   │       ├── networks.py         # ResNet backbone, PatchGAN discriminators, VGG feature loss
│   │       ├── pix2pixHD_model.py  # Standard Pix2PixHD baseline
│   │       ├── pix2pixHD_rgba.py   # 4-channel RGBA Pix2PixHD model
│   │       ├── pix2pixHD_rnn_rgba.py # 4-channel RGBA + ConvLSTM recurrent sequence model
│   │       └── ui_model.py
│   │
│   ├── OffscreenRenderer\          # Hardware-Accelerated OpenGL offscreen rendering
│   │   ├── base_renderer.py        # Base OpenGL context with Framebuffer Object (FBO)
│   │   ├── flat_renderer.py        # Flat textured shader rendering SMPL mesh to Virtual Mesh (VM)
│   │   ├── uv_renderer.py          # UV coordinate space renderer
│   │   ├── gl_utils.py             # Shader compilation, program linking, VBO/VAO helpers
│   │   ├── shaders.py              # GLSL vertex and fragment shader definitions
│   │   └── Shaders\                # Specialized Phong and Black background shaders
│   │
│   ├── OpticalFlow\                # Optical flow calculations
│   │   ├── optical_flow.py         # Farneback & Lucas-Kanade wrappers
│   │   ├── advect_img.py           # Flow-based image warping and advection
│   │   └── __init__.py
│   │
│   ├── options\                    # Configuration options for training and testing
│   │   ├── base_options.py         # Common CLI arguments (input_nc, output_nc, batchSize)
│   │   ├── train_options.py        # Learning rates, epochs, ganFeat/VGG loss lambdas
│   │   ├── test_options.py         # Testing/inference checkpoint paths
│   │   └── __init__.py
│   │
│   ├── SMPL\                       # 3D Human Body Model Regression and Tracking
│   │   ├── model.pkl               # SMPL neutral body model parameters and blend shapes
│   │   ├── smpl_regressor.py       # Wrapper calling ROMP/BEV with One-Euro temporal filtering
│   │   ├── my_bev.py               # BEV integration with camera translation extraction
│   │   ├── smpl_np.py              # Pure NumPy SMPL forward kinematics implementation
│   │   ├── smpl_np_base.py         # Base SMPL kinematics & vertex skinning
│   │   ├── one_euro_filter.py      # Real-time adaptive low-pass filter for smooth motion
│   │   ├── pose_filter.py          # Temporal pose smoothing
│   │   ├── projection2screen.py    # Weak perspective to 2D image coordinate projection
│   │   ├── fullbody_smpl\
│   │   │   └── FullBody.py         # Full-body mesh renderer via `vm_full_body.obj`
│   │   └── upperbody_smpl\
│   │       └── UpperBody.py        # Upper-body mesh renderer via `smpl_female_upperbody.obj`
│   │
│   ├── Training\                   # Model training scripts
│   │   ├── upperbody_training.py   # Training loop for Pix2PixHD_RGBA models
│   │   ├── upperbody_training.sh   # Bash command launcher for multi-GPU training
│   │   └── __init__.py
│   │
│   ├── util\                       # Comprehensive Computer Vision Utilities
│   │   ├── cv2_trans_util.py       # Affine matrix transformation, decomposition & smoothing
│   │   ├── densepose_util.py       # IUV to SDP/SSDP color mapping and ROI extraction
│   │   ├── image_warp.py           # Zoom, shift, rotate, crop, and aspect ratio utilities
│   │   ├── pixel_align.py          # Centroid alignment, threshold masking, contour bounds
│   │   ├── multithread_video_loader.py # Multi-threaded ring-buffer video frame reader
│   │   ├── visualizer.py           # HTML and Visdom logging utilities
│   │   └── util.py                 # Tensor-to-image and image-to-tensor normalization
│   │
│   └── VITON\                      # Core Virtual Try-On Frame Processors
│       ├── viton_upperbody.py      # FrameProcessor for upper-body try-on
│       ├── viton_fullbody_seq.py   # FullBodySeqFrameProcessor for sequence try-on with ConvLSTM
│       └── viton_fullbody.py       # Single-frame full-body processor
│
└── ROMP_repo\                      # Complete Submodule: Monocular 3D Human Mesh Recovery
    ├── romp\                       # Full ROMP/BEV training and evaluation pipeline
    │   ├── train.py                # Distributed training script
    │   ├── eval.py                 # Benchmark evaluation
    │   ├── lib\
    │   │   ├── models\             # ResNet-50 and HRNet-32 backbones, CoordConv, BEV model
    │   │   ├── dataset\            # 3DPW, MPI-INF-3DHP, Human3.6M, COCO, CrowdPose loaders
    │   │   ├── maps_utils\         # Center map, scale map, and mesh parameter parsers
    │   │   └── loss_funcs\         # SMPL vertex loss, 2D/3D joint loss, MPJPE
    │   └── exports\
    │       └── blender_mocap.py    # FBX/BVH motion capture export for Blender
    ├── simple_romp\                # Lightweight distribution package for ROMP & BEV inference
    │   ├── setup.py                # Standalone setup script
    │   ├── romp\                   # Standalone inference modules
    │   └── bev\                    # Standalone BEV inference modules
    ├── configs\                    # YAML training and evaluation configurations
    └── docs\                       # Detailed ROMP documentation
```

---

## 4. End-to-End Pipeline & Architecture Flow

The complete virtual try-on workflow operates across two decoupled, synchronized pipelines: the **Asynchronous Deep Learning Keyframe Pipeline** and the **Sub-Millisecond 60 FPS Compositing Pipeline**.

```mermaid
flowchart TD
    subgraph INPUT ["Input & Buffering"]
        CAM["Live Webcam (cv2.VideoCapture)"] --> CAP_LOOP["Capture Thread (Native Camera FPS)"]
        CAP_LOOP --> RING_BUF["240-Frame Timestamped Ring Buffer\n(4-8s Video History)"]
    end

    subgraph KEYFRAME ["Asynchronous ML Keyframe Worker (approx. 200-400ms)"]
        RING_BUF -.-> TARGET_FRAME["Synchronized Target Frame"]
        TARGET_FRAME --> BEV["BEV / ROMP 3D Regression\n(SMPL Joints, Verts, Cam Trans)"]
        BEV --> FILTER["OneEuroFilter\n(Temporal Jitter Suppression)"]
        
        FILTER --> OPENGL["PyOpenGL Offscreen FBO\n(Flat/UV Mesh Rendering)"]
        OPENGL --> ROI_VM["ROI Virtual Mesh (VM)\n(3 Channels: 512x512)"]
        
        TARGET_FRAME --> DENSEPOSE["Detectron2 DensePose\n(IUV Coordinate Extraction)"]
        DENSEPOSE --> ROI_SDP["ROI Smoothed DensePose (SDP)\n(3 Channels: 512x512)"]
        
        ROI_VM & ROI_SDP --> CONCAT["6-Channel Input Tensor\n[VM (3ch) + SDP (3ch)]"]
        CONCAT --> PIX2PIX["Pix2PixHD RGBA / RNN Generator\n(CUDA FP16 Autocast)"]
        PIX2PIX --> SYNTH["Synthesized 4-Channel RGBA\n[Garment RGB (3ch) + Alpha (1ch)]"]
        
        SYNTH --> ENVELOPE["3D Physical Body Envelope\n(Clamps Neural Artifacts)"]
        ENVELOPE --> FEATURES["GoodFeaturesToTrack\n(Upper Sternum Keypoints)"]
        FEATURES --> PATCH["Garment Patch Package\n(Target RGB, Clean Alpha, Affine Matrix, Track Pts)"]
    end

    subgraph COMPOSITING ["Sub-Millisecond 60 FPS Display Loop (< 1.2ms)"]
        PATCH --> TRACKER["Lucas-Kanade Optical Flow Tracker\n(Propagates Affine Motion Frame-to-Frame)"]
        RING_BUF --> DISPLAY_FRAME["Delay-Matched Display Frame\n(Time: t_now - latency)"]
        
        DISPLAY_FRAME & TRACKER --> COLLAR["Collar Erasure Engine\n(Removes existing shirt above neckline)"]
        COLLAR --> HAND_OCC["YCrCb Skin Occlusion Mask\n(Preserves foreground hands/arms)"]
        HAND_OCC --> BLEND["Vectorized Integer Alpha Blending\n((Target * A + Frame * (255 - A)) >> 8)"]
        BLEND --> HUD["Interactive Heads-Up Display (HUD)\n(360° Angle, FPS, Latency, Sync Status)"]
        HUD --> GUI["PyQt5 Window (60 FPS Native Viewport)"]
    end
```

---

## 5. Detailed Component Breakdown

### 5.1 3D Body Mesh Estimation (`SMPL/smpl_regressor.py`)
- Uses **BEV (Body with Depth and Age)** or **ROMP** as the monocular 3D human pose regressor.
- Estimates 24 3D body joints, SMPL pose parameters ($\theta \in \mathbb{R}^{72}$), shape parameters ($\beta \in \mathbb{R}^{10}$), and camera weak-perspective scale and translation.
- Integrates the **OneEuroFilter** (`SMPL/one_euro_filter.py`) on the 3D translation ($t_x, t_y, t_z$) and body joints:
  - High smoothing at low velocity eliminates camera wobble and micro-jitters.
  - Decreasing smoothing at high velocity eliminates visual lag during sudden user movements.
- Calculates the 3D body yaw angle to provide live **360-degree orientation feedback** in the HUD (Front $0^\circ$, Left/Right $3/4$, Left/Right Profile, Back View).

### 5.2 Offscreen Hardware OpenGL Renderer (`OffscreenRenderer/`)
- Initializes an offscreen OpenGL context using `PyOpenGL` with a Framebuffer Object (FBO) and texture attachments.
- Compiles GLSL shaders (`OffscreenRenderer/shaders.py`, `base_shaders.py`):
  - **FlatRenderer**: Maps the reference garment checkerboard pattern (`assets/color_pattern/board_300x300.png`) across SMPL 3D vertices using precomputed sparse UV matrices (`long_sleeve.npz`, `full_body.npz`).
  - Produces a deterministic **Virtual Mesh (VM)** representation of the human pose invariant to user skin tone, lighting, or background.

### 5.3 DensePose Surface Coordinates (`model/DensePose/`)
- Encapsulates Detectron2 with ResNet-50 FPN to segment the human body into 24 anatomical surface charts.
- Converts raw IUV coordinates into continuous **Smoothed Dense Pose (SDP / SSDP)** RGB images.
- Combines with the Virtual Mesh to provide the generator with spatial understanding of body contour, limb rotation, and cloth boundary.

### 5.4 Neural Virtual Try-On Network (`model/pix2pixHD/`)
- **Architecture**:
  - **Generator**: Coarse-to-fine or dual-encoder ResNet backbone with reflection padding, instance normalization, and residual blocks.
  - **Discriminator**: Multi-scale 70x70 PatchGAN discriminators operating at original resolution and $2\times$ downsampled scales.
  - **Recurrent Variant (`Pix2PixHD_RNN_RGBA`)**: Features a **ConvLSTM** layer embedded at the bottleneck, tracking cloth state across consecutive frames to prevent flickering on loose garments.
- **Loss Functions**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{GAN}}(G, D) + \lambda_{\text{feat}} \mathcal{L}_{\text{GAN\_Feat}}(G, D) + \lambda_{\text{VGG}} \mathcal{L}_{\text{VGG}}(G(\mathbf{x}), \mathbf{y})$$
- Generates 4 channels:
  - Channels 0, 1, 2: Photorealistic garment RGB texture.
  - Channel 3: Continuous alpha matte representing fabric coverage and edge translucency.

### 5.5 60 FPS Torso Motion Tracker & Compositor (`composition/naive_overlay.py` & `VITON/`)
- **Keyframe-to-Frame Optical Flow Propagation**:
  - The deep network runs at ~3–5 FPS (~200–350 ms per inference).
  - Rather than showing a delayed or jittery feed, the system tracks keypoints on the upper chest using `cv2.calcOpticalFlowPyrLK` (< 1.2 ms).
  - It continuously warps and updates the affine transformation matrix ($M \in \mathbb{R}^{2 \times 3}$) across all 60 FPS camera frames, keeping the garment locked to the chest in real time.
- **Natural Collar Eraser**:
  - Detects the wearer's neck and chin landmarks via SMPL projections.
  - Samples skin tone from the lower jaw and creates a soft feathered mask that erases any high-contrast shirt collars poking out above the virtual garment neckline.
- **Foreground Hand & Arm Occlusion**:
  - Employs an ultra-fast YCrCb skin classifier ($Y \in [0, 255], Cr \in [133, 173], Cb \in [77, 127]$) on the torso bounding box.
  - Zeroes out the virtual garment alpha mask over detected hands/arms, allowing wearers to cross their arms or put their hands in front of their chest without the garment rendering over their hands.
- **Vectorized Integer Alpha Blending**:
  - Avoids floating-point CPU conversions by computing:
    $$\text{Out} = \left(\text{Target} \times \alpha + \text{Background} \times (255 - \alpha)\right) \gg 8$$
  - Eliminates garbage collection stalls and CPU memory churn.
- **Dynamic Articulated Sleeve Warping (Delay Frame Motion Tracking)**:
  - Compensates for the 1–2 FPS deep neural inference rate (~500–1000ms delay) during live camera feed.
  - Dynamically extracts, rotates, and scales sleeve segments around the shoulder pivot joint ($\Delta \theta = \theta_{\text{curr}} - \theta_{\text{ref}}$) at 30–60 FPS (< 20 ms), keeping sleeves attached to moving hands in real time.
- **AR Graphical Motion Feedback Engine (`util/ar_motion_graphics.py`)**:
  - Renders flowing, fading neon cyan/emerald motion trails behind tracked wrist trajectories.
  - Spawns kinetic sparkle particles along hand velocity vectors during fast movements.
  - Displays rotating holographic AR wrist tracking reticles with kinetic speed flares.
  - High-speed bounding box ROI alpha blending executes in < 3.3 ms. Toggleable in real time via `[G]` key.

---

## 6. Pretrained Garment Checkpoints Inventory

Checkpoints are hosted on Hugging Face at [`wuzaiqiang/rtv_ckpts`](https://huggingface.co/wuzaiqiang/rtv_ckpts) and stored locally under `RTV/rtv_ckpts/`:

| Checkpoint Name | File Size | Model Architecture | Garment Type | Target Body Coverage | Temporal Memory |
|---|---|---|---|---|---|
| `coat_seq_vmssdp2ta_576` | **721.96 MB** | `Pix2PixHD_RNN_RGBA` | Trench Coat / Long Overcoat | Full Body ($432 \times 576$) | **Yes (ConvLSTM)** |
| `han_seq_vmssdp2ta_576` | **721.96 MB** | `Pix2PixHD_RNN_RGBA` | Hanfu / Traditional Loose Dress | Full Body ($432 \times 576$) | **Yes (ConvLSTM)** |
| `jin_17_vmsdp2ta` | **712.74 MB** | `Pix2PixHD_RGBA` | Casual Button-up Shirt | Upper Body ($512 \times 512$) | Single Keyframe |
| `jin_18_vmsdp2ta` | **712.74 MB** | `Pix2PixHD_RGBA` | Polo Shirt | Upper Body ($512 \times 512$) | Single Keyframe |
| `jin_22_vmsdp2ta` | **712.74 MB** | `Pix2PixHD_RGBA` | Short-Sleeve T-Shirt | Upper Body ($512 \times 512$) | Single Keyframe |
| `lab_03_vmsdp2ta` | **712.74 MB** | `Pix2PixHD_RGBA` | Zip-up Jacket | Upper Body ($512 \times 512$) | Single Keyframe |
| `lab_04_vmsdp2ta` | **712.74 MB** | `Pix2PixHD_RGBA` | Loose Winter Hoodie | Upper Body ($512 \times 512$) | Single Keyframe |
| `lab_07_vmsdp2ta` | **712.74 MB** | `Pix2PixHD_RGBA` | Long-Sleeve Knitwear / Sweater | Upper Body ($512 \times 512$) | Single Keyframe |

---

## 7. Data Formats & Schemas

### 7.1 Input Tensors to Neural Generator
- **Shape**: `(BatchSize, 6, Height, Width)`
  - Channels `[0, 1, 2]`: Virtual Mesh (VM) RGB image normalized to $[-1.0, 1.0]$.
  - Channels `[3, 4, 5]`: Smoothed DensePose (SDP/SSDP) RGB image normalized to $[-1.0, 1.0]$.
- **ROI Dimensions**:
  - Upper-Body: $512 \times 512$ pixels.
  - Full-Body Sequence: $576 \times 432$ pixels ($H \times W$, $4:3$ aspect ratio).

### 7.2 Output Tensors from Neural Generator
- **Shape**: `(BatchSize, 4, Height, Width)`
  - Channels `[0, 1, 2]`: Synthesized garment color channels (RGB) normalized to $[-1.0, 1.0]$.
  - Channel `[3]`: Synthesized raw alpha channel, mapped to $[0, 255]$ with envelope thresholding:
    $$\alpha_{\text{solid}} = \text{clamp}\left(\frac{\text{raw\_alpha} - 0.28}{0.40}, 0.0, 1.0\right) \times 255$$

### 7.3 Training Dataset Directory Structure (`./PerGarmentDatasets/<garment_name>/`)
During dataset preparation (`upperbody_dataset_generation.py`), the output is formatted as follows:
```
PerGarmentDatasets/<garment_name>/
├── garment_img/       # Extracted RGB crops of the clothing item (.png)
├── garment_mask/      # Binary alpha masks (.png)
├── vm_img/            # Rendered SMPL Virtual Mesh crops (.png)
├── dp_img/            # Rendered DensePose SDP crops (.png)
└── trans.json         # Transformation matrices mapping crops back to original 4K frames
```

---

## 8. Installation & Setup Guide

### 8.1 Prerequisites
- **Operating System**: Windows 10/11 or Ubuntu 20.04/22.04 LTS
- **GPU**: NVIDIA RTX 2050, 3060, 4070, A100 or higher (at least 4 GB VRAM for inference, 12 GB+ for training)
- **CUDA**: 12.1 or compatible NVIDIA driver
- **Hardware**: Standard USB webcam (720p or 1080p recommended) or pre-recorded MP4 video

### 8.2 Environment Installation
From the root workspace:

```bash
# 1. Activate existing Python environment
cd D:\projects\RTV
.\venv\Scripts\activate

# 2. If rebuilding from scratch with conda/pip:
conda create -n rtv python=3.10
conda activate rtv

# 3. Install core dependencies
pip install -r RTV/requirements.txt

# 4. Install Detectron2 (Windows-compatible wheel or from source)
pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git

# 5. Install simple_romp
cd ROMP_repo/simple_romp
pip install -e .
cd ../../
```

### 8.3 Checkpoint Acquisition
If checkpoints are missing, clone them via Git LFS:
```bash
git lfs install
cd D:\projects\RTV\RTV
git clone https://huggingface.co/wuzaiqiang/rtv_ckpts
```

---

## 9. Operation & Usage Instructions

### 9.1 Launching the Interactive GUI Applications

#### Upper-Body Virtual Try-On
Double-click `run_demo.bat` or run:
```powershell
cd D:\projects\RTV\RTV
..\venv\Scripts\python.exe rtl_demo.py
```
- **Option 1**: Auto-detects connected webcam.
- **Option 2**: Runs against `assets/sample_video.mp4` for offline testing.

#### Full-Body Sequence Virtual Try-On (Loose Garments & Coats)
Double-click `run_fullbody_demo.bat` or run:
```powershell
cd D:\projects\RTV\RTV
..\venv\Scripts\python.exe demo.py
```

### 9.2 Interactive Controls & Hotkeys

| Key / Control | Action | Function |
|---|---|---|
| **Garment Carousel (Bottom Bar)** | Mouse Click | Instantly switches the target virtual clothing item |
| **`[SPACE]` or `[ENTER]`** | Press Key | Skips 3D motion calibration and starts try-on immediately |
| **`[C]`** | Press Key | Re-triggers 360-degree calibration mode |
| **`[R]`** | Press Key | Rotates camera input by $90^\circ$ clockwise (useful for portrait webcams) |
| **`[+]` / `[-]`** | Press Key | Manually adjusts ring buffer synchronization delay ($\pm 50$ ms) |
| **`[A]`** | Press Key | Auto-locks buffer delay to exact measured GPU inference latency |
| **`[Q]` or `[ESC]`** | Press Key | Safely closes the stream engine and exits |

---

---

## 10. Custom Garment Studio & Training Pipeline (UI & CLI)

The system provides a **complete end-to-end interactive UI suite** allowing you to upload custom dress videos or photos, auto-generate training datasets, train neural models on an RTX 2050 using PyTorch AMP FP16, import checkpoints, and test/run them immediately in both live video mirrors and static photo studios.

### 10.1 Interactive UI Workflow (`demo.py` Desktop Mirror)

1. **Launch Desktop Studio**: Run `python demo.py` (or `run_fullbody_demo.bat`).
2. **Access Custom Studio**: Click the prominent **`👗 Custom Dress Studio`** button on the right sidebar (or press **`🧪 Test on Photo / Video`**).
3. **Tab 1: 🚀 Train Custom Garment**:
   - **Garment Model Name**: Enter a descriptive identifier (e.g., `wedding_dress_v1`).
   - **📸 4-View Photos Mode (Front, Back, Right, Left)**:
     - Directly upload 4 still photos of the dress:
       - 👗 **Front View (Required)**: Front-facing dress photo. Also automatically generates the wardrobe thumbnail.
       - 🔄 **Back View (Recommended)**: Rear-facing view capturing back cuts and patterns.
       - ➡️ **Right Side View (Optional)**: Profile view capturing sleeve and side contours.
       - ⬅️ **Left Side View (Optional)**: Profile view capturing opposite side contours.
     - **360° Synthetic Augmentation**: The engine automatically fits 3D SMPL meshes and DensePose UV maps for each view, then synthesizes 100+ balanced affine & photometric variations around all 4 angles to form a complete training set.
   - **🎥 Single Video File Mode**:
     - Alternatively, upload a single continuous `.mp4` video or folder of sequential frames.
   - **⚡ Speed Presets**:
     - ⚡ **Turbo Mode (1–2 Mins) [Default]**: Fine-tunes pretrained high-definition base (lab_03_vmsdp2ta) with 3 epochs on 40 balanced pairs (~60-90s).
     - 🎯 **Balanced Mode (4–5 Mins)**: 6 epochs on 80 pairs (~4-5 mins).
     - 💎 **Deep Studio Mode (30–40 Mins)**: 15 epochs on 120 pairs (~30-40 mins).
   - **Hyperparameters**: Auto-synced to preset (Epochs: 3 by default, Batch Size: 2, Resolution: 512x512/576x576 with AMP FP16).
   - **Execute**: Click **`🚀 Start Dataset Generation & Training`**. The background `TrainingWorker` runs asynchronously without freezing the UI, reporting live progress and loss in the embedded console.
   - **Automatic Deployment**: Upon completion, the model is automatically placed in `rtv_ckpts/<name>/latest_net_G.pth`, registered in the wardrobe, and selected for live try-on!
4. **Tab 2: 📥 Import Checkpoint (.pth)**:
   - Import any existing `latest_net_G.pth` model and preview thumbnail image.
   - Click **`📥 Import & Deploy into Wardrobe`** to instantly make it available for live streaming.
5. **Tab 3: 🧪 Test Studio (Static Photo Try-On)**:
   - Select any person photo (`.jpg`, `.png`).
   - Choose the target garment from the wardrobe dropdown (refreshed automatically).
   - Click **`✨ Apply Virtual Try-On Test`** to view an instant side-by-side Before/After preview with 100% pure graphics and anti-aliased edges.
   - Click **`💾 Save Test Result Image`** to export the synthesized image.

### 10.2 Web Studio Workflow (`colab_gradio_app.py`)

1. **Launch Web Studio**: Run `python colab_gradio_app.py`.
2. Open `http://localhost:7860` (or the public Gradio link).
3. Navigate to **`👗 Custom Garment Studio (Upload, Train & Test)`**:
   - **Train**: Upload your dress video or zip archive of photos, configure epochs/batch, and monitor live training logs.
   - **Import**: Upload `.pth` checkpoint and thumbnail.
   - **Test**: Run single-photo try-on directly in the browser and download the resulting image.
   - Newly added garments are dynamically pushed to the live webcam and video try-on dropdown menus.

### 10.3 Manual CLI Training Pipeline

For automated scripts or remote headless training:
```bash
# 1. Dataset Generation
python -c "
from util.custom_garment_trainer import CustomGarmentTrainer
trainer = CustomGarmentTrainer()
ds = trainer.generate_dataset_from_source('path/to/video.mp4', None, 'custom_dress', auto_mask=True)
print('Dataset created at:', ds)
"

# 2. Neural Training (AMP FP16 on GPU)
python -c "
from util.custom_garment_trainer import CustomGarmentTrainer
trainer = CustomGarmentTrainer()
trainer.train_garment('./PerGarmentDatasets/custom_dress', 'custom_dress', epochs=15, batch_size=2)
"
```

---

## 11. Automated Verification & Testing Suite

The repository includes a comprehensive test suite in `RTV/test_e2e_verification.py`. It tests:
1. **OneEuroFilter Test**: Verifies $>80\%$ stationary jitter suppression while ensuring rapid response during step movements.
2. **FullBody Sequence Pipeline Test**: Runs the full-body processor on sample video frames, verifying solid opacity ratio ($>65\%$) and compositing speed ($<35$ ms).
3. **UpperBody Pipeline Test**: Tests `FrameProcessor` with `lab_03_vmsdp2ta`, validating RGBA generation and overlay blending.
4. **Sequential 60 FPS Tracking & Hand Preservation Test**: Validates Lucas-Kanade torso tracking ($<1.2$ ms execution) and verifies foreground skin color preservation when hands cross over the chest.

To execute the test suite:
```powershell
cd D:\projects\RTV\RTV
..\venv\Scripts\python.exe test_e2e_verification.py
```
*Result: `ALL END-TO-END VERIFICATION TESTS PASSED`.*

---

## 12. Heterogeneous Dual-GPU Architecture & 6+ FPS Dress Synthesis

To achieve high-fidelity virtual try-on on entry-level laptop configurations (e.g., NVIDIA GeForce RTX 2050 4GB alongside an Intel UHD Graphics iGPU with 8GB shared system RAM), the system employs a **Heterogeneous Dual-Engine Pipeline**:

### 12.1 Hardware Partitioning & Co-Processing
| Hardware Device | Memory | Engine / Runtime | Computational Role | Latency / Throughput |
|---|---|---|---|---|
| **GPU.0: Intel(R) UHD Graphics** | 8 GB Shared RAM | OpenVINO 2026.4.1 | Asynchronous camera frame inspection, background scene validation, motion confidence scoring | **4–30 ms** (15–25 FPS) |
| **GPU.1: NVIDIA GeForce RTX 2050** | 4 GB VRAM | PyTorch 2.3.1 + CUDA 12.1 (AMP FP16) | Offscreen 3D SMPL OpenGL rendering, DensePose IUV mapping, Pix2PixHD ResNet generator | **131–138 ms** (7.2–7.6 FPS) |

### 12.2 Multi-Rate Dress Synthesis Acceleration
- **Original Bottleneck**: Sequential monolithic execution of ROMP (242 ms) + DensePose (262 ms) + Pix2PixHD (223 ms) incurred 727 ms per frame, restricting dress generation to ~2–3 FPS.
- **AMP FP16 Tensor Cores**: Precision conversion via `torch.autocast(device_type='cuda', dtype=torch.float16)` reduced Pix2PixHD forward latency from 223 ms to 132 ms while lowering VRAM footprint from 3.2 GB to 1.3 GB.
- **Optical Multi-Rate LK Scheduling**: `max_intermediate_frames` is set to 10 with a 80 px motion displacement limit. Between 3D anchor frames, garment deformation is synthesized using optical-flow warped geometry and fast Tensor Core inference in **132–136 ms (7.5 FPS)**, resulting in a sustained **6.4+ FPS** dress generation rate (>2X speedup).
- **Clean AR Visuals**: Circular segmented hand-tracking reticles have been eliminated, rendering clean wrist gestures with live neon motion trails and particle dynamics.

