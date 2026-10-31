# RTV - Production Real-Time Virtual Try-On Engine

Production-grade, photorealistic, temporally consistent 3D Virtual Try-On system targeting the visual quality of **Decart Anywear / Lucy V-TON** with locked 30 FPS neural rendering, 3D Physics-Informed Neural Garment Fields (PBNF), Reference-Conditioned Cross-Attention Synthesis, and Learned Zero-Shot Foreground Occlusion on **NVIDIA L40S** (Ada Lovelace, 48 GB GDDR6 ECC).

---

## ⚡ What's New in the Upgraded Neural Engine

1. **3D Physics-Informed Neural Garment Field (PBNF):**
   - Dynamic cloth wrinkle and fold kinematics driven by joint angular velocities ($\dot{\theta}$) and gravitational inertia ($\vec{g} = [0, -9.8, 0]^T$).
   - Direct PyTorch CUDA screen rasterization producing 3D Virtual Mesh coordinates, surface normals, depth maps, and UV atlases without OpenGL context overhead.
2. **Reference-Conditioned Cross-Attention Synthesis:**
   - Spatial-temporal cross-attention transferring high-frequency textures, embroidery, zippers, and weave patterns from any arbitrary reference garment image into 3D deformed geometry.
   - Operating at high resolution ($1024 \times 768$ / $1280 \times 720$ native) with FlashAttention-2 / FP16 acceleration.
3. **Learned Zero-Shot Foreground Occlusion:**
   - Fully replaces brittle YCrCb color thresholding with an adaptive neural reasoner invariant across all Fitzpatrick skin tones (I–VI), rings, watches, necklaces, and long flowing hair.
   - Dynamic ambient illumination & color temperature harmonization matching user room lighting.
4. **Decoupled 30 FPS Multi-Rate Streaming Pipeline:**
   - Asynchronous double-buffered CUDA streams ensuring locked 30.0 FPS output (33.3 ms per frame) with zero frame duplication, stutter, or keyframe popping.
5. **Comprehensive Automated Verification Suite:**
   - 14 rigorous movement stress scenarios evaluated via `python tests/benchmark_rtv_engine.py`.

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOGESHWARAN-FAF/RTV.git
cd RTV
```

### 2. Environment Setup

#### Linux / Remote Cloud Host (NVIDIA L40S / A100 / H100 / RTX 4090)
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA 12.1+ acceleration
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies (FastAPI, Uvicorn[standard], Websockets, Gradio)
pip install -r RTV/requirements.txt

# Install ROMP 3D pose submodule
pip install -e ROMP_repo/simple_romp

# (Optional) Install Detectron2 without compilation check
git clone --depth 1 https://github.com/facebookresearch/detectron2.git
sed -i 's/ext_modules=get_extensions()/ext_modules=[]/g' detectron2/setup.py
pip install --no-build-isolation -e detectron2
```

#### Windows (1-Click Automated Setup)
Simply double-click **`setup_new_system.bat`**.

---

## 🚀 Running the Upgraded Virtual Try-On Studio

### Interactive Gradio Web Studio (Browser UI & 30 FPS Live Mirror)
```bash
cd RTV
python colab_gradio_app.py --share
```
- Open the printed local URL (`http://localhost:7860`) or public Gradio Live link.
- Click **"Start Live Mirror"** to experience the 30 FPS real-time virtual try-on feed.
- Switch seamlessly between the 3D Neural PBNF wardrobe garments (`Puffer Coat`, `Biker Jacket`, `Silk Dress`, `Hoodie`, `Tailored Suit`) or upload your own reference garment!

---

## 🧪 Automated Benchmarking

Run the 14-scenario automated performance and latency benchmark:
```bash
cd RTV
python tests/benchmark_rtv_engine.py --iterations 100
```

---

## 📁 Repository Structure
```
├── RTV/                                # Core virtual try-on engine
│   ├── colab_gradio_app.py             # Web Studio, FastAPI & 30 FPS WebSocket Mirror
│   ├── physics/                        # 3D Physics-Informed Neural Garment Field (PBNF)
│   │   ├── garment_field.py            # Dynamic cloth wrinkles, gravity & GPU rasterizer
│   │   └── __init__.py
│   ├── model/
│   │   └── reference_synthesis/        # Lucy V-TON / Decart Anywear Synthesis Engine
│   │       ├── reference_encoder.py    # Multi-scale reference texture feature pyramid
│   │       ├── spatial_temporal_generator.py # Cross-attention generator & temporal bridge
│   │       └── __init__.py
│   ├── composition/
│   │   ├── neural_occlusion.py         # Zero-shot occlusion & ambient tone matching
│   │   ├── naive_overlay.py            # Hermite smoothstep SIMD compositing interface
│   │   └── __init__.py
│   ├── VITON/
│   │   ├── viton_neural_stream.py      # Upgraded 30 FPS Neural Virtual Try-On Processor
│   │   ├── viton_fullbody_seq.py       # Full-body sequence multi-rate engine
│   │   └── viton_upperbody.py          # Upper-body specialized processor
│   ├── tests/
│   │   └── benchmark_rtv_engine.py     # 14-scenario latency & VRAM benchmark suite
│   ├── SMPL/                           # 3D parametric human body models (model.pkl)
│   ├── rtv_ckpts/                      # Pretrained garment checkpoints
│   ├── assets/                         # Preview images, icons, and test assets
│   └── requirements.txt                # Updated package requirements
├── ROMP_repo/                          # Monocular 3D SMPL pose recovery module
├── download_models.py                  # Hugging Face model weight downloader
├── download_models.bat
├── package_project.bat
└── README.md
```

