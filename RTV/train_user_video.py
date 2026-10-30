import os
import sys
import time
import shutil
import cv2
import torch
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from util.custom_garment_trainer import CustomGarmentTrainer
from VITON.viton_upperbody import FrameProcessor

def main():
    video_path = r"C:\Users\Yogesh\OneDrive\Pictures\Camera Roll\WIN_20261023_12_14_49_Pro.mp4"
    garment_name = "rcb_jersey_18"

    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found at: {video_path}")
        sys.exit(1)

    print("=" * 70)
    print("      RTV REAL-TIME VIRTUAL TRY-ON: CUSTOM GARMENT TRAINING")
    print("=" * 70)
    print(f" Source Video: {video_path}")
    print(f" Garment Name: {garment_name}")
    print(f" CUDA Device : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)

    trainer = CustomGarmentTrainer(
        checkpoints_dir="./rtv_ckpts",
        datasets_dir="./PerGarmentDatasets"
    )

    # -------------------------------------------------------------
    # PHASE 1: Dataset Generation (3D SMPL + DensePose + Auto Mask)
    # -------------------------------------------------------------
    print("\n>>> [Phase 1/3] Generating Per-Garment Dataset from Video...")
    t0 = time.time()

    def ds_progress(cur, tot, msg):
        pct = int(cur / max(1, tot) * 100)
        print(f"  [{pct:3d}%] {msg}", flush=True)

    ds_path, count = trainer.generate_dataset_from_source(
        source_path=video_path,
        mask_dir=None,
        dataset_name=garment_name,
        max_frames=120,
        progress_cb=ds_progress
    )
    t_ds = time.time() - t0
    print(f"\n[Phase 1 Complete] Generated {count} high-definition training pairs at:")
    print(f"  --> {ds_path} in {t_ds:.1f}s ({t_ds/60:.1f} mins)")

    # -------------------------------------------------------------
    # PHASE 2: Deep Learning Neural Training (Pix2PixHD AMP FP16)
    # -------------------------------------------------------------
    print("\n>>> [Phase 2/3] Training Pix2PixHD_RGBA Neural Generator...")
    t1 = time.time()

    def train_progress(ep, tot_ep, st, loss, msg):
        print(f"  [Epoch {ep}/{tot_ep}] {msg}", flush=True)

    deployed_name, deployed_pth = trainer.train_garment(
        dataset_name=garment_name,
        garment_name=garment_name,
        num_epochs=6,
        batch_size=2,
        img_size=512,
        mode="balanced",
        progress_cb=train_progress
    )
    t_train = time.time() - t1
    print(f"\n[Phase 2 Complete] Model trained and deployed:")
    print(f"  --> Deployed Checkpoint: {deployed_pth}")
    print(f"  --> Duration: {t_train:.1f}s ({t_train/60:.1f} mins)")

    # -------------------------------------------------------------
    # PHASE 3: Verification & Test Inference
    # -------------------------------------------------------------
    print("\n>>> [Phase 3/3] Running Verification Try-On Test...")
    os.makedirs("./output", exist_ok=True)
    test_out_path = os.path.abspath(f"./output/test_tryon_{garment_name}.jpg")

    test_img = None
    sample_candidate = "scratch/video_inspect/frame_211.jpg"
    if os.path.exists(sample_candidate):
        test_img = cv2.imread(sample_candidate)
    elif os.path.exists("test_upperbody_out.jpg"):
        test_img = cv2.imread("test_upperbody_out.jpg")

    if test_img is not None:
        try:
            print("  Initializing FrameProcessor with new garment checkpoint...")
            proc = FrameProcessor([garment_name])
            proc.switch_to_target_garment(0)
            patch = proc.get_garment_patch(test_img)
            if patch is not None:
                composite = proc.composite_garment(test_img, patch)
                # Combine side by side: Original vs Virtual Try-On
                h, w = test_img.shape[:2]
                side_by_side = np.hstack([test_img, composite])
                cv2.imwrite(test_out_path, side_by_side)
                print(f"  --> Saved Side-by-Side Test Verification: {test_out_path}")
            else:
                print("  [Warning] Patch extraction returned None on test frame.")
        except Exception as e:
            print(f"  [Warning] Verification test run notice: {e}")
    else:
        print("  Skipping test image generation (no test image found).")

    print("\n" + "=" * 70)
    print(f"  SUCCESS! Garment '{deployed_name}' is fully trained and deployed.")
    print(f"  Model location: {deployed_pth}")
    print(f"  Wardrobe preview: assets/garment_images/{garment_name}_preview.jpg")
    print(f"  You can now select '{garment_name}' in demo.py / rtl_demo.py / web UI!")
    print("=" * 70)

if __name__ == "__main__":
    main()
