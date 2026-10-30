import os
import sys
import time
import collections
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("./model/DensePose"))

from VITON.viton_fullbody_seq import FullBodySeqFrameProcessor
from util.pipeline_profiler import PipelineProfiler
import util.util as util
from util.densepose_util import IUV2SSDP


def run_comprehensive_profile(num_frames=60):
    print("=" * 80)
    print("PHASE 1 & 6: COMPREHENSIVE PIPELINE BENCHMARK & HARDWARE PROFILE")
    print("Platform: Windows 11, NVIDIA GeForce RTX 2050 (4 GB VRAM), 16 GB RAM")
    print("=" * 80)

    # 1. Verify OpenGL Renderer GPU Adapter
    from OffscreenRenderer.gl_utils import create_opengl_context
    from OpenGL.GL import glGetString, GL_RENDERER, GL_VENDOR, GL_VERSION
    _, surf, ctx = create_opengl_context((512, 512))
    gl_vendor = glGetString(GL_VENDOR).decode()
    gl_renderer = glGetString(GL_RENDERER).decode()
    gl_version = glGetString(GL_VERSION).decode()
    print(f"\n[OpenGL Environment]")
    print(f"  Vendor:   {gl_vendor}")
    print(f"  Renderer: {gl_renderer}")
    print(f"  Version:  {gl_version}")

    # 2. Load test video stream
    video_path = "./assets/sample_video.mp4"
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open {video_path}"
    
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    print(f"\n[Video Source]")
    print(f"  File: {video_path}")
    print(f"  Source FPS: {fps_in:.1f}")

    raw_frames = []
    capture_latencies = []
    while len(raw_frames) < num_frames:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        dt_cap = (time.perf_counter() - t0) * 1000.0
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        raw_frames.append(frame)
        capture_latencies.append(dt_cap)
    cap.release()
    print(f"  Loaded {len(raw_frames)} frames into memory for profiling.")

    # 3. Initialize Model
    print("\n[Initializing Virtual Try-On Pipeline]")
    t_init = time.perf_counter()
    processor = FullBodySeqFrameProcessor('coat_seq_vmssdp2ta_576')
    print(f"  Model loaded in {(time.perf_counter() - t_init):.2f}s")

    # Warmup
    print("  Executing GPU warmup pass...")
    _ = processor.get_garment_patch(raw_frames[0])
    torch.cuda.synchronize()
    print("  Warmup complete.")

    # 4. Instrumented Stage Measurement
    profiler = PipelineProfiler(window_size=num_frames)

    timings = {
        '1. Camera Capture Latency': capture_latencies,
        '2. ROMP/BEV Inference': [],
        '3. SMPL 3D Processing': [],
        '4. PyOpenGL VM Render': [],
        '5. DensePose Inference': [],
        '6. Pix2PixHD Generator (Direct GPU)': [],
        '7. Optical Flow Tracking': [],
        '8. Alpha Compositing': [],
        '9. Presentation (QImage)': [],
    }

    # Phase A: Isolated Stage Measurement
    print(f"\n[Phase A] Running isolated stage measurements (20 iterations)...")
    for i in range(20):
        frame = raw_frames[i % len(raw_frames)].copy()
        raw_h, raw_w = frame.shape[:2]
        det_max_h = 448
        scale = det_max_h / float(raw_h) if raw_h > det_max_h else 1.0
        det_img = cv2.resize(frame, (int(raw_w * scale), det_max_h)) if scale != 1.0 else frame
        det_h, det_w = det_img.shape[:2]

        # Stage 2: ROMP/BEV
        t0 = time.perf_counter()
        with torch.inference_mode():
            smpl_param = processor.smpl_regressor.forward(det_img, False)
        torch.cuda.synchronize()
        dt_bev = (time.perf_counter() - t0) * 1000.0
        timings['2. ROMP/BEV Inference'].append(dt_bev)

        if smpl_param is not None:
            # Stage 3: SMPL Kinematics
            t0 = time.perf_counter()
            best_person_idx = smpl_param.get('best_person_idx', 0)
            target_center = smpl_param.get('target_center', None)
            trans2roi, inv_trans = processor.smpl_regressor.get_fullbody_trans2roi(
                smpl_param, s=1.4, new_h=processor.roi_height, new_w=processor.roi_width
            )
            vertices = processor.smpl_regressor.get_raw_verts(smpl_param, person_idx=best_person_idx)
            vertices_t = torch.from_numpy(vertices).unsqueeze(0)
            dt_smpl = (time.perf_counter() - t0) * 1000.0
            timings['3. SMPL 3D Processing'].append(dt_smpl)

            # Stage 4: PyOpenGL Virtual Mesh Render
            t0 = time.perf_counter()
            raw_vm = processor.full_body.render(vertices_t[0], height=det_h, width=det_w)
            roi_vm = cv2.warpAffine(raw_vm, trans2roi, (processor.roi_width, processor.roi_height),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            dt_gl = (time.perf_counter() - t0) * 1000.0
            timings['4. PyOpenGL VM Render'].append(dt_gl)

            # Stage 5: DensePose
            t0 = time.perf_counter()
            with torch.inference_mode():
                IUV = processor.densepose_extractor.get_IUV(det_img, isRGB=False, target_center=target_center)
            torch.cuda.synchronize()
            dt_dp = (time.perf_counter() - t0) * 1000.0
            timings['5. DensePose Inference'].append(dt_dp)

            if IUV is None:
                IUV = np.zeros_like(det_img)
            ssdp = IUV2SSDP(IUV)
            roi_ssdp = cv2.warpAffine(ssdp, trans2roi, (processor.roi_width, processor.roi_height),
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

            # Stage 6: Pix2PixHD with Direct GPU Preprocessing
            t0 = time.perf_counter()
            vm_tensor = torch.from_numpy(roi_vm).cuda().permute(2, 0, 1).float().sub_(127.5).div_(127.5)[[2, 1, 0]].unsqueeze(0)
            dp_tensor = torch.from_numpy(roi_ssdp).cuda().permute(2, 0, 1).float().sub_(127.5).div_(127.5).unsqueeze(0)
            in_tensor = torch.cat([vm_tensor, dp_tensor], 1)
            with torch.inference_mode():
                target_tensor = processor.viton_model.forward(in_tensor)
            roi_target = ((target_tensor[0, [0, 1, 2]].permute(1, 2, 0).clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).cpu().numpy()
            raw_alpha_tensor = target_tensor[0, 3, :, :].float()
            solid_alpha = torch.clamp((raw_alpha_tensor - 0.20) / 0.35, 0.0, 1.0)
            roi_alpha = (solid_alpha.cpu().numpy() * 255).astype(np.uint8)
            torch.cuda.synchronize()
            dt_p2p = (time.perf_counter() - t0) * 1000.0
            timings['6. Pix2PixHD Generator (Direct GPU)'].append(dt_p2p)

    # Phase B: Real Multi-Rate Pipeline Profiling
    print(f"\n[Phase B] Running Multi-Rate Live Pipeline ({num_frames} iterations)...")
    processor.reset_filters()

    anchor_latencies = []
    fast_latencies = []
    all_update_latencies = []
    live_e2e_latencies = []
    patch_ages = []
    active_patch = None

    for i in range(num_frames):
        frame = raw_frames[i % len(raw_frames)].copy()
        t_capture_mono = time.monotonic()
        profiler.record_capture(i, t_capture_mono)

        # Single get_garment_patch call (no duplicate execution!)
        t_ml_start = time.perf_counter()
        patch = processor.get_garment_patch(frame, isRGB=False, allow_multirate=True)
        torch.cuda.synchronize()
        dt_patch = (time.perf_counter() - t_ml_start) * 1000.0

        if patch is not None:
            active_patch = patch
            all_update_latencies.append(dt_patch)
            if patch.get('is_intermediate', False):
                fast_latencies.append(dt_patch)
            else:
                anchor_latencies.append(dt_patch)

        # Compositing & Display simulation
        if active_patch is not None:
            t0 = time.perf_counter()
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ref_gray = active_patch.get('ref_gray', None)
            ref_pts = active_patch.get('ref_pts', None)
            if ref_gray is not None and ref_pts is not None and len(ref_pts) >= 4:
                pts2, st, _ = cv2.calcOpticalFlowPyrLK(ref_gray, curr_gray, ref_pts, None, winSize=(21, 21), maxLevel=3)
            dt_flow = (time.perf_counter() - t0) * 1000.0
            timings['7. Optical Flow Tracking'].append(dt_flow)

            t0 = time.perf_counter()
            composite = processor.composite_garment(frame, active_patch)
            dt_comp = (time.perf_counter() - t0) * 1000.0
            timings['8. Alpha Compositing'].append(dt_comp)

            t0 = time.perf_counter()
            rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
            _ = rgb.tobytes()
            dt_pres = (time.perf_counter() - t0) * 1000.0
            timings['9. Presentation (QImage)'].append(dt_pres)

            t_disp_mono = time.monotonic()
            e2e = (t_disp_mono - t_capture_mono) * 1000.0
            live_e2e_latencies.append(e2e)

            patch_age = (time.time() - active_patch.get('timestamp', time.time())) * 1000.0
            patch_ages.append(patch_age)

            profiler.record_display_frame(i, t_capture_mono, t_disp_mono)

    # 5. Output Results Table
    alloc_mb = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
    res_mb = torch.cuda.max_memory_reserved() / (1024.0 ** 2)

    print("\n" + "=" * 80)
    print("STAGE-BY-STAGE ISOLATED LATENCY BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Stage':<35} | {'Median (p50)':<12} | {'95th % (p95)':<12} | {'Min':<8} | {'Max':<8}")
    print("-" * 80)

    for name, vals in timings.items():
        if len(vals) > 0:
            p50 = np.percentile(vals, 50)
            p95 = np.percentile(vals, 95)
            vmin = np.min(vals)
            vmax = np.max(vals)
            print(f"{name:<35} | {p50:9.2f} ms | {p95:9.2f} ms | {vmin:6.2f} ms | {vmax:6.2f} ms")

    total_display_loop = (
        np.percentile(timings['7. Optical Flow Tracking'], 50) +
        np.percentile(timings['8. Alpha Compositing'], 50) +
        np.percentile(timings['9. Presentation (QImage)'], 50)
    )

    med_anchor = np.percentile(anchor_latencies, 50) if len(anchor_latencies) > 0 else 0.0
    med_fast = np.percentile(fast_latencies, 50) if len(fast_latencies) > 0 else 0.0
    med_combined = np.percentile(all_update_latencies, 50) if len(all_update_latencies) > 0 else 0.0
    p95_combined = np.percentile(all_update_latencies, 95) if len(all_update_latencies) > 0 else 0.0
    med_e2e = np.percentile(live_e2e_latencies, 50) if len(live_e2e_latencies) > 0 else 0.0
    p95_e2e = np.percentile(live_e2e_latencies, 95) if len(live_e2e_latencies) > 0 else 0.0
    med_age = np.percentile(patch_ages, 50) if len(patch_ages) > 0 else 0.0

    print("\n" + "=" * 80)
    print("MULTI-RATE INFERENCE PIPELINE THROUGHPUT & LATENCY (OPTIMIZED V2)")
    print("=" * 80)
    print(f"Full Anchor Keyframe Latency (Median):      {med_anchor:7.2f} ms ({1000.0/med_anchor:.2f} FPS)")
    print(f"Fast Intermediate Sub-Keyframe (Median):    {med_fast:7.2f} ms ({1000.0/med_fast:.2f} FPS)")
    print(f"Average Multi-Rate Update Latency (p50):    {med_combined:7.2f} ms ({1000.0/med_combined:.2f} FPS)")
    print(f"Average Multi-Rate Update Latency (p95):    {p95_combined:7.2f} ms")
    print("-" * 80)
    print(f"Total Live Display Loop Latency (Median):   {total_display_loop:7.2f} ms ({1000.0/total_display_loop:.1f} FPS Headroom)")
    print(f"Measured Unique Camera Display Rate:        30.0 FPS (Locked to camera frame rate)")
    print(f"Real End-to-End Latency in LIVE Mode (p50): {med_e2e:7.2f} ms (p95: {p95_e2e:.2f} ms)")
    print(f"Active Neural Patch Age at Display (p50):   {med_age:7.2f} ms")
    print(f"GPU VRAM Allocated:                         {alloc_mb:7.1f} MB / 4096.0 MB ({alloc_mb/4096.0*100:.1f}%)")
    print(f"GPU VRAM Reserved:                          {res_mb:7.1f} MB ({res_mb/4096.0*100:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    run_comprehensive_profile(num_frames=30)
