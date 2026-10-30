import os
import sys
import time
import threading
import collections
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("./model/DensePose"))

from VITON.viton_fullbody_seq import FullBodySeqFrameProcessor
from util.pipeline_profiler import PipelineProfiler


def run_multirate_live_benchmark(duration_seconds=60.0, target_fps=30.0):
    print("=" * 85)
    print("RTV V3: MULTI-RATE REAL-TIME VIRTUAL TRY-ON PERFORMANCE BENCHMARK")
    print("Target Platform: Windows 11, NVIDIA GeForce RTX 2050 (4 GB VRAM), 16 GB RAM")
    print(f"Benchmark Configuration: Duration = {duration_seconds:.1f}s | Target Camera FPS = {target_fps:.1f}")
    print("=" * 85)

    # 1. Load video frames
    video_path = "./assets/sample_video.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    raw_frames = []
    while len(raw_frames) < 180:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        raw_frames.append(frame)
    cap.release()
    print(f"[Dataset] Pre-loaded {len(raw_frames)} source video frames ({raw_frames[0].shape[1]}x{raw_frames[0].shape[0]}).")

    # 2. Initialize Model & Pipeline
    print("[Pipeline] Initializing FullBodySeqFrameProcessor ('coat_seq_vmssdp2ta_576')...")
    t_init = time.perf_counter()
    processor = FullBodySeqFrameProcessor('coat_seq_vmssdp2ta_576')
    print(f"  Initialized in {(time.perf_counter() - t_init):.2f}s")

    # 3. GPU Warmup Pass
    print("[Warmup] Executing GPU inference warmup pass...")
    _ = processor.get_garment_patch(raw_frames[0])
    torch.cuda.synchronize()
    print("  GPU Warmup complete.")

    # 4. Multi-rate Architecture Setup
    profiler = PipelineProfiler(window_size=120)

    running = True
    new_frame_event = threading.Event()
    latest_frame_lock = threading.Lock()
    latest_frame_packet = None

    patch_lock = threading.Lock()
    latest_patch = None

    # Telemetry metrics collection
    reanchor_triggers = 0
    tracking_confidences = []
    tracking_drifts = []
    tracking_fps_samples = []

    # Camera Capture Thread
    def capture_worker():
        nonlocal latest_frame_packet
        frame_idx = 0
        seq = 0
        frame_interval = 1.0 / target_fps
        t_next = time.monotonic()

        while running:
            t_now = time.monotonic()
            if t_now < t_next:
                time.sleep(max(0.0005, t_next - t_now))
                continue

            t_next += frame_interval
            frame = raw_frames[frame_idx % len(raw_frames)]
            frame_idx += 1
            seq += 1

            mono_ts = time.monotonic()
            wall_ts = time.time()
            profiler.record_capture(seq, mono_ts)

            packet = {
                'seq': seq,
                'mono_ts': mono_ts,
                'wall_ts': wall_ts,
                'frame': frame
            }

            with latest_frame_lock:
                latest_frame_packet = packet

            new_frame_event.set()

    # Neural ML Synthesis Worker
    def ml_worker():
        nonlocal latest_patch, reanchor_triggers
        last_processed_seq = -1

        while running:
            target_packet = None
            with latest_frame_lock:
                if latest_frame_packet is not None and latest_frame_packet['seq'] > last_processed_seq:
                    target_packet = latest_frame_packet

            if target_packet is None:
                time.sleep(0.005)
                continue

            # Adaptive Re-Anchoring Logic
            tr_info = processor.get_tracking_info()
            needs_reanchor = tr_info.get('needs_reanchor', False)

            with patch_lock:
                has_patch = latest_patch is not None
                patch_age = (time.time() - latest_patch.get('timestamp', 0.0)) if has_patch else 999.0

            if not needs_reanchor and has_patch and patch_age < 0.25:
                time.sleep(0.005)
                continue

            if needs_reanchor:
                reanchor_triggers += 1

            target_frame = target_packet['frame'].copy()
            target_seq = target_packet['seq']
            target_ts = target_packet['wall_ts']
            target_mono_ts = target_packet['mono_ts']
            last_processed_seq = target_seq

            t_ml_start = time.perf_counter()
            with torch.inference_mode():
                patch = processor.get_garment_patch(target_frame, isRGB=False, allow_multirate=True)
            torch.cuda.synchronize()
            dt_ml = (time.perf_counter() - t_ml_start) * 1000.0

            if patch is not None:
                patch['timestamp'] = target_ts
                patch['mono_ts'] = target_mono_ts
                patch['seq'] = target_seq
                patch['inference_time_ms'] = dt_ml

                with patch_lock:
                    latest_patch = patch

            profiler.record_ml_inference(dt_ml)
            time.sleep(0.002)

    # Launch background threads
    cap_thread = threading.Thread(target=capture_worker, daemon=True)
    ml_thread = threading.Thread(target=ml_worker, daemon=True)
    cap_thread.start()
    ml_thread.start()

    # Wait for first patch
    print("[Pipeline] Waiting for initial anchor patch...")
    while latest_patch is None and running:
        time.sleep(0.05)
    print("  Initial neural anchor ready. Starting live 30 FPS display & tracking benchmark...")

    # Display & Tracking Loop (Runs on every camera frame arrival)
    start_time = time.monotonic()
    last_disp_seq = -1
    last_report_time = start_time
    total_frames_presented = 0

    while running and (time.monotonic() - start_time) < duration_seconds:
        got_event = new_frame_event.wait(timeout=0.04)
        new_frame_event.clear()

        loop_start = time.monotonic()
        curr_packet = None
        with latest_frame_lock:
            if latest_frame_packet is not None and latest_frame_packet['seq'] != last_disp_seq:
                curr_packet = latest_frame_packet

        if curr_packet is None:
            time.sleep(0.001)
            continue

        last_disp_seq = curr_packet['seq']
        curr_frame = curr_packet['frame']
        cap_mono_ts = curr_packet['mono_ts']

        patch_to_use = None
        with patch_lock:
            if latest_patch is not None:
                patch_to_use = latest_patch

        # Live Garment Tracking & Compositing
        t_track_start = time.perf_counter()
        if patch_to_use is not None:
            composite = processor.composite_garment(curr_frame, patch_to_use)
            patch_age_ms = max(0.0, (time.time() - patch_to_use.get('timestamp', time.time())) * 1000.0)
            profiler.record_stage('patch_age', patch_age_ms)
        else:
            composite = curr_frame
            patch_age_ms = 0.0

        t_track_ms = (time.perf_counter() - t_track_start) * 1000.0
        profiler.record_stage('tracking_lat', t_track_ms)
        profiler.record_stage('compositing', t_track_ms)
        profiler.record_tracking_update(t_track_ms, loop_start)

        # Presentation (QImage RGB byte simulation)
        t_pres_start = time.perf_counter()
        rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
        _ = rgb.tobytes()
        t_pres_ms = (time.perf_counter() - t_pres_start) * 1000.0
        profiler.record_stage('presentation', t_pres_ms)

        disp_mono_ts = time.monotonic()
        profiler.record_display_frame(last_disp_seq, cap_mono_ts, disp_mono_ts)
        total_frames_presented += 1

        # Track quality diagnostics
        tr_info = processor.get_tracking_info()
        tracking_confidences.append(tr_info.get('valid_ratio', 1.0))
        tracking_drifts.append(tr_info.get('disp', 0.0))
        if profiler.tracking_fps > 0:
            tracking_fps_samples.append(profiler.tracking_fps)

        # Live console telemetry update every 5 seconds
        if disp_mono_ts - last_report_time >= 5.0:
            elapsed = disp_mono_ts - start_time
            print(f"  [{elapsed:4.1f}s / {duration_seconds:.0f}s] "
                  f"Cam: {profiler.camera_fps:4.1f} FPS | "
                  f"Disp: {profiler.unique_display_fps:4.1f} FPS | "
                  f"Tracking: {profiler.tracking_fps:4.1f} FPS ({t_track_ms:4.1f}ms) | "
                  f"Neural: {profiler.ml_fps:3.1f} FPS ({profiler.ml_latency_ms:3.0f}ms) | "
                  f"Age: {patch_age_ms:4.0f}ms | "
                  f"E2E Lat: {profiler.end_to_end_latency_ms:4.1f}ms")
            last_report_time = disp_mono_ts

    running = False
    cap_thread.join(timeout=1.0)
    ml_thread.join(timeout=1.0)

    # 5. Output Final Comprehensive Benchmark Results Table
    e2e_p50, e2e_p95 = profiler.get_stage_percentiles('end_to_end_lat')
    track_p50, track_p95 = profiler.get_stage_percentiles('tracking_lat')
    patch_p50, patch_p95 = profiler.get_stage_percentiles('patch_age')
    pres_p50, pres_p95 = profiler.get_stage_percentiles('presentation')
    alloc_mb, res_mb, total_mb = profiler.get_vram_stats()

    avg_track_fps = float(np.mean(tracking_fps_samples)) if len(tracking_fps_samples) > 0 else profiler.tracking_fps
    min_track_fps = float(np.percentile(tracking_fps_samples, 5)) if len(tracking_fps_samples) > 0 else avg_track_fps
    avg_conf = float(np.mean(tracking_confidences)) if len(tracking_confidences) > 0 else 1.0
    avg_drift = float(np.mean(tracking_drifts)) if len(tracking_drifts) > 0 else 0.0

    print("\n" + "=" * 85)
    print("FINAL BENCHMARK RESULTS: 3-INDEPENDENT-METRIC RTV V3 PIPELINE")
    print("=" * 85)
    print(f"{'Performance Metric':<45} | {'Measured Value':<35}")
    print("-" * 85)
    print(f"{'1. Unique Camera Frames Presented (Disp FPS)':<45} | {profiler.unique_display_fps:6.2f} FPS")
    print(f"{'2. Garment Tracking/Compositing (Tracking FPS)':<45} | {avg_track_fps:6.2f} FPS (Min: {min_track_fps:.2f} FPS)")
    print(f"{'3. Fresh Neural Garment Outputs (Neural FPS)':<45} | {profiler.ml_fps:6.2f} FPS")
    print("-" * 85)
    print(f"{'End-to-End Latency (p50 / p95)':<45} | {e2e_p50:6.2f} ms / {e2e_p95:.2f} ms")
    print(f"{'Active Neural Patch Age at Display (p50 / p95)':<45} | {patch_p50:6.2f} ms / {patch_p95:.2f} ms")
    print(f"{'Garment Tracking & Compositing Latency (p50)':<45} | {track_p50:6.2f} ms (p95: {track_p95:.2f} ms)")
    print(f"{'Neural Anchor Generation Latency (p50)':<45} | {profiler.ml_latency_ms:6.2f} ms")
    print(f"{'Presentation Latency (p50)':<45} | {pres_p50:6.2f} ms")
    print("-" * 85)
    print(f"{'Tracking Optical Flow Confidence (Average)':<45} | {avg_conf * 100:6.1f} %")
    print(f"{'Tracking Optical Flow Drift (Average)':<45} | {avg_drift:6.2f} px")
    print(f"{'Adaptive Re-Anchor Triggers':<45} | {reanchor_triggers} triggers")
    print(f"{'NVIDIA GPU VRAM Allocated':<45} | {alloc_mb:6.1f} MB / {total_mb:.1f} MB ({alloc_mb/total_mb*100:.1f}%)")
    print(f"{'NVIDIA GPU VRAM Reserved':<45} | {res_mb:6.1f} MB ({res_mb/total_mb*100:.1f}%)")
    print("=" * 85)

    assert avg_track_fps >= 20.0, f"Tracking FPS failed acceptance criteria: {avg_track_fps:.2f} < 20.0 FPS"
    print("\n>>> ACCEPTANCE CRITERIA MET: Sustained 20–30 FPS Real-Time Virtual Try-On Achieved! <<<\n")


if __name__ == "__main__":
    duration = 60.0
    if len(sys.argv) > 1 and sys.argv[1].replace('.', '', 1).isdigit():
        duration = float(sys.argv[1])
    run_multirate_live_benchmark(duration_seconds=duration, target_fps=30.0)
