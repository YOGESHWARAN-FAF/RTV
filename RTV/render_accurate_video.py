#!/usr/bin/env python3
"""
RTV Real-Time Virtual Try-On - Accurate 30 FPS Video Generator
=============================================================
Processes any user or sample video with high accuracy by pacing the video
processing, generating dense neural keyframes, and synthesizing all missing
intermediate frames via bidirectional flow morphing.
Outputs a clean, solid, photorealistic video at rock-solid 30 FPS.
"""

import os
import sys
import time
import argparse
import subprocess
import shutil
import cv2
import numpy as np
import torch

# Ensure working directory is always RTV directory
RTV_DIR = os.path.abspath(os.path.dirname(__file__))
if os.getcwd() != RTV_DIR:
    os.chdir(RTV_DIR)
if RTV_DIR not in sys.path:
    sys.path.insert(0, RTV_DIR)

from VITON.viton_fullbody_seq import FullBodySeqFrameProcessor
from VITON.viton_upperbody import FrameProcessor


def render_accurate_video(
    input_video_path,
    garment_name="coat_seq_vmssdp2ta_576",
    output_video_path="output/accurate_tryon_30fps.mp4",
    keyframe_stride=1,
    output_fps=30.0,
    max_frames=0,
    progress_callback=None
):
    """
    Renders an accurate virtual try-on video at 30 FPS.
    - Takes whatever time is required per frame to achieve ultra-realistic quality.
    - Computes dense neural keyframes every `keyframe_stride` frames.
    - Accurately generates all missing intermediate frames via bidirectional flow morphing.
    - Enforces 100% solid, non-mesh, non-shaking cloth output at 30.0 FPS.
    """
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video file not found: {input_video_path}")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if max_frames > 0:
        total_frames = min(total_frames, max_frames)

    print("=" * 70)
    print("RTV ACCURATE 30 FPS VIDEO GENERATOR (SLOW PACED, DENSE SYNTHESIS)")
    print(f"Input:       {input_video_path} ({width}x{height}, {total_frames} frames)")
    print(f"Garment:     {garment_name}")
    print(f"Stride:      Keyframe every {keyframe_stride} frame(s) with missing frame generation")
    print(f"Output:      {output_video_path} at {output_fps:.1f} FPS")
    print("=" * 70)

    # Determine processor type
    is_upperbody = any(ub_id in garment_name for ub_id in ['lab_', 'jin_'])
    if is_upperbody:
        print(f"[RTV] Initializing UpperBody AMP FP16 Engine for '{garment_name}'...")
        processor = FrameProcessor([garment_name])
        processor.switch_to_target_garment(0)
    else:
        print(f"[RTV] Initializing FullBody Sequence Engine for '{garment_name}'...")
        processor = FullBodySeqFrameProcessor(garment_name)
    if hasattr(processor, 'reset_filters'):
        processor.reset_filters()
    if hasattr(processor, 'smpl_regressor') and hasattr(processor.smpl_regressor, 'reset_tracking'):
        processor.smpl_regressor.reset_tracking()

    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
    temp_raw_path = output_video_path.replace(".mp4", "_temp_raw.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(temp_raw_path, fourcc, output_fps, (width, height))

    # Read all required frames into memory / stream buffer
    print("[RTV] Reading video frames...")
    frames = []
    while len(frames) < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    total_frames = len(frames)
    print(f"[RTV] Loaded {total_frames} frames into processing pipeline.")

    # Generate neural keyframe patches
    # Keyframe indices: 0, stride, 2*stride, ..., total_frames - 1
    keyframe_indices = list(range(0, total_frames, max(1, keyframe_stride)))
    if (total_frames - 1) not in keyframe_indices:
        keyframe_indices.append(total_frames - 1)

    print(f"[RTV] Generating {len(keyframe_indices)} neural keyframes at slow paced accuracy...")
    keyframes = {}
    t_start = time.time()

    last_valid_patch = None
    for ki_idx, f_idx in enumerate(keyframe_indices):
        t_k0 = time.time()
        try:
            patch = processor.get_garment_patch(frames[f_idx], isRGB=False, allow_multirate=False)
        except TypeError:
            patch = processor.get_garment_patch(frames[f_idx], isRGB=False)

        if patch is not None:
            last_valid_patch = patch
            keyframes[f_idx] = patch
        elif last_valid_patch is not None:
            keyframes[f_idx] = last_valid_patch
        else:
            keyframes[f_idx] = None

        dt_k = (time.time() - t_k0) * 1000.0
        elapsed = time.time() - t_start
        rate = (ki_idx + 1) / max(0.001, elapsed)
        rem = (len(keyframe_indices) - (ki_idx + 1)) / max(0.001, rate)
        status_msg = f"Keyframe {ki_idx+1}/{len(keyframe_indices)} (Frame {f_idx}) - {dt_k:.1f}ms [ETA: {rem:.1f}s]"
        print(f"  -> {status_msg}")
        if progress_callback is not None:
            progress_callback(ki_idx + 1, len(keyframe_indices), status_msg)

    # Backward fill for any initial None keyframes
    valid_keys = [k for k in keyframe_indices if keyframes.get(k) is not None]
    if valid_keys:
        first_valid = keyframes[valid_keys[0]]
        for k in keyframe_indices:
            if keyframes.get(k) is None:
                keyframes[k] = first_valid

    # Synthesize missing intermediate frames with bidirectional morphing & compose
    print("\n[RTV] Synthesizing missing frames and composing photorealistic 30 FPS video...")
    rendered_frames = []

    for idx in range(total_frames):
        # Find preceding and succeeding keyframes
        prev_k = max([k for k in keyframe_indices if k <= idx], default=0)
        next_k = min([k for k in keyframe_indices if k >= idx], default=prev_k)

        patch_prev = keyframes.get(prev_k, None)
        patch_next = keyframes.get(next_k, None)

        if prev_k == next_k or patch_next is None:
            # Exact keyframe
            if patch_prev is not None:
                composite = processor.composite_garment(frames[idx], patch_prev, use_exact_inv=True)
            else:
                composite = frames[idx].copy()
        else:
            # Missing intermediate frame: calculate morphing alpha factor
            alpha = float(idx - prev_k) / float(next_k - prev_k)
            composite = processor.composite_garment(
                frames[idx], patch_prev, next_patch=patch_next, alpha=alpha
            )

        out_writer.write(composite)
        rendered_frames.append(composite)

        if idx % 10 == 0 or idx == total_frames - 1:
            print(f"  -> Rendered frame {idx+1}/{total_frames} (Missing frame morphing active)")

    out_writer.release()
    total_time = time.time() - t_start
    print(f"\n[RTV] Rendering Complete! Processed {total_frames} frames in {total_time:.1f}s ({total_frames/total_time:.2f} proc FPS)")

    # Final H.264 30 FPS MP4 Conversion via FFmpeg
    final_output = output_video_path
    if shutil.which("ffmpeg"):
        try:
            print(f"[RTV] Encoding final H.264 30 FPS MP4 via FFmpeg: {final_output}")
            cmd = [
                "ffmpeg", "-y", "-i", temp_raw_path,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-r", str(output_fps),
                "-movflags", "+faststart", final_output
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if os.path.exists(temp_raw_path) and os.path.abspath(temp_raw_path) != os.path.abspath(final_output):
                os.remove(temp_raw_path)
        except Exception as e:
            print(f"[RTV] FFmpeg warning (using raw MP4): {e}")
            if os.path.exists(temp_raw_path) and not os.path.exists(final_output):
                os.rename(temp_raw_path, final_output)
    else:
        if os.path.exists(temp_raw_path) and not os.path.exists(final_output):
            os.rename(temp_raw_path, final_output)

    print(f"[RTV] SUCCESS: Output saved to: {final_output} at 30 FPS.")
    return final_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTV Accurate 30 FPS Virtual Try-On Video Generator")
    parser.add_argument("--input", "-i", type=str, default="assets/sample_video.mp4", help="Path to input video file")
    parser.add_argument("--garment", "-g", type=str, default="coat_seq_vmssdp2ta_576", help="Target garment model name")
    parser.add_argument("--output", "-o", type=str, default="output/accurate_tryon_30fps.mp4", help="Path to output video file")
    parser.add_argument("--stride", "-s", type=int, default=3, help="Keyframe stride (e.g. 1 for all-neural, 3 for dense morphing)")
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS (default: 30.0)")
    parser.add_argument("--max_frames", "-m", type=int, default=0, help="Max frames to process (0 for full video)")

    args = parser.parse_args()
    render_accurate_video(
        input_video_path=args.input,
        garment_name=args.garment,
        output_video_path=args.output,
        keyframe_stride=args.stride,
        output_fps=args.fps,
        max_frames=args.max_frames
    )

