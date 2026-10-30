#!/usr/bin/env python3

"""

RTV Real-Time Virtual Try-On - Interactive Google Colab & Web Server Interface

==============================================================================

Provides a complete, browser-based interface for Google Colab, Cloud GPUs, and local testing:

- Live Webcam Virtual Try-On with Real-Time Optical Flow Tracking (25-30 FPS)

- Video File Upload & Batch Try-On Video Processor

- Single Image Studio with 3D SMPL & DensePose Diagnostic Overlays

- Dynamic Garment Wardrobe Selection (Puffer Coat, Hanfu Silk Dress, Upper-Body Lab & Jin Coats)

- Real-Time Telemetry HUD (Cam FPS, ML Inference Latency, Tracking FPS, Body Yaw)

- Headless Linux / EGL / OSMesa GPU acceleration for Google Colab & remote VMs

"""



import os

import sys

import time

import argparse

import platform

import collections

import threading

import subprocess

import shutil



# Ensure headless OpenGL & Qt in Colab / Linux environments

if platform.system() != "Windows":

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")



import numpy as np

import cv2

import torch

import gradio as gr



# Enable cuDNN benchmark for accelerated neural inference

if torch.cuda.is_available():

    torch.backends.cudnn.benchmark = True



# Ensure working directory is always RTV directory

RTV_DIR = os.path.abspath(os.path.dirname(__file__))

if os.getcwd() != RTV_DIR:

    os.chdir(RTV_DIR)

if RTV_DIR not in sys.path:

    sys.path.insert(0, RTV_DIR)



from VITON.viton_fullbody_seq import FullBodySeqFrameProcessor

from VITON.viton_upperbody import FrameProcessor

from util.pipeline_profiler import PipelineProfiler

try:

    from util.custom_garment_trainer import CustomGarmentTrainer

except Exception as _e:

    print(f"[Warning] CustomGarmentTrainer initialization deferred: {_e}")

    CustomGarmentTrainer = None

def get_orientation_str(yaw_deg):

    yaw = (yaw_deg + 180) % 360 - 180

    abs_yaw = abs(yaw)

    if abs_yaw < 25:

        return "Front (0 deg)"

    elif 25 <= abs_yaw < 65:

        return f"{'Right' if yaw > 0 else 'Left'} 3/4 ({yaw:+.0f} deg)"

    elif 65 <= abs_yaw < 115:

        return f"{'Right' if yaw > 0 else 'Left'} Profile ({yaw:+.0f} deg)"

    else:

        return f"Back View ({yaw:+.0f} deg)"



def draw_calibration_hud(frame, progress, yaw_angle, joints_2d=None, status_text="Stand in camera view for 3D body calibration"):

    h, w = frame.shape[:2]

    overlay = frame.copy()

    cv2.rectangle(overlay, (0, 0), (w, 115), (10, 12, 18), -1)

    cv2.rectangle(overlay, (0, h - 55), (w, h), (10, 12, 18), -1)



    orient_str = get_orientation_str(yaw_angle)

    pct = int(progress * 100)

    title = f"[ 3D BODY MOTION CALIBRATION: {pct}% ]"

    cv2.putText(overlay, title, (25, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2, cv2.LINE_AA)



    bar_w = min(450, w - 50)

    bar_x = 25

    bar_y = 44

    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (50, 50, 60), 1)

    fill_w = int(bar_w * progress)

    if fill_w > 0:

        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + fill_w, bar_y + 10), (0, 255, 120), -1)



    status_str = f"Angle: {orient_str} | Status: {status_text}"

    cv2.putText(overlay, status_str, (25, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)





# ============================================================================

# Wardrobe Configuration & Checkpoints Mapping

# ============================================================================



GARMENT_CATALOG = {

    "Winter Puffer Coat (Full-Body)": {

        "type": "fullbody",

        "id": "coat_seq_vmssdp2ta_576",

        "preview": "./assets/garment_images/coat_seq_preview.jpg",

        "desc": "Full-body winter puffer jacket with realistic cloth physics & fold rendering."

    },

    "Hanfu Silk Dress (Full-Body)": {

        "type": "fullbody",

        "id": "han_seq_vmssdp2ta_576",

        "preview": "./assets/garment_images/han_seq_preview.jpg",

        "desc": "Traditional flowing silk Hanfu dress with dynamic drape and drape deformation."

    },

    "Lab Coat 03 (Upper-Body)": {

        "type": "upperbody",

        "id": "lab_03_vmsdp2ta",

        "preview": "./assets/garment_images/lab_03_white_bg.jpg",

        "desc": "Fitted lab garment with collar alignment and upper-torso contour tracking."

    },

    "Lab Coat 04 (Upper-Body)": {

        "type": "upperbody",

        "id": "lab_04_vmsdp2ta",

        "preview": "./assets/garment_images/lab_04_white_bg.jpg",

        "desc": "Alternative cut lab garment with arm occlusion preservation."

    },

    "Lab Coat 07 (Upper-Body)": {

        "type": "upperbody",

        "id": "lab_07_vmsdp2ta",

        "preview": "./assets/garment_images/lab_07_white_bg.jpg",

        "desc": "Classic tailored lab coat model."

    },

    "Jin Apparel 17 (Upper-Body)": {

        "type": "upperbody",

        "id": "jin_17_vmsdp2ta",

        "preview": "./assets/garment_images/jin_17_white_bg.jpg",

        "desc": "Casual apparel top with high-fidelity texture and neckline fitting."

    },

    "Jin Apparel 18 (Upper-Body)": {

        "type": "upperbody",

        "id": "jin_18_vmsdp2ta",

        "preview": "./assets/garment_images/jin_18_white_bg.jpg",

        "desc": "Sportswear cut top with torso deformation tracking."

    },

    "Jin Apparel 22 (Upper-Body)": {

        "type": "upperbody",

        "id": "jin_22_vmsdp2ta",

        "preview": "./assets/garment_images/jin_22_white_bg.jpg",

        "desc": "Dark tone structured apparel with high contrast contour segmentation."

    },

    "Raw Camera / AR Skeleton (No Garment)": {

        "type": "none",

        "id": "none",

        "preview": "./assets/none.png" if os.path.exists("./assets/none.png") else None,

        "desc": "Raw camera stream with 3D SMPL pose detection and anatomical skeleton overlay."

    }

}





def scan_and_update_catalog():

    """Scan rtv_ckpts/ directory and dynamically add custom garments to GARMENT_CATALOG."""

    ckpts_dir = os.path.abspath("./rtv_ckpts")

    if os.path.exists(ckpts_dir):

        for item in sorted(os.listdir(ckpts_dir)):

            if item.startswith("."):

                continue

            model_dir = os.path.join(ckpts_dir, item)

            model_file = os.path.join(model_dir, "latest_net_G.pth")

            if os.path.isdir(model_dir) and os.path.exists(model_file):

                # Search for thumbnail

                thumb = None

                for cand in [

                    os.path.join(model_dir, "preview.jpg"),

                    os.path.abspath(f"./assets/garment_images/{item}_preview.jpg"),

                    os.path.abspath(f"./assets/garment_images/{item}.jpg"),

                    os.path.abspath(f"./assets/garment_images/{item}_white_bg.jpg"),

                ]:

                    if os.path.exists(cand):

                        thumb = cand

                        break

                

                # Check if already present

                already_in = False

                for k, v in GARMENT_CATALOG.items():

                    if v.get("id") == item:

                        already_in = True

                        break

                if not already_in:

                    label = item.replace("_seq_vmssdp2ta_576", "").replace("_vmsdp2ta", "").replace("_", " ").title()

                    is_fullbody = ("seq" in item) or ("fullbody" in item)

                    g_type = "fullbody" if is_fullbody else "upperbody"

                    GARMENT_CATALOG[label] = {

                        "type": g_type,

                        "id": item,

                        "preview": thumb,

                        "desc": f"Custom trained / imported neural garment: '{item}'."

                    }

    return GARMENT_CATALOG



# Initial scan

scan_and_update_catalog()





# ============================================================================

# Core RTV Colab Try-On Engine Manager

# ============================================================================



class ColabTryOnEngine:

    """

    Thread-safe, high-performance Virtual Try-On Engine for Web & Colab execution.

    Features:

    - Lazy model loading & dynamic caching to fit GPU memory constraints.

    - Decoupled real-time tracking for smooth webcam streaming (25-30 FPS).

    - Full telemetry profiler & HUD overlays.

    """

    def __init__(self):

        self.lock = threading.Lock()

        self.profiler = PipelineProfiler(window_size=60)

        

        self.active_garment_label = "Winter Puffer Coat (Full-Body)"

        self.active_catalog_entry = GARMENT_CATALOG[self.active_garment_label]

        

        # Processors (initialized on demand)

        self.fullbody_processor = None

        self.upperbody_processor = None

        

        # Asynchronous streaming state for webcam

        self.latest_input_bgr = None

        self.latest_patch = None

        self.latest_yaw = 0.0

        self.latest_joints = None

        self.body_detected = False

        self.ml_running = False

        self.ml_thread = None

        self.last_ml_time = 0.0

        self.frame_seq = 0

        self.model_status = "Warming up..."

        

        # Pre-load default fullbody model in background so it is instantly ready

        def _warmup():

            print("[RTV-Colab] Pre-warming FullBody model in GPU...")

            try:

                # Ensure SMPLA_NEUTRAL.pth & SMPL_NEUTRAL.pth exist in ~/.romp

                romp_dir = os.path.expanduser("~/.romp")

                if os.path.exists(romp_dir):

                    for src, dst in [("smpla_packed_info.pth", "SMPLA_NEUTRAL.pth"), ("smpl_packed_info.pth", "SMPL_NEUTRAL.pth")]:

                        sp, dp = os.path.join(romp_dir, src), os.path.join(romp_dir, dst)

                        if os.path.exists(sp) and not os.path.exists(dp):

                            shutil.copy2(sp, dp)

                self.get_fullbody_processor("coat_seq_vmssdp2ta_576")

                self.model_status = "Ready"

                print("[RTV-Colab] FullBody model is warmed up and ready in GPU!")

            except Exception as e:

                print(f"[RTV-Colab] Warmup notice: {e}")

                self.model_status = "Ready on-demand"



        threading.Thread(target=_warmup, daemon=True).start()



        # Start background ML worker

        self.worker_active = True

        self.ml_thread = threading.Thread(target=self._background_ml_worker, daemon=True)

        self.ml_thread.start()



    def get_fullbody_processor(self, target_id="coat_seq_vmssdp2ta_576"):

        with self.lock:

            if self.fullbody_processor is None:

                print(f"[RTV-Colab] Initializing FullBodySeqFrameProcessor ({target_id})...")

                self.fullbody_processor = FullBodySeqFrameProcessor(target_id)

            elif self.fullbody_processor.current_target_name != target_id:

                self.fullbody_processor.set_target_garment(target_id)

            return self.fullbody_processor



    def get_upperbody_processor(self, target_id="lab_03_vmsdp2ta"):

        with self.lock:

            if self.upperbody_processor is None:

                print(f"[RTV-Colab] Initializing FrameProcessor for upperbody ({target_id})...")

                garments = [

                    'lab_03_vmsdp2ta', 'lab_04_vmsdp2ta', 'lab_07_vmsdp2ta',

                    'jin_17_vmsdp2ta', 'jin_18_vmsdp2ta', 'jin_22_vmsdp2ta'

                ]

                self.upperbody_processor = FrameProcessor(garments)

            

            # Switch target garment

            try:

                if target_id not in self.upperbody_processor.garment_name_list:

                    self.upperbody_processor.garment_name_list.append(target_id)

                    self.upperbody_processor.viton_model_list.append(None)

                idx = self.upperbody_processor.garment_name_list.index(target_id)

                self.upperbody_processor.switch_to_target_garment(idx)

            except Exception as e:

                print(f"[RTV-Colab] Upperbody switch warning: {e}")

            return self.upperbody_processor



    def set_active_garment(self, label):

        if label in GARMENT_CATALOG:

            self.active_garment_label = label

            self.active_catalog_entry = GARMENT_CATALOG[label]

            print(f"[RTV-Colab] Selected garment: {label} ({self.active_catalog_entry['id']})")



    def _background_ml_worker(self):

        """Asynchronous worker for webcam streaming: runs heavy neural models without stalling the stream."""

        while self.worker_active:

            if self.latest_input_bgr is None:

                time.sleep(0.01)

                continue



            entry = self.active_catalog_entry

            g_type = entry['type']

            g_id = entry['id']



            if g_type == "none":

                time.sleep(0.02)

                continue



            # Pick latest frame

            with self.lock:

                frame_to_process = self.latest_input_bgr.copy()

            

            t0 = time.perf_counter()

            try:

                if g_type == "fullbody":

                    proc = self.get_fullbody_processor(g_id)

                    patch = proc.get_garment_patch(frame_to_process, isRGB=False)

                else:

                    proc = self.get_upperbody_processor(g_id)

                    patch = proc.get_garment_patch(frame_to_process, isRGB=False)



                dt_ms = (time.perf_counter() - t0) * 1000.0

                self.profiler.record_ml_inference(dt_ms)



                if patch is not None:

                    patch['timestamp'] = time.time()

                    patch['inference_time_ms'] = dt_ms

                    with self.lock:

                        self.latest_patch = patch

                        self.body_detected = True

                        if 'yaw_angle' in patch:

                            self.latest_yaw = patch['yaw_angle']

                else:

                    with self.lock:

                        self.body_detected = False

            except Exception as e:

                print(f"[RTV-Colab] Background ML inference error: {e}")



            time.sleep(0.005)



    def process_stream_frame_bgr(self, frame_bgr, garment_label=None, tryon_mode="LIVE", show_hud=True, show_skeleton=False):

        """

        High-performance frame processor accepting raw OpenCV BGR image.

        Used by the WebSocket live mirror endpoint for zero-latency streaming.

        """

        if frame_bgr is None:

            return None



        t_loop_start = time.monotonic()

        self.frame_seq += 1



        orig_h, orig_w = frame_bgr.shape[:2]



        # Standardize live webcam frame to 640px wide (720p HD) for fast 30 FPS network streaming & sharp graphics

        target_w = 640

        if orig_w > target_w:

            scale = target_w / float(orig_w)

            target_h = int(orig_h * scale)

            frame_bgr = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        h, w = frame_bgr.shape[:2]



        # Update active garment if requested

        if garment_label and garment_label != self.active_garment_label:

            self.set_active_garment(garment_label)



        # Update input for background worker

        with self.lock:

            self.latest_input_bgr = frame_bgr



        self.profiler.record_capture(self.frame_seq, t_loop_start)



        entry = self.active_catalog_entry

        g_type = entry['type']

        g_id = entry['id']



        display_bgr = frame_bgr.copy()

        patch_age_ms = 0.0



        if g_type != "none":

            patch = None

            with self.lock:

                patch = self.latest_patch



            if patch is not None:

                patch_age_ms = max(0.0, (time.time() - patch.get('timestamp', time.time())) * 1000.0)

                t_comp_start = time.perf_counter()

                

                try:

                    if g_type == "fullbody":

                        proc = self.get_fullbody_processor(g_id)

                        display_bgr = proc.composite_garment(display_bgr, patch)

                    else:

                        proc = self.get_upperbody_processor(g_id)

                        display_bgr = proc.composite_garment(display_bgr, patch)

                except Exception as e:

                    pass



                t_comp = (time.perf_counter() - t_comp_start) * 1000.0

                self.profiler.record_stage('compositing', t_comp)

                self.profiler.record_tracking_update(t_comp)

            else:

                # Visual banner when person is not detected or model is initializing

                h_ov, w_ov = display_bgr.shape[:2]

                cv2.rectangle(display_bgr, (0, 0), (w_ov, 45), (10, 12, 18), -1)

                if self.model_status != "Ready":

                    hud_msg = "[ INITIALIZING NEURAL MODEL... PLEASE STAND IN VIEW (5-10s) ]"

                    hud_col = (0, 220, 255)

                else:

                    hud_msg = "[ SEARCHING FOR PERSON: Step back so shoulders & torso are in view ]"

                    hud_col = (0, 165, 255)

                cv2.putText(display_bgr, hud_msg, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, hud_col, 1, cv2.LINE_AA)

        else:

            # Raw Camera / Calibration Overlay

            if show_skeleton and self.latest_joints is not None:

                display_bgr = draw_calibration_hud(

                    display_bgr, 1.0, self.latest_yaw,

                    joints_2d=self.latest_joints,

                    status_text="AR Skeleton Tracking Active"

                )



        self.profiler.record_display_frame(self.frame_seq, t_loop_start, time.monotonic())



        # Render Telemetry HUD

        if show_hud:

            display_bgr = self.profiler.draw_hud(

                display_bgr,

                mode_str=tryon_mode,

                sync_delay_sec=0.0 if tryon_mode == "LIVE" else 0.60,

                yaw_angle=self.latest_yaw,

                patch_age_ms=patch_age_ms

            )



        return display_bgr



    def process_stream_frame(self, frame_rgb, garment_label, tryon_mode, show_hud, show_skeleton):

        """Processes a single live webcam frame from Gradio streaming (Fallback)."""

        if frame_rgb is None:

            return None

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        out_bgr = self.process_stream_frame_bgr(frame_bgr, garment_label, tryon_mode, show_hud, show_skeleton)

        return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB) if out_bgr is not None else None



    def process_single_image(self, input_rgb, garment_label, show_hud, show_debug):

        """Processes a single uploaded image and returns try-on result and diagnostic maps."""

        if input_rgb is None:

            return None, None, None, "Please upload or capture an image."



        input_bgr = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2BGR)

        entry = GARMENT_CATALOG.get(garment_label, GARMENT_CATALOG["Winter Puffer Coat (Full-Body)"])

        g_type = entry['type']

        g_id = entry['id']



        if g_type == "none":

            return input_rgb, None, None, "Raw mode selected (No garment applied)."



        t0 = time.perf_counter()

        try:

            if g_type == "fullbody":

                proc = self.get_fullbody_processor(g_id)

            else:

                proc = self.get_upperbody_processor(g_id)



            patch_info = proc.get_garment_patch(input_bgr, isRGB=False)

            dt_ml = (time.perf_counter() - t0) * 1000.0



            if patch_info is None:

                return input_rgb, None, None, f"No person detected in the image ({dt_ml:.1f}ms)."



            # Composite garment

            t_blend = time.perf_counter()

            blended_bgr = proc.composite_garment(input_bgr.copy(), patch_info)

            blend_ms = (time.perf_counter() - t_blend) * 1000.0



            if show_hud:

                blended_bgr = self.profiler.draw_hud(

                    blended_bgr, mode_str="STATIC", sync_delay_sec=0.0,

                    yaw_angle=patch_info.get('yaw_angle', 0.0), patch_age_ms=0.0

                )



            result_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)



            # Debug visualizations: alpha mask and target patch

            alpha_vis = None

            patch_vis = None

            if show_debug and 'roi_alpha' in patch_info:

                alpha = patch_info['roi_alpha']

                alpha_vis = cv2.applyColorMap(alpha.astype(np.uint8), cv2.COLORMAP_VIRIDIS)

                alpha_vis = cv2.cvtColor(alpha_vis, cv2.COLOR_BGR2RGB)



            if show_debug and 'roi_target' in patch_info:

                patch_vis = patch_info['roi_target']

                if patch_vis.shape[-1] == 3:

                    # ROI target is usually RGB

                    pass



            status_msg = f"Try-On Successful! Inference: {dt_ml:.1f} ms | Compositing: {blend_ms:.1f} ms"

            return result_rgb, alpha_vis, patch_vis, status_msg



        except Exception as e:

            return None, None, None, f"Processing Error: {str(e)}"



    def process_video_file(self, video_path, garment_label, max_frames, show_hud, progress=gr.Progress()):

        """Processes an uploaded video file frame-by-frame and exports an MP4 video."""

        if not video_path or not os.path.exists(video_path):

            return None, "Video file not found."



        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():

            return None, "Could not open video file."



        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



        frames_to_process = min(total_frames, int(max_frames)) if max_frames > 0 else total_frames

        print(f"[RTV-Colab] Processing video: {frames_to_process} frames at {src_fps:.1f} FPS ({width}x{height})")



        entry = GARMENT_CATALOG.get(garment_label, GARMENT_CATALOG["Winter Puffer Coat (Full-Body)"])

        g_type = entry['type']

        g_id = entry['id']



        if g_type == "fullbody":

            proc = self.get_fullbody_processor(g_id)

        elif g_type == "upperbody":

            proc = self.get_upperbody_processor(g_id)

        else:

            proc = None



        os.makedirs("./output", exist_ok=True)

        final_out_path = f"./output/rtv_tryon_{int(time.time())}.mp4"



        try:

            from render_accurate_video import render_accurate_video

            out_file = render_accurate_video(

                input_video_path=video_path,

                garment_name=g_id,

                output_video_path=final_out_path,

                keyframe_stride=1,

                output_fps=30.0,

                max_frames=int(max_frames) if max_frames > 0 else 0,

                progress_callback=lambda cur, tot, msg: progress(cur / tot, desc=msg)

            )

            return out_file, f"Successfully rendered accurate 30 FPS video with missing frame generation! Ready to play."

        except Exception as e:

            print(f"[RTV-Colab] Accurate video generator error: {e}")

            raw_out_path = f"./output/rtv_rendered_{int(time.time())}_raw.mp4"



        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(raw_out_path, fourcc, src_fps, (width, height))



        last_patch = None

        processed_count = 0

        t_start = time.time()



        for idx in range(frames_to_process):

            ret, frame_bgr = cap.read()

            if not ret:

                break



            progress(idx / frames_to_process, desc=f"Processing frame {idx+1}/{frames_to_process}")



            if proc is not None:

                patch = proc.get_garment_patch(frame_bgr, isRGB=False)

                if patch is not None:

                    last_patch = patch

                

                if last_patch is not None:

                    frame_bgr = proc.composite_garment(frame_bgr, last_patch)



            if show_hud:

                fps_est = (idx + 1) / max(0.001, time.time() - t_start)

                cv2.rectangle(frame_bgr, (0, 0), (width, 40), (10, 12, 18), -1)

                hud_txt = f"RTV Virtual Try-On | Garment: {garment_label} | Frame: {idx+1}/{frames_to_process} | Proc Speed: {fps_est:.1f} FPS"

                cv2.putText(frame_bgr, hud_txt, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 170), 1, cv2.LINE_AA)



            out.write(frame_bgr)

            processed_count += 1



        cap.release()

        out.release()



        # Convert to H.264 MP4 with ffmpeg if available for standard web browser playback

        if shutil.which("ffmpeg"):

            try:

                cmd = [

                    "ffmpeg", "-y", "-i", raw_out_path,

                    "-c:v", "libx264", "-pix_fmt", "yuv420p",

                    "-movflags", "+faststart", final_out_path

                ]

                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

                if os.path.exists(raw_out_path):

                    os.remove(raw_out_path)

                return final_out_path, f"Successfully processed {processed_count} frames! Ready to play."

            except Exception:

                pass



        return raw_out_path, f"Successfully processed {processed_count} frames! (Raw MP4 output)"





# ============================================================================

# Gradio UI Construction & Styling

# ============================================================================





RTV_LIVE_MIRROR_JS = r'''

(function() {

    if (window._rtvEngineReady) return;

    window._rtvEngineReady = true;



    let rtvWs = null;

    let rtvRunning = false;

    let rtvInFlight = 0;

    const MAX_IN_FLIGHT = 60; // 2 seconds of network pipeline buffering (eliminates 5 FPS network stall)



    // Jitter Playback Buffer (Smooth 30.0 FPS locked stream with 2.0s buffer)

    let rtvPlaybackQueue = [];

    let rtvPlaybackTimer = null;

    let rtvSendTimer = null;

    let rtvPlaybackActive = false;

    const TARGET_BUFFER_FRAMES = 30; // ~1.0s buffer for responsive butter-smooth 30.0 FPS playback

    let rtvDispCount = 0;

    let rtvLastFpsTime = performance.now();

    let rtvSendTime = 0;



    const rtvCanvas = document.createElement('canvas');

    const rtvCtx = rtvCanvas.getContext('2d');



    window.rtvStartMirror = async function() {

        if (rtvRunning) return;

        const btnStart = document.getElementById('rtv_btn_start');

        const btnStop = document.getElementById('rtv_btn_stop');

        const statusBadge = document.getElementById('rtv_ws_status');

        const video = document.getElementById('rtv_cam_preview');

        const placeholderCam = document.getElementById('rtv_cam_placeholder');

        const placeholderMirror = document.getElementById('rtv_mirror_placeholder');

        const mirrorImg = document.getElementById('rtv_mirror_display');



        if (statusBadge) {

            statusBadge.innerText = '🟡 Requesting Camera...';

            statusBadge.style.color = '#fbbf24';

        }



        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {

            const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';

            if (!isLocal && window.location.protocol !== 'https:') {

                alert("🔒 Browser Security Error:\n\nCamera access requires a SECURE HTTPS connection.\n\n👉 Please open your Cloudflare HTTPS link (e.g. https://xxxx.trycloudflare.com) shown in your Kaggle notebook output!");

                if (statusBadge) {

                    statusBadge.innerText = '❌ HTTPS Link Required';

                    statusBadge.style.color = '#f87171';

                }

                return;

            } else {

                alert("Camera API is not supported or blocked by browser permissions.");

                if (statusBadge) {

                    statusBadge.innerText = '❌ Camera Blocked';

                    statusBadge.style.color = '#f87171';

                }

                return;

            }

        }



        try {

            const stream = await navigator.mediaDevices.getUserMedia({

                video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },

                audio: false

            });

            if (video) {

                video.srcObject = stream;

                await video.play();

                video.style.display = 'block';

            }

            if (placeholderCam) placeholderCam.style.display = 'none';

        } catch (err) {

            alert('Camera access error: ' + err.message + '\n\nPlease click the camera icon in your browser address bar to allow camera access.');

            if (statusBadge) {

                statusBadge.innerText = '❌ Camera Permission Denied';

                statusBadge.style.color = '#f87171';

            }

            return;

        }



        if (statusBadge) {

            statusBadge.innerText = '🟡 Connecting to GPU...';

            statusBadge.style.color = '#38bdf8';

        }



        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';

        const wsUrl = `${proto}//${window.location.host}/api/live_mirror`;



        try {

            rtvWs = new WebSocket(wsUrl);

            rtvWs.binaryType = 'blob';

        } catch (e) {

            alert('WebSocket connection failed: ' + e.message);

            if (statusBadge) {

                statusBadge.innerText = '❌ WS Failed';

                statusBadge.style.color = '#f87171';

            }

            return;

        }



        rtvWs.onopen = function() {

            rtvRunning = true;

            rtvPlaybackActive = false;

            rtvInFlight = 0;

            rtvPlaybackQueue = [];

            rtvDispCount = 0;

            rtvLastFpsTime = performance.now();



            if (statusBadge) {

                statusBadge.innerText = '🟡 Buffering Smooth 30 FPS Stream...';

                statusBadge.style.color = '#fbbf24';

            }

            if (btnStart) {

                btnStart.disabled = true;

                btnStart.style.opacity = '0.5';

                btnStart.style.cursor = 'not-allowed';

            }

            if (btnStop) {

                btnStop.disabled = false;

                btnStop.style.opacity = '1.0';

                btnStop.style.background = '#ef4444';

                btnStop.style.color = '#ffffff';

                btnStop.style.cursor = 'pointer';

            }

            if (mirrorImg) mirrorImg.style.display = 'block';

            if (placeholderMirror) placeholderMirror.style.display = 'none';

            window.rtvSendConfig();



            // Continuous 30 FPS camera capture timer (fires every 33.3ms)

            if (rtvSendTimer) clearInterval(rtvSendTimer);

            rtvSendTimer = setInterval(rtvContinuousSendTick, 33);

        };



        rtvWs.onmessage = function(event) {

            if (event.data instanceof Blob) {

                rtvInFlight = Math.max(0, rtvInFlight - 1);



                // Add incoming rendered frame to smooth playback buffer

                rtvPlaybackQueue.push(event.data);



                // Start playback once target buffer is filled (1.5-2.0s delay for smooth 30 FPS)

                if (!rtvPlaybackActive) {

                    const pct = Math.min(100, Math.round((rtvPlaybackQueue.length / TARGET_BUFFER_FRAMES) * 100));

                    if (statusBadge) {

                        statusBadge.innerText = `🟡 Buffering Ultra-Smooth 30 FPS (${pct}%)...`;

                        statusBadge.style.color = '#fbbf24';

                    }

                    if (rtvPlaybackQueue.length >= TARGET_BUFFER_FRAMES) {

                        rtvPlaybackActive = true;

                        if (statusBadge) {

                            statusBadge.innerText = '🟢 Ultra-Smooth 30 FPS Stream Active';

                            statusBadge.style.color = '#00ffaa';

                        }

                        if (rtvPlaybackTimer) clearInterval(rtvPlaybackTimer);

                        rtvPlaybackTimer = setInterval(rtvRenderNextBufferedFrame, 33.33); // 30 FPS locked playback

                    }

                }

            }

        };



        rtvWs.onerror = function(err) {

            console.error('[RTV] WebSocket error:', err);

            if (statusBadge) {

                statusBadge.innerText = '⚠️ WS Error';

                statusBadge.style.color = '#f87171';

            }

        };



        rtvWs.onclose = function() {

            if (rtvRunning) {

                window.rtvStopMirror();

                if (statusBadge) {

                    statusBadge.innerText = '⚪ Disconnected';

                    statusBadge.style.color = '#94a3b8';

                }

            }

        };

    };



    function rtvContinuousSendTick() {

        if (!rtvRunning || !rtvWs || rtvWs.readyState !== WebSocket.OPEN) return;

        if (rtvInFlight >= MAX_IN_FLIGHT) return; // Only throttle if network buffer exceeds 2 seconds



        const video = document.getElementById('rtv_cam_preview');

        if (!video || video.readyState < 2) return;



        const vw = video.videoWidth || 640;

        const vh = video.videoHeight || 480;

        const targetW = 480;

        let targetH = Math.round(targetW * (vh / vw));

        if (targetH % 2 !== 0) targetH -= 1;



        if (rtvCanvas.width !== targetW || rtvCanvas.height !== targetH) {

            rtvCanvas.width = targetW;

            rtvCanvas.height = targetH;

        }



        // Draw standard natural camera frame directly (preserves true body symmetry & sleeve orientation)

        rtvCtx.drawImage(video, 0, 0, targetW, targetH);



        rtvInFlight++;

        rtvCanvas.toBlob(function(blob) {

            if (blob && rtvWs && rtvWs.readyState === WebSocket.OPEN) {

                rtvWs.send(blob);

            } else {

                rtvInFlight = Math.max(0, rtvInFlight - 1);

            }

        }, 'image/jpeg', 0.70);

    }



    function rtvRenderNextBufferedFrame() {

        if (!rtvRunning) return;

        const mirrorImg = document.getElementById('rtv_mirror_display');

        const fpsBadge = document.getElementById('rtv_fps_badge');

        const rttBadge = document.getElementById('rtv_rtt_badge');



        if (rtvPlaybackQueue.length > 0) {

            // Pop frame from queue

            const blob = rtvPlaybackQueue.shift();



            // Drift catchup: keep buffer stable around 1.0-1.5s (30-45 frames)

            if (rtvPlaybackQueue.length > 50) {

                rtvPlaybackQueue.shift();

            }



            const url = URL.createObjectURL(blob);

            if (mirrorImg) {

                const prev = mirrorImg.src;

                mirrorImg.src = url;

                if (prev && prev.startsWith('blob:')) {

                    URL.revokeObjectURL(prev);

                }

            }



            rtvDispCount++;

            const now = performance.now();

            if (now - rtvLastFpsTime >= 1000) {

                const fps = (rtvDispCount * 1000 / (now - rtvLastFpsTime)).toFixed(1);

                if (fpsBadge) {

                    fpsBadge.innerText = `⚡ Mirror: ${fps} FPS (Ultra-Smooth)`;

                }

                rtvDispCount = 0;

                rtvLastFpsTime = now;

            }



            if (rttBadge) {

                const bufSec = (rtvPlaybackQueue.length * 33.33 / 1000).toFixed(1);

                rttBadge.innerText = `⏱️ Buffer: ${bufSec}s (${rtvPlaybackQueue.length} f)`;

            }

        }

    }



    window.rtvStopMirror = function() {

        rtvRunning = false;

        rtvPlaybackActive = false;

        rtvInFlight = 0;

        rtvPlaybackQueue = [];

        if (rtvSendTimer) {

            clearInterval(rtvSendTimer);

            rtvSendTimer = null;

        }

        if (rtvPlaybackTimer) {

            clearInterval(rtvPlaybackTimer);

            rtvPlaybackTimer = null;

        }

        if (rtvWs) {

            try { rtvWs.close(); } catch(e){}

            rtvWs = null;

        }

        const video = document.getElementById('rtv_cam_preview');

        if (video && video.srcObject) {

            try {

                video.srcObject.getTracks().forEach(t => t.stop());

            } catch(e){}

            video.srcObject = null;

        }

        const btnStart = document.getElementById('rtv_btn_start');

        const btnStop = document.getElementById('rtv_btn_stop');

        const statusBadge = document.getElementById('rtv_ws_status');

        const fpsBadge = document.getElementById('rtv_fps_badge');

        const rttBadge = document.getElementById('rtv_rtt_badge');



        if (btnStart) {

            btnStart.disabled = false;

            btnStart.style.opacity = '1.0';

            btnStart.style.cursor = 'pointer';

        }

        if (btnStop) {

            btnStop.disabled = true;

            btnStop.style.opacity = '0.5';

            btnStop.style.background = '#374151';

            btnStop.style.color = '#9ca3af';

            btnStop.style.cursor = 'not-allowed';

        }

        if (statusBadge) {

            statusBadge.innerText = '⚪ Standby';

            statusBadge.style.color = '#94a3b8';

        }

        if (fpsBadge) fpsBadge.innerText = '⚡ Mirror: 0.0 FPS';

        if (rttBadge) rttBadge.innerText = '📶 Latency: -- ms';

    };



    window.rtvSendConfig = function() {

        if (!rtvWs || rtvWs.readyState !== WebSocket.OPEN) return;

        const garmentEl = document.getElementById('rtv_ws_garment');

        const hudEl = document.getElementById('rtv_ws_hud');

        const garment = garmentEl ? garmentEl.value : 'Winter Puffer Coat (Full-Body)';

        const hud = hudEl ? hudEl.checked : true;

        rtvWs.send(JSON.stringify({ action: 'set_config', garment: garment, hud: hud }));

    };



    window.rtvCaptureSnapshot = function() {

        const mirrorImg = document.getElementById('rtv_mirror_display');

        if (!mirrorImg || !mirrorImg.src) {

            alert('Please start the mirror first before capturing a snapshot.');

            return;

        }

        const a = document.createElement('a');

        a.href = mirrorImg.src;

        a.download = 'rtv_tryon_snapshot.jpg';

        document.body.appendChild(a);

        a.click();

        document.body.removeChild(a);

    };



    // Event delegation for clicks

    document.addEventListener('click', function(e) {

        if (!e.target) return;

        const btnStart = e.target.closest('#rtv_btn_start');

        if (btnStart) {

            e.preventDefault();

            if (typeof window.rtvStartMirror === 'function') {

                window.rtvStartMirror();

            }

            return;

        }

        const btnStop = e.target.closest('#rtv_btn_stop');

        if (btnStop) {

            e.preventDefault();

            if (typeof window.rtvStopMirror === 'function') {

                window.rtvStopMirror();

            }

            return;

        }

        const btnSnap = e.target.closest('#rtv_btn_snap');

        if (btnSnap) {

            e.preventDefault();

            if (typeof window.rtvCaptureSnapshot === 'function') {

                window.rtvCaptureSnapshot();

            }

            return;

        }

    }, true);



    console.log('[RTV] Ultra-Smooth 30 FPS Mirror Engine Initialized & Attached to Window!');

})();

'''



def build_gradio_ui(engine: ColabTryOnEngine):

    garment_choices = list(GARMENT_CATALOG.keys())

    

    # Custom CSS for Sleek Dark Cyber Aesthetic

    custom_css = """

    body { background-color: #0a0b10 !important; color: #e2e8f0 !important; font-family: 'Segoe UI', system-ui, sans-serif; }

    .gradio-container { max-width: 1400px !important; margin: auto; }

    .header-box {

        background: linear-gradient(135deg, #10121d 0%, #171b2e 100%);

        border: 1px solid #232946;

        border-radius: 12px;

        padding: 24px;

        margin-bottom: 20px;

        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);

    }

    .header-title {

        font-size: 2.2rem;

        font-weight: 800;

        background: linear-gradient(90deg, #00ffaa, #00d2ff);

        -webkit-background-clip: text;

        -webkit-text-fill-color: transparent;

        margin-bottom: 6px;

    }

    .status-badge {

        display: inline-block;

        padding: 4px 12px;

        border-radius: 20px;

        font-size: 0.85rem;

        font-weight: 600;

        background: rgba(0, 255, 170, 0.12);

        color: #00ffaa;

        border: 1px solid rgba(0, 255, 170, 0.3);

        margin-right: 8px;

    }

    .control-card {

        background: #11131c;

        border: 1px solid #1f2338;

        border-radius: 10px;

        padding: 16px;

    }

    .gallery-preview {

        border-radius: 8px;

        border: 1px solid #252a42;

    }

    """



    gpu_info = "GPU: " + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Mode (CUDA not detected)")

    vram_info = f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB" if torch.cuda.is_available() else "RAM: System"



    with gr.Blocks(title="RTV Real-Time Virtual Try-On", css=custom_css, theme=gr.themes.Monochrome()) as demo:

        # Header Section

        with gr.Row():

            with gr.Column(elem_classes=["header-box"]):

                gr.Markdown(f"""

                <div class="header-title">RTV : Real-Time Virtual Try-On Studio</div>

                <div style="margin-top: 6px;">

                    <span class="status-badge">⚡ {gpu_info}</span>

                    <span class="status-badge">💾 {vram_info}</span>

                    <span class="status-badge">🚀 PyTorch {torch.__version__}</span>

                    <span class="status-badge">🎯 25-30 FPS Multi-Rate Pipeline</span>

                </div>

                <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 10px;">

                    High-performance neural cloth synthesis using ROMP/BEV 3D human pose, DensePose UV mapping, 

                    SMPL offscreen mesh rendering, and real-time Lucas-Kanade optical flow tracking.

                </p>

                """)



        # Main Tabs

        with gr.Tabs() as tabs:

            # ----------------------------------------------------------------

            # TAB 1: Live Webcam Try-On

            # ----------------------------------------------------------------

            with gr.TabItem("🎥 Live Webcam Try-On"):

                garment_options_html = "".join([f'<option value="{k}">{k}</option>' for k in garment_choices])



                realtime_mirror_html = f"""

                <div class="mirror-container" style="background:#0b0f19; border:1px solid #1e293b; border-radius:12px; padding:18px; margin-bottom:18px; box-shadow:0 8px 30px rgba(0,0,0,0.5); font-family:system-ui,-apple-system,sans-serif;">

                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; border-bottom:1px solid #1e293b; padding-bottom:10px; flex-wrap:wrap; gap:10px;">

                        <div>

                            <span style="font-size:1.25rem; font-weight:700; color:#00ffaa; letter-spacing:0.5px;">⚡ High-Speed Real-Time Mirror</span>

                            <span style="color:#94a3b8; font-size:0.85rem; margin-left:10px;">30 FPS Pipelined WebSocket Engine (HD 720p)</span>

                        </div>

                        <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">

                            <span id="rtv_ws_status" class="status-badge" style="background:rgba(148,163,184,0.15); color:#94a3b8; border:1px solid #334155;">⚪ Standby</span>

                            <span id="rtv_fps_badge" class="status-badge" style="background:rgba(0,255,170,0.12); color:#00ffaa; border:1px solid rgba(0,255,170,0.3);">⚡ Mirror: 0.0 FPS</span>

                            <span id="rtv_rtt_badge" class="status-badge" style="background:rgba(56,189,248,0.12); color:#38bdf8; border:1px solid rgba(56,189,248,0.3);">📶 Latency: -- ms</span>

                        </div>

                    </div>

                    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin-bottom:16px;">

                        <div style="background:#030712; border-radius:10px; padding:12px; border:1px solid #1f293d; text-align:center;">

                            <div style="font-weight:600; color:#38bdf8; margin-bottom:8px; font-size:0.9rem; display:flex; justify-content:space-between;">

                                <span>📹 Your Camera (Mirrored)</span>

                                <span style="color:#64748b; font-size:0.8rem;">640x480 HD</span>

                            </div>

                            <div style="position:relative; width:100%; aspect-ratio:4/3; background:#000; border-radius:8px; overflow:hidden; display:flex; align-items:center; justify-content:center;">

                                <video id="rtv_cam_preview" autoplay playsinline muted style="width:100%; height:100%; object-fit:cover; transform:scaleX(-1); display:none;"></video>

                                <div id="rtv_cam_placeholder" style="color:#94a3b8; font-size:0.95rem; padding:20px;">

                                    📷 Click <b>"▶ Start Live Mirror"</b> below to activate camera

                                </div>

                            </div>

                        </div>

                        <div style="background:#030712; border-radius:10px; padding:12px; border:1px solid rgba(0,255,170,0.3); text-align:center; box-shadow:0 0 20px rgba(0,255,170,0.05);">

                            <div style="font-weight:600; color:#00ffaa; margin-bottom:8px; font-size:0.9rem; display:flex; justify-content:space-between;">

                                <span>✨ Real-Time Virtual Try-On Mirror</span>

                                <span style="color:#00ffaa; font-size:0.8rem;">Kaggle GPU Live</span>

                            </div>

                            <div style="position:relative; width:100%; aspect-ratio:4/3; background:#000; border-radius:8px; overflow:hidden; display:flex; align-items:center; justify-content:center;">

                                <img id="rtv_mirror_display" style="width:100%; height:100%; object-fit:cover; display:none;" alt="Live Mirror"/>

                                <div id="rtv_mirror_placeholder" style="color:#94a3b8; font-size:0.95rem; padding:20px;">

                                    ✨ Real-time virtual try-on feed will stream here at 30 FPS locked

                                </div>

                            </div>

                        </div>

                    </div>

                    <div style="display:flex; flex-wrap:wrap; gap:12px; align-items:center; justify-content:space-between; background:#111827; padding:14px; border-radius:10px; border:1px solid #1f293d;">

                        <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">

                            <button type="button" id="rtv_btn_start" onclick="if(window.rtvStartMirror){{window.rtvStartMirror();}}else{{alert('Initializing camera engine, please click again in 1 second.');}}" style="background:linear-gradient(135deg, #059669 0%, #10b981 100%); color:#fff; border:none; border-radius:6px; padding:10px 22px; font-weight:700; font-size:0.95rem; cursor:pointer; box-shadow:0 4px 12px rgba(16,185,129,0.3); transition:all 0.2s;">

                                ▶ Start Live Mirror

                            </button>

                            <button type="button" id="rtv_btn_stop" onclick="if(window.rtvStopMirror)window.rtvStopMirror();" disabled style="background:#374151; color:#9ca3af; border:none; border-radius:6px; padding:10px 18px; font-weight:600; font-size:0.95rem; cursor:not-allowed; transition:all 0.2s;">

                                ⏹ Stop

                            </button>

                            <button type="button" id="rtv_btn_snap" onclick="if(window.rtvCaptureSnapshot)window.rtvCaptureSnapshot();" style="background:#1e293b; color:#e2e8f0; border:1px solid #334155; border-radius:6px; padding:10px 14px; font-weight:600; font-size:0.9rem; cursor:pointer;">

                                📸 Snapshot

                            </button>

                        </div>

                        <div style="display:flex; gap:16px; align-items:center; flex-wrap:wrap;">

                            <div style="display:flex; align-items:center; gap:8px;">

                                <label style="color:#cbd5e1; font-size:0.9rem; font-weight:600;">Garment:</label>

                                <select id="rtv_ws_garment" onchange="if(window.rtvSendConfig)window.rtvSendConfig();" style="background:#0f172a; color:#00ffaa; border:1px solid #334155; border-radius:6px; padding:8px 12px; font-weight:600; font-size:0.9rem; cursor:pointer; outline:none;">

                                    {garment_options_html}

                                </select>

                            </div>

                            <label style="color:#cbd5e1; font-size:0.88rem; cursor:pointer; display:flex; align-items:center; gap:6px;">

                                <input type="checkbox" id="rtv_ws_hud" checked onchange="if(window.rtvSendConfig)window.rtvSendConfig();" style="cursor:pointer; accent-color:#00ffaa;">

                                Show HUD

                            </label>

                        </div>

                    </div>



                </div>

                """

                gr.HTML(realtime_mirror_html + f"<script>\n{RTV_LIVE_MIRROR_JS}\n</script>")



                with gr.Accordion("📸 Instant Snapshot Try-On (High-Res Photo Mode)", open=False):

                    with gr.Row():

                        with gr.Column(scale=4):

                            snap_input = gr.Image(sources=["webcam", "upload"], type="numpy", label="Capture Photo or Upload", height=340)

                            snap_garment = gr.Dropdown(choices=garment_choices, value="Winter Puffer Coat (Full-Body)", label="Garment")

                            snap_btn = gr.Button("✨ Render High-Res Snapshot Try-On", variant="primary")

                        with gr.Column(scale=4):

                            snap_output = gr.Image(type="numpy", label="High-Res Result", height=340)

                    snap_btn.click(

                        fn=lambda img, g: (engine.process_single_image(img, g, True, False)[0] if img is not None else None),

                        inputs=[snap_input, snap_garment],

                        outputs=[snap_output]

                    )



            # ----------------------------------------------------------------

            # TAB 2: Video File Try-On

            # ----------------------------------------------------------------

            with gr.TabItem("🎞️ Video File Try-On"):

                gr.Markdown("""

                ### Video Try-On (Sample Video Pre-Loaded)

                Click **"⚡ Render Virtual Try-On Video"** below to see the model fit the garment on the sample footage!

                """)

                with gr.Row():

                    with gr.Column(scale=4):

                        sample_video_path = "./assets/sample_video.mp4" if os.path.exists("./assets/sample_video.mp4") else None

                        video_upload = gr.Video(

                            value=sample_video_path,

                            label="Source Video (Sample Video Pre-Loaded)",

                            height=360

                        )

                        sample_btn = gr.Button("📁 Reload Bundled Sample Video", variant="secondary")

                        

                        video_garment_select = gr.Dropdown(

                            choices=garment_choices,

                            value="Winter Puffer Coat (Full-Body)",

                            label="Garment Choice"

                        )

                        max_frames_slider = gr.Slider(

                            minimum=30, maximum=1200, value=300, step=30,

                            label="Max Frames to Process (up to 1200 frames)",

                            info="Select frame limit for Kaggle/Colab processing (up to 1200 frames = ~40 seconds)"

                        )

                        video_hud_toggle = gr.Checkbox(value=True, label="Render HUD Banner on Output Video")

                        process_video_btn = gr.Button("⚡ Render Virtual Try-On Video", variant="primary")



                    with gr.Column(scale=4):

                        video_output = gr.Video(label="Rendered Try-On Video", height=360)

                        video_status = gr.Markdown("Ready to process video.")



                def load_sample_video():

                    sample_path = "./assets/sample_video.mp4"

                    if os.path.exists(sample_path):

                        return sample_path, "Sample video loaded successfully."

                    return None, "Sample video not found in assets/."



                sample_btn.click(

                    fn=load_sample_video,

                    inputs=[],

                    outputs=[video_upload, video_status]

                )



                process_video_btn.click(

                    fn=engine.process_video_file,

                    inputs=[video_upload, video_garment_select, max_frames_slider, video_hud_toggle],

                    outputs=[video_output, video_status]

                )



            # ----------------------------------------------------------------

            # TAB 3: Single Photo Studio & Diagnostics

            # ----------------------------------------------------------------

            with gr.TabItem("📸 Single Photo Studio"):

                gr.Markdown("""

                ### Single Photo Try-On (Sample Photo Pre-Loaded)

                Click **"✨ Apply Virtual Try-On"** below to test the neural network on a single image!

                """)

                with gr.Row():

                    with gr.Column(scale=3):

                        sample_frame_path = "./assets/sample_frame.jpg" if os.path.exists("./assets/sample_frame.jpg") else None

                        image_input = gr.Image(

                            value=sample_frame_path,

                            type="numpy",

                            label="Portrait / Full-Body Photo (Sample Frame Pre-Loaded)"

                        )

                        photo_garment_select = gr.Dropdown(

                            choices=garment_choices,

                            value="Winter Puffer Coat (Full-Body)",

                            label="Select Garment"

                        )

                        photo_hud_check = gr.Checkbox(value=True, label="Include Telemetry HUD Overlay")

                        photo_debug_check = gr.Checkbox(value=True, label="Extract Alpha Mask & UV Maps")

                        tryon_btn = gr.Button("✨ Apply Virtual Try-On", variant="primary")



                    with gr.Column(scale=4):

                        photo_output = gr.Image(type="numpy", label="Virtual Try-On Result")

                        photo_status = gr.Markdown("Upload an image and click Apply.")



                with gr.Accordion("🔍 Neural Diagnostic Inspection (Alpha Mask & Garment ROI)", open=False):

                    with gr.Row():

                        with gr.Column():

                            alpha_output = gr.Image(type="numpy", label="Garment Alpha Mask (Opacity Heatmap)")

                        with gr.Column():

                            roi_output = gr.Image(type="numpy", label="Synthesized Garment ROI (Pix2PixHD)")



                tryon_btn.click(

                    fn=engine.process_single_image,

                    inputs=[image_input, photo_garment_select, photo_hud_check, photo_debug_check],

                    outputs=[photo_output, alpha_output, roi_output, photo_status]

                )



            # ----------------------------------------------------------------

            # TAB 4: System Diagnostics & Performance

            # ----------------------------------------------------------------

            with gr.TabItem("📊 Hardware & Architecture"):

                with gr.Row():

                    with gr.Column():

                        gr.Markdown("""

                        ### RTV Architecture Highlights

                        - **Decoupled Multi-Rate Architecture**: Display loop runs at 25-30 FPS locked to camera/webcam arrival.

                        - **Lucas-Kanade Motion Tracking**: Garment patch propagates smoothly on every live frame without stalling.

                        - **OneEuroFilter Jitter Suppression**: Eliminates torso & neck high-frequency vibrations.

                        - **Headless GPU Offscreen Rendering**: Uses PyOpenGL with EGL/OSMesa on Linux cloud instances.

                        - **Solid Alpha Compositing**: Fast ROI-bounded alpha blending preserves hands and background fidelity.

                        """)

                    with gr.Column():

                        device_details = f"""

                        ### Active Environment Diagnostics

                        - **Operating System:** {platform.system()} {platform.release()} ({platform.machine()})

                        - **Python Executable:** `{sys.executable}`

                        - **PyTorch Version:** `{torch.__version__}`

                        - **CUDA Available:** `{torch.cuda.is_available()}`

                        - **Device Name:** `{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}`

                        - **CUDA Compute:** `{torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A'}`

                        - **PyOpenGL Platform:** `{os.environ.get('PYOPENGL_PLATFORM', 'default')}`

                        """

                        gr.Markdown(device_details)



            # ----------------------------------------------------------------

            # TAB 5: 👗 Custom Garment Studio (Upload, Train & Test)

            # ----------------------------------------------------------------

            with gr.TabItem("👗 Custom Garment Studio (Upload, Train & Test)"):

                gr.Markdown("""

                <div class="header-box" style="padding: 16px; margin-bottom: 16px;">

                    <h3 style="margin: 0; color: #00ffaa;">👗 Custom Garment Studio</h3>

                    <p style="color: #94a3b8; margin: 4px 0 0 0;">

                        Upload your custom dress video or photos, train Pix2PixHD neural checkpoints with mixed precision (AMP FP16) on your GPU, 

                        import existing <code>.pth</code> models, and test instantly with before & after try-on comparison!

                    </p>

                </div>

                """)

                with gr.Tabs():

                    # Sub-Tab 1: 4-View Photos

                    with gr.TabItem("📸 Train from 4-View Photos (Front, Back, Right, Left)"):

                        gr.Markdown("""

                        Upload still photos of your dress from **Front**, **Back**, **Right**, and **Left** views.

                        The system will compute 3D SMPL poses, DensePose UV mappings, and synthesize a balanced 360° dataset.

                        """)

                        with gr.Row():

                            with gr.Column(scale=4):

                                mv_train_name = gr.Textbox(value="my_custom_dress", label="Garment Model Name", info="Unique name in rtv_ckpts/")

                                with gr.Row():

                                    img_front = gr.Image(type="numpy", label="👗 FRONT View (Required)")

                                    img_back = gr.Image(type="numpy", label="🔄 BACK View (Recommended)")

                                with gr.Row():

                                    img_right = gr.Image(type="numpy", label="➡️ RIGHT SIDE View (Optional)")

                                    img_left = gr.Image(type="numpy", label="⬅️ LEFT SIDE View (Optional)")

                                mv_speed_mode = gr.Dropdown(

                                    choices=[

                                        "⚡ Turbo Mode (1–2 Mins) [Default - Recommended]",

                                        "🎯 Balanced Mode (4–5 Mins)",

                                        "💎 Deep Studio Mode (30–40 Mins)"

                                    ],

                                    value="⚡ Turbo Mode (1–2 Mins) [Default - Recommended]",

                                    label="⚡ Training Speed Mode"

                                )

                                with gr.Row():

                                    mv_epochs = gr.Slider(minimum=1, maximum=50, value=3, step=1, label="Epochs (Fine-Tuning)")

                                    mv_batch = gr.Slider(minimum=1, maximum=4, value=2, step=1, label="Batch Size (AMP FP16)")

                                    mv_res = gr.Dropdown(choices=["512", "576"], value="512", label="Resolution")

                                btn_train_mv = gr.Button("🚀 Train from 4-View Photos (⚡ Turbo Active)", variant="primary")



                            with gr.Column(scale=4):

                                train_log_mv = gr.Textbox(

                                    label="Live Multi-View Training Console",

                                    lines=16,

                                    interactive=False,

                                    placeholder="4-View training logs will stream here..."

                                )



                    # Sub-Tab 2: Video / Zip

                    with gr.TabItem("🎥 Train from Video (.mp4) / Zip of Frames"):

                        with gr.Row():

                            with gr.Column(scale=4):

                                custom_train_name = gr.Textbox(value="custom_designer_dress", label="Garment Model Name", info="Identifier for model folder in rtv_ckpts/")

                                custom_video_input = gr.File(label="Upload Dress Video (.mp4, .mov) or Zip of Photos", file_types=[".mp4", ".mov", ".avi", ".zip"])

                                custom_auto_mask = gr.Checkbox(value=True, label="Auto-Segment Dress Masks using 3D SMPL Pose & DensePose (Recommended)")

                                custom_speed_mode = gr.Dropdown(

                                    choices=[

                                        "⚡ Turbo Mode (1–2 Mins) [Default - Recommended]",

                                        "🎯 Balanced Mode (4–5 Mins)",

                                        "💎 Deep Studio Mode (30–40 Mins)"

                                    ],

                                    value="⚡ Turbo Mode (1–2 Mins) [Default - Recommended]",

                                    label="⚡ Training Speed Mode"

                                )

                                with gr.Row():

                                    custom_epochs = gr.Slider(minimum=1, maximum=50, value=3, step=1, label="Epochs (Fine-Tuning)")

                                    custom_batch = gr.Slider(minimum=1, maximum=4, value=2, step=1, label="Batch Size (AMP FP16)")

                                    custom_res = gr.Dropdown(choices=["512", "576"], value="512", label="Resolution")

                                btn_start_train = gr.Button("🚀 Start Dataset Generation & Training (⚡ Turbo Active)", variant="primary")



                            with gr.Column(scale=4):

                                train_log_output = gr.Textbox(

                                    label="Live Training Console & Status",

                                    lines=14,

                                    interactive=False,

                                    placeholder="Training logs will appear here..."

                                )



                    # Sub-Tab 3: Import

                    with gr.TabItem("📥 Import Checkpoint (.pth)"):

                        with gr.Row():

                            with gr.Column(scale=4):

                                import_name = gr.Textbox(value="my_imported_dress", label="Garment Display Name")

                                import_pth = gr.File(label="Upload Model Checkpoint (.pth)", file_types=[".pth"])

                                import_thumb = gr.Image(label="Upload Preview Thumbnail (Optional)", type="filepath")

                                btn_import_deploy = gr.Button("📥 Import & Deploy Checkpoint", variant="primary")

                            with gr.Column(scale=4):

                                import_status = gr.Markdown("Upload an external `.pth` checkpoint and thumbnail image to add it immediately into the wardrobe.")



                    # Sub-Tab 4: Test Studio

                    with gr.TabItem("🧪 Test Studio (Static Photo Try-On)"):

                        with gr.Row():

                            with gr.Column(scale=4):

                                sample_p = "./assets/sample_frame.jpg" if os.path.exists("./assets/sample_frame.jpg") else None

                                test_photo_in = gr.Image(value=sample_p, type="numpy", label="Person Photo")

                                test_garment_select = gr.Dropdown(

                                    choices=list(GARMENT_CATALOG.keys()),

                                    value="Winter Puffer Coat (Full-Body)",

                                    label="Select Target Garment"

                                )

                                btn_refresh_catalog = gr.Button("🔄 Refresh Available Garments", variant="secondary")

                                btn_run_test_garment = gr.Button("✨ Test Virtual Try-On", variant="primary")

                            with gr.Column(scale=4):

                                test_photo_out = gr.Image(type="numpy", label="Virtual Try-On Result")

                                test_status = gr.Markdown("Ready to test.")



                def handle_gradio_train_multiview(front_im, back_im, right_im, left_im, garment_name, speed_mode, epochs, batch, res):

                    if front_im is None:

                        yield "❌ Error: Please upload at least the FRONT view photo of the dress."

                        return

                    if not garment_name or not garment_name.strip():

                        yield "❌ Error: Please enter a garment model name."

                        return



                    mode = "turbo"

                    if "Balanced" in str(speed_mode):

                        mode = "balanced"

                    elif "Deep" in str(speed_mode):

                        mode = "deep"



                    import queue

                    import threading



                    views_dict = {

                        "front": front_im,

                        "back": back_im,

                        "right": right_im,

                        "left": left_im

                    }

                    trainer = CustomGarmentTrainer()

                    loaded_names = [k.upper() for k, v in views_dict.items() if v is not None]

                    logs = [

                        f"🚀 Initiating 4-View training pipeline for '{garment_name}' ({speed_mode})...",

                        f"Detected {len(loaded_names)} views: {', '.join(loaded_names)}"

                    ]

                    yield "\n".join(logs)



                    msg_q = queue.Queue()



                    def _p(*args):

                        if len(args) >= 5:

                            ep, tot_ep, step, loss, msg = args[:5]

                            msg_q.put(f"⚡ [Epoch {ep}/{tot_ep}] {msg}")

                        elif len(args) == 3:

                            cur, tot, msg = args

                            msg_q.put(f"[{cur}/{tot}] {msg}")

                        elif len(args) > 0:

                            msg_q.put(str(args[-1]))



                    def _e(*args):

                        if len(args) >= 5:

                            ep, tot_ep, step, loss, msg = args[:5]

                            msg_q.put(f"✨ [Epoch {ep}/{tot_ep} Complete] {msg}")

                        elif len(args) > 0:

                            msg_q.put(str(args[-1]))



                    def _worker():

                        try:

                            msg_q.put(f"Phase 1: Synthesizing 360° training dataset with 3D SMPL pose & DensePose UV maps ({mode.upper()} mode)...")

                            ds_path, count = trainer.generate_dataset_from_multiview(

                                views_dict, dataset_name=garment_name, samples_per_view=None, mode=mode, progress_cb=_p

                            )

                            msg_q.put(f"Dataset ready at {ds_path} ({count} pairs). Phase 2: Starting neural training ({mode.upper()} mode, {epochs} epochs)...")

                            name, ckpt = trainer.train_garment(

                                dataset_name=garment_name, garment_name=garment_name,

                                num_epochs=int(epochs), batch_size=int(batch),

                                img_size=int(res), mode=mode, progress_cb=_p, epoch_cb=_e

                            )

                            scan_and_update_catalog()

                            msg_q.put(f"✅ Success! Multi-view model '{name}' deployed into wardrobe at: {ckpt}")

                        except Exception as ex:

                            msg_q.put(f"❌ Training error: {str(ex)}")

                        finally:

                            msg_q.put("__DONE__")



                    th = threading.Thread(target=_worker, daemon=True)

                    th.start()



                    while True:

                        try:

                            item = msg_q.get(timeout=0.25)

                            if item == "__DONE__":

                                break

                            logs.append(item)

                            yield "\n".join(logs)

                        except queue.Empty:

                            if not th.is_alive():

                                break

                    yield "\n".join(logs)



                def handle_gradio_train(file_obj, garment_name, auto_mask, speed_mode, epochs, batch, res):

                    if not file_obj:

                        yield "❌ Error: Please upload a training video (.mp4) or zip archive of images."

                        return

                    if not garment_name or not garment_name.strip():

                        yield "❌ Error: Please enter a garment model name."

                        return



                    mode = "turbo"

                    if "Balanced" in str(speed_mode):

                        mode = "balanced"

                    elif "Deep" in str(speed_mode):

                        mode = "deep"

                    

                    import queue

                    import threading



                    file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)

                    if file_path.endswith('.zip'):

                        import zipfile

                        clean_name = garment_name.strip().replace(" ", "_").lower()

                        extract_dir = os.path.abspath(f"./scratch/unpacked_{clean_name}")

                        os.makedirs(extract_dir, exist_ok=True)

                        with zipfile.ZipFile(file_path, 'r') as z:

                            z.extractall(extract_dir)

                        src_input = extract_dir

                    else:

                        src_input = file_path



                    trainer = CustomGarmentTrainer()

                    logs = [f"🚀 Initiating training pipeline for '{garment_name}' ({speed_mode})..."]

                    yield "\n".join(logs)



                    msg_q = queue.Queue()



                    def _p(*args):

                        if len(args) >= 5:

                            ep, tot_ep, step, loss, msg = args[:5]

                            msg_q.put(f"⚡ [Epoch {ep}/{tot_ep}] {msg}")

                        elif len(args) == 3:

                            cur, tot, msg = args

                            msg_q.put(f"[{cur}/{tot}] {msg}")

                        elif len(args) > 0:

                            msg_q.put(str(args[-1]))



                    def _e(*args):

                        if len(args) >= 5:

                            ep, tot_ep, step, loss, msg = args[:5]

                            msg_q.put(f"✨ [Epoch {ep}/{tot_ep} Complete] {msg}")

                        elif len(args) > 0:

                            msg_q.put(str(args[-1]))



                    def _worker():

                        try:

                            msg_q.put("Phase 1: Generating 3D SMPL pose & DensePose UV dataset...")

                            ds_path = trainer.generate_dataset_from_source(

                                src_input, None, garment_name, auto_mask=auto_mask, progress_cb=_p

                            )

                            msg_q.put(f"Dataset ready at {ds_path}. Phase 2: Starting neural training ({mode.upper()} mode, {epochs} epochs)...")

                            name, ckpt = trainer.train_garment(

                                dataset_name=garment_name, garment_name=garment_name,

                                num_epochs=int(epochs), batch_size=int(batch),

                                img_size=int(res), mode=mode, progress_cb=_p, epoch_cb=_e

                            )

                            scan_and_update_catalog()

                            msg_q.put(f"✅ Success! Trained model '{name}' deployed into wardrobe at: {ckpt}")

                        except Exception as ex:

                            msg_q.put(f"❌ Training error: {str(ex)}")

                        finally:

                            msg_q.put("__DONE__")



                    th = threading.Thread(target=_worker, daemon=True)

                    th.start()



                    while True:

                        try:

                            item = msg_q.get(timeout=0.25)

                            if item == "__DONE__":

                                break

                            logs.append(item)

                            yield "\n".join(logs)

                        except queue.Empty:

                            if not th.is_alive():

                                break

                    yield "\n".join(logs)





                def handle_gradio_import(pth_obj, thumb_path, garment_name):

                    if not pth_obj:

                        return "❌ Error: Please upload a .pth checkpoint file."

                    if not garment_name or not garment_name.strip():

                        return "❌ Error: Please enter a garment display name."

                    trainer = CustomGarmentTrainer()

                    src_pth = pth_obj.name if hasattr(pth_obj, 'name') else str(pth_obj)

                    try:

                        clean_name, dest = trainer.import_checkpoint(src_pth, garment_name, preview_path=thumb_path)

                        scan_and_update_catalog()

                        return f"✅ Garment '{clean_name}' successfully deployed into wardrobe at `{dest}`!"

                    except Exception as ex:

                        return f"❌ Import error: {str(ex)}"



                def handle_refresh_dropdowns():

                    scan_and_update_catalog()

                    choices = list(GARMENT_CATALOG.keys())

                    return (

                        gr.update(choices=choices),

                        gr.update(choices=choices),

                        gr.update(choices=choices),

                        gr.update(choices=choices)

                    )



                def handle_test_tryon(person_img, garment_label):

                    if person_img is None:

                        return None, "❌ Please select or upload a person image."

                    if not garment_label or garment_label not in GARMENT_CATALOG:

                        return None, f"❌ Garment '{garment_label}' not found."

                    res, _, _, msg = engine.process_single_image(

                        person_img, garment_label, show_hud=False, return_debug=False

                    )

                    return res, f"✨ Try-on completed for: {garment_label}"



                def _update_epochs_preset(mode_str):

                    if "Turbo" in str(mode_str):

                        return 3

                    elif "Balanced" in str(mode_str):

                        return 6

                    else:

                        return 15



                mv_speed_mode.change(fn=_update_epochs_preset, inputs=[mv_speed_mode], outputs=[mv_epochs])

                custom_speed_mode.change(fn=_update_epochs_preset, inputs=[custom_speed_mode], outputs=[custom_epochs])



                btn_train_mv.click(

                    fn=handle_gradio_train_multiview,

                    inputs=[img_front, img_back, img_right, img_left, mv_train_name, mv_speed_mode, mv_epochs, mv_batch, mv_res],

                    outputs=[train_log_mv]

                )



                btn_start_train.click(

                    fn=handle_gradio_train,

                    inputs=[custom_video_input, custom_train_name, custom_auto_mask, custom_speed_mode, custom_epochs, custom_batch, custom_res],

                    outputs=[train_log_output]

                )



                btn_import_deploy.click(

                    fn=handle_gradio_import,

                    inputs=[import_pth, import_thumb, import_name],

                    outputs=[import_status]

                )



                btn_refresh_catalog.click(

                    fn=handle_refresh_dropdowns,

                    inputs=[],

                    outputs=[snap_garment, video_garment_select, photo_garment_select, test_garment_select]

                )



                btn_run_test_garment.click(

                    fn=handle_test_tryon,

                    inputs=[test_photo_in, test_garment_select],

                    outputs=[test_photo_out, test_status]

                )



    return demo





# ============================================================================

# Main Entry Point

# ============================================================================



def launch_cloud_tunnel(port=7860):
    return

def main():

    import json

    from fastapi import FastAPI, WebSocket, WebSocketDisconnect

    import uvicorn



    default_host = "127.0.0.1" if platform.system() == "Windows" else "0.0.0.0"

    parser = argparse.ArgumentParser(description="RTV Real-Time Virtual Try-On - Colab & Web App")

    parser.add_argument("--share", action="store_true", default=True, help="Create a public shareable Gradio link (recommended for Colab)")

    parser.add_argument("--port", type=int, default=7860, help="Local server port")

    parser.add_argument("--server_name", type=str, default=default_host, help="Server host IP")

    parser.add_argument("--no_share", action="store_true", help="Disable public link sharing")

    args = parser.parse_args()

    # Reset sys.argv so internal libraries/argparse instances don't see Gradio launcher flags

    sys.argv = [sys.argv[0]]



    # Launch Cloudflare tunnel for reliable port 443 ingress on Kaggle

    launch_cloud_tunnel(args.port)



    print("=" * 70)

    print("  RTV Real-Time Virtual Try-On : Interactive Studio")

    print("=" * 70)

    print("  HOW TO ACCESS IN YOUR BROWSER:")

    print(f"  👉 Local URL:   http://localhost:{args.port}  or  http://127.0.0.1:{args.port}")

    print("  ⚠️ NOTE: Do NOT use 0.0.0.0 in your browser! Use http://localhost:7860")

    print(f"  PyTorch CUDA: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    print("=" * 70)



    engine = ColabTryOnEngine()



    fastapi_app = FastAPI(title="RTV Real-Time Virtual Try-On")



    @fastapi_app.websocket("/api/live_mirror")

    async def live_mirror_ws(websocket: WebSocket):

        await websocket.accept()

        current_garment = [engine.active_garment_label]

        show_hud = [True]



        import queue as thread_queue

        in_queue = thread_queue.Queue(maxsize=120)

        out_queue = thread_queue.Queue(maxsize=120)

        stop_event = threading.Event()



        def _gpu_worker_thread():

            while not stop_event.is_set():

                try:

                    raw_bytes = in_queue.get(timeout=0.05)

                except thread_queue.Empty:

                    continue

                try:

                    np_arr = np.frombuffer(raw_bytes, np.uint8)

                    frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    if frame_bgr is not None:

                        out_bgr = engine.process_stream_frame_bgr(

                            frame_bgr,

                            garment_label=current_garment[0],

                            tryon_mode="LIVE",

                            show_hud=show_hud[0]

                        )

                        success, enc = cv2.imencode(".jpg", out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 78])

                        if success:

                            if out_queue.full():

                                try:

                                    out_queue.get_nowait()

                                except thread_queue.Empty:

                                    pass

                            out_queue.put(enc.tobytes())

                except Exception as ex:

                    print(f"[RTV-Colab] GPU worker error: {ex}")



        worker_th = threading.Thread(target=_gpu_worker_thread, daemon=True)

        worker_th.start()



        async def _sender():

            while not stop_event.is_set():

                try:

                    try:

                        enc_bytes = out_queue.get_nowait()

                    except thread_queue.Empty:

                        await asyncio.sleep(0.005)

                        continue

                    await websocket.send_bytes(enc_bytes)

                except (WebSocketDisconnect, Exception):

                    stop_event.set()

                    break



        sender_task = asyncio.create_task(_sender())



        try:

            while not stop_event.is_set():

                msg = await websocket.receive()

                if "bytes" in msg and msg["bytes"]:

                    if in_queue.full():

                        try:

                            in_queue.get_nowait()

                        except thread_queue.Empty:

                            pass

                    in_queue.put(msg["bytes"])

                elif "text" in msg and msg["text"]:

                    try:

                        payload = json.loads(msg["text"])

                        if payload.get("action") == "set_config":

                            if "garment" in payload:

                                current_garment[0] = payload["garment"]

                                engine.set_active_garment(current_garment[0])

                            if "hud" in payload:

                                show_hud[0] = bool(payload["hud"])

                    except Exception:

                        pass

        except (WebSocketDisconnect, Exception):

            pass

        finally:

            stop_event.set()

            sender_task.cancel()



    demo = build_gradio_ui(engine)

    demo.queue()

    mount_kwargs = {

        "path": "/",

        "head": f"<script>\n{RTV_LIVE_MIRROR_JS}\n</script>"

    }

    import inspect

    sig_params = inspect.signature(gr.mount_gradio_app).parameters

    filtered_kwargs = {k: v for k, v in mount_kwargs.items() if k in sig_params}

    app = gr.mount_gradio_app(fastapi_app, demo, **filtered_kwargs)

    if getattr(args, 'share', False):
        def _launch_share_tunnel():
            time.sleep(2.5)
            try:
                try:
                    from gradio.networking import setup_tunnel
                except ImportError:
                    from gradio.tunneling import setup_tunnel
                import secrets
                share_token = secrets.token_urlsafe(32)
                import inspect
                sig = inspect.signature(setup_tunnel)
                call_args = []
                for name, param in sig.parameters.items():
                    if name in ('local_host', 'host'):
                        call_args.append('127.0.0.1')
                    elif name in ('local_port', 'port'):
                        call_args.append(args.port)
                    elif name == 'share_token':
                        call_args.append(share_token)
                    elif name == 'share_server_address':
                        call_args.append(None)
                    elif name == 'share_server_tls_certificate':
                        call_args.append(None)
                    else:
                        if param.default is not inspect.Parameter.empty:
                            continue
                        call_args.append(None)
                tunnel_url = setup_tunnel(*call_args)
                print('\n' + '=' * 70)
                print(f'  👉 PUBLIC GRADIO LIVE LINK: {tunnel_url}')
                print(f'  👉 LOCAL BROWSER LINK:      http://localhost:{args.port}')
                print('=' * 70 + '\n')
            except Exception as e:
                print(f'\n[RTV-Colab] Note on public link: {e}')
                print(f'[RTV-Colab] You can connect locally via http://localhost:{args.port}')
                print(f'[RTV-Colab] Or from your PC using SSH tunneling: ssh -L {args.port}:localhost:{args.port} administrator@miphi-aidaptiv\n')

        threading.Thread(target=_launch_share_tunnel, daemon=True).start()



    print(f"[RTV-Colab] Starting ASGI Server with Live WebSocket Mirror on port {args.port}...")

    uvicorn.run(app, host=args.server_name, port=args.port, log_level="warning")





if __name__ == "__main__":

    main()