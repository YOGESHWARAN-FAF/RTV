import sys
import os

# Ensure working directory is always RTV directory
RTV_DIR = os.path.abspath(os.path.dirname(__file__))
if os.getcwd() != RTV_DIR:
    os.chdir(RTV_DIR)
if RTV_DIR not in sys.path:
    sys.path.insert(0, RTV_DIR)

import time
import collections
import threading
import ctypes
import cv2
import numpy as np
import torch

# High-precision Windows Multimedia Timer (1ms granularity)
try:
    winmm = ctypes.WinDLL('winmm')
    winmm.timeBeginPeriod(1)
except Exception:
    winmm = None

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QEvent
from PyQt5.QtWidgets import (
    QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QSizePolicy,
    QHBoxLayout, QScrollArea, QListWidget, QListWidgetItem
)
from PyQt5.QtGui import QImage, QPixmap, QIcon

from util.image_warp import crop2_169, resize_img
from VITON.viton_upperbody import FrameProcessor
from util.pipeline_profiler import PipelineProfiler


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

    # Semi-transparent top and bottom banner
    cv2.rectangle(overlay, (0, 0), (w, 115), (10, 12, 18), -1)
    cv2.rectangle(overlay, (0, h - 55), (w, h), (10, 12, 18), -1)

    orient_str = get_orientation_str(yaw_angle)
    pct = int(progress * 100)
    title = f"[ 3D UPPER-BODY CALIBRATION: {pct}% ]"
    cv2.putText(overlay, title, (25, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2, cv2.LINE_AA)

    # Progress bar
    bar_w = min(450, w - 50)
    bar_x = 25
    bar_y = 44
    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (50, 50, 60), 1)
    fill_w = int(bar_w * progress)
    if fill_w > 0:
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + fill_w, bar_y + 10), (0, 255, 120), -1)

    # Anatomical Joint Tracking & AR Skeleton Overlay
    sh_locked = False
    hip_locked = False
    hand_locked = False

    if joints_2d is not None and len(joints_2d) >= 22:
        # Bone connections
        bones = [
            (16, 18), (18, 20),  # Left arm: shoulder -> elbow -> wrist
            (17, 19), (19, 21),  # Right arm: shoulder -> elbow -> wrist
            (16, 12), (17, 12),  # Shoulders to neck
            (12, 15),            # Neck to head
            (16, 1), (17, 2),    # Shoulders to hips
            (1, 2),              # Pelvis / Hips
            (1, 4), (4, 7),      # Left leg: hip -> knee -> ankle
            (2, 5), (5, 8),      # Right leg: hip -> knee -> ankle
        ]
        for j1, j2 in bones:
            pt1 = (int(np.clip(joints_2d[j1][0], 0, w - 1)), int(np.clip(joints_2d[j1][1], 0, h - 1)))
            pt2 = (int(np.clip(joints_2d[j2][0], 0, w - 1)), int(np.clip(joints_2d[j2][1], 0, h - 1)))
            cv2.line(overlay, pt1, pt2, (0, 255, 120), 2, cv2.LINE_AA)

        # Draw glowing reticles on Shoulders, Hips, and Hands
        key_landmarks = [
            (16, (0, 255, 0), "L.Shoulder"),
            (17, (0, 255, 0), "R.Shoulder"),
            (1, (0, 200, 255), "L.Hip"),
            (2, (0, 200, 255), "R.Hip"),
            (20, (255, 0, 255), "L.Hand"),
            (21, (255, 0, 255), "R.Hand"),
            (12, (0, 255, 255), "Neck"),
        ]
        for j_idx, col, name in key_landmarks:
            pt = (int(np.clip(joints_2d[j_idx][0], 0, w - 1)), int(np.clip(joints_2d[j_idx][1], 0, h - 1)))
            cv2.circle(overlay, pt, 6, col, -1, cv2.LINE_AA)
            cv2.circle(overlay, pt, 11, col, 1, cv2.LINE_AA)

        sh_dist = np.hypot(joints_2d[16][0] - joints_2d[17][0], joints_2d[16][1] - joints_2d[17][1])
        hip_dist = np.hypot(joints_2d[1][0] - joints_2d[2][0], joints_2d[1][1] - joints_2d[2][1])
        sh_locked = sh_dist > 25
        hip_locked = hip_dist > 15
        hand_locked = (joints_2d[20][1] > 20) and (joints_2d[21][1] > 20)
    else:
        # AR Silhouette Guide when joints not yet regressed
        cx, cy = w // 2, int(h * 0.48)
        head_r = int(min(w, h) * 0.12)
        cv2.ellipse(overlay, (cx, cy - int(head_r * 1.1)), (head_r, int(head_r * 1.35)), 0, 0, 360, (0, 215, 255), 1, cv2.LINE_AA)
        sh_w = int(min(w, h) * 0.35)
        cv2.ellipse(overlay, (cx, cy + head_r), (sh_w, int(head_r * 1.5)), 0, 180, 360, (0, 215, 255), 1, cv2.LINE_AA)

    # On-screen Checklist Badges
    badge_sh = "[✓] SHOULDERS: LOCKED" if sh_locked else "[ ] SHOULDERS: MAPPING..."
    badge_hip = "[✓] HIPS: LOCKED" if hip_locked else "[ ] HIPS: MAPPING..."
    badge_hnd = "[✓] HANDS: LOCKED" if hand_locked else "[ ] HANDS: MAPPING..."
    col_sh = (0, 255, 120) if sh_locked else (160, 160, 160)
    col_hip = (0, 255, 120) if hip_locked else (160, 160, 160)
    col_hnd = (0, 255, 120) if hand_locked else (160, 160, 160)

    cv2.putText(overlay, badge_sh, (25, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.46, col_sh, 1, cv2.LINE_AA)
    cv2.putText(overlay, badge_hip, (220, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.46, col_hip, 1, cv2.LINE_AA)
    cv2.putText(overlay, badge_hnd, (380, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.46, col_hnd, 1, cv2.LINE_AA)

    status_str = f"Angle: {orient_str} | Status: {status_text}"
    cv2.putText(overlay, status_str, (25, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    hint_text = "[SPACE] / [ENTER]: Start Live Mirror  |  [C]: Recalibrate"
    cv2.putText(overlay, hint_text, (25, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    return frame

def draw_synchronizing_hud(frame, progress, cached_patches, rem_sec):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), (10, 12, 18), -1)

    pct = int(progress * 100)
    title = f"[ SYNCHRONIZING REALTIME LOOKAHEAD BUFFER: {pct}% ]"
    cv2.putText(overlay, title, (25, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 255), 2, cv2.LINE_AA)

    bar_w = min(450, w - 50)
    cv2.rectangle(overlay, (25, 46), (25 + bar_w, 56), (50, 50, 60), 1)
    fill_w = int(bar_w * progress)
    if fill_w > 0:
        cv2.rectangle(overlay, (25, 46), (25 + fill_w, 56), (0, 255, 120), -1)

    info_str = f"Buffering lookahead frames ({cached_patches} 3D frames ready | {rem_sec:.1f}s)... Stand steady"
    cv2.putText(overlay, info_str, (25, 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 240, 255), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    return frame


class VitonStreamEngine(QThread):
    """
    High-Performance Asynchronous Upper-Body Virtual Try-On Engine.
    - Captures camera frames with monotonic timestamps & sequence IDs.
    - Decoupled Pipeline:
        1. Camera Capture Worker: Continuous 0-wait grabbing.
        2. ML Inference Worker: Runs ROMP/BEV, DensePose & Pix2PixHD asynchronously.
        3. Rendering & Compositing Stage: Forward LK optical flow tracking.
        4. Dual Operating Modes:
           - LIVE (Default): Zero delay (< 30ms latency), 30 FPS display locked to camera.
           - SYNC LOCKED: Lookahead delay matching measured neural latency.
    """
    frameCaptured = pyqtSignal(QImage)

    def __init__(self, garment_id_list, source=None, delay_seconds=0.60):
        super().__init__()
        self.source = source
        self.rotate_angle = 0
        self.running = True

        # Pipeline state machine: CALIBRATING -> SYNCHRONIZING -> ACTIVE_TRYON
        self.state = "CALIBRATING"
        self.mode = "LIVE"
        self.calibration_progress = 0.0
        self.sync_start_time = 0.0
        self.body_yaw_angle = 0.0
        self.body_detected = False
        self.calibration_joints = None

        self.cap = self.get_video_source()

        # Telemetry & Performance Profiling Monitor
        self.profiler = PipelineProfiler(window_size=60)

        # Threading synchronization primitives
        self.new_frame_event = threading.Event()
        self.latest_frame_lock = threading.Lock()
        self.latest_frame_packet = None
        self.capture_seq = 0
        self.last_displayed_seq = -1

        # Bounded frame buffer (stores max 60 frames = 2.0s history)
        self.frame_buffer = collections.deque(maxlen=60)
        self.buffer_lock = threading.Lock()

        # Dynamically measured sync delay (calibrated from measured ML latency)
        self.delay_seconds = delay_seconds
        self.measured_sync_delay = delay_seconds

        self.frame_processor = FrameProcessor(garment_id_list)

        self.patch_buffer = collections.deque(maxlen=60)
        self.latest_patch = None
        self.patch_lock = threading.Lock()

        self.ml_thread = threading.Thread(target=self._ml_worker_loop, daemon=True)
        self.ml_thread.start()

        # Paced Slow Streaming & Missing Frame Generation
        self.slow_streaming = False
        self.slow_fps = 4.0

        self.cap_thread = threading.Thread(target=self._capture_worker_loop, daemon=True)
        self.cap_thread.start()

    def toggle_slow_streaming(self):
        """Toggle paced slow streaming for dense neural synthesis & smooth missing frame generation."""
        self.slow_streaming = not self.slow_streaming
        status = f"ENABLED ({self.slow_fps:.1f} FPS slow streaming for ultra-dense neural keyframes & smooth missing frame generation)" if self.slow_streaming else "DISABLED (Normal camera speed)"
        print(f"[RTV] Paced Slow Streaming: {status}")

    def toggle_mode(self):
        """Toggle between LIVE low-latency mirror and SYNC lookahead mode."""
        if self.mode == "LIVE":
            self.mode = "SYNC"
            if self.delay_seconds <= 0.05:
                self.delay_seconds = self.measured_sync_delay
            print(f"[RTV] Mode switched to: SYNC LOCKED ({int(self.delay_seconds * 1000)}ms delay buffer)")
        else:
            self.mode = "LIVE"
            print("[RTV] Mode switched to: LIVE (Real-time forward LK tracking, zero delay)")

    def recalibrate(self):
        print("[RTV] Recalibrating 3D body skeleton (shoulders, hips, hands)...")
        self.state = "CALIBRATING"
        self.calibration_progress = 0.0
        self.sync_start_time = 0.0
        self.calibration_joints = None
        if hasattr(self.frame_processor, 'smpl_regressor') and hasattr(self.frame_processor.smpl_regressor, 'reset_tracking'):
            self.frame_processor.smpl_regressor.reset_tracking()
        if hasattr(self.frame_processor, 'reset_filters'):
            self.frame_processor.reset_filters()
        with self.patch_lock:
            self.patch_buffer.clear()
            self.latest_patch = None

    def force_start_tryon(self):
        print(f"[RTV] Manual trigger: Starting Virtual Try-On in {self.mode} mode...")
        self.calibration_progress = 1.0
        self.body_detected = True
        if self.mode == "LIVE":
            self.state = "ACTIVE_TRYON"
        else:
            self.state = "SYNCHRONIZING"
            self.sync_start_time = time.time()

    def set_target_id(self, garment_id):
        print(f"[RTV] Switching active garment ID: {garment_id}")
        self.frame_processor.set_target_garment(garment_id)
        with self.patch_lock:
            self.patch_buffer.clear()
            self.latest_patch = None

    def toggle_rotation(self):
        self.rotate_angle = (self.rotate_angle + 90) % 360
        print(f"[RTV] Camera rotation set to: {self.rotate_angle} deg")

    def adjust_delay(self, delta):
        self.delay_seconds = max(0.20, min(3.0, self.delay_seconds + delta))
        print(f"[RTV] Buffer delay adjusted to: {int(self.delay_seconds * 1000)} ms")

    def reset_to_measured_delay(self):
        self.delay_seconds = self.measured_sync_delay
        print(f"[RTV] Buffer delay reset to measured optimal value: {int(self.delay_seconds * 1000)} ms")

    def get_video_source(self):
        if self.source is not None:
            if isinstance(self.source, int) or str(self.source).isdigit():
                return cv2.VideoCapture(int(self.source))
            elif os.path.exists(str(self.source)):
                print(f"[RTV] Loading video source: {self.source}")
                return cv2.VideoCapture(str(self.source))

        if len(sys.argv) > 1:
            arg = sys.argv[1]
            if arg.isdigit():
                return cv2.VideoCapture(int(arg))
            elif os.path.exists(arg):
                print(f"[RTV] Loading video source: {arg}")
                return cv2.VideoCapture(arg)

        for idx in [0, 1]:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.mean() > 5:
                    print(f"[RTV] Auto-selected active webcam on index {idx} (mean brightness: {frame.mean():.1f})")
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    return cap
                cap.release()

        fallback_video = os.path.abspath("./assets/sample_video.mp4")
        if os.path.exists(fallback_video):
            print(f"[RTV] No active webcam found. Using sample video dataset: {fallback_video}")
            return cv2.VideoCapture(fallback_video)

        print("[RTV] Defaulting to camera index 0")
        return cv2.VideoCapture(0)

    def _capture_worker_loop(self):
        """
        Dedicated Camera Capture Worker:
        - Reads frames continuously with monotonic timestamps and sequence IDs.
        - Updates bounded ring buffer and signals display loop immediately.
        - Never blocks on neural network inference or UI rendering.
        """
        cap_fps = self.cap.get(cv2.CAP_PROP_FPS)
        is_video_file = cap_fps > 0 and not np.isnan(cap_fps) and cap_fps < 120 and self.cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
        target_fps = float(cap_fps) if is_video_file else 30.0
        frame_period = 1.0 / target_fps
        last_frame_time = time.monotonic()

        while self.running:
            if is_video_file:
                eff_fps = self.slow_fps if self.slow_streaming else target_fps
                eff_period = 1.0 / max(0.5, eff_fps)
                elapsed = time.monotonic() - last_frame_time
                if elapsed < eff_period:
                    time.sleep(max(0.001, eff_period - elapsed))
                last_frame_time = time.monotonic()

            t_read_start = time.perf_counter()
            ret, frame = self.cap.read()
            t_read = (time.perf_counter() - t_read_start) * 1000.0

            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

            mono_ts = time.monotonic()
            wall_ts = time.time()
            self.capture_seq += 1

            self.profiler.record_stage('capture_lat', t_read)
            self.profiler.record_capture(self.capture_seq, mono_ts)

            if self.rotate_angle == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotate_angle == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif self.rotate_angle == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            frame = cv2.flip(frame, 1)

            packet = {
                'seq': self.capture_seq,
                'mono_ts': mono_ts,
                'wall_ts': wall_ts,
                'frame': frame
            }

            with self.latest_frame_lock:
                self.latest_frame_packet = packet

            with self.buffer_lock:
                self.frame_buffer.append(packet)

            # Signal display loop that a new frame has arrived
            self.new_frame_event.set()

            if not is_video_file:
                time.sleep(0.002)

    def _ml_worker_loop(self):
        """
        Asynchronous Deep Learning Worker:
        - Processes newest appropriate frame without holding locks.
        - Calculates optimal synchronization delay from measured neural latency.
        - Associates output patch with source sequence ID and timestamp.
        """
        last_processed_seq = -1

        while self.running:
            target_packet = None

            if self.mode == "LIVE" or self.state == "CALIBRATING":
                # In LIVE mode, grab the latest available frame
                with self.latest_frame_lock:
                    if self.latest_frame_packet is not None and self.latest_frame_packet['seq'] > last_processed_seq:
                        target_packet = self.latest_frame_packet
                if target_packet is not None:
                    # Adaptive re-anchoring check in ACTIVE_TRYON
                    if self.state == "ACTIVE_TRYON":
                        tr_info = self.frame_processor.get_tracking_info()
                        with self.patch_lock:
                            has_patch = self.latest_patch is not None
                            patch_age = (time.time() - self.latest_patch.get('timestamp', 0.0)) if has_patch else 999.0

                        needs_reanchor = tr_info.get('needs_reanchor', False)
                        # Re-anchor immediately if tracking degraded or heartbeat (>0.10s) elapsed for 8-10+ FPS synthesis
                        if not needs_reanchor and has_patch and patch_age < 0.10:
                            time.sleep(0.005)
                            continue

                    target_frame = target_packet['frame'].copy()
                    target_ts = target_packet['wall_ts']
                    target_mono_ts = target_packet['mono_ts']
                    target_seq = target_packet['seq']
                    last_processed_seq = target_seq
                else:
                    time.sleep(0.005)
                    continue
            else:
                # In SYNC mode: Lookahead scheduling
                with self.buffer_lock:
                    if len(self.frame_buffer) == 0:
                        time.sleep(0.01)
                        continue
                    cam_newest_ts = self.frame_buffer[-1]['wall_ts']
                    target_disp_ts = time.time() - self.delay_seconds

                    with self.patch_lock:
                        if len(self.patch_buffer) > 0:
                            last_patch_ts = max(p.get('timestamp', 0.0) for p in self.patch_buffer)
                        else:
                            last_patch_ts = target_disp_ts - 0.20

                    stride = 0.28
                    next_target_ts = max(target_disp_ts + 0.30, last_patch_ts + stride)
                    next_target_ts = min(cam_newest_ts, next_target_ts)

                    if next_target_ts - last_patch_ts < 0.08 and last_patch_ts >= cam_newest_ts - 0.06:
                        time.sleep(0.01)
                        continue

                    best_diff = float('inf')
                    for p in self.frame_buffer:
                        diff = abs(p['wall_ts'] - next_target_ts)
                        if diff < best_diff:
                            best_diff = diff
                            target_packet = p

                if target_packet is not None:
                    target_frame = target_packet['frame'].copy()
                    target_ts = target_packet['wall_ts']
                    target_mono_ts = target_packet['mono_ts']
                    target_seq = target_packet['seq']
                else:
                    time.sleep(0.01)
                    continue

            t0 = time.perf_counter()

            if self.state == "CALIBRATING":
                try:
                    raw_h, raw_w = target_frame.shape[:2]
                    det_max_h = 512
                    scale = det_max_h / float(raw_h) if raw_h > det_max_h else 1.0
                    det_img = cv2.resize(target_frame, (int(raw_w * scale), det_max_h)) if scale != 1.0 else target_frame
                    with torch.inference_mode():
                        smpl_param = self.frame_processor.smpl_regressor.forward(det_img, False)
                    if smpl_param is not None and 'center_confs' in smpl_param:
                        confs = smpl_param['center_confs']
                        conf = confs.max().item() if len(confs) > 0 else 0.0
                        if conf >= 0.07:
                            self.body_detected = True
                            rot_rad = self.frame_processor.smpl_regressor.get_rotation_angle(smpl_param)
                            if rot_rad is not None:
                                self.body_yaw_angle = float(np.degrees(rot_rad))

                            best_idx = smpl_param.get('best_person_idx', 0)
                            raw_joints = smpl_param['pj2d_org'][best_idx].cpu().numpy()
                            self.calibration_joints = raw_joints / scale if scale != 1.0 else raw_joints
                            self.calibration_progress = min(1.0, self.calibration_progress + 0.20)
                        else:
                            self.body_detected = False
                            self.calibration_progress = max(0.0, self.calibration_progress - 0.01)
                    else:
                        self.body_detected = False
                        self.calibration_progress = max(0.0, self.calibration_progress - 0.01)

                    if self.calibration_progress >= 1.0:
                        if self.mode == "LIVE":
                            print("[RTV] 3D Skeleton Calibration Complete! Virtual Try-On Active (LIVE Mode, Zero Lag)...")
                            self.state = "ACTIVE_TRYON"
                        else:
                            print(f"[RTV] 3D Skeleton Calibration Complete! Buffering lookahead ({int(self.delay_seconds*1000)}ms)...")
                            self.state = "SYNCHRONIZING"
                            self.sync_start_time = time.time()
                except Exception as e:
                    print(f"[RTV] Calibration warning: {e}")

                dt_cal = (time.perf_counter() - t0) * 1000.0
                self.profiler.record_ml_inference(dt_cal)
                time.sleep(0.01)
                continue

            # State is SYNCHRONIZING or ACTIVE_TRYON
            if self.frame_processor.viton_model is None:
                time.sleep(0.02)
                continue

            try:
                patch = self.frame_processor.get_garment_patch(target_frame)
                dt_ml = (time.perf_counter() - t0) * 1000.0
                self.profiler.record_ml_inference(dt_ml)

                # Dynamically calibrate measured optimal lookahead delay
                measured_lookahead = max(0.50, min(1.20, (dt_ml * 1.25) / 1000.0))
                self.measured_sync_delay = measured_lookahead

                if patch is not None:
                    patch['timestamp'] = target_ts
                    patch['mono_ts'] = target_mono_ts
                    patch['seq'] = target_seq
                    patch['inference_time_ms'] = dt_ml

                    with self.patch_lock:
                        self.latest_patch = patch
                        self.patch_buffer.append(patch)
                        cutoff = time.time() - self.delay_seconds - 3.0
                        if len(self.patch_buffer) > 20:
                            self.patch_buffer = collections.deque(
                                [p for p in self.patch_buffer if p.get('timestamp', 0.0) >= cutoff],
                                maxlen=60
                            )

                    if self.state == "SYNCHRONIZING":
                        elapsed_sync = time.time() - self.sync_start_time
                        with self.patch_lock:
                            buf_len = len(self.patch_buffer)
                        if elapsed_sync >= self.delay_seconds and buf_len >= 2:
                            self.state = "ACTIVE_TRYON"
                            print(f"[RTV] Lookahead Synchronized! Try-On Active (SYNC Mode, Delay={int(self.delay_seconds*1000)}ms).")
            except Exception as e:
                print(f"[RTV] ML worker exception: {e}")

            time.sleep(0.002)

    def run(self):
        """
        Paced Display Loop:
        - Locks display refresh to camera frame arrival (30 FPS for 30 FPS camera).
        - Eliminates duplicate rendering and artificial FPS counters.
        - Delivers true low-latency (< 30ms) interactive experience.
        """
        while self.running:
            self.new_frame_event.wait(timeout=0.035)
            self.new_frame_event.clear()

            if not self.running:
                break

            loop_start = time.monotonic()

            if self.state == "CALIBRATING":
                chosen_frame = None
                with self.latest_frame_lock:
                    if self.latest_frame_packet is not None:
                        chosen_frame = self.latest_frame_packet['frame'].copy()

                if chosen_frame is not None:
                    status_str = "Body Detected - Hold Steady" if self.body_detected else "Looking for person..."
                    display_frame = draw_calibration_hud(
                        chosen_frame, self.calibration_progress, self.body_yaw_angle,
                        joints_2d=self.calibration_joints, status_text=status_str
                    )
                    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    h, w, c = rgb_frame.shape
                    self.frameCaptured.emit(QImage(rgb_frame.data, w, h, c * w, QImage.Format_RGB888).copy())

            elif self.state == "SYNCHRONIZING":
                chosen_frame = None
                with self.latest_frame_lock:
                    if self.latest_frame_packet is not None:
                        chosen_frame = self.latest_frame_packet['frame'].copy()

                if chosen_frame is not None:
                    elapsed = max(0.0, time.time() - self.sync_start_time) if self.sync_start_time > 0 else 0.0
                    sync_progress = min(1.0, elapsed / max(0.1, self.delay_seconds))
                    with self.patch_lock:
                        buf_len = len(self.patch_buffer)
                    rem_sec = max(0.0, self.delay_seconds - elapsed)
                    display_frame = draw_synchronizing_hud(chosen_frame, sync_progress, buf_len, rem_sec)
                    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    h, w, c = rgb_frame.shape
                    self.frameCaptured.emit(QImage(rgb_frame.data, w, h, c * w, QImage.Format_RGB888).copy())

            else:
                # ACTIVE_TRYON
                chosen_frame = None
                capture_mono_ts = 0.0
                frame_seq = -1

                if self.mode == "LIVE":
                    # LIVE MODE: Display newest live camera frame immediately (< 30ms latency)
                    with self.latest_frame_lock:
                        if self.latest_frame_packet is not None:
                            packet = self.latest_frame_packet
                            frame_seq = packet['seq']
                            if frame_seq == self.last_displayed_seq:
                                time.sleep(0.001)
                                continue
                            chosen_frame = packet['frame'].copy()
                            capture_mono_ts = packet['mono_ts']

                    if chosen_frame is not None:
                        patch_to_use = None
                        with self.patch_lock:
                            if self.latest_patch is not None:
                                patch_to_use = self.latest_patch

                        t_comp_start = time.perf_counter()
                        if patch_to_use is not None:
                            display_frame = self.frame_processor.composite_garment(chosen_frame, patch_to_use)
                        else:
                            display_frame = chosen_frame

                        t_comp = (time.perf_counter() - t_comp_start) * 1000.0
                        self.profiler.record_stage('compositing', t_comp)
                        self.profiler.record_tracking_update(t_comp)
                        self.profiler.record_display_frame(frame_seq, capture_mono_ts, loop_start)

                        patch_age_ms = 0.0
                        if patch_to_use is not None:
                            p_ts = patch_to_use.get('timestamp', time.time())
                            patch_age_ms = max(0.0, (time.time() - p_ts) * 1000.0)

                        display_frame = self.profiler.draw_hud(
                            display_frame, mode_str="LIVE", sync_delay_sec=0.0, yaw_angle=self.body_yaw_angle,
                            patch_age_ms=patch_age_ms
                        )
                        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                        h, w, c = rgb_frame.shape
                        self.frameCaptured.emit(QImage(rgb_frame.data, w, h, c * w, QImage.Format_RGB888).copy())

                else:
                    # SYNC LOCKED MODE: Lookahead buffered replay
                    target_timestamp = time.time() - self.delay_seconds
                    with self.buffer_lock:
                        if len(self.frame_buffer) > 0:
                            best_diff = float('inf')
                            for p in self.frame_buffer:
                                diff = abs(p['wall_ts'] - target_timestamp)
                                if diff < best_diff:
                                    best_diff = diff
                                    chosen_frame = p['frame'].copy()
                                    capture_mono_ts = p['mono_ts']
                                    frame_seq = p['seq']

                    if chosen_frame is not None:
                        prev_patch = None
                        next_patch = None
                        alpha = 0.0

                        with self.patch_lock:
                            if len(self.patch_buffer) > 0:
                                recent = [p for p in self.patch_buffer if abs(p.get('timestamp', 0.0) - target_timestamp) < 2.5]
                                if len(recent) > 0:
                                    recent.sort(key=lambda x: x.get('timestamp', 0.0))
                                    cand_prev = [p for p in recent if p.get('timestamp', 0.0) <= target_timestamp]
                                    cand_next = [p for p in recent if p.get('timestamp', 0.0) > target_timestamp]
                                    if len(cand_prev) > 0:
                                        prev_patch = cand_prev[-1]
                                    if len(cand_next) > 0:
                                        next_patch = cand_next[0]
                                    if prev_patch is not None and next_patch is not None:
                                        tp = prev_patch.get('timestamp', 0.0)
                                        tn = next_patch.get('timestamp', 0.0)
                                        if tn > tp:
                                            alpha = float(np.clip((target_timestamp - tp) / (tn - tp), 0.0, 1.0))
                                    elif prev_patch is None and next_patch is not None:
                                        prev_patch = next_patch
                                        next_patch = None
                            if prev_patch is None and self.latest_patch is not None:
                                prev_patch = self.latest_patch

                        t_comp_start = time.perf_counter()
                        if prev_patch is not None:
                            display_frame = self.frame_processor.composite_garment(
                                chosen_frame, prev_patch, next_patch=next_patch, alpha=alpha
                            )
                        else:
                            display_frame = chosen_frame

                        t_comp = (time.perf_counter() - t_comp_start) * 1000.0
                        self.profiler.record_stage('compositing', t_comp)
                        self.profiler.record_tracking_update(t_comp)
                        self.profiler.record_display_frame(frame_seq, capture_mono_ts, loop_start)

                        patch_age_ms = 0.0
                        if prev_patch is not None:
                            p_ts = prev_patch.get('timestamp', time.time())
                            patch_age_ms = max(0.0, (time.time() - p_ts) * 1000.0)

                        display_frame = self.profiler.draw_hud(
                            display_frame, mode_str="SYNC", sync_delay_sec=self.delay_seconds, yaw_angle=self.body_yaw_angle,
                            patch_age_ms=patch_age_ms
                        )
                        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                        h, w, c = rgb_frame.shape
                        self.frameCaptured.emit(QImage(rgb_frame.data, w, h, c * w, QImage.Format_RGB888).copy())

    def stop(self):
        self.running = False
        self.new_frame_event.set()
        if self.cap is not None:
            self.cap.release()


class CameraApp(QMainWindow):
    def __init__(self, source=None):
        super().__init__()
        self.setWindowTitle("Real-Time Upper-Body Virtual Try-On Mirror (High-Performance)")
        self.setGeometry(70, 70, 1200, 800)
        self.setMinimumSize(600, 400)

        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        # Video Viewport
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("background-color: #0d0d12; border-radius: 8px;")
        main_layout.addWidget(self.image_label, 1)

        # Right Column: Garment Selection Gallery
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(240)
        scroll_area.setStyleSheet("background-color: #15151e; border: none; border-radius: 6px;")

        container_list = QWidget()
        layout_list = QVBoxLayout(container_list)
        layout_list.setContentsMargins(6, 6, 6, 6)

        title_garment = QLabel("VIRTUAL WARDROBE")
        title_garment.setAlignment(Qt.AlignCenter)
        title_garment.setStyleSheet("color: #00ffaa; font-weight: bold; font-size: 13px; margin: 6px 0;")
        layout_list.addWidget(title_garment)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SingleSelection)
        self.list_widget.setIconSize(QSize(150, 150))
        self.list_widget.setStyleSheet("""
            QListWidget { background: #1a1a24; border: 1px solid #282836; border-radius: 6px; color: #fff; }
            QListWidget::item { padding: 8px; border-bottom: 1px solid #252535; }
            QListWidget::item:selected { background: #005533; border: 1px solid #00ffaa; border-radius: 4px; }
        """)

        item0 = QListWidgetItem()
        item0.setText("0: Raw Camera")
        if os.path.exists('./assets/none.png'):
            item0.setIcon(QIcon('./assets/none.png'))
        item0.setData(Qt.UserRole, -1)
        self.list_widget.addItem(item0)

        garment_name_list = ['lab_03', 'lab_04', 'lab_07', 'jin_17', 'jin_18', 'jin_22']
        for i, garment in enumerate(garment_name_list):
            item = QListWidgetItem()
            item.setText(f"{i+1}: {garment}")
            thumb_path = f'./assets/garment_images/{garment}_white_bg.jpg'
            if os.path.exists(thumb_path):
                item.setIcon(QIcon(thumb_path))
            item.setData(Qt.UserRole, i)
            self.list_widget.addItem(item)

        layout_list.addWidget(self.list_widget)
        self.list_widget.itemSelectionChanged.connect(self.on_selection_changed)
        self.list_widget.installEventFilter(self)

        scroll_area.setWidget(container_list)
        main_layout.addWidget(scroll_area)

        self.setCentralWidget(main_widget)
        self.setStyleSheet("background-color: #0a0a0f;")

        processed_garments = [g + '_vmsdp2ta' for g in garment_name_list]
        self.stream_engine = VitonStreamEngine(processed_garments, source=source)
        self.stream_engine.frameCaptured.connect(self.update_image)
        self.stream_engine.start()

        if self.list_widget.count() > 1:
            self.list_widget.setCurrentRow(1)

    def on_selection_changed(self):
        selected_items = self.list_widget.selectedItems()
        if selected_items:
            item_id = selected_items[0].data(Qt.UserRole)
            self.stream_engine.set_target_id(item_id)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            key = event.key()
            if key == Qt.Key_Space or key == Qt.Key_Return:
                self.stream_engine.force_start_tryon()
                return True
            if obj == self.list_widget and Qt.Key_0 <= key <= Qt.Key_9:
                idx = key - Qt.Key_0
                if idx < self.list_widget.count():
                    self.list_widget.setCurrentRow(idx)
                    return True
        return super(CameraApp, self).eventFilter(obj, event)

    def update_image(self, q_img):
        try:
            t0 = time.perf_counter()
            pixmap = QPixmap.fromImage(q_img)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation))
            dt = (time.perf_counter() - t0) * 1000.0
            if hasattr(self, 'stream_engine') and hasattr(self.stream_engine, 'profiler'):
                self.stream_engine.profiler.record_gui_refresh()
                self.stream_engine.profiler.record_stage('presentation', dt)
        except Exception as e:
            print(f"[RTV] GUI update warning: {e}")

    def keyPressEvent(self, e):
        key = e.key()
        if key == Qt.Key_Q or key == Qt.Key_Escape:
            print("[RTV] Exiting application...")
            self.close()
        elif key == Qt.Key_F or key == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        elif key == Qt.Key_M or key == Qt.Key_Tab:
            self.stream_engine.toggle_mode()
        elif key == Qt.Key_P:
            self.stream_engine.toggle_slow_streaming()
        elif key == Qt.Key_R:
            self.stream_engine.toggle_rotation()
        elif key == Qt.Key_BracketRight:
            self.stream_engine.adjust_delay(+0.05)
        elif key == Qt.Key_BracketLeft:
            self.stream_engine.adjust_delay(-0.05)
        elif key == Qt.Key_S:
            self.stream_engine.reset_to_measured_delay()
        elif key == Qt.Key_C:
            self.stream_engine.recalibrate()
        elif key == Qt.Key_Space or key == Qt.Key_Return:
            self.stream_engine.force_start_tryon()
        elif Qt.Key_0 <= key <= Qt.Key_9:
            idx = key - Qt.Key_0
            if idx < self.list_widget.count():
                self.list_widget.setCurrentRow(idx)

    def closeEvent(self, event):
        self.stream_engine.stop()
        self.stream_engine.wait()
        if winmm is not None:
            try:
                winmm.timeEndPeriod(1)
            except Exception:
                pass
        print("[RTV] Stream engine stopped")
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    src = sys.argv[1] if len(sys.argv) > 1 else None
    window = CameraApp(source=src)
    window.show()
    sys.exit(app.exec_())
