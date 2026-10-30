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

try:
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QEvent
    from PyQt5.QtWidgets import (
        QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QSizePolicy,
        QHBoxLayout, QScrollArea, QListWidget, QListWidgetItem, QDialog, QTabWidget,
        QPushButton, QLineEdit, QFileDialog, QSpinBox, QProgressBar, QTextEdit,
        QMessageBox, QComboBox, QCheckBox, QFormLayout, QGroupBox, QRadioButton, QFrame
    )
    from PyQt5.QtGui import QImage, QPixmap, QIcon
except ImportError:
    # Headless / Cloud environment without PyQt5 GUI
    Qt = QThread = pyqtSignal = QSize = QEvent = None
    QApplication = QLabel = QMainWindow = QVBoxLayout = QWidget = QSizePolicy = None
    QHBoxLayout = QScrollArea = QListWidget = QListWidgetItem = QDialog = QTabWidget = None
    QPushButton = QLineEdit = QFileDialog = QSpinBox = QProgressBar = QTextEdit = None
    QMessageBox = QComboBox = QCheckBox = QFormLayout = QGroupBox = QRadioButton = QFrame = None
    QImage = QPixmap = QIcon = None

from VITON.viton_fullbody_seq import FullBodySeqFrameProcessor
from util.pipeline_profiler import PipelineProfiler
from util.ar_motion_graphics import ARMotionGraphics
from util.intel_uhd_coprocessor import IntelUHDCoProcessor
from util.custom_garment_trainer import CustomGarmentTrainer


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
    title = f"[ 3D BODY MOTION CALIBRATION: {pct}% ]"
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

    hint_text = "[SPACE] / [ENTER]: Start Live Mirror  |  [C]: Recalibrate  |  [G]: Toggle AR Motion FX"
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


class VitonFullbodyStreamEngine(QThread):
    """
    High-Performance Asynchronous Real-Time Virtual Try-On Engine.
    - Captures camera frames with monotonic timestamps & incrementing sequence IDs.
    - Decoupled Pipeline:
        1. Camera Capture Worker: Continuously grabs frames, 0-blocking.
        2. ML Inference Worker: Runs ROMP/BEV, DensePose & Pix2PixHD asynchronously.
        3. Rendering & Compositing Stage: Uses forward Lucas-Kanade optical flow on every live frame.
        4. Dual Operating Modes:
           - LIVE (Default): Zero delay (< 30ms latency), 100% camera-locked 30 FPS display.
           - SYNC LOCKED: Dynamically calibrated lookahead delay matching measured neural latency.
    """
    frameCaptured = pyqtSignal(QImage)

    def __init__(self, target_name='coat_seq_vmssdp2ta_576', source=None, delay_seconds=0.22):
        super().__init__()
        self.source = source
        self.rotate_angle = 0
        self.running = True

        # Pipeline state machine: CALIBRATING -> SYNCHRONIZING -> ACTIVE_TRYON
        self.state = "CALIBRATING"
        self.mode = "LIVE"  # LIVE Mode default: Zero latency (<30ms), 60 FPS real-time sync with camera
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

        # Bounded frame buffer (stores max 60 frames = 2.0s of history)
        self.frame_buffer = collections.deque(maxlen=60)
        self.buffer_lock = threading.Lock()

        # Dynamically measured sync delay (calibrated from measured ML latency)
        self.delay_seconds = delay_seconds
        self.measured_sync_delay = delay_seconds

        self.frame_processor = FullBodySeqFrameProcessor(target_name)

        self.patch_buffer = collections.deque(maxlen=60)
        self.latest_patch = None
        self.patch_lock = threading.Lock()
        self.ml_lock = threading.Lock()

        # Intel UHD Graphics (GPU.0) Co-Processor for heterogeneous offloading
        self.intel_coprocessor = IntelUHDCoProcessor()

        # Paced Slow Streaming & Missing Frame Generation
        self.slow_streaming = False
        self.slow_fps = 4.0

        # AR Graphical Motion & Visual Feedback Engine (Disabled for 100% pure camera pixels)
        self.ar_motion_enabled = False
        self.ar_motion = ARMotionGraphics()

        self.ml_thread = threading.Thread(target=self._ml_worker_loop, daemon=True)
        self.ml_thread.start()

        self.cap_thread = threading.Thread(target=self._capture_worker_loop, daemon=True)
        self.cap_thread.start()

    def toggle_slow_streaming(self):
        """Toggle paced slow streaming for dense neural synthesis & smooth missing frame generation."""
        self.slow_streaming = not self.slow_streaming
        status = f"ENABLED ({self.slow_fps:.1f} FPS slow streaming for ultra-dense neural keyframes & smooth missing frame generation)" if self.slow_streaming else "DISABLED (Normal camera speed)"
        print(f"[RTV] Paced Slow Streaming: {status}")

    def toggle_ar_motion(self):
        """Toggle active AR graphical motion trails and particles during delay frames."""
        self.ar_motion_enabled = not self.ar_motion_enabled
        state_str = "ENABLED (Live Neon Trails + Particles + Cyber Reticles)" if self.ar_motion_enabled else "DISABLED"
        print(f"[RTV] AR Graphical Motion FX: {state_str}")

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

    def set_target_garment(self, target_name):
        print(f"[RTV] Switching active garment: {target_name}")
        with self.ml_lock:
            self.frame_processor.set_target_garment(target_name)
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

            # Asynchronously offload frame inspection to Intel UHD Graphics (GPU.0)
            self.intel_coprocessor.process_frame_async(frame)

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
                        # Re-anchor immediately if tracking degraded or drift occurred; allow 6-8+ FPS generation
                        if not needs_reanchor and has_patch and patch_age < 0.12:
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

                    stride = 0.14
                    next_target_ts = max(target_disp_ts + 0.15, last_patch_ts + stride)
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
                with self.ml_lock:
                    patch = self.frame_processor.get_garment_patch(target_frame)
                dt_ml = (time.perf_counter() - t0) * 1000.0
                self.profiler.record_ml_inference(dt_ml)

                # Dynamically calibrate measured optimal lookahead delay (1.25x inference latency)
                measured_lookahead = max(0.18, min(0.60, (dt_ml * 1.25) / 1000.0))
                self.measured_sync_delay = measured_lookahead
                if self.mode == "SYNC":
                    # Adaptively steer buffer delay towards measured optimal lookahead
                    diff = measured_lookahead - self.delay_seconds
                    if abs(diff) > 0.03:
                        self.delay_seconds += float(np.clip(diff * 0.12, -0.015, 0.015))

                if patch is not None:
                    patch['timestamp'] = target_ts
                    patch['mono_ts'] = target_mono_ts
                    patch['seq'] = target_seq
                    patch['inference_time_ms'] = dt_ml

                    with self.patch_lock:
                        self.latest_patch = patch
                        self.patch_buffer.append(patch)
                        # Keep patch buffer bounded to avoid memory bloat
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
            # Wait for camera worker to capture a new frame (up to 35ms timeout)
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
                                # Frame already displayed, skip redundant render
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

                        # AR Graphical Motion Effects (Neon Trails, Kinetic Sparkles, Reticles)
                        if self.ar_motion_enabled:
                            tr_info = self.frame_processor.get_tracking_info()
                            wl = tr_info.get('wrist_l', None)
                            wr = tr_info.get('wrist_r', None)
                            self.ar_motion.update(wl, wr, dt=0.033)
                            display_frame = self.ar_motion.draw_effects(display_frame)

                        t_comp = (time.perf_counter() - t_comp_start) * 1000.0
                        self.profiler.record_stage('compositing', t_comp)
                        self.profiler.record_tracking_update(t_comp)
                        self.profiler.record_display_frame(frame_seq, capture_mono_ts, loop_start)

                        patch_age_ms = 0.0
                        if patch_to_use is not None:
                            p_ts = patch_to_use.get('timestamp', time.time())
                            patch_age_ms = max(0.0, (time.time() - p_ts) * 1000.0)

                        # Draw telemetry HUD with Dual-GPU status
                        display_frame = self.profiler.draw_hud(
                            display_frame, mode_str="LIVE", sync_delay_sec=0.0, yaw_angle=self.body_yaw_angle,
                            patch_age_ms=patch_age_ms, dual_gpu_status=self.intel_coprocessor.get_status_str()
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
                        if frame_seq == self.last_displayed_seq:
                            time.sleep(0.001)
                            continue
                        self.last_displayed_seq = frame_seq

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

                        # AR Graphical Motion Effects (Neon Trails, Kinetic Sparkles, Reticles)
                        if self.ar_motion_enabled:
                            tr_info = self.frame_processor.get_tracking_info()
                            wl = tr_info.get('wrist_l', None)
                            wr = tr_info.get('wrist_r', None)
                            self.ar_motion.update(wl, wr, dt=0.033)
                            display_frame = self.ar_motion.draw_effects(display_frame)

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
                            patch_age_ms=patch_age_ms, dual_gpu_status=self.intel_coprocessor.get_status_str()
                        )
                        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                        h, w, c = rgb_frame.shape
                        self.frameCaptured.emit(QImage(rgb_frame.data, w, h, c * w, QImage.Format_RGB888).copy())

    def stop(self):
        self.running = False
        self.new_frame_event.set()
        if hasattr(self, 'intel_coprocessor'):
            self.intel_coprocessor.stop()
        if self.cap is not None:
            self.cap.release()


class TrainingWorker(QThread):
    progress_signal = pyqtSignal(int, int, str)
    epoch_signal = pyqtSignal(int, int, int, float, str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, trainer, source_path, mask_dir, garment_name, epochs=3, batch_size=2, img_size=512, views_dict=None, mode="turbo"):
        super().__init__()
        self.trainer = trainer
        self.source_path = source_path
        self.mask_dir = mask_dir
        self.garment_name = garment_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.views_dict = views_dict
        self.mode = mode
        self.stop_requested = threading.Event()

    def run(self):
        try:
            if self.views_dict:
                self.progress_signal.emit(0, 100, f"Synthesizing 360° dataset ({self.mode.upper()} mode) from Front, Back, Right, Left photos...")
                ds_path, count = self.trainer.generate_dataset_from_multiview(
                    self.views_dict,
                    dataset_name=self.garment_name,
                    samples_per_view=None,
                    mode=self.mode,
                    progress_cb=lambda cur, tot, msg: self.progress_signal.emit(int((cur / max(1, tot)) * 40), 100, msg)
                )
            else:
                self.progress_signal.emit(0, 100, "Starting dataset generation from custom dress media...")
                ds_path, count = self.trainer.generate_dataset_from_source(
                    self.source_path,
                    mask_dir=self.mask_dir,
                    dataset_name=self.garment_name,
                    max_frames=120,
                    progress_cb=lambda cur, tot, msg: self.progress_signal.emit(int((cur / max(1, tot)) * 40), 100, msg)
                )
            if self.stop_requested.is_set():
                self.finished_signal.emit(False, "Dataset generation stopped by user.")
                return

            self.progress_signal.emit(40, 100, f"Dataset ready ({count} pairs)! Training Pix2PixHD on RTX 2050 (AMP FP16, Mode: {self.mode})...")

            def train_cb(ep, tot_ep, st, loss, msg):
                total_target_steps = max(1, tot_ep * max(1, count))
                step_fraction = min(1.0, max(0.0, st / total_target_steps))
                ep_fraction = (ep / max(1, tot_ep)) if ep > 0 else 0.0
                effective_prog = max(ep_fraction, step_fraction)
                pct = min(99, 40 + int(effective_prog * 59))
                self.epoch_signal.emit(ep, tot_ep, st, loss, msg)
                self.progress_signal.emit(pct, 100, msg)

            name, out_pth = self.trainer.train_garment(
                dataset_name=self.garment_name,
                garment_name=self.garment_name,
                num_epochs=self.epochs,
                batch_size=self.batch_size,
                img_size=self.img_size,
                mode=self.mode,
                progress_cb=train_cb,
                stop_event=self.stop_requested
            )
            self.finished_signal.emit(True, f"Model '{name}' successfully trained & deployed to wardrobe!\nLocation: {out_pth}")
        except Exception as e:
            self.finished_signal.emit(False, str(e))

    def stop(self):
        self.stop_requested.set()


class CustomGarmentDialog(QDialog):
    garment_added_signal = pyqtSignal(str)

    def __init__(self, stream_engine, parent=None):
        super().__init__(parent)
        self.stream_engine = stream_engine
        self.trainer = CustomGarmentTrainer()
        self.worker = None

        self.photo_views = {"front": None, "back": None, "right": None, "left": None}
        self.view_preview_labels = {}

        self.setWindowTitle("Custom Dress Studio - Upload, Train & Test")
        self.resize(920, 720)
        self.setStyleSheet("""
            QDialog { background-color: #0e1017; color: #e2e8f0; font-family: Segoe UI, sans-serif; }
            QTabWidget::pane { border: 1px solid #1f2438; background: #131622; border-radius: 6px; }
            QTabBar::tab { background: #181c2b; color: #94a3b8; padding: 10px 18px; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 3px; font-weight: bold; }
            QTabBar::tab:selected { background: #00aa66; color: #ffffff; }
            QLineEdit, QSpinBox, QComboBox { background: #1a1f30; border: 1px solid #2d354d; border-radius: 4px; padding: 6px 10px; color: #ffffff; }
            QPushButton { background: #00aa66; color: #ffffff; border-radius: 4px; padding: 8px 14px; font-weight: bold; }
            QPushButton:hover { background: #00cc7a; }
            QPushButton:disabled { background: #262c3d; color: #64748b; }
            QProgressBar { background: #1a1f30; border: 1px solid #2d354d; border-radius: 4px; text-align: center; color: #ffffff; }
            QProgressBar::chunk { background: #00ffaa; border-radius: 3px; }
            QTextEdit { background: #0a0d14; border: 1px solid #1e2436; border-radius: 4px; color: #00ffaa; font-family: Consolas, monospace; font-size: 11px; }
            QGroupBox { border: 1px solid #23293d; border-radius: 6px; margin-top: 12px; padding-top: 12px; font-weight: bold; color: #38bdf8; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QRadioButton { color: #e2e8f0; font-weight: bold; }
        """)

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self._build_train_tab()
        self._build_import_tab()
        self._build_test_tab()

    def _build_train_tab(self):
        tab = QWidget()
        l = QVBoxLayout(tab)
        l.setSpacing(10)

        group_inputs = QGroupBox("1. Custom Dress Media (Upload Images: Front, Back, Right, Left)")
        v_inputs = QVBoxLayout(group_inputs)

        f_name = QFormLayout()
        self.txt_train_name = QLineEdit("custom_designer_dress")
        f_name.addRow("Garment Model Name:", self.txt_train_name)
        v_inputs.addLayout(f_name)

        # Source mode toggle
        h_mode = QHBoxLayout()
        self.radio_multiview = QRadioButton("📸 4-View Photos (Front, Back, Right, Left) [Recommended for Stills]")
        self.radio_multiview.setChecked(True)
        self.radio_multiview.toggled.connect(self._toggle_train_mode)
        self.radio_video = QRadioButton("🎥 Single Video File (.mp4) / Continuous Image Folder")
        h_mode.addWidget(self.radio_multiview)
        h_mode.addWidget(self.radio_video)
        v_inputs.addLayout(h_mode)

        # Container for 4-View Photos
        self.container_multiview = QWidget()
        v_mv = QVBoxLayout(self.container_multiview)
        v_mv.setContentsMargins(0, 4, 0, 4)
        v_mv.setSpacing(8)

        lbl_mv_hint = QLabel("💡 Upload still photos of the dress from Front, Back, Right, and Left angles. The neural engine extracts 3D SMPL pose, DensePose UV mapping, and synthesizes 100+ training pairs automatically.")
        lbl_mv_hint.setStyleSheet("color: #94a3b8; font-size: 11px;")
        lbl_mv_hint.setWordWrap(True)
        v_mv.addWidget(lbl_mv_hint)

        h_cards = QHBoxLayout()
        h_cards.setSpacing(8)

        view_specs = [
            ("front", "👗 FRONT (Required)", "#00ffaa"),
            ("back", "🔄 BACK (Recommended)", "#38bdf8"),
            ("right", "➡️ RIGHT SIDE", "#a78bfa"),
            ("left", "⬅️ LEFT SIDE", "#f472b6")
        ]

        for key, title, col in view_specs:
            card = QFrame()
            card.setStyleSheet("""
                QFrame {
                    background: #111420;
                    border: 1px solid #1e2438;
                    border-radius: 6px;
                }
            """)
            c_lay = QVBoxLayout(card)
            c_lay.setContentsMargins(6, 6, 6, 6)
            c_lay.setSpacing(6)

            c_lbl = QLabel(title)
            c_lbl.setStyleSheet(f"color: {col}; font-weight: bold; font-size: 11px;")
            c_lbl.setAlignment(Qt.AlignCenter)
            c_lay.addWidget(c_lbl)

            lbl_prev = QLabel(f"No {key.title()}\nPhoto")
            lbl_prev.setAlignment(Qt.AlignCenter)
            lbl_prev.setStyleSheet("background: #090b12; border: 1px dashed #263048; border-radius: 4px; color: #64748b; font-size: 10px; min-height: 110px;")
            c_lay.addWidget(lbl_prev)
            self.view_preview_labels[key] = lbl_prev

            btn_b = QPushButton("Browse...")
            btn_b.setStyleSheet("background: #1e2538; color: #e2e8f0; font-size: 11px; padding: 5px;")
            btn_b.clicked.connect(lambda _, k=key: self._browse_single_view(k))
            c_lay.addWidget(btn_b)

            h_cards.addWidget(card)

        v_mv.addLayout(h_cards)

        h_quick = QHBoxLayout()
        btn_quick_folder = QPushButton("📁 Or Select Folder with front/back/right/left images")
        btn_quick_folder.setStyleSheet("background: #161b2b; color: #38bdf8; font-size: 11px; padding: 6px; border: 1px solid #28334f;")
        btn_quick_folder.clicked.connect(self._browse_multiview_folder)
        h_quick.addWidget(btn_quick_folder)

        self.lbl_mv_status = QLabel("⚠️ 0/4 Views Loaded (FRONT view required to train)")
        self.lbl_mv_status.setStyleSheet("color: #f59e0b; font-weight: bold; font-size: 11px;")
        h_quick.addWidget(self.lbl_mv_status)
        v_mv.addLayout(h_quick)

        v_inputs.addWidget(self.container_multiview)

        # Container for Video
        self.container_video = QWidget()
        v_vid = QVBoxLayout(self.container_video)
        v_vid.setContentsMargins(0, 4, 0, 4)
        f_vid = QFormLayout()

        h_src = QHBoxLayout()
        self.txt_src_path = QLineEdit()
        self.txt_src_path.setPlaceholderText("Select video file (.mp4, .mov) or folder of dress photos...")
        btn_browse_src = QPushButton("Browse...")
        btn_browse_src.clicked.connect(self._browse_train_source)
        h_src.addWidget(self.txt_src_path)
        h_src.addWidget(btn_browse_src)
        f_vid.addRow("Training Video / Images:", h_src)

        self.chk_auto_mask = QCheckBox("Automatically segment dress masks using 3D pose & DensePose (Recommended)")
        self.chk_auto_mask.setChecked(True)
        f_vid.addRow("", self.chk_auto_mask)

        h_mask = QHBoxLayout()
        self.txt_mask_path = QLineEdit()
        self.txt_mask_path.setPlaceholderText("Optional: Select mask directory (.png alpha masks)...")
        self.txt_mask_path.setEnabled(False)
        self.chk_auto_mask.toggled.connect(lambda c: self.txt_mask_path.setEnabled(not c))
        btn_browse_mask = QPushButton("Browse...")
        btn_browse_mask.clicked.connect(self._browse_train_mask)
        h_mask.addWidget(self.txt_mask_path)
        h_mask.addWidget(btn_browse_mask)
        f_vid.addRow("Custom Masks Folder:", h_mask)
        v_vid.addLayout(f_vid)

        self.container_video.hide()
        v_inputs.addWidget(self.container_video)

        l.addWidget(group_inputs)

        group_hyper = QGroupBox("2. Fast RTX 2050 Neural Training Configuration")
        v_hyper = QVBoxLayout(group_hyper)

        # Speed Mode Preset selector
        h_preset = QHBoxLayout()
        h_preset.addWidget(QLabel("⚡ Speed Preset:"))
        self.combo_preset = QComboBox()
        self.combo_preset.addItems([
            "⚡ Turbo Mode (1–2 Mins) [Default - Recommended]",
            "🎯 Balanced Mode (4–5 Mins)",
            "💎 Deep Studio Mode (30–40 Mins)"
        ])
        self.combo_preset.currentIndexChanged.connect(self._on_preset_changed)
        h_preset.addWidget(self.combo_preset, 1)
        v_hyper.addLayout(h_preset)

        self.lbl_preset_desc = QLabel("⚡ Turbo Mode: Fine-tunes high-resolution pretrained base in 1–2 minutes using RTX 2050 Tensor Cores.")
        self.lbl_preset_desc.setStyleSheet("color: #00ffaa; font-size: 11px;")
        v_hyper.addWidget(self.lbl_preset_desc)

        h_opts = QHBoxLayout()
        h_opts.addWidget(QLabel("Epochs:"))
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 100)
        self.spin_epochs.setValue(3)
        h_opts.addWidget(self.spin_epochs)

        h_opts.addWidget(QLabel("Batch Size:"))
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 4)
        self.spin_batch.setValue(2)
        h_opts.addWidget(self.spin_batch)

        h_opts.addWidget(QLabel("Resolution:"))
        self.combo_size = QComboBox()
        self.combo_size.addItems(["512x512", "576x576"])
        h_opts.addWidget(self.combo_size)

        v_hyper.addLayout(h_opts)
        l.addWidget(group_hyper)

        group_prog = QGroupBox("3. Training Progress & Live Console")
        v_prog = QVBoxLayout(group_prog)
        self.pbar = QProgressBar()
        self.pbar.setValue(0)
        self.lbl_status = QLabel("Ready. Select training media and click Start.")
        self.lbl_status.setStyleSheet("color: #38bdf8;")
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFixedHeight(130)

        v_prog.addWidget(self.pbar)
        v_prog.addWidget(self.lbl_status)
        v_prog.addWidget(self.console)
        l.addWidget(group_prog)

        h_btns = QHBoxLayout()
        self.btn_start_train = QPushButton("🚀 Start Dataset Generation & Training")
        self.btn_start_train.setStyleSheet("background: #00aa66; font-size: 13px; padding: 10px;")
        self.btn_start_train.clicked.connect(self._start_training)

        self.btn_stop_train = QPushButton("⏹️ Cancel")
        self.btn_stop_train.setStyleSheet("background: #ef4444; font-size: 13px; padding: 10px;")
        self.btn_stop_train.setEnabled(False)
        self.btn_stop_train.clicked.connect(self._stop_training)

        h_btns.addWidget(self.btn_start_train)
        h_btns.addWidget(self.btn_stop_train)
        l.addLayout(h_btns)

        self.tabs.addTab(tab, "🚀 Train Custom Garment")

    def _build_import_tab(self):
        tab = QWidget()
        l = QVBoxLayout(tab)
        l.setSpacing(14)

        group = QGroupBox("Import Pre-Trained Model Checkpoint (.pth)")
        form = QFormLayout(group)

        self.txt_import_name = QLineEdit("my_imported_dress")
        form.addRow("Garment Display Name:", self.txt_import_name)

        h_model = QHBoxLayout()
        self.txt_import_model = QLineEdit()
        self.txt_import_model.setPlaceholderText("Select latest_net_G.pth or model folder...")
        btn_browse_model = QPushButton("Browse...")
        btn_browse_model.clicked.connect(self._browse_import_model)
        h_model.addWidget(self.txt_import_model)
        h_model.addWidget(btn_browse_model)
        form.addRow("Model Checkpoint (.pth):", h_model)

        h_thumb = QHBoxLayout()
        self.txt_import_thumb = QLineEdit()
        self.txt_import_thumb.setPlaceholderText("Optional: Select preview thumbnail image (.jpg, .png)...")
        btn_browse_thumb = QPushButton("Browse...")
        btn_browse_thumb.clicked.connect(self._browse_import_thumb)
        h_thumb.addWidget(self.txt_import_thumb)
        h_thumb.addWidget(btn_browse_thumb)
        form.addRow("Thumbnail Preview:", h_thumb)

        l.addWidget(group)

        lbl_info = QLabel("💡 Imported checkpoints will be copied into rtv_ckpts/ and added to the wardrobe immediately.")
        lbl_info.setStyleSheet("color: #94a3b8; font-style: italic;")
        l.addWidget(lbl_info)

        l.addStretch()

        btn_deploy = QPushButton("📥 Import & Deploy into Wardrobe")
        btn_deploy.setStyleSheet("background: #0088ff; font-size: 13px; padding: 12px;")
        btn_deploy.clicked.connect(self._import_checkpoint)
        l.addWidget(btn_deploy)

        self.tabs.addTab(tab, "📥 Import Checkpoint (.pth)")

    def _build_test_tab(self):
        tab = QWidget()
        l = QVBoxLayout(tab)
        l.setSpacing(10)

        group_ctrl = QGroupBox("Select Photo & Garment to Test")
        form = QFormLayout(group_ctrl)

        h_photo = QHBoxLayout()
        self.txt_test_photo = QLineEdit()
        self.txt_test_photo.setPlaceholderText("Select photo of person to try-on (.jpg, .png)...")
        btn_browse_photo = QPushButton("Browse...")
        btn_browse_photo.clicked.connect(self._browse_test_photo)
        h_photo.addWidget(self.txt_test_photo)
        h_photo.addWidget(btn_browse_photo)
        form.addRow("Person Photo:", h_photo)

        h_gar = QHBoxLayout()
        self.combo_test_garment = QComboBox()
        self._refresh_test_garments()
        btn_refresh_list = QPushButton("🔄 Refresh")
        btn_refresh_list.clicked.connect(self._refresh_test_garments)
        h_gar.addWidget(self.combo_test_garment, 1)
        h_gar.addWidget(btn_refresh_list)
        form.addRow("Target Garment:", h_gar)

        l.addWidget(group_ctrl)

        btn_run_test = QPushButton("✨ Apply Virtual Try-On Test")
        btn_run_test.setStyleSheet("background: #00aa66; font-size: 13px; padding: 10px;")
        btn_run_test.clicked.connect(self._run_single_test)
        l.addWidget(btn_run_test)

        # Side-by-side preview viewports
        h_views = QHBoxLayout()
        self.lbl_orig_view = QLabel("Original Image")
        self.lbl_orig_view.setAlignment(Qt.AlignCenter)
        self.lbl_orig_view.setStyleSheet("background: #121520; border: 1px solid #1f2438; border-radius: 6px; min-height: 280px;")

        self.lbl_tryon_view = QLabel("Virtual Try-On Result")
        self.lbl_tryon_view.setAlignment(Qt.AlignCenter)
        self.lbl_tryon_view.setStyleSheet("background: #121520; border: 1px solid #00aa66; border-radius: 6px; min-height: 280px;")

        h_views.addWidget(self.lbl_orig_view)
        h_views.addWidget(self.lbl_tryon_view)
        l.addLayout(h_views)

        self.btn_save_test = QPushButton("💾 Save Test Result Image")
        self.btn_save_test.setEnabled(False)
        self.btn_save_test.clicked.connect(self._save_test_result)
        l.addWidget(self.btn_save_test)

        self.last_test_result = None
        self.tabs.addTab(tab, "🧪 Test Studio (Static Photo Try-On)")

    def _toggle_train_mode(self):
        is_mv = self.radio_multiview.isChecked()
        self.container_multiview.setVisible(is_mv)
        self.container_video.setVisible(not is_mv)

    def _browse_single_view(self, key):
        f, _ = QFileDialog.getOpenFileName(self, f"Select {key.upper()} View Photo of Dress", "", "Images (*.jpg *.png *.jpeg *.bmp);;All Files (*)")
        if f:
            self.photo_views[key] = f
            pix = QPixmap(f)
            if not pix.isNull():
                lbl = self.view_preview_labels.get(key)
                if lbl:
                    lbl.setPixmap(pix.scaled(lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    lbl.setToolTip(f)
            self._update_multiview_status()
            print(f"[RTV Custom Studio] Loaded {key} view: {f}")

    def _browse_multiview_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select Folder Containing Front/Back/Right/Left Images")
        if not d:
            return
        matched = []
        for fname in os.listdir(d):
            low = fname.lower()
            if not low.endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                continue
            fpath = os.path.join(d, fname)
            for key in ["front", "back", "right", "left"]:
                if key in low and self.photo_views[key] is None:
                    self.photo_views[key] = fpath
                    pix = QPixmap(fpath)
                    if not pix.isNull():
                        lbl = self.view_preview_labels.get(key)
                        if lbl:
                            lbl.setPixmap(pix.scaled(lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                            lbl.setToolTip(fpath)
                    matched.append(f"{key.upper()}: {fname}")
                    break
        self._update_multiview_status()
        if matched:
            print(f"[RTV Custom Studio] Auto-detected views: {', '.join(matched)}")

    def _update_multiview_status(self):
        count = sum(1 for v in self.photo_views.values() if v)
        names = [k.upper() for k, v in self.photo_views.items() if v]
        if "FRONT" in names:
            self.lbl_mv_status.setText(f"✅ {count}/4 Views Loaded: {', '.join(names)}")
            self.lbl_mv_status.setStyleSheet("color: #00ffaa; font-weight: bold; font-size: 11px;")
        else:
            self.lbl_mv_status.setText(f"⚠️ {count}/4 Views Loaded (FRONT view required to train)")
            self.lbl_mv_status.setStyleSheet("color: #f59e0b; font-weight: bold; font-size: 11px;")

    def _browse_train_source(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*)")
        if not f:
            d = QFileDialog.getExistingDirectory(self, "Or Select Image Sequence Directory")
            if d: self.txt_src_path.setText(d)
        else:
            self.txt_src_path.setText(f)

    def _browse_train_mask(self):
        d = QFileDialog.getExistingDirectory(self, "Select Garment Masks Directory")
        if d: self.txt_mask_path.setText(d)

    def _browse_import_model(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Model Checkpoint", "", "PyTorch Checkpoints (*.pth);;All Files (*)")
        if f: self.txt_import_model.setText(f)

    def _browse_import_thumb(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Thumbnail Image", "", "Images (*.jpg *.png *.jpeg);;All Files (*)")
        if f: self.txt_import_thumb.setText(f)

    def _browse_test_photo(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Person Photo", "", "Images (*.jpg *.png *.jpeg);;All Files (*)")
        if f:
            self.txt_test_photo.setText(f)
            pix = QPixmap(f)
            if not pix.isNull():
                self.lbl_orig_view.setPixmap(pix.scaled(self.lbl_orig_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _refresh_test_garments(self):
        self.combo_test_garment.clear()
        ckpts_dir = os.path.abspath("./rtv_ckpts")
        if os.path.exists(ckpts_dir):
            for item in sorted(os.listdir(ckpts_dir)):
                if item.startswith("."):
                    continue
                model_path = os.path.join(ckpts_dir, item, "latest_net_G.pth")
                if os.path.isdir(os.path.join(ckpts_dir, item)) and os.path.exists(model_path):
                    self.combo_test_garment.addItem(item)

    def _on_preset_changed(self, idx):
        if idx == 0:  # Turbo Mode
            self.spin_epochs.setValue(3)
            self.lbl_preset_desc.setText("⚡ Turbo Mode: Fine-tunes high-resolution pretrained base in 1–2 minutes using RTX 2050 Tensor Cores.")
            self.lbl_preset_desc.setStyleSheet("color: #00ffaa; font-size: 11px;")
        elif idx == 1:  # Balanced Mode
            self.spin_epochs.setValue(6)
            self.lbl_preset_desc.setText("🎯 Balanced Mode: Transfer learning with 6 epochs (4–5 minutes).")
            self.lbl_preset_desc.setStyleSheet("color: #38bdf8; font-size: 11px;")
        else:  # Deep Studio Mode
            self.spin_epochs.setValue(15)
            self.lbl_preset_desc.setText("💎 Deep Studio Mode: Deep convergence with 15 epochs (30–40 minutes).")
            self.lbl_preset_desc.setStyleSheet("color: #f59e0b; font-size: 11px;")

    def _start_training(self):
        name = self.txt_train_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing Name", "Please enter a garment model name.")
            return

        epochs = self.spin_epochs.value()
        batch = self.spin_batch.value()
        size = 576 if "576" in self.combo_size.currentText() else 512

        idx = self.combo_preset.currentIndex()
        if idx == 0:
            mode = "turbo"
        elif idx == 1:
            mode = "balanced"
        else:
            mode = "deep"

        if self.radio_multiview.isChecked():
            if not self.photo_views.get("front") or not os.path.exists(self.photo_views["front"]):
                QMessageBox.warning(self, "Missing Front View", "Please upload at least the FRONT view photo of the dress.")
                return
            valid_views = {k: v for k, v in self.photo_views.items() if v and os.path.exists(v)}
            self.btn_start_train.setEnabled(False)
            self.btn_stop_train.setEnabled(True)
            self.console.clear()
            self.pbar.setValue(0)
            self.lbl_status.setText(f"Initializing 360° {mode.upper()} training pipeline for {len(valid_views)} views...")

            self.worker = TrainingWorker(
                self.trainer,
                source_path=None,
                mask_dir=None,
                garment_name=name,
                epochs=epochs,
                batch_size=batch,
                img_size=size,
                views_dict=valid_views,
                mode=mode
            )
        else:
            src = self.txt_src_path.text().strip()
            if not src or not os.path.exists(src):
                QMessageBox.warning(self, "Missing Source", "Please select a valid training video or image directory.")
                return
            mask_dir = None if self.chk_auto_mask.isChecked() else self.txt_mask_path.text().strip()
            self.btn_start_train.setEnabled(False)
            self.btn_stop_train.setEnabled(True)
            self.console.clear()
            self.pbar.setValue(0)
            self.lbl_status.setText(f"Initializing {mode.upper()} training worker...")

            self.worker = TrainingWorker(
                self.trainer,
                source_path=src,
                mask_dir=mask_dir,
                garment_name=name,
                epochs=epochs,
                batch_size=batch,
                img_size=size,
                views_dict=None,
                mode=mode
            )

        self.worker.progress_signal.connect(self._on_train_progress)
        self.worker.epoch_signal.connect(self._on_train_epoch)
        self.worker.finished_signal.connect(self._on_train_finished)
        self.worker.start()

    def _stop_training(self):
        if self.worker:
            self.worker.stop()
            self.lbl_status.setText("Stopping training...")

    def _on_train_progress(self, val, total, msg):
        self.pbar.setValue(val)
        self.lbl_status.setText(msg)
        self.console.append(msg)

    def _on_train_epoch(self, ep, tot_ep, st, loss, msg):
        self.console.append(f"==> {msg}")

    def _on_train_finished(self, success, msg):
        self.btn_start_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)
        self.lbl_status.setText(msg)
        self.console.append(f"\n[DONE] {msg}")
        if success:
            QMessageBox.information(self, "Training Complete", msg)
            name = self.txt_train_name.text().strip().replace(" ", "_").lower()
            self.garment_added_signal.emit(name)
            self._refresh_test_garments()
        else:
            QMessageBox.critical(self, "Training Notice", msg)

    def _import_checkpoint(self):
        src = self.txt_import_model.text().strip()
        name = self.txt_import_name.text().strip()
        thumb = self.txt_import_thumb.text().strip()
        if not src or not os.path.exists(src):
            QMessageBox.warning(self, "Missing Checkpoint", "Please select a valid .pth model file or folder.")
            return
        if not name:
            QMessageBox.warning(self, "Missing Name", "Please enter a garment name.")
            return

        try:
            clean_name, dest = self.trainer.import_checkpoint(src, name, preview_path=thumb if thumb else None)
            QMessageBox.information(self, "Import Successful", f"Garment '{clean_name}' successfully added to wardrobe!")
            self.garment_added_signal.emit(clean_name)
            self._refresh_test_garments()
        except Exception as e:
            QMessageBox.critical(self, "Import Error", str(e))

    def _run_single_test(self):
        photo_path = self.txt_test_photo.text().strip()
        garment = self.combo_test_garment.currentText()
        if not photo_path or not os.path.exists(photo_path):
            QMessageBox.warning(self, "Missing Photo", "Please select an image file to test.")
            return
        if not garment:
            QMessageBox.warning(self, "Missing Garment", "Please select a garment.")
            return

        img = cv2.imread(photo_path)
        if img is None:
            QMessageBox.warning(self, "Read Error", "Failed to load test image.")
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            comp = None
            if self.stream_engine is not None and hasattr(self.stream_engine, 'ml_lock'):
                with self.stream_engine.ml_lock:
                    self.stream_engine.frame_processor.set_target_garment(garment)
                    patch = self.stream_engine.frame_processor.get_garment_patch(img)
                    if patch is not None:
                        comp = self.stream_engine.frame_processor.composite_garment(img, patch)
            elif self.stream_engine is not None:
                self.stream_engine.frame_processor.set_target_garment(garment)
                patch = self.stream_engine.frame_processor.get_garment_patch(img)
                if patch is not None:
                    comp = self.stream_engine.frame_processor.composite_garment(img, patch)
            else:
                proc = FullBodySeqFrameProcessor(garment)
                patch = proc.get_garment_patch(img)
                if patch is not None:
                    comp = proc.composite_garment(img, patch)

            if comp is None:
                comp = img.copy()

            self.last_test_result = comp
            self.btn_save_test.setEnabled(True)

            rgb = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
            h, w, c = rgb.shape
            qimg = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            self.lbl_tryon_view.setPixmap(pix.scaled(self.lbl_tryon_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            QMessageBox.critical(self, "Inference Notice", f"Test try-on encountered an error:\n{str(e)}")
        finally:
            QApplication.restoreOverrideCursor()

    def _save_test_result(self):
        if self.last_test_result is None:
            return
        dest, _ = QFileDialog.getSaveFileName(self, "Save Try-On Image", "tryon_test_result.jpg", "Images (*.jpg *.png)")
        if dest:
            cv2.imwrite(dest, self.last_test_result)
            QMessageBox.information(self, "Saved", f"Test result saved to: {dest}")


class CameraApp(QMainWindow):
    def __init__(self, source=None):
        super().__init__()
        self.setWindowTitle("Real-Time Virtual Try-On Mirror (High-Performance)")
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
        scroll_area.setFixedWidth(260)
        scroll_area.setStyleSheet("background-color: #15151e; border: none; border-radius: 6px;")

        container_list = QWidget()
        layout_list = QVBoxLayout(container_list)
        layout_list.setContentsMargins(6, 6, 6, 6)
        layout_list.setSpacing(8)

        # Studio Control Buttons
        self.btn_studio = QPushButton("👗 Custom Dress Studio")
        self.btn_studio.setStyleSheet("""
            QPushButton { background: #00aa66; color: #fff; font-weight: bold; font-size: 12px; padding: 10px; border-radius: 6px; }
            QPushButton:hover { background: #00cc7a; }
        """)
        self.btn_studio.clicked.connect(self.open_custom_studio)
        layout_list.addWidget(self.btn_studio)

        self.btn_test_studio = QPushButton("🧪 Test on Photo / Video")
        self.btn_test_studio.setStyleSheet("""
            QPushButton { background: #1f2438; color: #38bdf8; font-weight: bold; font-size: 11px; padding: 7px; border-radius: 5px; border: 1px solid #2d354d; }
            QPushButton:hover { background: #2a324d; }
        """)
        self.btn_test_studio.clicked.connect(self.open_test_studio)
        layout_list.addWidget(self.btn_test_studio)

        title_garment = QLabel("VIRTUAL WARDROBE")
        title_garment.setAlignment(Qt.AlignCenter)
        title_garment.setStyleSheet("color: #00ffaa; font-weight: bold; font-size: 12px; margin-top: 6px;")
        layout_list.addWidget(title_garment)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SingleSelection)
        self.list_widget.setIconSize(QSize(140, 140))
        self.list_widget.setStyleSheet("""
            QListWidget { background: #1a1a24; border: 1px solid #282836; border-radius: 6px; color: #fff; }
            QListWidget::item { padding: 6px; border-bottom: 1px solid #252535; }
            QListWidget::item:selected { background: #005533; border: 1px solid #00ffaa; border-radius: 4px; }
        """)

        layout_list.addWidget(self.list_widget)
        self.list_widget.itemSelectionChanged.connect(self.on_selection_changed)
        self.list_widget.installEventFilter(self)

        scroll_area.setWidget(container_list)
        main_layout.addWidget(scroll_area)

        self.setCentralWidget(main_widget)
        self.setStyleSheet("background-color: #0a0a0f;")

        self.stream_engine = VitonFullbodyStreamEngine(target_name='coat_seq_vmssdp2ta_576', source=source)
        self.stream_engine.frameCaptured.connect(self.update_image)
        self.stream_engine.start()

        self.custom_dialog = None
        self.load_wardrobe(select_name="coat_seq_vmssdp2ta_576")

    def load_wardrobe(self, select_name=None):
        self.list_widget.clear()

        # Item 0: Raw Camera
        item0 = QListWidgetItem("0: Raw Camera (No Garment)")
        if os.path.exists('./assets/none.png'):
            item0.setIcon(QIcon('./assets/none.png'))
        item0.setData(Qt.UserRole, "none")
        self.list_widget.addItem(item0)

        # Scan rtv_ckpts/ directory dynamically
        ckpts_dir = os.path.abspath("./rtv_ckpts")
        discovered = []
        if os.path.exists(ckpts_dir):
            for item in sorted(os.listdir(ckpts_dir)):
                model_dir = os.path.join(ckpts_dir, item)
                if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, "latest_net_G.pth")):
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

                    # Format friendly label
                    clean_title = item.replace("_seq_vmssdp2ta_576", "").replace("_vmsdp2ta", "").replace("_", " ").title()
                    discovered.append((clean_title, item, thumb))

        selected_row = 1
        for idx, (label, model_id, thumb_path) in enumerate(discovered):
            w_item = QListWidgetItem(f"{idx+1}: {label}")
            if thumb_path and os.path.exists(thumb_path):
                w_item.setIcon(QIcon(thumb_path))
            w_item.setData(Qt.UserRole, model_id)
            self.list_widget.addItem(w_item)

            if select_name and select_name == model_id:
                selected_row = idx + 1

        if self.list_widget.count() > selected_row:
            self.list_widget.setCurrentRow(selected_row)

    def open_custom_studio(self):
        if self.custom_dialog is None:
            self.custom_dialog = CustomGarmentDialog(self.stream_engine, self)
            self.custom_dialog.garment_added_signal.connect(self.on_custom_garment_added)
        self.custom_dialog.tabs.setCurrentIndex(0)
        self.custom_dialog.show()
        self.custom_dialog.raise_()
        self.custom_dialog.activateWindow()

    def open_test_studio(self):
        if self.custom_dialog is None:
            self.custom_dialog = CustomGarmentDialog(self.stream_engine, self)
            self.custom_dialog.garment_added_signal.connect(self.on_custom_garment_added)
        self.custom_dialog.tabs.setCurrentIndex(2)
        self.custom_dialog.show()
        self.custom_dialog.raise_()
        self.custom_dialog.activateWindow()

    def on_custom_garment_added(self, garment_name):
        print(f"[RTV] Custom garment added: {garment_name}. Reloading wardrobe...")
        self.load_wardrobe(select_name=garment_name)
        self.stream_engine.set_target_garment(garment_name)

    def on_selection_changed(self):
        selected_items = self.list_widget.selectedItems()
        if selected_items:
            target_name = selected_items[0].data(Qt.UserRole)
            self.stream_engine.set_target_garment(target_name)

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
        elif key == Qt.Key_G:
            self.stream_engine.toggle_ar_motion()
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
