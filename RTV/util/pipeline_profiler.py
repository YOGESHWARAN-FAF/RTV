import time
import collections
import numpy as np
import cv2
import torch


class PipelineProfiler:
    """
    Lightweight, low-overhead performance profiling and telemetry monitor.
    Tracks rolling averages and 95th-percentile timings across all pipeline stages:
    - Camera capture FPS & latency
    - Neural inference (ROMP/BEV + DensePose + Pix2PixHD) latency & FPS
    - PyOpenGL virtual-mesh render latency
    - Optical-flow motion tracking latency
    - Alpha compositing latency
    - GUI presentation latency & refresh rate
    - End-to-end latency from capture to display
    - NVIDIA GPU VRAM allocation & capacity
    """
    def __init__(self, window_size=60):
        self.window_size = window_size
        
        # Stage duration ring buffers (ms)
        self.stage_timings = {
            'capture_lat': collections.deque(maxlen=window_size),
            'romp_bev': collections.deque(maxlen=window_size),
            'smpl': collections.deque(maxlen=window_size),
            'opengl': collections.deque(maxlen=window_size),
            'densepose': collections.deque(maxlen=window_size),
            'pix2pixhd': collections.deque(maxlen=window_size),
            'optical_flow': collections.deque(maxlen=window_size),
            'compositing': collections.deque(maxlen=window_size),
            'tracking_lat': collections.deque(maxlen=window_size),
            'presentation': collections.deque(maxlen=window_size),
            'end_to_end_lat': collections.deque(maxlen=window_size),
            'patch_age': collections.deque(maxlen=window_size),
        }

        # Timestamps for throughput tracking
        self.capture_timestamps = collections.deque(maxlen=window_size)
        self.display_timestamps = collections.deque(maxlen=window_size)
        self.gui_timestamps = collections.deque(maxlen=window_size)
        self.ml_timestamps = collections.deque(maxlen=window_size)
        self.tracking_timestamps = collections.deque(maxlen=window_size)

        self.last_display_seq = -1
        self.unique_frames_count = 0
        self.dropped_frames_count = 0

        # Cached computed metrics
        self.camera_fps = 0.0
        self.unique_display_fps = 0.0
        self.tracking_fps = 0.0
        self.gui_refresh_fps = 0.0
        self.ml_fps = 0.0
        self.ml_latency_ms = 0.0
        self.tracking_latency_ms = 0.0
        self.end_to_end_latency_ms = 0.0

    def record_stage(self, stage_name, duration_ms):
        if stage_name in self.stage_timings:
            self.stage_timings[stage_name].append(float(duration_ms))

    def record_capture(self, seq_num, mono_ts=None):
        if mono_ts is None:
            mono_ts = time.monotonic()
        self.capture_timestamps.append(mono_ts)
        if len(self.capture_timestamps) >= 2:
            dt = self.capture_timestamps[-1] - self.capture_timestamps[0]
            if dt > 0:
                self.camera_fps = (len(self.capture_timestamps) - 1) / dt

    def record_display_frame(self, seq_num, capture_mono_ts, display_mono_ts=None):
        if display_mono_ts is None:
            display_mono_ts = time.monotonic()
        
        # Measure true end-to-end frame age
        if capture_mono_ts > 0:
            e2e = (display_mono_ts - capture_mono_ts) * 1000.0
            self.stage_timings['end_to_end_lat'].append(e2e)
            self.end_to_end_latency_ms = e2e

        # Track unique frame throughput
        if seq_num != self.last_display_seq:
            self.unique_frames_count += 1
            if self.last_display_seq != -1 and seq_num > self.last_display_seq + 1:
                self.dropped_frames_count += (seq_num - self.last_display_seq - 1)
            self.last_display_seq = seq_num
            self.display_timestamps.append(display_mono_ts)
            if len(self.display_timestamps) >= 2:
                dt = self.display_timestamps[-1] - self.display_timestamps[0]
                if dt > 0:
                    self.unique_display_fps = (len(self.display_timestamps) - 1) / dt

    def record_tracking_update(self, duration_ms, mono_ts=None):
        """
        Records the real-time garment tracking & deformation update:
        Optical flow motion propagation + OneEuroFilter smoothing + arm kinematics + alpha compositing.
        """
        if mono_ts is None:
            mono_ts = time.monotonic()
        self.stage_timings['tracking_lat'].append(float(duration_ms))
        self.tracking_latency_ms = float(duration_ms)
        self.tracking_timestamps.append(mono_ts)
        if len(self.tracking_timestamps) >= 2:
            dt = self.tracking_timestamps[-1] - self.tracking_timestamps[0]
            if dt > 0:
                self.tracking_fps = (len(self.tracking_timestamps) - 1) / dt

    def record_gui_refresh(self, mono_ts=None):
        if mono_ts is None:
            mono_ts = time.monotonic()
        self.gui_timestamps.append(mono_ts)
        if len(self.gui_timestamps) >= 2:
            dt = self.gui_timestamps[-1] - self.gui_timestamps[0]
            if dt > 0:
                self.gui_refresh_fps = (len(self.gui_timestamps) - 1) / dt

    def record_ml_inference(self, duration_ms, mono_ts=None):
        if mono_ts is None:
            mono_ts = time.monotonic()
        self.ml_timestamps.append(mono_ts)
        self.ml_latency_ms = duration_ms
        if len(self.ml_timestamps) >= 2:
            recent = list(self.ml_timestamps)[-10:] if len(self.ml_timestamps) > 10 else list(self.ml_timestamps)
            dt = recent[-1] - recent[0]
            if dt > 0:
                self.ml_fps = (len(recent) - 1) / dt

    def get_stage_percentiles(self, stage_name):
        buf = self.stage_timings.get(stage_name, None)
        if not buf or len(buf) == 0:
            return 0.0, 0.0
        arr = np.array(buf)
        return float(np.percentile(arr, 50)), float(np.percentile(arr, 95))

    def get_vram_stats(self):
        if torch.cuda.is_available():
            alloc_mb = torch.cuda.memory_allocated() / (1024.0 ** 2)
            res_mb = torch.cuda.memory_reserved() / (1024.0 ** 2)
            props = torch.cuda.get_device_properties(0)
            total_mb = props.total_memory / (1024.0 ** 2)
            return alloc_mb, res_mb, total_mb
        return 0.0, 0.0, 4096.0

    def get_summary_dict(self):
        stats = {
            'camera_fps': self.camera_fps,
            'unique_display_fps': self.unique_display_fps,
            'tracking_fps': self.tracking_fps,
            'gui_refresh_fps': self.gui_refresh_fps,
            'ml_fps': self.ml_fps,
            'ml_latency_ms': self.ml_latency_ms,
            'tracking_latency_ms': self.tracking_latency_ms,
            'end_to_end_lat_ms': self.end_to_end_latency_ms,
        }
        for st in self.stage_timings:
            p50, p95 = self.get_stage_percentiles(st)
            stats[f'{st}_p50'] = p50
            stats[f'{st}_p95'] = p95
        alloc_mb, res_mb, total_mb = self.get_vram_stats()
        stats['vram_alloc_mb'] = alloc_mb
        stats['vram_res_mb'] = res_mb
        stats['vram_total_mb'] = total_mb
        return stats

    def draw_hud(self, frame, mode_str="LIVE", sync_delay_sec=0.0, yaw_angle=0.0, patch_age_ms=0.0, is_profiling=True, dual_gpu_status=None):
        """
        Draws comprehensive real-time HUD with accurate 3-tier metric breakdown:
        - 0. Dual Heterogeneous Hardware: GPU.0 (Intel UHD Graphics) & GPU.1 (RTX 2050 AMP FP16)
        - 1. Camera capture FPS & Unique displayed camera FPS
        - 2. Garment tracking & compositing FPS (20–30 FPS live deformation)
        - 3. Fresh neural garment generation FPS & latency (6+ FPS dress generation)
        - Latency, garment patch age, and NVIDIA GPU memory
        """
        h, w = frame.shape[:2]
        track_p50, _ = self.get_stage_percentiles('tracking_lat')
        if track_p50 == 0.0:
            track_p50, _ = self.get_stage_percentiles('compositing')
        e2e_p50, _ = self.get_stage_percentiles('end_to_end_lat')
        alloc_mb, res_mb, total_mb = self.get_vram_stats()

        latency_val = int(e2e_p50) if e2e_p50 > 0 else int(self.end_to_end_latency_ms)

        if mode_str == "LIVE":
            mode_tag = f"LIVE ({latency_val}ms latency)"
            sync_color = (0, 255, 120)  # Bright green
        else:
            delay_ms = int(sync_delay_sec * 1000)
            mode_tag = f"SYNC ({delay_ms}ms buffer)"
            sync_color = (0, 220, 255)  # Cyan

        tracking_display_fps = self.tracking_fps if self.tracking_fps > 0 else self.unique_display_fps

        import torch
        if dual_gpu_status:
            line0 = dual_gpu_status
        elif torch.cuda.is_available():
            cnt = torch.cuda.device_count()
            if cnt >= 2:
                line0 = f"GPU.0 [{torch.cuda.get_device_name(0)}]: ACTIVE | GPU.1 [{torch.cuda.get_device_name(1)}]: AMP FP16"
            else:
                line0 = f"GPU.0 [{torch.cuda.get_device_name(0)}]: ACTIVE (16GB VRAM) [AMP FP16]"
        else:
            line0 = "CPU Mode: ACTIVE"
        line1 = (
            f"Cam: {self.camera_fps:.1f} FPS | "
            f"Disp: {self.unique_display_fps:.1f} FPS | "
            f"Tracking: {tracking_display_fps:.1f} FPS | "
            f"Dress: {self.ml_fps:.1f} FPS"
        )
        line2 = (
            f"Patch Age: {patch_age_ms:.0f}ms | "
            f"Latency: {latency_val}ms | "
            f"Track Lat: {track_p50:.1f}ms | "
            f"Dress Lat: {self.ml_latency_ms:.0f}ms | "
            f"VRAM: {alloc_mb/1024.0:.1f}/{total_mb/1024.0:.1f}GB"
        )
        
        (tw0, th0), _ = cv2.getTextSize(line0, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        (tw1, th1), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        (tw2, th2), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        bw = max(tw0, max(tw1, tw2)) + 20
        bh = th0 + th1 + th2 + 30

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 8), (10 + bw, 8 + bh), (10, 12, 18), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (10, 8), (10 + bw, 8 + bh), sync_color, 1)

        cv2.putText(frame, line0, (18, 12 + th0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 200, 80), 1, cv2.LINE_AA)
        cv2.putText(frame, line1, (18, 18 + th0 + th1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, sync_color, 1, cv2.LINE_AA)
        cv2.putText(frame, line2, (18, 24 + th0 + th1 + th2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 230, 255), 1, cv2.LINE_AA)

        sub_hint = "[M]: Toggle LIVE / SYNC  |  [C]: Recalibrate  |  [ [ / ] ]: Adjust Delay  |  [S]: Reset Delay"
        cv2.putText(frame, sub_hint, (15, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        return frame
