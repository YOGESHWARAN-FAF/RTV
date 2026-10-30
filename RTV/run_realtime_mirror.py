#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import threading
import queue
import json
import re
import inspect
import asyncio
import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import gradio as gr
import uvicorn

RTV_DIR = os.path.abspath(os.path.dirname(__file__)) if os.path.exists("OffscreenRenderer") else "/kaggle/working/RTV"
if os.path.exists(RTV_DIR):
    os.chdir(RTV_DIR)
    if RTV_DIR not in sys.path:
        sys.path.insert(0, RTV_DIR)

# 1. Apply Mesa & Linux GLSL Shader fixes
try:
    rf = os.path.join(RTV_DIR, "OffscreenRenderer/off_screen_render.py")
    if os.path.exists(rf):
        with open(rf, "r", encoding="utf-8") as f:
            c = f.read()
        if "np.int8" in c:
            with open(rf, "w", encoding="utf-8") as f:
                f.write(c.replace("np.int8", "np.uint8"))

    for sf in ["OffscreenRenderer/shaders.py", "OffscreenRenderer/uv_shaders.py"]:
        sp = os.path.join(RTV_DIR, sf)
        if os.path.exists(sp):
            with open(sp, "r", encoding="utf-8") as f:
                sc = f.read()
            if "out float gl_FragDepth;" in sc:
                with open(sp, "w", encoding="utf-8") as f:
                    f.write(sc.replace("out float gl_FragDepth;", "// out float gl_FragDepth;"))

    import shutil
    rd = os.path.expanduser("~/.romp")
    if os.path.exists(rd):
        for s, d in [("smpla_packed_info.pth", "SMPLA_NEUTRAL.pth"), ("smpl_packed_info.pth", "SMPL_NEUTRAL.pth")]:
            src_f, dst_f = os.path.join(rd, s), os.path.join(rd, d)
            if os.path.exists(src_f) and not os.path.exists(dst_f):
                shutil.copy2(src_f, dst_f)
except Exception as e:
    print(f"[RTV] Setup notice: {e}")

# GPU Configuration: Check for Dual Tesla T4 (T4 x 2)
DEV_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
if DEV_COUNT >= 2:
    GPU_SUMMARY = f"Dual GPU (T4x2): GPU 0 ({torch.cuda.get_device_name(0)}) + GPU 1 ({torch.cuda.get_device_name(1)}) [AMP FP16]"
elif DEV_COUNT == 1:
    GPU_SUMMARY = f"Single GPU: {torch.cuda.get_device_name(0)} (16GB VRAM) [AMP FP16]"
else:
    GPU_SUMMARY = "CPU Mode (CUDA not active)"

from colab_gradio_app import ColabTryOnEngine, GARMENT_CATALOG, launch_cloud_tunnel

engine = ColabTryOnEngine()

def process_live_frame_bgr(frame_bgr, garment_label=None, show_hud=True):
    if frame_bgr is None:
        return None
    orig_h, orig_w = frame_bgr.shape[:2]
    # 720p HD processing standard: 640px width for fast 30 FPS network streaming & pristine clarity
    target_w = 640
    if orig_w > target_w:
        scale = target_w / float(orig_w)
        target_h = int(orig_h * scale)
        frame_bgr = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    if garment_label and garment_label != engine.active_garment_label:
        engine.set_active_garment(garment_label)

    with engine.lock:
        engine.latest_input_bgr = frame_bgr

    entry = engine.active_catalog_entry
    g_type = entry['type']
    g_id = entry['id']
    display_bgr = frame_bgr.copy()
    patch_age_ms = 0.0

    if g_type != "none":
        patch = None
        with engine.lock:
            patch = engine.latest_patch
        if patch is not None:
            patch_age_ms = max(0.0, (time.time() - patch.get('timestamp', time.time())) * 1000.0)
            try:
                if g_type == "fullbody":
                    proc = engine.get_fullbody_processor(g_id)
                    display_bgr = proc.composite_garment(display_bgr, patch)
                else:
                    proc = engine.get_upperbody_processor(g_id)
                    display_bgr = proc.composite_garment(display_bgr, patch)
            except Exception:
                pass
        else:
            h_ov, w_ov = display_bgr.shape[:2]
            cv2.rectangle(display_bgr, (0, 0), (w_ov, 45), (10, 12, 18), -1)
            hud_msg = "[ INITIALIZING 3D MODEL... STAND IN CAMERA VIEW ]" if engine.model_status != "Ready" else "[ SEARCHING FOR PERSON: Step back so torso is in view ]"
            cv2.putText(display_bgr, hud_msg, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1, cv2.LINE_AA)

    if show_hud:
        display_bgr = engine.profiler.draw_hud(
            display_bgr,
            mode_str="LIVE",
            sync_delay_sec=2.0,
            yaw_angle=engine.latest_yaw,
            patch_age_ms=patch_age_ms
        )
    return display_bgr

fastapi_app = FastAPI(title="RTV Real-Time Virtual Try-On")

@fastapi_app.websocket("/api/live_mirror")
async def live_mirror_ws(websocket: WebSocket):
    await websocket.accept()
    current_garment = [engine.active_garment_label]
    show_hud = [True]

    in_queue = queue.Queue(maxsize=120)
    out_queue = queue.Queue(maxsize=120)
    stop_event = threading.Event()

    def _gpu_worker_thread():
        """Dedicated GPU worker running at maximum pipeline throughput."""
        while not stop_event.is_set():
            try:
                raw_bytes = in_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                np_arr = np.frombuffer(raw_bytes, np.uint8)
                frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame_bgr is not None:
                    out_bgr = process_live_frame_bgr(frame_bgr, garment_label=current_garment[0], show_hud=show_hud[0])
                    success, enc = cv2.imencode(".jpg", out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
                    if success:
                        if out_queue.full():
                            try:
                                out_queue.get_nowait()
                            except queue.Empty:
                                pass
                        out_queue.put(enc.tobytes())
            except Exception as ex:
                print(f"[RTV] GPU worker error: {ex}")

    worker_th = threading.Thread(target=_gpu_worker_thread, daemon=True)
    worker_th.start()

    async def _sender():
        while not stop_event.is_set():
            try:
                try:
                    enc_bytes = out_queue.get_nowait()
                except queue.Empty:
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
                    except queue.Empty:
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

garment_choices = list(GARMENT_CATALOG.keys())
garment_options_html = "".join([f'<option value="{k}">{k}</option>' for k in garment_choices])

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

HTML_TEMPLATE = f"""
<div class="mirror-container" style="background:#0b0f19; border:1px solid #1e293b; border-radius:12px; padding:18px; margin-bottom:18px; box-shadow:0 8px 30px rgba(0,0,0,0.5); font-family:system-ui,-apple-system,sans-serif;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; border-bottom:1px solid #1e293b; padding-bottom:10px; flex-wrap:wrap; gap:10px;">
        <div>
            <span style="font-size:1.25rem; font-weight:700; color:#00ffaa; letter-spacing:0.5px;">⚡ High-Speed Real-Time Mirror</span>
            <span style="color:#94a3b8; font-size:0.85rem; margin-left:10px;">Ultra-Smooth 30 FPS Pipelined Engine (HD 720p - 2.0s Jitter Buffer)</span>
        </div>
        <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
            <span id="rtv_ws_status" style="background:rgba(148,163,184,0.15); color:#94a3b8; border:1px solid #334155; padding:4px 12px; border-radius:20px; font-size:0.85rem; font-weight:600;">⚪ Standby</span>
            <span id="rtv_fps_badge" style="background:rgba(0,255,170,0.12); color:#00ffaa; border:1px solid rgba(0,255,170,0.3); padding:4px 12px; border-radius:20px; font-size:0.85rem; font-weight:600;">⚡ Mirror: 0.0 FPS</span>
            <span id="rtv_rtt_badge" style="background:rgba(56,189,248,0.12); color:#38bdf8; border:1px solid rgba(56,189,248,0.3); padding:4px 12px; border-radius:20px; font-size:0.85rem; font-weight:600;">📶 Latency: -- ms</span>
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
                <span style="color:#00ffaa; font-size:0.8rem;">Kaggle GPU T4x2 Live</span>
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
                    __GARMENTS__
                </select>
            </div>
            <label style="color:#cbd5e1; font-size:0.88rem; cursor:pointer; display:flex; align-items:center; gap:6px;">
                <input type="checkbox" id="rtv_ws_hud" checked onchange="if(window.rtvSendConfig)window.rtvSendConfig();" style="cursor:pointer; accent-color:#00ffaa;">
                Show Diagnostic HUD
            </label>
        </div>
    </div>
</div>
"""

realtime_mirror_html = HTML_TEMPLATE.replace("__GARMENTS__", garment_options_html) + f"<script>\n{RTV_LIVE_MIRROR_JS}\n</script>"

custom_css = """
body { background-color: #0a0b10 !important; color: #e2e8f0 !important; }
.header-box { background: linear-gradient(135deg, #10121d 0%, #171b2e 100%); border: 1px solid #232946; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
.header-title { font-size: 2rem; font-weight: 800; background: linear-gradient(90deg, #00ffaa, #00d2ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.status-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; background: rgba(0, 255, 170, 0.12); color: #00ffaa; border: 1px solid rgba(0, 255, 170, 0.3); margin-right: 8px; }
"""

with gr.Blocks(title="RTV Real-Time Virtual Try-On", css=custom_css, theme=gr.themes.Monochrome()) as demo:
    with gr.Row():
        with gr.Column(elem_classes=["header-box"]):
            gr.Markdown(f"""
            <div class="header-title">RTV : Real-Time Virtual Try-On Studio</div>
            <div style="margin-top: 8px;">
                <span class="status-badge">⚡ {GPU_SUMMARY}</span>
                <span class="status-badge">🚀 30 FPS Pipelined Mirror (2.0s Jitter Buffer)</span>
                <span class="status-badge">🎯 720p HD Graphics</span>
            </div>
            """)

    with gr.Tabs():
        with gr.TabItem("🎥 Live Webcam Try-On"):
            gr.HTML(realtime_mirror_html)
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

        with gr.TabItem("🎞️ Video File Try-On"):
            with gr.Row():
                with gr.Column(scale=4):
                    sample_video_path = "./assets/sample_video.mp4" if os.path.exists("./assets/sample_video.mp4") else None
                    video_upload = gr.Video(value=sample_video_path, label="Source Video (Sample Video Pre-Loaded)", height=340)
                    video_garment_select = gr.Dropdown(choices=garment_choices, value="Winter Puffer Coat (Full-Body)", label="Garment Choice")
                    max_frames_slider = gr.Slider(minimum=30, maximum=1200, value=120, step=30, label="Max Frames to Process (up to 1200 frames)")
                    video_hud_toggle = gr.Checkbox(value=True, label="Render HUD Banner on Output Video")
                    process_video_btn = gr.Button("⚡ Render Virtual Try-On Video", variant="primary")
                with gr.Column(scale=4):
                    video_output = gr.Video(label="Rendered Try-On Video", height=340)
                    video_status = gr.Markdown("Ready to process video.")
            process_video_btn.click(
                fn=engine.process_video_file,
                inputs=[video_upload, video_garment_select, max_frames_slider, video_hud_toggle],
                outputs=[video_output, video_status]
            )

        with gr.TabItem("📸 Single Photo Studio"):
            with gr.Row():
                with gr.Column(scale=4):
                    sample_img_path = "./assets/sample_frame.jpg" if os.path.exists("./assets/sample_frame.jpg") else None
                    photo_upload = gr.Image(value=sample_img_path, type="numpy", label="Upload Person Photo", height=340)
                    photo_garment_select = gr.Dropdown(choices=garment_choices, value="Winter Puffer Coat (Full-Body)", label="Target Garment")
                    photo_hud_toggle = gr.Checkbox(value=True, label="Show Diagnostic HUD")
                    render_photo_btn = gr.Button("✨ Render Virtual Try-On", variant="primary")
                with gr.Column(scale=4):
                    photo_result = gr.Image(type="numpy", label="Virtual Try-On Result", height=340)
                    photo_status = gr.Markdown("Ready to test.")
            render_photo_btn.click(
                fn=lambda img, g, h: (engine.process_single_image(img, g, h, False)[0], "Complete"),
                inputs=[photo_upload, photo_garment_select, photo_hud_toggle],
                outputs=[photo_result, photo_status]
            )

demo.queue()

mount_kwargs = {
    "path": "/",
    "head": f"<script>\n{RTV_LIVE_MIRROR_JS}\n</script>"
}
sig_params = inspect.signature(gr.mount_gradio_app).parameters
filtered_kwargs = {k: v for k, v in mount_kwargs.items() if k in sig_params}
app = gr.mount_gradio_app(fastapi_app, demo, **filtered_kwargs)

if __name__ == "__main__":
    launch_cloud_tunnel(7860)
    print("\n" + "=" * 70)
    print("  RTV Real-Time Virtual Try-On Studio (ASGI WebSocket 30 FPS Mode)")
    print("=" * 70)
    print(f"  {GPU_SUMMARY}")
    print("  Port: 7860")
    print("=" * 70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="warning")
