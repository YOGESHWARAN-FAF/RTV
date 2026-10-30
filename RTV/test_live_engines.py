import sys
import os
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from demo import VitonFullbodyStreamEngine
from rtl_demo import VitonStreamEngine


def sleep_process_events(sec):
    t_end = time.time() + sec
    while time.time() < t_end:
        QApplication.processEvents()
        time.sleep(0.01)

def test_fullbody_engine():
    print("=== Testing VitonFullbodyStreamEngine (demo.py) ===")
    engine = VitonFullbodyStreamEngine(
        target_name='coat_seq_vmssdp2ta_576',
        source='./assets/sample_video.mp4',
        delay_seconds=0.60
    )
    
    received_frames = []
    def on_frame(q_img):
        received_frames.append(time.time())

    engine.frameCaptured.connect(on_frame)
    engine.start()

    # Let it run for 4 seconds
    print("Engine started, capturing and running for 5 seconds...")
    sleep_process_events(5.0)

    # Force start try-on
    engine.force_start_tryon()
    sleep_process_events(4.0)

    # Toggle to SYNC mode and test
    engine.toggle_mode()
    sleep_process_events(3.0)

    # Toggle back to LIVE mode
    engine.toggle_mode()
    sleep_process_events(3.0)

    stats = engine.profiler.get_summary_dict()
    print("\nTelemetry Stats for VitonFullbodyStreamEngine:")
    print(f"  Camera FPS:             {stats['camera_fps']:.1f}")
    print(f"  Unique Display FPS:     {stats['unique_display_fps']:.1f}")
    print(f"  ML Latency:             {stats['ml_latency_ms']:.1f} ms ({stats['ml_fps']:.1f} FPS)")
    print(f"  Compositing p50:        {stats['compositing_p50']:.2f} ms")
    print(f"  Total frames displayed: {len(received_frames)}")
    print(f"  Mode:                   {engine.mode}")

    engine.stop()
    engine.wait()
    print("VitonFullbodyStreamEngine shutdown cleanly!\n")
    assert len(received_frames) > 20, f"Expected >20 frames, got {len(received_frames)}"


def test_upperbody_engine():
    print("=== Testing VitonStreamEngine (rtl_demo.py) ===")
    garments = ['lab_03_vmsdp2ta', 'lab_04_vmsdp2ta']
    engine = VitonStreamEngine(
        garment_id_list=garments,
        source='./assets/sample_video.mp4',
        delay_seconds=0.60
    )

    received_frames = []
    def on_frame(q_img):
        received_frames.append(time.time())

    engine.frameCaptured.connect(on_frame)
    engine.start()

    print("Upper-body engine running for 5 seconds...")
    sleep_process_events(5.0)

    engine.force_start_tryon()
    sleep_process_events(4.0)

    stats = engine.profiler.get_summary_dict()
    print("\nTelemetry Stats for VitonStreamEngine:")
    print(f"  Camera FPS:             {stats['camera_fps']:.1f}")
    print(f"  Unique Display FPS:     {stats['unique_display_fps']:.1f}")
    print(f"  ML Latency:             {stats['ml_latency_ms']:.1f} ms ({stats['ml_fps']:.1f} FPS)")
    print(f"  Compositing p50:        {stats['compositing_p50']:.2f} ms")
    print(f"  Total frames displayed: {len(received_frames)}")

    engine.stop()
    engine.wait()
    print("VitonStreamEngine shutdown cleanly!\n")
    assert len(received_frames) > 20, f"Expected >20 frames, got {len(received_frames)}"


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    test_fullbody_engine()
    test_upperbody_engine()
    print("=" * 60)
    print("ALL LIVE STREAM ENGINES VERIFIED SUCCESSFULLY!")
    print("=" * 60)
