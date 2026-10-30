import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from colab_gradio_app import ColabTryOnEngine, GARMENT_CATALOG

def test_engine():
    print("=== Testing ColabTryOnEngine ===")
    engine = ColabTryOnEngine()

    # 1. Test Single Image
    print("Testing process_single_image with test frame...")
    test_img = cv2.imread("assets/sample_video.mp4") # let's grab 1 frame from sample_video
    cap = cv2.VideoCapture("assets/sample_video.mp4")
    ret, frame_bgr = cap.read()
    cap.release()
    assert ret, "Could not read frame from assets/sample_video.mp4"
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    res_rgb, alpha_vis, roi_vis, status = engine.process_single_image(
        frame_rgb,
        garment_label="Winter Puffer Coat (Full-Body)",
        show_hud=True,
        show_debug=True
    )
    print(f"Single image result status: {status}")
    assert res_rgb is not None, "Failed to get single image try-on result"
    print(f"Result shape: {res_rgb.shape}")

    # 2. Test Stream frame processing
    print("Testing process_stream_frame...")
    stream_res = engine.process_stream_frame(
        frame_rgb,
        garment_label="Winter Puffer Coat (Full-Body)",
        tryon_mode="LIVE",
        show_hud=True,
        show_skeleton=True
    )
    assert stream_res is not None, "Failed to get stream frame result"
    print(f"Stream result shape: {stream_res.shape}")

    # 3. Test Video File Processing (5 frames)
    print("Testing process_video_file (5 frames)...")
    out_video, v_status = engine.process_video_file(
        "assets/sample_video.mp4",
        garment_label="Winter Puffer Coat (Full-Body)",
        max_frames=5,
        show_hud=True
    )
    print(f"Video result status: {v_status}")
    assert out_video is not None and os.path.exists(out_video), f"Output video not generated: {out_video}"
    print(f"Output video generated at: {out_video}")

    engine.worker_active = False
    print("\nALL COLAB APP ENGINE TESTS PASSED!")

if __name__ == "__main__":
    test_engine()
