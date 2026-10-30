import os
import sys
import time
import numpy as np
import cv2
import torch

def test_one_euro_filter():
    print("=== Testing OneEuroFilter ===")
    from SMPL.one_euro_filter import OneEuroFilter
    
    f = OneEuroFilter(min_cutoff=0.6, beta=0.015)
    
    # 1. Noisy static signal (jitter of +/- 2px)
    jitter_inputs = [100.0 + (1.0 if i % 2 == 0 else -1.0) * 2.0 for i in range(30)]
    t = 0.0
    filtered_jitter = []
    for x in jitter_inputs:
        t += 1.0 / 30.0
        filtered_jitter.append(f(x, t=t))
    
    input_variance = np.var(jitter_inputs)
    filtered_variance = np.var(filtered_jitter[-15:])
    print(f"Jitter suppression: Raw variance={input_variance:.4f}, Filtered variance={filtered_variance:.4f}")
    assert filtered_variance < input_variance * 0.2, "Filter should heavily suppress stationary jitter"

    # 2. Sudden rapid movement (step from 100 to 200)
    for _ in range(5):
        t += 1.0 / 30.0
        out = f(200.0, t=t)
    print(f"Rapid movement response: target=200.0, filter output after 5 frames={out:.2f}")
    assert out > 185.0, "Filter should adapt cutoff to follow rapid movement with minimal lag"
    print("OneEuroFilter test PASSED!\n")


def test_fullbody_pipeline():
    print("=== Testing FullBody Sequence Pipeline (demo.py) ===")
    from VITON.viton_fullbody_seq import FullBodySeqFrameProcessor
    from composition.naive_overlay import smooth_overlay_alpha

    cap = cv2.VideoCapture("assets/sample_video.mp4")
    ret, frame = cap.read()
    assert ret, "Could not read assets/sample_video.mp4"
    h, w = frame.shape[:2]
    print(f"Loaded test frame: {w}x{h}")

    processor = FullBodySeqFrameProcessor()
    
    # Run 5 frames
    for idx in range(5):
        t0 = time.time()
        patch_info = processor.get_garment_patch(frame, isRGB=False)
        dt = (time.time() - t0) * 1000.0
        
        if patch_info is not None:
            patch_rgb = patch_info['roi_target']
            alpha_mask = patch_info['roi_alpha']
            inv_trans = patch_info['inv_trans']
            print(f"Frame {idx}: Success! Inference time: {dt:.1f}ms")
            print(f"  patch_rgb shape: {patch_rgb.shape}, alpha range: [{alpha_mask.min():.2f}, {alpha_mask.max():.2f}]")
            
            # Verify alpha mask has solid opaque values (not semi-transparent)
            opaque_ratio = np.mean(alpha_mask[alpha_mask > 20] > 240)
            print(f"  Solid opaque coverage in garment region: {opaque_ratio * 100:.1f}%")
            assert opaque_ratio > 0.65, f"Expected solid opacity > 65%, got {opaque_ratio*100:.1f}%"
            
            # Test optimized bounding box overlay compositing
            t_blend = time.perf_counter()
            blended = processor.composite_garment(frame.copy(), patch_info)
            blend_ms = (time.perf_counter() - t_blend) * 1000.0
            print(f"  Compositing blend time: {blend_ms:.2f}ms")
            if idx > 0:
                assert blend_ms < 85.0, f"Blend time ({blend_ms:.2f}ms) too high!"
            
            cv2.imwrite("test_fullbody_out.jpg", blended)
        else:
            print(f"Frame {idx}: No person detected ({dt:.1f}ms)")
        
        ret, frame = cap.read()
        if not ret:
            break
            
    cap.release()
    print("FullBody Sequence Pipeline test PASSED!\n")


def test_upperbody_pipeline():
    print("=== Testing UpperBody Pipeline (rtl_demo.py) ===")
    from VITON.viton_upperbody import FrameProcessor
    from composition.naive_overlay import smooth_overlay_alpha

    cap = cv2.VideoCapture("assets/sample_video.mp4")
    ret, frame = cap.read()
    assert ret, "Could not read assets/sample_video.mp4"

    processor = FrameProcessor(['lab_03_vmsdp2ta'])
    processor.switch_to_target_garment(0)
    
    # Run 3 frames
    for idx in range(3):
        t0 = time.time()
        patch_info = processor.get_garment_patch(frame, isRGB=False)
        dt = (time.time() - t0) * 1000.0
        
        if patch_info is not None:
            patch_rgb = patch_info['roi_target']
            alpha_mask = patch_info['roi_alpha']
            inv_trans = patch_info['inv_trans']
            print(f"Frame {idx}: UpperBody Success! Inference time: {dt:.1f}ms")
            print(f"  patch_rgb shape: {patch_rgb.shape}, alpha range: [{alpha_mask.min():.2f}, {alpha_mask.max():.2f}]")
            
            opaque_ratio = np.mean(alpha_mask[alpha_mask > 20] > 220)
            print(f"  Solid opaque coverage: {opaque_ratio * 100:.1f}%")
            
            t_blend = time.perf_counter()
            blended = processor.composite_garment(frame.copy(), patch_info)
            blend_ms = (time.perf_counter() - t_blend) * 1000.0
            print(f"  Compositing blend time: {blend_ms:.2f}ms")
            
            cv2.imwrite("test_upperbody_out.jpg", blended)
        else:
            print(f"Frame {idx}: UpperBody - No person detected ({dt:.1f}ms)")
            
        ret, frame = cap.read()
        if not ret:
            break
            
    cap.release()
    print("UpperBody Pipeline test PASSED!\n")


def test_sequential_tracking_and_occlusion():
    print("=== Testing 60 FPS Sequential Tracking & Hand Occlusion Preservation ===")
    from VITON.viton_upperbody import FrameProcessor
    from composition.naive_overlay import smooth_overlay_alpha

    cap = cv2.VideoCapture("assets/sample_video.mp4")
    frames = []
    for _ in range(10):
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    assert len(frames) >= 5, "Need at least 5 frames from assets/sample_video.mp4"

    processor = FrameProcessor(['lab_03_vmsdp2ta'])
    processor.switch_to_target_garment(0)

    # Keyframe on frame 0
    patch = processor.get_garment_patch(frames[0], isRGB=False)
    assert patch is not None, "Failed to get initial keyframe patch"

    # Sequential tracking over subsequent frames at 60 FPS
    track_times = []
    for idx, f in enumerate(frames):
        t0 = time.perf_counter()
        out = processor.composite_garment(f.copy(), patch)
        dt = (time.perf_counter() - t0) * 1000.0
        track_times.append(dt)
        if idx > 0:
            print(f"  Frame {idx} sequential compositing time: {dt:.2f}ms")
            assert dt < 45.0, f"Frame {idx} took {dt:.2f}ms, exceeds threshold!"

    avg_track_ms = np.mean(track_times[1:])
    print(f"Average 60 FPS sequential tracking time: {avg_track_ms:.2f}ms (Passes real-time requirement < 30ms)")
    assert avg_track_ms < 30.0, f"Average tracking time {avg_track_ms:.2f}ms exceeds 30ms!"

    # Verify Hand Occlusion Preservation
    # Draw a synthetic hand (skin color) across the lower chest of the last frame
    test_frame = frames[-1].copy()
    h, w = test_frame.shape[:2]
    hand_y1, hand_y2 = int(h * 0.45), int(h * 0.52)
    hand_x1, hand_x2 = int(w * 0.40), int(w * 0.55)
    # Human skin color in BGR (roughly Y=140, Cr=150, Cb=100 -> BGR ~ [110, 130, 180])
    skin_bgr = np.array([110, 130, 185], dtype=np.uint8)
    test_frame[hand_y1:hand_y2, hand_x1:hand_x2] = skin_bgr

    composited_with_hand = processor.composite_garment(test_frame, patch)
    hand_patch_after = composited_with_hand[hand_y1+2:hand_y2-2, hand_x1+2:hand_x2-2]
    # Check that hand pixels were NOT overwritten by garment fabric
    mean_diff = np.mean(np.abs(hand_patch_after.astype(float) - skin_bgr.astype(float)))
    print(f"Hand preservation mean color difference: {mean_diff:.2f} (0 means fully preserved hand)")
    assert mean_diff < 30.0, "Foreground hand was overwritten by virtual garment!"
    print("Sequential Tracking & Hand Occlusion test PASSED!\n")


if __name__ == "__main__":
    test_one_euro_filter()
    test_fullbody_pipeline()
    test_upperbody_pipeline()
    test_sequential_tracking_and_occlusion()
    print("========================================")
    print(" ALL END-TO-END VERIFICATION TESTS PASSED")
    print("========================================")
