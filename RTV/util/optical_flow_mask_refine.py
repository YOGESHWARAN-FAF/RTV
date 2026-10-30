import cv2
import numpy as np
from tqdm import tqdm
from util.multithread_video_loader import MultithreadVideoLoader
from Graphonomy.human_parser import HumanParser
def compute_confidence(mask):
    # Example confidence computation: distance from mask boundary
    dist_transform = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    max_dist = dist_transform.max()
    confidence = dist_transform / max_dist
    return confidence
def improve_masks_with_optical_flow(video_path, max_height,human_parser: HumanParser):
    video_loader = MultithreadVideoLoader(video_path,max_height)
    improved_masks = []
    frame0=video_loader.cap()
    prev_frame = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    improved_masks.append(human_parser.GetAtrGarmentMask(frame0,isRGB=False).astype(np.uint8))

    for i in tqdm(range(1, len(video_loader)),desc='Optical flow mask refinement'):
        frame_t = video_loader.cap()
        mask_t = human_parser.GetAtrGarmentMask(frame_t,isRGB=False)
        current_frame = cv2.cvtColor(frame_t, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow from prev_frame to current_frame
        flow = cv2.calcOpticalFlowFarneback(prev_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Generate remap grid
        h, w = flow.shape[:2]
        flow_map_x, flow_map_y = np.meshgrid(np.arange(w), np.arange(h))
        flow_map = np.array([flow_map_x + flow[..., 0], flow_map_y + flow[..., 1]], dtype=np.float32)

        # Warp the previous mask using the optical flow
        # Warp the previous mask using the optical flow
        warped_mask = cv2.remap(improved_masks[-1].astype(np.float32), flow_map[0], flow_map[1],
                                interpolation=cv2.INTER_LINEAR)
        warped_mask = (warped_mask > 0.5).astype(np.uint8)  # Threshold to get boolean mask

        # Compute confidence scores for both masks
        warped_confidence = compute_confidence(warped_mask)
        current_confidence = compute_confidence(mask_t.astype(np.uint8))

        # Combine masks based on confidence scores
        combined_mask = np.where(
            warped_confidence > current_confidence,
            warped_mask,
            mask_t.astype(np.uint8)
        )
        improved_masks.append(warped_mask.astype(bool))

        prev_frame = current_frame

    return improved_masks