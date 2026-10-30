import numpy as np
import cv2


def get_mask_bounds(mask: np.ndarray):

    rows, cols = np.where(mask)
    if rows.size == 0 or cols.size == 0:
        return None
    return  cols.min(), cols.max(),rows.min(), rows.max(),


def mask2roi(mask:np.ndarray, new_h=1024, new_w=768,s=1.2,y_shift=0):
    height,width=mask.shape

    bbox = get_mask_bounds(mask)
    if bbox is None:
        return None, None
    x_min, x_max,y_min, y_max = bbox
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    if (y_max - y_min) / ((x_max - x_min) + 1e-5) > (new_h / new_w):  # Too tall
        half_y = (y_max - y_center) * s
        half_x = half_y * new_w / new_h
    else:
        half_x = (x_max - x_center) * s
        half_y = half_x * new_h / new_w
    y_center = y_center + (y_max - y_center) * y_shift


    src = np.zeros([3, 2], np.float32)
    center = np.array([x_center, y_center], np.float32)
    src[0, :] = center + np.array([-half_x, half_y], np.float32)
    src[1, :] = center + np.array([-half_x, -half_y], np.float32)
    src[2, :] = center + np.array([half_x, -half_y], np.float32)

    dst = np.zeros([3, 2], np.float32)
    dst[0, :] = np.array([0, new_h - 1], np.float32)
    dst[1, :] = np.array([0, 0], np.float32)
    dst[2, :] = np.array([new_w - 1, 0], np.float32)
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    inv_trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))

    return trans, inv_trans