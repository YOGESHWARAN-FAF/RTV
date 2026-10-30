import numpy as np

def decompose_affine(M):
    tx = float(M[0, 2])
    ty = float(M[1, 2])
    sx = float(np.hypot(M[0, 0], M[0, 1]))
    sy = float(np.hypot(M[1, 0], M[1, 1]))
    s = (sx + sy) * 0.5
    theta = float(np.arctan2(M[1, 0], M[0, 0]))
    return np.array([tx, ty, s, theta], dtype=np.float32)

def compose_affine(params):
    tx, ty, s, theta = params
    cos_t = np.cos(theta) * s
    sin_t = np.sin(theta) * s
    return np.array([[cos_t, -sin_t, tx],
                     [sin_t,  cos_t, ty]], dtype=np.float32)

def get_inverse_trans(trans):
    full_matrix = np.vstack([trans, [0, 0, 1]])

    # Compute the inverse of the 3x3 matrix
    inverse_matrix = np.linalg.inv(full_matrix)

    # Extract the top 2 rows for use with cv2.warpAffine
    inverse_transform_matrix = inverse_matrix[:2, :]
    return inverse_transform_matrix

class TemporalSmoothing:
    """
    Adaptive velocity-aware temporal smoothing.
    - Operates in Lie/polar similarity space (tx, ty, scale, theta) to preserve orthogonality.
    - When user is stationary: smooths micro-jitter.
    - When user moves or turns: adapts dynamically to 0-lag tracking with no inertia delay.
    """
    def __init__(self, c=0.6, snap_threshold=20.0):
        self.past_trans = None
        self.base_c = c
        self.snap_threshold = snap_threshold

    def __call__(self, trans):
        if trans is None:
            if self.past_trans is not None:
                return self.past_trans.copy(), get_inverse_trans(self.past_trans.copy())
            return None, None

        if self.past_trans is None:
            self.past_trans = trans.copy().astype(np.float32)
            return trans, get_inverse_trans(trans)

        # Compute translation displacement between consecutive frames
        dx = trans[0, 2] - self.past_trans[0, 2]
        dy = trans[1, 2] - self.past_trans[1, 2]
        dist = np.sqrt(dx * dx + dy * dy)

        # Scale difference
        s_curr = np.sqrt(trans[0, 0]**2 + trans[0, 1]**2)
        s_past = np.sqrt(self.past_trans[0, 0]**2 + self.past_trans[0, 1]**2)
        s_diff = abs(s_curr - s_past)

        # Adaptive responsiveness: if turning or moving fast, snap immediately
        if dist > self.snap_threshold or s_diff > 0.15:
            effective_c = 0.05  # 95% immediate response
        elif dist > 8.0 or s_diff > 0.05:
            effective_c = 0.20  # Fast response for medium motion
        else:
            effective_c = self.base_c  # Smooth jitter for stationary state

        p_curr = decompose_affine(trans)
        p_past = decompose_affine(self.past_trans)
        d_theta = (p_curr[3] - p_past[3] + np.pi) % (2.0 * np.pi) - np.pi
        p_smooth = np.zeros(4, dtype=np.float32)
        p_smooth[0:3] = p_curr[0:3] * (1.0 - effective_c) + p_past[0:3] * effective_c
        p_smooth[3] = p_past[3] + d_theta * (1.0 - effective_c)
        smoothed = compose_affine(p_smooth)
        self.past_trans = smoothed.copy()
        return smoothed, get_inverse_trans(smoothed)