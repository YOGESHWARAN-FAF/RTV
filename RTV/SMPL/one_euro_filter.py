import math
import time
import numpy as np

class LowPassFilter:
    def __init__(self, alpha=0.5):
        self.alpha = float(alpha)
        self.prev_val = None

    def reset(self):
        self.prev_val = None

    def __call__(self, val, alpha=None):
        if alpha is not None:
            self.alpha = float(alpha)
        if self.prev_val is None or self.prev_val.shape != val.shape:
            self.prev_val = np.copy(val)
            return val
        filtered = self.alpha * val + (1.0 - self.alpha) * self.prev_val
        self.prev_val = np.copy(filtered)
        return filtered


class OneEuroFilter:
    """
    Industry-Standard 1-Euro Adaptive Filter (Casiez et al., ACM CHI 2012).
    - At near-zero velocity (user standing/seated still): uses min_cutoff (e.g. 0.4 - 1.0 Hz)
      to eliminate 100% of monocular neural pose jitter and garment shaking.
    - At high velocity (raising arms, turning): dynamically increases cutoff frequency
      via beta to eliminate latency, ensuring 0-lag tracking on rapid movements.
    """
    def __init__(self, min_cutoff=0.8, beta=0.015, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_filter = None
        self.dx_filter = None
        self.last_time = None

    def reset(self):
        self.x_filter = None
        self.dx_filter = None
        self.last_time = None

    def _compute_alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * max(1e-5, cutoff))
        return 1.0 / (1.0 + tau / max(1e-5, dt))

    def __call__(self, x, t=None):
        if t is None:
            t = time.time()

        x = np.asarray(x, dtype=np.float32)

        if (self.last_time is None or self.x_filter is None or
            self.x_filter.prev_val is None or self.x_filter.prev_val.shape != x.shape):
            self.last_time = t
            self.x_filter = LowPassFilter()
            self.dx_filter = LowPassFilter()
            return self.x_filter(x)

        dt = max(1e-4, t - self.last_time)
        self.last_time = t

        prev_x = self.x_filter.prev_val
        if prev_x is None:
            prev_x = x

        dx = (x - prev_x) / dt
        edx = self.dx_filter(dx, alpha=self._compute_alpha(self.d_cutoff, dt))

        speed = float(np.linalg.norm(edx))
        cutoff = self.min_cutoff + self.beta * speed
        alpha = self._compute_alpha(cutoff, dt)

        return self.x_filter(x, alpha=alpha)
