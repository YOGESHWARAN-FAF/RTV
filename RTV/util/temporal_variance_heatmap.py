import cv2
import numpy as np
import os

class TemporalVarianceHeatmap:
    def __init__(self,window_size=10):
        self.window_size = window_size
        self.frames=[]

    def __call__(self, img):
        self.frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32))
        if len(self.frames)>self.window_size:
            self.frames.pop(0)
        stack = np.stack(self.frames, axis=0)  # Shape: (window_size, H, W)
        variance = np.var(stack, axis=0)  # Per-pixel temporal variance
        #print(np.max(variance), np.min(variance))
        # Normalize and apply colormap
        var_norm = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX)
        #var_norm=np.clip(var_norm,0,1)*255
        var_uint8 = var_norm.astype(np.uint8)
        heatmap = cv2.applyColorMap(var_uint8, cv2.COLORMAP_JET)
        return heatmap