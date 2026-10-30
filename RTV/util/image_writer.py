import cv2
import numpy as np

def save_image(path:str,img:np.ndarray,isRGB=True):
    if isRGB and img.ndim==3:
        img=img[:,:,[2,1,0]]
    cv2.imwrite(path,img)