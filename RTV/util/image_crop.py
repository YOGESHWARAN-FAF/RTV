import numpy as np
def crop(img: np.ndarray, up=0.2, down=0.02, left=0.0, right=0.0):
    new_img=img.copy()
    height, width = new_img.shape[:2]
    n_row = int(height * up)
    n_left = int(width * left)
    n_right = int(width * right)
    n_down = int(height * down)
    new_img=new_img[n_row:height-n_down,n_left:width-n_right:,:]
    return new_img

def crop_16_9(img: np.ndarray, up=0.2, left=0.0, right=0.0):
    new_img=img.copy()
    height, width = new_img.shape[:2]
    n_row = int(height * up)
    n_left = int(width * left)
    n_right = int(width * right)
    n_down = int(height - (width - n_right - n_left)*16/9 -n_row)
    new_img=new_img[n_row:height-n_down,n_left:width-n_right:,:]
    return new_img

def crop_4_3(img: np.ndarray, up=0.2, left=0.0, right=0.0):
    new_img=img.copy()
    height, width = new_img.shape[:2]
    n_row = int(height * up)
    n_left = int(width * left)
    n_right = int(width * right)
    n_down = int(height - (width - n_right - n_left)*4/3 -n_row)
    new_img=new_img[n_row:height-n_down,n_left:width-n_right:,:]
    return new_img
