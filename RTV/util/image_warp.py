import cv2
import numpy as np

def crop_resize2size(img:np.ndarray, h, w):
    img_h,img_w = img.shape[:2]
    if img_h*w>img_w*h:#too tall
        delta = img_h - img_w*h/w
        img = img[int(delta / 2):img_h - int(delta / 2), :, :]
    else:
        delta = img_w - img_h*w/h
        img = img[:, int(delta / 2):img_w - int(delta / 2), :]
    img=cv2.resize(img,(w,h))
    return img


def crop2_43(img: np.ndarray) -> np.ndarray:
    h,w=img.shape[:2]
    ratio=0.5
    if 3*h>4*w:#too tall
        delta=h-w*4/3
        img=img[int(delta*ratio):h-int(delta*(1-ratio)),:,:]
    else:
        delta = w-h*3/4
        img = img[:,int(delta / 2):w - int(delta / 2), :]
    return img

def crop2_169(img: np.ndarray) -> np.ndarray:
    h,w=img.shape[:2]
    if 9*h>16*w:#too tall
        delta=h-w*16/9
        img=img[int(delta/2):h-int(delta/2),:,:]
    else:
        delta = w-h*9/16
        img = img[:,int(delta / 2):w - int(delta / 2), :]
    return img

def resize_img(img: np.ndarray, max_height=1024) -> np.ndarray:
    h,w=img.shape[:2]
    if h>w:
        new_h=max_height
        new_w=int(w*new_h/h)
        img=cv2.resize(img,(new_w,new_h))
    return img


def pad2square(img,size=512):
    width, height = img.shape[1], img.shape[0]
    if width>height:
        zero_pad = np.zeros([int((width-height)/2),width,3],np.uint8)
        img = np.concatenate([zero_pad,img,zero_pad],0)
    elif height>width:
        zero_pad = np.zeros([height,int((height-width) / 2), 3],np.uint8)
        img = np.concatenate([zero_pad, img, zero_pad], 1)
    img = cv2.resize(img,(size,size))
    return img

def center_crop(img, dim):
    """Returns center cropped image
    Args:
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped
    """
    width, height = img.shape[1], img.shape[0]


    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
    return crop_img


def scale_image(img, factor=1):
    """Returns resize image by scale factor.
    This helps to retain resolution ratio while resizing.
    Args:
    img: image to be scaled
    factor: scale factor to resize
    """
    return cv2.resize(img, (int(img.shape[1] * factor), int(img.shape[0] * factor)))


def zoom_in(img, factor):
    width, height = img.shape[1], img.shape[0]
    img = scale_image(img, factor)
    img =center_crop(img,(width,height))
    return img

def zoom_out(img, factor):
    width, height = img.shape[1], img.shape[0]
    width_new, height_new = int(width*factor), int(height*factor)
    zero_canvas=np.zeros((height_new,width_new,3),np.uint8)
    start_row=int((height_new-height)/2)
    start_col=int((width_new-width)/2)
    zero_canvas[start_row:start_row+height,start_col:start_col+width,:]=img
    new_img = cv2.resize(zero_canvas,(width,height))

    return new_img


def shift_image_right(img, factor):
    num_rows, num_cols = img.shape[:2]

    shift_x = factor*num_cols

    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, 0]])
    img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))
    return img_translation

def shift_image_down(img, factor):
    num_rows, num_cols = img.shape[:2]

    shift_y = factor*num_rows

    translation_matrix = np.float32([[1, 0, 0], [0, 1, shift_y]])
    img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))
    return img_translation


def shift_image(img, factor_x, factor_y):
    num_rows, num_cols = img.shape[:2]

    shift_x = factor_x*num_cols
    shift_y = factor_y * num_rows

    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))
    return img_translation

def rotate_image(image, angle):
    angle = angle*180/np.pi
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result
