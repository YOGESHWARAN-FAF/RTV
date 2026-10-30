import cv2
import numpy as np
from Graphonomy.human_parser import HumanParser
from Graphonomy.dataset_settings import dataset_settings

cihp_label = dataset_settings['cihp']['label']
import random


def slide_window_garment_mask_refine(human_parser: HumanParser, num_window, raw_image):
    trans2roi=np.zeros((2,3),np.float32)
    trans2roi[0,0]=1
    trans2roi[1, 1] = 1
    height = raw_image.shape[0]
    width = raw_image.shape[1]
    trans_list = []
    bias = width * 0.1
    txs = np.linspace(int(-bias), int(bias), num_window).tolist()
    tys = np.linspace(int(-bias), int(bias), num_window).tolist()
    for i in range(num_window):
        for j in range(num_window):

            tx = int(txs[i])
            ty = int(tys[j])
            new_trans = trans2roi.copy()
            new_trans[0, 2] += tx
            new_trans[1, 2] += ty

            trans_list.append(new_trans)
    raw_mask_list = []
    raw_valid_mask_list = []
    for i in range(len(trans_list)):
        roi_img = cv2.warpAffine(raw_image, trans_list[i], (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
        garment_mask = get_garment_mask(human_parser, roi_img).astype(np.uint8) * 255
        raw_mask = cv2.warpAffine(garment_mask, get_inverse_trans(trans_list[i]), (width, height),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0))
        raw_valid_mask = get_trans_mask(trans_list[i], height, width)
        raw_mask_list.append((raw_mask > 128).astype(np.uint8))
        raw_valid_mask_list.append(raw_valid_mask)
    sum_mask = np.zeros((height, width), dtype=np.uint8)
    sum_valid_mask = np.zeros((height, width), dtype=np.uint8)
    for i in range(num_window*num_window):
        sum_mask += raw_mask_list[i]
        sum_valid_mask += raw_valid_mask_list[i]
    result_mask = sum_mask.astype(np.float32) > (sum_valid_mask.astype(np.float32) * 0.5)
    #result_mask = raw_valid_mask_list[-1].astype(np.float32) * 0.5 > 0
    return result_mask


def get_trans_mask(trans, height, width):
    valid_mask = np.ones((height, width), dtype=np.uint8)
    raw_valid_mask = cv2.warpAffine(valid_mask, get_inverse_trans(trans), (width, height), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0))
    return raw_valid_mask


def get_garment_mask(human_parser: HumanParser, roi_img):


    garment_mask = human_parser.GetAtrGarmentMask(roi_img,isRGB=False)

    return garment_mask


def get_inverse_trans(trans):
    full_matrix = np.vstack([trans, [0, 0, 1]])

    # Compute the inverse of the 3x3 matrix
    inverse_matrix = np.linalg.inv(full_matrix)

    # Extract the top 2 rows for use with cv2.warpAffine
    inverse_transform_matrix = inverse_matrix[:2, :]
    return inverse_transform_matrix
