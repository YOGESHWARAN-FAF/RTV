import numpy as np
import sys
import cv2

sys.path.append('./util/cpp_extensions/build')


#import example  # Replace 'example' with the name of your module

#1, 2 = Torso, 3 = Right Hand, 4 = Left Hand, 5 = Left Foot, 6 = Right Foot, 7, 9 = Upper Leg Right,
# 8, 10 = Upper Leg Left, 11, 13 = Lower Leg Right, 12, 14 = Lower Leg Left, 15, 17 = Upper Arm Left,
# 16, 18 = Upper Arm Right, 19, 21 = Lower Arm Left, 20, 22 = Lower Arm Right, 23, 24 = Head

def IUV2UpperBodyMask(IUV: np.ndarray):
    upper_index = [1, 2, 15, 17, 16, 18, 19, 21, 20, 22, 23, 24]#[1, 2, 3, 4, 15, 17, 16, 18, 19, 21, 20, 22, 23, 24]
    h, w = IUV.shape[:2]
    mask = np.zeros((h, w), bool)
    for i in upper_index:
        mask[IUV[:, :, 0] == i] = True
    return mask


def IUV2UpperBodyRoiTrans(IUV: np.ndarray,roi_size=1024,roi_ratio=1.2):
    mask = IUV2UpperBodyMask(IUV)
    h,w = IUV.shape[:2]
    y_indices, x_indices = np.where(mask == True)

    # Calculate the centroid
    centroid_x = np.mean(x_indices)
    centroid_y = np.mean(y_indices)
    # Finding the indices of non-zero elements
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    size=max([ymax-centroid_y, centroid_y-ymin,xmax-centroid_x, centroid_x-xmin])

    src = np.zeros([3, 2], np.float32)
    center = np.array([centroid_x,centroid_y],np.float32)#Joints[9] * 0.8 + Joints[12] * 0.2
    size = size * roi_ratio
    src[0, :] = center + np.array([-size, size], np.float32)
    src[1, :] = center + np.array([-size, -size], np.float32)
    src[2, :] = center + np.array([size, -size], np.float32)

    dst = np.zeros([3, 2], np.float32)
    dst[0, :] = np.array([0, roi_size - 1], np.float32)
    dst[1, :] = np.array([0, 0], np.float32)
    dst[2, :] = np.array([roi_size - 1, 0], np.float32)
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    inv_trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return trans, inv_trans


def IUV2Img(IUV: np.ndarray):
    IUV = IUV.astype(np.float32)
    IUV[:, :, 0] /= 24.0
    IUV[:, :, 0] *= 255
    IUV = IUV.astype(np.uint8)
    return IUV


def IUV2UpperBodyImg(IUV: np.ndarray):
    remove_index=[5,6,7,9,8,10,11,13,12,14,23,24]
    torso_index = [1, 2]
    result=IUV.copy()
    for i in remove_index:
        result[IUV[:,:,0]==i]=0
    result = result.astype(np.float32)
    result[:, :, 0] /= 24.0
    result[:, :, 0] *= 255
    result = result.astype(np.uint8)


    for i in torso_index:
        result[IUV[:, :, 0] == i] = 255

    return result


def IUV2TorsoLeg(IUV: np.ndarray):
    torsoleg_index = [1, 2, 5, 6, 7, 9, 8, 10, 11, 13, 12, 14]
    result = IUV.copy()
    result = result.astype(np.float32)
    result[:, :, 0] /= 24
    result[:, :, 0] *= 255
    result = result.astype(np.uint8)
    for i in torsoleg_index:
        result[IUV[:, :, 0] == i] = 255
    return result


def IUV2SDP(IUV: np.ndarray):
    torsoleg_index = [1, 2, 5, 6, 7, 9, 8, 10, 11, 13, 12, 14]

    # Copy and normalize the IUV array
    result = IUV.copy().astype(np.float32)
    result[:, :, 0] = (result[:, :, 0] / 24) * 255
    result = result.astype(np.uint8)

    # Create a mask for the torso and leg indices
    torsoleg_mask = np.isin(IUV[:, :, 0], torsoleg_index)

    # Apply the mask to set the corresponding pixels to 255
    result[torsoleg_mask] = 255

    return result

#1, 2 = Torso, 3 = Right Hand, 4 = Left Hand, 5 = Left Foot, 6 = Right Foot, 7, 9 = Upper Leg Right,
# 8, 10 = Upper Leg Left, 11, 13 = Lower Leg Right, 12, 14 = Lower Leg Left, 15, 17 = Upper Arm Left,
# 16, 18 = Upper Arm Right, 19, 21 = Lower Arm Left, 20, 22 = Lower Arm Right, 23, 24 = Head
def IUV2SSDP_old(IUV: np.ndarray):
    torsoleghead_index = [1, 2, 5, 6, 7, 9, 8, 10, 11, 13, 12, 14, 23,24]
    left_arm_index= [4,15,17,19,21]
    right_arm_index = [3,16,18,20,22]
    result = IUV.copy()
    result = result.astype(np.float32)
    result[:, :, 0] /= 24
    result[:, :, 0] *= 255
    result = result.astype(np.uint8)
    for i in torsoleghead_index:
        result[IUV[:, :, 0] == i] = 255
    for i in left_arm_index:
        result[IUV[:, :, 0] == i] = np.array([255,0,0], np.uint8)
    for i in right_arm_index:
        result[IUV[:, :, 0] == i] = np.array([0,0, 255], np.uint8)
    return result

def IUV2SSDP_deprecated(IUV: np.ndarray):
    torsoleghead_index = [1, 2, 5, 6, 7, 9, 8, 10, 11, 13, 12, 14, 23, 24]
    left_arm_index = [4, 15, 17, 19, 21]
    right_arm_index = [3, 16, 18, 20, 22]

    result = IUV.copy().astype(np.float32)
    result[:, :, 0] = (result[:, :, 0] / 24) * 255
    result = result.astype(np.uint8)

    # Create masks for each index group
    torsoleghead_mask = np.isin(IUV[:, :, 0], torsoleghead_index)
    left_arm_mask = np.isin(IUV[:, :, 0], left_arm_index)
    right_arm_mask = np.isin(IUV[:, :, 0], right_arm_index)

    # Apply masks to result
    result[torsoleghead_mask] = 255
    result[left_arm_mask] = np.array([255, 0, 0], np.uint8)
    result[right_arm_mask] = np.array([0, 0, 255], np.uint8)

    return result



def IUV2SSDP(IUV: np.ndarray):
    return IUV2SSDP_new(IUV)
#1, 2 = Torso, 3 = Right Hand, 4 = Left Hand, 5 = Left Foot, 6 = Right Foot, 7, 9 = Upper Leg Right,
# 8, 10 = Upper Leg Left, 11, 13 = Lower Leg Right, 12, 14 = Lower Leg Left, 15, 17 = Upper Arm Left,
# 16, 18 = Upper Arm Right, 19, 21 = Lower Arm Left, 20, 22 = Lower Arm Right, 23, 24 = Head
def IUV2SSDP_new(IUV: np.ndarray):
    torsoleghead_index = [1, 2, 5, 6, 7, 9, 8, 10, 11, 13, 12, 14, 23, 24]
    left_upper_arm_index = [15, 17]
    left_lower_arm_index = [19, 21]
    left_hand_index = [4,]
    right_upper_arm_index = [16, 18]
    right_lower_arm_index = [20, 22]
    right_hand_index = [3,]

    result = IUV.copy().astype(np.float32)
    result[:, :, 0] = (result[:, :, 0] / 24) * 255
    result = result.astype(np.uint8)

    # Create masks for each index group
    torsoleghead_mask = np.isin(IUV[:, :, 0], torsoleghead_index)
    left_upper_arm_mask = np.isin(IUV[:, :, 0], left_upper_arm_index)
    left_lower_arm_mask = np.isin(IUV[:, :, 0], left_lower_arm_index)
    left_hand_mask = np.isin(IUV[:, :, 0], left_hand_index)
    right_upper_arm_mask = np.isin(IUV[:, :, 0], right_upper_arm_index)
    right_lower_arm_mask = np.isin(IUV[:, :, 0], right_lower_arm_index)
    right_hand_mask = np.isin(IUV[:, :, 0], right_hand_index)

    # Apply masks to result
    result[torsoleghead_mask] = 255
    result[left_upper_arm_mask] = np.array([255, 0, 0], np.uint8)
    result[left_lower_arm_mask] = np.array([255, 255, 0], np.uint8)
    result[left_hand_mask] = np.array([255, 0, 255], np.uint8)

    result[right_upper_arm_mask] = np.array([0, 0, 255], np.uint8)
    result[right_lower_arm_mask] = np.array([0, 255, 255], np.uint8)
    result[right_hand_mask] = np.array([0, 255, 0], np.uint8)

    return result