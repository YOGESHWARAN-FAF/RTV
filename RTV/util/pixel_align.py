import numpy as np
import torch
import cv2
from util.landmark_annotator import GetLandmarkCoord

joint_list = ['thorax', 'rShoulder', 'lShoulder', 'lElbow', 'rElbow', 'lHip', 'rHip', 'pelvis']
mpii_id = {'thorax': 7, 'rShoulder': 12, 'lShoulder': 13, 'lElbow': 14, 'rElbow': 11, 'lHip': 3, 'rHip': 2, 'pelvis': 6}

class JointAlignment:
    def __init__(self):
        self.scale_list = []
        self.translation_list = []
        self.avg_scale_list = []
        self.avg_translation_list = []

    def __len__(self):
        return len(self.scale_list)

    def record_trans(self, joint_heatmap, joint_2d):
        xs, ys, _ = GetLandmarkCoord(joint_heatmap)
        xys = [np.array([xs[i], ys[i]], np.float32) for i in range(len(xs))]

        xys_reduced = [xys[mpii_id[j]] for j in joint_list]
        joint_2d = [np.array(j, np.float32) for j in joint_2d]
        src= np.concatenate([np.expand_dims(j2d,0) for j2d in joint_2d ],axis=0)
        dst = np.concatenate([np.expand_dims(j2d,0) for j2d in xys_reduced ],axis=0)
        c_src =src.mean(0)
        c_dst = dst.mean(0)
        src_centered = src - c_src
        dst_centered = dst - c_dst
        A0 = np.zeros((2,2),np.float32)
        A1 = np.zeros((2, 2), np.float32)
        for i in range(src.shape[0]):
            A0+=np.matmul(np.expand_dims(dst_centered[i],1),np.expand_dims(src_centered[i],0))
        for i in range(src.shape[0]):
            A1+=np.matmul(np.expand_dims(src_centered[i],1),np.expand_dims(src_centered[i],0))
        A1_inv = np.linalg.inv(A1)
        A=np.matmul(A0,A1_inv)
        u, s, vh = np.linalg.svd(A, full_matrices=True)
        s0=s[0]
        s1=s[1]
        avg_s = np.sqrt(s0*s1)
        self.scale_list.append(avg_s)
        result = np.diag([avg_s,avg_s])
        final_result = np.zeros((2,3),np.float32)
        final_result[:,[0,1]] = result

        final_result[:,2] = c_dst - result.dot(c_src)
        self.translation_list.append(c_dst - result.dot(c_src))

    def offline_smooth(self):
        length = len(self.scale_list)
        radius = 20
        for i in range(length):
            count = 0
            sum = 0.0
            for j in range(i - radius, i + radius + 1):
                if length > j >= 0:
                    sum += self.scale_list[j]
                    count += 1
            self.avg_scale_list.append(sum / count)
        radius = 15
        for i in range(length):
            count = 0
            sum = np.zeros(2).astype(np.float32)
            for j in range(i - radius, i + radius + 1):
                if length > j >= 0:
                    sum += self.translation_list[j]
                    count += 1
            self.avg_translation_list.append(sum / count)
    def get_smoothed_trans(self,i):
        scale = self.avg_scale_list[i]
        translation = self.avg_translation_list[i]
        trans = np.zeros((2,3),np.float32)
        trans[:,[0,1]] = np.diag([scale,scale])
        trans[:,2] = translation
        return trans



def joint_align(joint_heatmap, joint_2d):
    xs, ys, _ = GetLandmarkCoord(joint_heatmap)
    xys = [np.array([xs[i], ys[i]], np.float32) for i in range(len(xs))]

    joint_2d = [np.array(j, np.float32) for j in joint_2d]
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = (joint_2d[joint_list.index('lElbow')] + joint_2d[joint_list.index('lShoulder')]) / 2.0
    src[1, :] = (joint_2d[joint_list.index('rElbow')] + joint_2d[joint_list.index('rShoulder')]) / 2.0
    src[2, :] = (joint_2d[joint_list.index('lHip')] + joint_2d[joint_list.index('rHip')] + joint_2d[
        joint_list.index('pelvis')]) / 3.0

    dst[0, :] = (xys[mpii_id['lElbow']] + xys[mpii_id['lShoulder']])/2.0
    dst[1, :] = (xys[mpii_id['rElbow']] + xys[mpii_id['rShoulder']]) / 2.0
    dst[2, :] = (xys[mpii_id['lHip']] + xys[mpii_id['rHip']] + xys[mpii_id['pelvis']]) / 3.0
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    #print('M:',trans)
    return trans



class AABB:
    def __init__(self, min_x, max_x, min_y, max_y):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y


class ScaleTransOptimizer:
    def __init__(self, ):
        pass

    def compute_transfrom(self, input_mask, syn_mask):
        input_aabb = self.get_aabb(input_mask)
        syn_aabb = self.get_aabb(syn_mask)
        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = np.array([syn_aabb.min_x, syn_aabb.min_y], np.float32)
        src[1, :] = np.array([syn_aabb.max_x, syn_aabb.min_y], np.float32)
        src[2, :] = np.array([syn_aabb.max_x, syn_aabb.max_y], np.float32)

        input_aabb.max_y = (syn_aabb.max_y - syn_aabb.min_y) / (syn_aabb.max_x - syn_aabb.min_x) * (
                    input_aabb.max_x - input_aabb.min_x) + input_aabb.min_y

        dst[0, :] = np.array([input_aabb.min_x, input_aabb.min_y], np.float32)
        dst[1, :] = np.array([input_aabb.max_x, input_aabb.min_y], np.float32)
        dst[2, :] = np.array([input_aabb.max_x, input_aabb.max_y], np.float32)
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        return trans

    def get_aabb(self, mask):
        hist_x = mask.sum(0).astype(np.float32)
        hist_y = mask.sum(1).astype(np.float32)
        min_x, max_x = self.get_bound(hist_x)
        min_y, max_y = self.get_bound(hist_y)
        return AABB(min_x, max_x, min_y, max_y)

    def get_bound(self, hist):
        length = hist.shape[0]
        max_value = hist.max()
        thr = 0.1
        idx = np.arange(length)
        bound = idx[hist > thr * max_value]
        lower_bound = bound.min()
        upper_bound = bound.max()
        return lower_bound, upper_bound


if __name__ == '__main__':
    h = 400
    w = 600
    source_mask = np.zeros((h, w), np.uint8)
    for i in range(h):
        for j in range(w):
            if 32 < (i - h / 2) * (i - h / 2) + (j - w / 2) * (j - w / 2) < 36:
                source_mask[i, j] = 1
    target_mask = np.zeros((h, w), np.uint8)
    for i in range(h):
        for j in range(w):
            if 32 < (i - h / 2) * (i - h / 2) + (j - w / 2) * (j - w / 2) < 36:
                target_mask[i, j] = 1
    stoptimizer = ScaleTransOptimizer(h, w)
    print(stoptimizer.compute_scale_trans(source_mask, target_mask))


class OverlapMaximizer:
    def __init__(self):
        self.prev_trans = None

    def align_image(self, target_img, input_mask, target_mask):
        if self.prev_trans is not None:
            target_img = translate_image(target_img, self.prev_trans)
            target_mask = get_threshold_mask(target_img)

        trans = pixel_align(input_mask, target_mask)

        print('prev_trans:', self.prev_trans)
        print('trans:', trans)
        # smoothing
        # if trans[0] * trans[0] + trans[1] * trans[1] < 18:
        #    trans = np.zeros(2, np.int64)

        np_synthesised = translate_image(target_img, trans)

        if self.prev_trans is not None:
            self.prev_trans = trans + self.prev_trans
            self.prev_trans = np.rint(self.prev_trans).astype(np.int64)
        else:
            self.prev_trans = trans
        return np_synthesised


class OfflineOverlapMaximizer:
    def __init__(self):
        self.prev_trans = None
        self.trans_history = []
        self.smoothed_trans = []
        self.target_img_list = []

    def record_align_trans(self, target_img, input_mask, target_mask):
        self.target_img_list.append(target_img.copy())
        if self.prev_trans is not None:
            target_img = translate_image(target_img, self.prev_trans)
            target_mask = get_threshold_mask(target_img)

        trans = pixel_align(input_mask, target_mask)

        # print('prev_trans:', self.prev_trans)
        # print('trans:', trans)
        # smoothing
        # if trans[0] * trans[0] + trans[1] * trans[1] < 18:
        #    trans = np.zeros(2, np.int64)

        if self.prev_trans is not None:
            self.prev_trans = trans + self.prev_trans
            self.prev_trans = np.rint(self.prev_trans).astype(np.int64)

        else:
            self.prev_trans = trans
        self.trans_history.append(self.prev_trans)

    def offline_smooth(self):
        length = len(self.trans_history)
        avg_trans_list = []
        radius = 15
        for i in range(length):
            count = 0
            sum = np.zeros(2, np.float32)
            for j in range(i - radius, i + radius + 1):
                if length > j >= 0:
                    sum += self.trans_history[j]
                    count += 1
            self.smoothed_trans.append(np.rint(sum / count).astype(np.int64))

    def get_smoothed_align(self, i, mask = None):
        if mask is None:
            target_img = translate_image(self.target_img_list[i], self.smoothed_trans[i])
            return target_img
        else:
            target_img = translate_image(self.target_img_list[i], self.smoothed_trans[i])
            mask = translate_image(mask, self.smoothed_trans[i])
            return target_img, mask

    def __len__(self):
        return len(self.target_img_list)


def get_threshold_mask(img) -> np.array:
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img / 255.0)
        mask = img.square().sum(2)
        th = 0.04
        synthesized_garment_mask = (mask > th).numpy().astype(np.uint8)
    else:
        img = img.cpu()
        if img.dim() == 4:
            mask = img[0]
        mask = mask.permute(1, 2, 0)
        mask = mask.square().sum(2)
        th = 0.04
        synthesized_garment_mask = (mask > th).numpy().astype(np.uint8)
    return synthesized_garment_mask


def translate_image(img: np.array, trans: np.array) -> np.array:
    h = img.shape[0]
    w = img.shape[1]
    x = trans[0]
    y = trans[1]
    # print('trans:',trans)
    # print('input:',img.shape)
    if x != 0:
        x_pad = np.zeros((h, abs(x), 3), np.uint8)
        if x > 0:
            img = img[:, :-abs(x), :]
            # print(img.shape)
            # print(x_pad.shape)
            img = np.concatenate([x_pad, img], axis=1)
        else:
            img = img[:, abs(x):, :]
            img = np.concatenate([img, x_pad], axis=1)
    if y != 0:
        y_pad = np.zeros((abs(y), w, 3), np.uint8)
        if y > 0:
            img = img[abs(y):, :, :]
            img = np.concatenate([img, y_pad], axis=0)
        else:
            img = img[:-abs(y), :, :]
            img = np.concatenate([y_pad, img], axis=0)
    # print('output:', img.shape)
    return img


def pixel_align(mask1: np.array, mask2: np.array):
    mask1 = mask1.astype(np.uint8)
    mask2 = mask2.astype(np.uint8)
    # print(mask1.dtype)
    # print(mask2.dtype)

    func_list = [move_r, move_l, move_u, move_d]
    vector_list = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    vector_list = [np.array(v, np.int64) for v in vector_list]
    move_history = []

    max_iou = compute_iou(mask1, mask2)
    while True:
        iou_r = compute_iou(mask1, move_r(mask2))
        iou_l = compute_iou(mask1, move_l(mask2))
        iou_u = compute_iou(mask1, move_u(mask2))
        iou_d = compute_iou(mask1, move_d(mask2))
        iou_list = [iou_r, iou_l, iou_u, iou_d]
        new_iou = max(iou_list)
        new_id = iou_list.index(new_iou)

        if new_iou > max_iou:
            mask2 = func_list[new_id](mask2)
            move_history.append(vector_list[new_id])
            max_iou = new_iou
        else:
            break
    move_vector = np.array([0, 0], np.int64)
    for m in move_history:
        move_vector = move_vector + m
    return move_vector


def compute_iou(mask1, mask2):
    mask1 = mask1.astype(np.float32)
    mask2 = mask2.astype(np.float32)
    i = (mask1 * mask2).sum()
    u = ((mask1 > 0.5) | (mask2 > 0.5)).astype(np.float32)
    u = u.sum() + 1e-4
    return i / u


def move_r(mask: np.array):
    mask = mask.copy()
    pad = np.zeros((mask.shape[0], 1), np.uint8)
    mask = mask[:, :-1]
    mask = np.concatenate([pad, mask], axis=1)
    return mask


def move_l(mask):
    mask = mask.copy()
    pad = np.zeros((mask.shape[0], 1), np.uint8)
    mask = mask[:, 1:]
    mask = np.concatenate([mask, pad], axis=1)
    return mask


def move_u(mask):
    mask = mask.copy()
    pad = np.zeros((1, mask.shape[1]), np.uint8)
    mask = mask[1:, :]
    mask = np.concatenate([mask, pad], axis=0)
    return mask


def move_d(mask):
    mask = mask.copy()
    pad = np.zeros((1, mask.shape[1]), np.uint8)
    mask = mask[:-1, :]
    mask = np.concatenate([pad, mask], axis=0)
    return mask
