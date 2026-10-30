import argparse
import glob
import logging
import os
import sys
from typing import Any, ClassVar, Dict, List
import torch
import copy
_dp_dir = os.path.dirname(os.path.abspath(__file__))
if _dp_dir not in sys.path:
    sys.path.insert(0, _dp_dir)

from detectron2.config import CfgNode, get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.instances import Instances
from detectron2.utils.logger import setup_logger

from densepose import add_densepose_config
from densepose.structures import DensePoseChartPredictorOutput, DensePoseEmbeddingPredictorOutput
from densepose.utils.logger import verbosity_to_level
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.densepose_outputs_vertex import (
    DensePoseOutputsTextureVisualizer,
    DensePoseOutputsVertexVisualizer,
    get_texture_atlases,
)
from densepose.vis.densepose_results import (
    DensePoseResultsContourVisualizer,
    DensePoseResultsFineSegmentationVisualizer,
    DensePoseResultsUVisualizer,
    DensePoseResultsVVisualizer,
)
from densepose.vis.densepose_results_textures import (
    DensePoseResultsVisualizerWithTexture,
    get_texture_atlas,
)
from densepose.vis.extractor import (
    CompoundExtractor,
    DensePoseOutputsExtractor,
    DensePoseResultExtractor,
    create_extractor,
)
from .apply_net import create_argument_parser, DumpAction

import torch
import torch.nn.functional as F
import numpy as np
import cv2

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

class DensePoseExtractor(DumpAction):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.dp_model = DumpAction()
        #self.dp_model.add_arguments(self.parser)
        self.args = self.parser.parse_args([])
        opts = ['INPUT.MIN_SIZE_TEST', 384, 'INPUT.MAX_SIZE_TEST', 512]
        self.cfg = os.path.join(_dp_dir, 'configs', 'densepose_rcnn_R_50_FPN_s1x.yaml')
        self.model = 'https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl'
        cfg = self.dp_model.setup_config(self.cfg, self.model, self.args, opts)
        if torch.cuda.device_count() >= 2:
            cfg.defrost()
            cfg.MODEL.DEVICE = 'cuda:1'
            cfg.freeze()
            self.device_str = 'cuda:1'
        else:
            self.device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.predictor = DefaultPredictor(cfg)
        self.palette = np.array(get_palette(25), np.uint8).reshape(-1,3)

    @staticmethod
    def _safe_place_tensor(dst_tensor, src_tensor, x_min, y_min, mask_w, mask_h, raw_w, raw_h):
        dst_y1 = max(0, min(raw_h, y_min))
        dst_y2 = max(0, min(raw_h, y_min + mask_h))
        dst_x1 = max(0, min(raw_w, x_min))
        dst_x2 = max(0, min(raw_w, x_min + mask_w))

        src_y1 = dst_y1 - y_min
        src_y2 = src_y1 + (dst_y2 - dst_y1)
        src_x1 = dst_x1 - x_min
        src_x2 = src_x1 + (dst_x2 - dst_x1)

        if dst_y2 > dst_y1 and dst_x2 > dst_x1:
            if dst_tensor.dim() == 2:
                dst_tensor[dst_y1:dst_y2, dst_x1:dst_x2] = src_tensor[src_y1:src_y2, src_x1:src_x2]
            elif dst_tensor.dim() == 3:
                dst_tensor[:, dst_y1:dst_y2, dst_x1:dst_x2] = src_tensor[:, src_y1:src_y2, src_x1:src_x2]

    def forward(self,img):
        img = img[:,:,[2,1,0]] # convert to BGR
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = self.predictor(img)["instances"]#BGR fromat
        image_fpath = "file_name"
        result = {"file_name": image_fpath}
        if outputs.has("scores"):
            result["scores"] = outputs.get("scores").cpu()
        if outputs.has("pred_boxes"):
            result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
            if outputs.has("pred_densepose"):
                if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
                    extractor = DensePoseResultExtractor()
                elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
                    extractor = DensePoseOutputsExtractor()
                result["pred_densepose"] = extractor(outputs)[0]

        if len(result['pred_boxes_XYXY']) == 0:
            return None

        box_list = (result['pred_boxes_XYXY'][0]).tolist()
        box_list = [int(round(coord)) for coord in box_list]
        x_min, y_min, x_max, y_max = box_list
        labels = result['pred_densepose'][0].labels
        uv = result['pred_densepose'][0].uv
        mask_h, mask_w = labels.shape
        raw_h, raw_w, _ = img.shape

        #convert label to float
        labels = labels.float()/24.0
        output_tensor = torch.zeros(3, raw_h, raw_w, device=self.device_str)
        self._safe_place_tensor(output_tensor[0], labels, x_min, y_min, mask_w, mask_h, raw_w, raw_h)
        self._safe_place_tensor(output_tensor[1:], uv, x_min, y_min, mask_w, mask_h, raw_w, raw_h)

        return output_tensor

    def get_soft_map(self,img, isRGB=False):
        if isRGB:
            img = img[:,:,[2,1,0]] # convert to BGR
        with torch.no_grad():
            outputs = self.predictor(img)["instances"]
        image_fpath = "file_name"
        result = {"file_name": image_fpath}
        if outputs.has("scores"):
            result["scores"] = outputs.get("scores").cpu()
        if outputs.has("pred_boxes"):
            result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
            if outputs.has("pred_densepose"):
                if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
                    extractor = DensePoseResultExtractor()
                    #print("yes")
                elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
                    extractor = DensePoseOutputsExtractor()
                coarse_segm = outputs.pred_densepose.coarse_segm
                fine_segm = outputs.pred_densepose.fine_segm

                #result["pred_densepose"] = extractor(outputs)[0]
        if len(result['pred_boxes_XYXY'])==0:
            return None
        max_id = self.get_max_index(result['pred_boxes_XYXY'])
        # print("Box:",result['pred_boxes_XYXY'][0])
        box_list = (result['pred_boxes_XYXY'][max_id]).tolist()
        box_list = [int(round(coord)) for coord in box_list]
        x_min, y_min, x_max, y_max = box_list

        w = max(int(x_max-x_min), 1)
        h = max(int(y_max-y_min), 1)
        # coarse segmentation
        coarse_segm_bbox = F.interpolate(
            coarse_segm,
            (h, w),
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)
        # combined coarse and fine segmentation
        #labels = (
        #        F.interpolate(fine_segm, (h, w), mode="bilinear", align_corners=False).argmax(dim=1)
        #        * (coarse_segm_bbox > 0).long()
        #)
        #print(F.interpolate(fine_segm, (h, w), mode="bilinear", align_corners=False).shape)
        #print((coarse_segm_bbox > 0).shape)
        soft_map = F.interpolate(fine_segm[[max_id]], (h, w), mode="bilinear", align_corners=False)#* (coarse_segm_bbox[[max_id]] > 0).long()
        soft_map = soft_map[0]#CHW
        raw_h, raw_w, _ = img.shape
        output_map = torch.zeros((25,raw_h,raw_w), dtype=torch.float32, device=self.device_str)
        for i in range(25):
            maxv=soft_map[i].max().item()
            minv = soft_map[i].min().item()
            if maxv > minv:
                soft_map[i] = (soft_map[i] - minv)/(maxv - minv)

        self._safe_place_tensor(output_map, soft_map, x_min, y_min, w, h, raw_w, raw_h)
        torsoleghead_index = [1, 2, 5, 6, 7, 9, 8, 10, 11, 13, 12, 14, 23, 24]
        left_arm_index = [4, 15, 17, 19, 21]
        right_arm_index = [3, 16, 18, 20, 22]
        r_channel = output_map[left_arm_index, :,:].max(dim=0)[0].cpu().numpy()
        b_channel = output_map[right_arm_index, :,:].max(dim=0)[0].cpu().numpy()
        g_channel = output_map[torsoleghead_index, :,:].max(dim=0)[0].cpu().numpy()
        result_img = np.concatenate((r_channel[:,:,np.newaxis],g_channel[:,:,np.newaxis],b_channel[:,:,np.newaxis]), axis=2)

        result_img=(result_img*255).astype(np.uint8)

        return result_img



    def get_IUV(self, img, isRGB=False, target_center=None):
        if isRGB:
            img = img[:, :, [2, 1, 0]]  # convert to BGR
        with torch.no_grad():
            outputs = self.predictor(img)["instances"]
        image_fpath = "file_name"
        result = {"file_name": image_fpath}
        if outputs.has("scores"):
            result["scores"] = outputs.get("scores").cpu()
        if outputs.has("pred_boxes"):
            result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
            if outputs.has("pred_densepose"):
                if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
                    extractor = DensePoseResultExtractor()
                elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
                    extractor = DensePoseOutputsExtractor()
                result["pred_densepose"] = extractor(outputs)[0]

        if len(result['pred_boxes_XYXY']) == 0:
            return None

        # Synchronize target subject selection with SMPL joints
        if target_center is not None and len(result['pred_boxes_XYXY']) > 1:
            boxes = result['pred_boxes_XYXY'].numpy()
            box_centers = np.stack([(boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5], axis=1)
            dists = np.linalg.norm(box_centers - np.array(target_center, dtype=np.float32), axis=1)
            target_id = int(np.argmin(dists))
        else:
            target_id = self.get_max_index(result['pred_boxes_XYXY'])

        box_list = (result['pred_boxes_XYXY'][target_id]).tolist()
        box_list = [int(round(coord)) for coord in box_list]
        x_min, y_min, x_max, y_max = box_list
        labels = result['pred_densepose'][target_id].labels
        uv = result['pred_densepose'][target_id].uv
        mask_h, mask_w = labels.shape
        raw_h, raw_w, _ = img.shape

        # convert label to float
        labels = labels.float()
        output_tensor = torch.zeros(3, raw_h, raw_w, device=self.device_str)
        self._safe_place_tensor(output_tensor[0], labels, x_min, y_min, mask_w, mask_h, raw_w, raw_h)
        self._safe_place_tensor(output_tensor[1:], uv * 255.0, x_min, y_min, mask_w, mask_h, raw_w, raw_h)
        IUV = output_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        return IUV

    def get_max_index(self, boxes):
        areas=[]
        for i in range(len(boxes)):
            box=boxes[i]
            area=(box[2]-box[0])*(box[3]-box[1])
            areas.append(area)
        return np.argmax(areas)


    def IUV2img(self,IUV:np.ndarray):
        IUV=IUV.astype(np.float32)
        IUV[:,:,0]/=24.0
        IUV[:,:,0]*=255
        IUV=IUV.astype(np.uint8)
        return IUV

    def get_dp_img(self,img,isRGB=False):
        return self.IUV2img(self.get_IUV(img,isRGB))

    def get_hand_mask(self,img):
        # input must be BGR
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = self.predictor(img)["instances"]#BGR fromat
        image_fpath = "file_name"
        result = {"file_name": image_fpath}
        if outputs.has("scores"):
            result["scores"] = outputs.get("scores").cpu()
        if outputs.has("pred_boxes"):
            result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
            if outputs.has("pred_densepose"):
                if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
                    extractor = DensePoseResultExtractor()
                elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
                    extractor = DensePoseOutputsExtractor()
                result["pred_densepose"] = extractor(outputs)[0]

        raw_h, raw_w, _ = img.shape
        if len(result['pred_boxes_XYXY']) == 0:
            return np.zeros([raw_h, raw_w]).astype(bool)

        box_list = (result['pred_boxes_XYXY'][0]).tolist()
        box_list = [int(round(coord)) for coord in box_list]
        x_min, y_min, x_max, y_max = box_list
        labels = result['pred_densepose'][0].labels
        uv = result['pred_densepose'][0].uv
        mask_h, mask_w = labels.shape

        labels = labels.cpu().numpy().astype(np.uint8)
        raw_labels = np.zeros([raw_h, raw_w], dtype=np.uint8)
        dst_y1, dst_y2 = max(0, min(raw_h, y_min)), max(0, min(raw_h, y_min + mask_h))
        dst_x1, dst_x2 = max(0, min(raw_w, x_min)), max(0, min(raw_w, x_min + mask_w))
        src_y1, src_x1 = dst_y1 - y_min, dst_x1 - x_min
        if dst_y2 > dst_y1 and dst_x2 > dst_x1:
            raw_labels[dst_y1:dst_y2, dst_x1:dst_x2] = labels[src_y1:src_y1 + (dst_y2 - dst_y1), src_x1:src_x1 + (dst_x2 - dst_x1)]

        hand_mask = (raw_labels==3)|(raw_labels==4)

        return hand_mask

    def get_vis_img(self,img_path):
        output_tensor = self.forward(img_path).cpu()
        output_tensor = output_tensor.permute(1,2,0)*255
        output_img = output_tensor.numpy().astype(np.uint8)


        cv2.imwrite('seg.jpg',output_img)

    def get_bbox(self,img, isRGB=False):
        if isRGB:
            img = img[:,:,[2,1,0]] # convert to BGR
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = self.predictor(img)["instances"]
        image_fpath = "file_name"
        result = {"file_name": image_fpath}
        if outputs.has("scores"):
            result["scores"] = outputs.get("scores").cpu()
        if outputs.has("pred_boxes"):
            result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
            if outputs.has("pred_densepose"):
                if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
                    extractor = DensePoseResultExtractor()
                elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
                    extractor = DensePoseOutputsExtractor()
                result["pred_densepose"] = extractor(outputs)[0]

        if len(result['pred_boxes_XYXY'])==0:
            return None, None
        max_id = self.get_max_index(result['pred_boxes_XYXY'])
        box_list = (result['pred_boxes_XYXY'][max_id]).tolist()
        res_box_list=copy.deepcopy(box_list)

        box_list = [int(round(coord)) for coord in box_list]
        x_min, y_min, x_max, y_max = box_list
        labels = result['pred_densepose'][max_id].labels
        uv = result['pred_densepose'][max_id].uv
        mask_h, mask_w = labels.shape
        raw_h, raw_w, _ = img.shape

        labels = labels.float()
        output_tensor = torch.zeros(3, raw_h, raw_w, device=self.device_str)
        self._safe_place_tensor(output_tensor[0], labels, x_min, y_min, mask_w, mask_h, raw_w, raw_h)
        self._safe_place_tensor(output_tensor[1:], uv * 255.0, x_min, y_min, mask_w, mask_h, raw_w, raw_h)
        IUV = output_tensor.permute(1,2,0).cpu().numpy().astype(np.uint8)

        return res_box_list, IUV


    def get_trans2roi(self,img,new_h, new_w,isRGB=False):
        bbox, IUV = self.get_bbox(img,isRGB)
        if bbox is None:
            return None, None
        x_min, y_min, x_max, y_max = bbox
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        if (y_max - y_min) / ((x_max - x_min) + 1e-5) > (new_h / new_w):  # Too tall
            half_y = (y_max - y_center) * 1.1
            half_x = half_y * new_w / new_h
        else:
            half_x = (x_max - x_center) * 1.1
            half_y = half_x * new_h / new_w
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
        roi_IUV = cv2.warpAffine(IUV, trans, (new_w, new_h),
                                 flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
        return trans, inv_trans, roi_IUV

# 1, 2 = Torso, 3 = Right Hand, 4 = Left Hand, 5 = Left Foot, 6 = Right Foot, 7, 9 = Upper Leg Right, 8, 10 = Upper Leg Left, 11, 13 = Lower Leg Right, 12, 14 = Lower Leg Left, 15, 17 = Upper Arm Left, 16, 18 = Upper Arm Right, 19, 21 = Lower Arm Left, 20, 22 = Lower Arm Right, 23, 24 = Head;