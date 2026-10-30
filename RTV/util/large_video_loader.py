import cv2
import numpy as np
# from util.garment_heatmap import HeatmapGenerator
import torch
import torchvision.transforms as transforms
import ffmpeg
class LargeVideoLoader:
    def __init__(self, path,max_height=None):
        if isinstance(path,list):
            self.path_list = path
        else:
            self.path_list=[path]
        self.cap_list=[cv2.VideoCapture(p) for p in self.path_list]
        self.nframe_list=[int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in self.cap_list]
        self.idx=0
        self.max_height=max_height

    def global2local(self,i):
        n_cap=0
        assert i<=self.__len__()-1
        for n in self.nframe_list:
            if i<=n-1:
                return n_cap, i
            else:
                i=i-n
                n_cap+=1



    def cap(self):
        n_cap,n_frame=self.global2local(self.idx)
        self.idx+=1


        #self.cap_list[n_cap].set(cv2.CAP_PROP_POS_FRAMES, n_frame - 1)
        res, frame = self.cap_list[n_cap].read()
        h=frame.shape[0]
        w=frame.shape[1]
        need_resize=False
        if self.max_height is not None:
            if h>self.max_height:
                need_resize=True
        if need_resize:
            tmp = h
            h = self.max_height
            w = int(w* h / tmp)
            frame=cv2.resize(frame,(w,h))
        return frame

    def __len__(self):
        nframes=0
        for n in self.nframe_list:
            nframes+=n
        return nframes

    def __del__(self):
        for cap in self.cap_list:
            cap.release()
