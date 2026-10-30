import time

import cv2
import numpy as np
import torch
from threading import Thread
import queue
class MultithreadVideoLoader:
    def __init__(self, path,max_height=None):
        self.stop = False
        if isinstance(path,list):
            self.path_list = path
        else:
            self.path_list=[path]
        self.cap_list=[cv2.VideoCapture(p) for p in self.path_list]
        self.fps_list = [cap.get(cv2.CAP_PROP_FPS) for cap in self.cap_list]
        self.nframe_list=[int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in self.cap_list]
        self.nframe=self.__len__()

        self.max_height=max_height

        self.buff_size=64
        self.frame_queue=queue.Queue(maxsize=self.buff_size)
        self.t=Thread(target=self.load_buff,args=())
        self.t.daemon=True
        self.t.start()

    def get_fps(self):
        return self.fps_list[0]




    def load_buff(self):
        i=0
        while True:
            if self.stop:
                break
            if i>self.nframe-1:
                break
            ret, frame = self.get_item(i)
            if not ret:
                break
            self.frame_queue.put(frame)
            i+=1
        while True:
            time.sleep(1)
            if self.stop:
                break


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
        try:
            result = self.frame_queue.get(timeout=1)
        except queue.Empty:
            result = None

        return result

    def get_item(self, idx):
        n_cap,n_frame=self.global2local(idx)


        #self.cap_list[n_cap].set(cv2.CAP_PROP_POS_FRAMES, n_frame - 1)
        #sequential loading is faster!
        res, frame = self.cap_list[n_cap].read()
        if not res:
            return res, frame
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
        return res, frame

    def __len__(self):
        nframes=0
        for n in self.nframe_list:
            nframes+=n
        return nframes

    def close(self):
        self.stop=True
        while not self.frame_queue.empty():
            self.frame_queue.get()
        self.t.join()
        for cap in self.cap_list:
            cap.release()

