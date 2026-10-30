import os

import cv2
import numpy as np
import imageio

class Image2VideoWriter():
    def __init__(self,quality=5):
        self.image_list = []
        self.quality=quality

    def append(self,image,isRGB=False):
        if not isRGB:
            image = image[:, :, [2,1,0]]
        self.image_list.append(image)

    def make_video(self,outvid=None, fps=5):


        writer = imageio.get_writer(outvid, fps=fps, codec='libx264',quality=self.quality)

        for image in self.image_list:
            writer.append_data(image)
        writer.close()

class StreamImage2Video():
    def __init__(self,outvid,fps=30,quality=5):
        self.outvid = outvid
        self.quality=quality

        self.fps = fps
        self.vid=imageio.get_writer(outvid, fps=fps, codec='libx264',quality=self.quality)
        self.image_list = []

    def append(self,image,isRGB=False):
        if not isRGB:
            image = image[:, :, [2,1,0]]
        self.vid.append_data(image)

    def make_video(self):

        self.vid.close()

