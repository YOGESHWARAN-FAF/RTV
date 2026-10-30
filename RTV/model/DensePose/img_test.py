import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..","..","..")))
from util.multithread_video_loader import MultithreadVideoLoader
from util.image2video import Image2VideoWriter
from util.densepose_util import IUV2Img, IUV2UpperBodyImg, IUV2TorsoLeg, IUV2SSDP, IUV2SDP
from tqdm import tqdm
import numpy as np
from util.image_caption import ImageCaption
from model.DensePose.densepose_extractor import DensePoseExtractor
from Graphonomy.cihp_human_parser import CihpHumanParser
import cv2

def crop(img:np.ndarray,up=0.,down=0,left=0.0,right=0.0):
    new_img=img.copy()
    height, width = new_img.shape[:2]
    n_row = int(height * up)
    n_left = int(width * left)
    n_right = int(width * right)
    n_down = int(height * down)
    new_img=new_img[n_row:height-n_down,n_left:width-n_right:,:]
    return new_img

def main(img_path):
    img=cv2.imread(img_path)
    img_name=os.path.split(img_path)[1].split(".")[0]


    densepose_extractor=DensePoseExtractor()
    human_parser=CihpHumanParser()

    frame=img
    frame=crop(frame)
    IUV = densepose_extractor.get_IUV(frame)
    if IUV is None:
        return
        #soft_map=densepose_extractor.get_soft_map(frame)
    dp_img = IUV2Img(IUV)
    ssdp_img = IUV2SSDP(IUV)
    frame = human_parser.BlurFaceBatch([frame,], isRGB=False)[0]
    result = np.concatenate((frame,dp_img),axis=1)

    cv2.imwrite(img_name+"_dp.jpg",result)


if __name__ == "__main__":
    #main('./VideoDatasets/jin/jin_16_test.mp4')
    main('./loose3.jpg')
    #v_path_list=[]
    #video0_dir='./tryon_videos/jin'
    #video1_dir = './tryon_videos/lab'
    #for i in range(16,27):
    #    video_name='jin_'+str(i).zfill(2)+'_train.mp4'
    #    v_path_list.append(os.path.join(video0_dir,video_name))
    #for v_path in v_path_list:
    #    main(v_path)