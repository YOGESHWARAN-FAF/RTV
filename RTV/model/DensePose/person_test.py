import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..","..","..")))
from util.multithread_video_loader import MultithreadVideoLoader
from util.image2video import Image2VideoWriter
from util.densepose_util import IUV2Img, IUV2UpperBodyImg, IUV2TorsoLeg, IUV2SSDP, IUV2SDP, IUV2SSDP_new
from tqdm import tqdm
import numpy as np
from util.image_caption import ImageCaption
from model.DensePose.densepose_extractor import DensePoseExtractor
from Graphonomy.cihp_human_parser import CihpHumanParser
from util.image_crop import crop_16_9



def main(v_path,up,left,right):
    def crop(img):
        return crop_16_9(img,up,left,right)
    video_loader=MultithreadVideoLoader(v_path,max_height=1024)
    video_name=os.path.split(v_path)[1].split(".")[0]

    video_writer=Image2VideoWriter()
    densepose_extractor=DensePoseExtractor()
    human_parser=CihpHumanParser()
    for i in tqdm(range(len(video_loader))):
        if i > 300:
            break
        frame=video_loader.cap()
        frame=crop(frame)
        IUV = densepose_extractor.get_IUV(frame)
        if IUV is None:
            continue
        #soft_map=densepose_extractor.get_soft_map(frame)
        dp_img = IUV2Img(IUV)
        ssdp_img = IUV2SSDP_new(IUV)
        frame = human_parser.BlurFaceBatch([frame,], isRGB=False)[0]
        result = np.concatenate((frame,dp_img),axis=1)

        video_writer.append(result)
    video_writer.make_video(os.path.join('./model/DensePose',video_name+'_test.mp4'),fps=30)


if __name__ == "__main__":
    main('./VideoData/azuma_1.MOV', up=0.15, left=0.15, right=0.17)
    main('./VideoData/azuma_2.MOV', up=0.15, left=0.15, right=0.17)

    #main('./VideoDatasets/jin/jin_16_test.mp4',up=0.25,left=0.13,right=0.17)
    #main('./VideoDatasets/jin/jin_25_train.mp4',up=0.25,left=0.13,right=0.17)
    #main('./VideoDatasets/LooseGarment/coat_01.mp4',up=0.16,left=0.11,right=0.10)
    #main('./VideoDatasets/LooseGarment/han_01.mp4',up=0.16,left=0.11,right=0.10)
    #main('./VideoDatasets/LooseGarment/korean_01.mp4',up=0.16,left=0.11,right=0.10)

