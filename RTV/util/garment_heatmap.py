import argparse
import os
#import pprint
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import tools.init_paths
from lib.config import cfg
from lib.config import update_config
#from core.loss import JointsMSELoss
#from core.function import validate
#from core.inference import get_max_preds
#from utils.utils import create_logger

#import dataset
#from datasets.mannequin import MannequinDataset
#import models
import lib.models as models

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        # required=True,
                        default='experiments/deepfashion2/hrnet/top1_only.yaml',
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args(args=[])
    return args


class HeatmapGenerator():
    def __init__(self, ckpt=None,useCUDA=True):
        self.useCUDA=useCUDA

        self.model = self._load_model(ckpt=ckpt)

    def __call__(self, img):
        return self.model(img)


    def _load_model(self, ckpt):
        if ckpt is None:
            #ckpt= 'output/deepfashion2/pose_hrnet/top1_forward_only/2022-08-08-20-32/model_best.pth'
            ckpt = 'pretrained_models/pose_hrnet/model_best.pth'
        if not os.path.exists(ckpt):
            print("Model not found, downloading from google drive...")
            import gdown
            path, name=os.path.split(ckpt)
            os.makedirs(path,exist_ok=True)
            id = "1voMopc-Cuq1Uls0ZAS-sDTCrHJ1M8zJJ"
            output = ckpt
            gdown.download(
                f"https://drive.google.com/uc?export=download&confirm=pbef&id="+id,
                output
            )
        args = parse_args()
        args.opts=[]
        update_config(cfg, args)



        # cudnn related setting
        cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

        model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
            cfg, is_train=False
        )
        model.load_state_dict(torch.load(ckpt), strict=True)
        model.eval()
        if torch.cuda.is_available() and self.useCUDA:
            model=model.cuda()
        return model