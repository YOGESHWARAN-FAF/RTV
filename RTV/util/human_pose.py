import argparse
import os
#import pprint
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
#import tools.init_paths
from lib.config import cfg
from lib.config import update_config
#from core.loss import JointsMSELoss
#from core.function import validate
#from core.inference import get_max_preds
#from utils.utils import create_logger

#import dataset
#from datasets.mannequin import MannequinDataset
import lib.models as models
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        # required=True,
                        default='experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml',
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


class HumanPose():
    def __init__(self, ckpt=None,useCUDA=True):
        self.useCUDA=useCUDA

        self.model = self._load_model(ckpt=ckpt)

    def __call__(self, img):
        return self.model(img)


    def _load_model(self, ckpt):
        if ckpt is None:
            #ckpt= 'output/deepfashion2/pose_hrnet/top1_forward_only/2022-08-08-20-32/model_best.pth'
            ckpt = 'pretrained_models/pose_hrnet/human_pose.pth'
        if not os.path.exists(ckpt):
            print("Model not found, downloading from google drive...")
            import gdown
            path, name=os.path.split(ckpt)
            os.makedirs(path,exist_ok=True)
            id = "1_wn2ifmoQprBrFvUCDedjPON4Y6jsN-v"
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
        if not self.useCUDA:
            model.load_state_dict(torch.load(ckpt,map_location=torch.device('cpu')), strict=True)
        else:
            model.load_state_dict(torch.load(ckpt), strict=True)

        model.eval()
        if torch.cuda.is_available() and self.useCUDA:
            model=model.cuda()
        return model


class HumanJointHeatmap:
    def __init__(self):
        self.joint_detector = HumanPose()
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        resize2 = transforms.Resize((256, 256))
        self.joint_pre_transform = transforms.Compose([normalize, resize2])
        self.post_transform = transforms.Resize((512, 512))
        self.joint_id_list = [2, 3, 6, 7, 11, 12, 13, 14]

    def forward(self,img):
        if not isinstance(img,torch.Tensor):
            img = img.astype(np.float32)
            img /= 255.0
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1)
            img = img.unsqueeze(0)
            if torch.cuda.is_available():
                img = img.cuda()
        with torch.no_grad():
            heatmaps = self.joint_detector(self.joint_pre_transform(img))
        heatmaps = heatmaps[:, self.joint_id_list, :, :]
        heatmaps = self.post_transform(heatmaps)
        return heatmaps


if __name__ == '__main__':
    human_pose =HumanPose()
    example_input = torch.zeros(1, 3, 256, 256)
    output = human_pose(example_input)
    print(output.shape)
