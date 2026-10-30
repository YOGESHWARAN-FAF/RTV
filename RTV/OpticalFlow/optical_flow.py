from torchvision.models.optical_flow import raft_large, raft_small
from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights
import torch
import torchvision.transforms.functional as F
import numpy as np
from torchvision.utils import flow_to_image


class OpticalFlow:
    def __init__(self, small=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not small:
            model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(self.device)
        else:
            model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(self.device)
        self.model = model.eval()
        self.model.requires_grad_(True)
        weights = Raft_Large_Weights.DEFAULT if not small else Raft_Small_Weights.DEFAULT
        self.transforms = weights.transforms()

    def diff_batch_forward(self,imgs,normalized=True):
        batch_size, length,_,_,_ = imgs.shape
        imgs1 = imgs[:,:-1].view(batch_size*(length-1),*imgs.shape[2:])
        imgs2 = imgs[:,1:].view(batch_size*(length-1),*imgs.shape[2:])
        if normalized:
            imgs1 = (imgs1 + 1.0)*0.5
            imgs2 = (imgs2 + 1.0) * 0.5

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        imgs1 = F.normalize(imgs1, mean=mean, std=std)
        imgs2 = F.normalize(imgs2, mean=mean, std=std)


        with torch.set_grad_enabled(True):
            flow = self.model(imgs1, imgs2)[-1]
            #print(flow[0].shape)
            #print(length)
            flow = flow.view(batch_size, length-1,*flow.shape[1:])
        return flow


    def batch_forward(self, img1, img2):
        h = img1.shape[2]
        w = img2.shape[3]
        #print(img1.shape,img2.shape)


        new_h = (h // 8) * 8
        new_w = (w // 8) * 8
        img1 = F.resize(img1, size=[new_h, new_w], antialias=False)
        img2 = F.resize(img2, size=[new_h, new_w], antialias=False)
        img1, img2 = self.transforms(img1, img2)
        with torch.no_grad():
            flow_list = self.model(img1.to(self.device), img2.to(self.device))
        predicted_flow = flow_list[-1]
        predicted_flow[:, 0, :, :] *= w / new_w
        predicted_flow[:, 1, :, :] *= h / new_h
        # each entry corresponds to the horizontal and vertical displacement of each pixel from the first image to the second image. Note that the predicted flows are in “pixel” unit, they are not normalized w.r.t. the dimensions of the images.
        predicted_flow = F.resize(predicted_flow, size=[h, w], antialias=False)
        return predicted_flow

    def __call__(self, img1, img2):
        h = img1.shape[0]
        w = img2.shape[1]
        assert img1.shape == img2.shape
        img1 = torch.from_numpy(img1.astype(np.float32)/255).permute(2,0,1).unsqueeze(0)
        img2 = torch.from_numpy(img2.astype(np.float32)/255).permute(2,0,1).unsqueeze(0)

        new_h = (h // 8) * 8
        new_w = (w // 8) * 8
        img1 = F.resize(img1, size=[new_h, new_w], antialias=False)
        img2 = F.resize(img2, size=[new_h, new_w], antialias=False)
        img1, img2 = self.transforms(img1, img2)
        with torch.no_grad():
            flow_list = self.model(img1.to(self.device), img2.to(self.device))
        predicted_flow = flow_list[-1]
        predicted_flow[:, 0, :, :] *= w / new_w
        predicted_flow[:, 1, :, :] *= h / new_h
        # each entry corresponds to the horizontal and vertical displacement of each pixel from the first image to the second image. Note that the predicted flows are in “pixel” unit, they are not normalized w.r.t. the dimensions of the images.
        predicted_flow = F.resize(predicted_flow, size=[h, w], antialias=False)
        return predicted_flow

    def get_flow_image(self,img1,img2):
        img_tensor = flow_to_image(self.__call__(img1,img2))
        img = img_tensor.squeeze(0).permute(1,2,0).cpu().numpy()
        img = img.astype(np.uint8)
        return img


