import numpy as np
import torch
import torch.nn as nn
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .ASAPNet_model import ASAPNetModel
import random


class ASAPNet_RGBA(ASAPNetModel):
    def name(self):
        return 'ASAPNet_RGBA'

    def forward(self, heatmaps, image, infer=False):
        # Encode Inputs
        #input_label, inst_map, real_image, feat_map = self.encode_input(label, inst, image, feat)
        real_input = image
        real_image = image[:, [0, 1, 2], :, :]
        real_mask = image[:, [3], :, :]

        # Fake Generation
        fake_output = self.netG.forward(heatmaps)
        fake_image = fake_output[:, [0, 1, 2], :, :]
        fake_mask = fake_output[:, [3], :, :]

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(heatmaps, fake_output, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)

        # Real Detection and Loss        
        pred_real = self.discriminate(heatmaps, real_input)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)        
        pred_fake = self.netD.forward(torch.cat((heatmaps, fake_output), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i]) - 1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                                       self.criterionFeat(pred_fake[i][j],
                                                          pred_real[i][j].detach()) * self.opt.lambda_feat

        # VGG feature matching loss
        loss_G_VGG = 0
        normalized = True
        if not self.opt.no_vgg_loss:
            if normalized:
                loss_G_VGG = self.criterionVGG((fake_image * 0.5 + 1.0), #* fake_mask,
                                               (real_image * 0.5 + 1.0) * real_mask) * self.opt.lambda_feat
            else:
                loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat
        loss_G_VGG +=  self.criterionVGG(real_mask, fake_mask) * self.opt.lambda_feat

        local_loss=False
        c=1.5
        if local_loss:
            local_loss_G_GAN, local_loss_G_GAN_Feat, local_loss_G_VGG, local_loss_D_real, local_loss_D_fake=self.localLoss(heatmaps,image,fake_output)
            loss_G_GAN+=local_loss_G_GAN*c
            loss_G_GAN_Feat+=local_loss_G_GAN_Feat*c
            loss_G_VGG+=local_loss_G_VGG*c
            loss_D_real+=local_loss_D_real*c
            loss_D_fake+=local_loss_D_fake*c


        # Only return the fake_B image if necessary to save BW
        return [self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake),
                None if not infer else fake_output]


    def localLoss(self,heatmaps, gt_output, fake_output):
        heatmaps, gt_output, fake_output = self.RandomCrop([heatmaps, gt_output, fake_output],size=360)
        heatmaps=nn.functional.interpolate(heatmaps, scale_factor=None,size=(1024,1024), mode='bilinear',align_corners=True)
        gt_output=nn.functional.interpolate(gt_output, scale_factor=None,size=(1024,1024), mode='bilinear',align_corners=True)
        fake_output=nn.functional.interpolate(fake_output, scale_factor=None,size=(1024,1024), mode='bilinear',align_corners=True)

        real_input = gt_output
        real_image = gt_output[:, [0, 1, 2], :, :]
        real_mask = gt_output[:, [3], :, :]

        fake_image = fake_output[:, [0, 1, 2], :, :]
        fake_mask = fake_output[:, [3], :, :]
        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(heatmaps, fake_output, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)

        # Real Detection and Loss
        pred_real = self.discriminate(heatmaps, real_input)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
        pred_fake = self.netD.forward(torch.cat((heatmaps, fake_output), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i]) - 1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                                       self.criterionFeat(pred_fake[i][j],
                                                          pred_real[i][j].detach()) * self.opt.lambda_feat

        # VGG feature matching loss
        loss_G_VGG = 0
        normalized = True
        if not self.opt.no_vgg_loss:
            if normalized:
                loss_G_VGG = self.criterionVGG((fake_image * 0.5 + 1.0) * fake_mask,
                                               (real_image * 0.5 + 1.0) * real_mask) * self.opt.lambda_feat
            else:
                loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat
        return [loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake]


    def localVGG(self, img1, img2):
        cropped1, cropped2 = self.RandomCrop([img1, img2],360)
        return self.criterionVGG(cropped1, cropped2)

    def RandomCrop(self, img_list, size):
        _, _, h, w = img_list[0].shape
        assert size<h and size<w
        h_start = random.randint(0, h - size)
        w_start = random.randint(0, w - size)
        cropped_list=[]
        for img in img_list:
            cropped_img = img[:,:,h_start:h_start + size, w_start:w_start + size]
            cropped_list.append(cropped_img)
        return cropped_list


    def inference(self, image=None):
        # Encode Inputs        
        image = Variable(image) if image is not None else None

        input_concat = image

        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(input_concat)
        else:
            fake_image = self.netG.forward(input_concat)
        return fake_image


class InferenceModel(ASAPNet_RGBA):
    def forward(self, inp):
        #label, inst = inp
        #return self.inference(label, inst)
        return self.inference(inp)
