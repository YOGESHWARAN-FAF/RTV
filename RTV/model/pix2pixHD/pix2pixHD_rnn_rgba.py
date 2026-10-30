import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .pix2pixHD_model import Pix2PixHDModel
from OpticalFlow.optical_flow import OpticalFlow
import torch.nn as nn

class Pix2PixHD_RNN_RGBA(Pix2PixHDModel):
    def name(self):
        return 'Pix2PixHD_RNN_RGBA'

    def reset(self):
        self.netG.reset()

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        input_nc = opt.input_nc  # opt.label_nc if opt.label_nc != 0 else opt.input_nc

        ##### define networks
        # Generator network
        netG_input_nc = input_nc
        # if not opt.no_instance:
        #    netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, 'rnn',
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            # if not opt.no_instance:
            #    netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        ### Encoder network
        if self.gen_features:
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder',
                                          opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)
        if self.opt.verbose:
            print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)

                # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)

            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake')

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:
                import sys
                if sys.version_info >= (3, 0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [value]
                        finetune_list.add(key.split('.')[0])
                print(
                    '------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))
            else:
                params = list(self.netG.parameters())
            if self.gen_features:
                params += list(self.netE.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optical_flow = OpticalFlow(small=True)
        self.l1loss = nn.L1Loss()

    def temporal_loss(self, fake_image,real_image):
        # 5D input
        batch_size, length,c,_,_ = real_image.shape
        if c==1:
            real_image = real_image[:,:,[0,0,0],:,:]
            fake_image = fake_image[:, :, [0, 0, 0], :, :]
        if length<=1:
            return 0
        else:
            real_flow = self.optical_flow.diff_batch_forward(real_image,normalized=True)
            fake_flow = self.optical_flow.diff_batch_forward(fake_image,normalized=True)
            return self.l1loss(fake_flow,real_flow.detach())


    def train_forward(self,input,image,infer=False):
        batch_size, length,c,h,w = input.shape
        #x = x.view(-1, *x.shape[2:])
        max_length_for_loss= 8
        if length>8:
            for i in range(length-8):
                with torch.no_grad():
                    _ = self.netG.inference_forward(input[:,i,:,:,:])
            input = input[:,length-8:,:,:,:]
            image = image[:, length - 8:, :, :, :]

        real_input = image
        real_image = image[:,:, [0, 1, 2], :, :]
        real_mask = image[:,:, [3], :, :]

        # Fake Generation
        fake_output = self.netG.forward(input)
        fake_image = fake_output[:,:, [0, 1, 2], :, :]
        fake_mask = fake_output[:,:, [3], :, :]

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(input.view(-1,*input.shape[2:]), fake_output.view(-1,*fake_output.shape[2:]), use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)

        # Real Detection and Loss
        pred_real = self.discriminate(input.view(-1,*input.shape[2:]), image.view(-1,*image.shape[2:]))
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
        pred_fake = self.netD.forward(torch.cat((input.view(-1,*input.shape[2:]), fake_output.view(-1,*fake_output.shape[2:])), dim=1))
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
        optical_loss = self.temporal_loss(fake_image, real_image) + self.temporal_loss(fake_mask, real_mask)
        if not self.opt.no_vgg_loss:
            fake_image = fake_image.view(-1,*fake_image.shape[2:])
            fake_mask = fake_mask.view(-1,*fake_mask.shape[2:])
            real_image = real_image.view(-1,*real_image.shape[2:])
            real_mask = real_mask.view(-1,*real_mask.shape[2:])
            if normalized:
                loss_G_VGG = self.criterionVGG((fake_image * 0.5 + 1.0),
                                               (real_image * 0.5 + 1.0) ) * self.opt.lambda_feat

            else:
                loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat
            loss_G_VGG += nn.L1Loss()(fake_mask, real_mask) * self.opt.lambda_feat

        # Only return the fake_B image if necessary to save BW

        loss_G_VGG +=optical_loss

        self.netG.reset()

        return [self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake),
                None if not infer else fake_output]






    def inference(self, image=None):
        if image is None:
            return None
        with torch.inference_mode():
            fake_image = self.netG.inference_forward(image)
        return fake_image



class InferenceModel(Pix2PixHD_RNN_RGBA):
    def forward(self, inp):
        #label, inst = inp
        #return self.inference(label, inst)
        return self.inference(inp)

        
