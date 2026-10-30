import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .pix2pixHD_model import Pix2PixHDModel


class Pix2PixHDModel_Align(Pix2PixHDModel):
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
        '''
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)
        '''
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, 'alignment',
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)




        if self.opt.verbose:
            print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)


        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)

            #self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            #self.criterionFeat = torch.nn.L1Loss()


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
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))





    def compute_trans_loss(self,heatmaps,gt_trans):
        loss = self.netG.trans_loss(heatmaps,gt_trans)
        return loss

    def get_affine_matrix(self,heatmaps):
        trans = self.netG.get_affine(heatmaps)
        return trans

    def forward(self, heatmaps, infer=False):
        # Encode Inputs
        # input_label, inst_map, real_image, feat_map = self.encode_input(label, inst, image, feat)


        # Fake Generation
        fake_image = self.netG.forward(heatmaps)

        # Fake Detection and Loss
        #pred_fake_pool = self.discriminate(joint_heatmaps, fake_image, use_pool=True)
        loss_D_fake = 0#self.criterionGAN(pred_fake_pool, False)

        # Real Detection and Loss
        #pred_real = self.discriminate(joint_heatmaps, real_image)
        loss_D_real = 0#self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
        #pred_fake = self.netD.forward(torch.cat((joint_heatmaps, fake_image), dim=1))
        loss_G_GAN = 0#self.criterionGAN(pred_fake, True)

        # GAN feature matching loss
        loss_G_GAN_Feat = 0


        # VGG feature matching loss
        loss_G_VGG = 0
        #if not self.opt.no_vgg_loss:
            #loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat





        # Only return the fake_B image if necessary to save BW
        return [self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake),
                None if not infer else fake_image]

    def inference(self, image,offset=False):

        with torch.no_grad():
            if offset:
                fake_image = self.netG.forward_with_offset(image)
            else:
                fake_image = self.netG.forward(image)

        return fake_image

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr



class InferenceModel_Align(Pix2PixHDModel_Align):
    def forward(self, inp):

        return self.inference(inp)
