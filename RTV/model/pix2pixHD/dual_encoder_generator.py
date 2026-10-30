import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
from .attention import MultiHeadCrossAttention, MultiHeadSelfAttention



class DualEncoderGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        self.input_nc=input_nc
        assert (n_blocks >= 0)
        super(DualEncoderGenerator, self).__init__()
        activation = nn.ReLU(True)

        encoder1 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            encoder1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]
        self.encoder1 = nn.Sequential(*encoder1)
        encoder2 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf),
                    activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            encoder2 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                         norm_layer(ngf * mult * 2), activation]
        self.encoder2 = nn.Sequential(*encoder1)

        down = [nn.Conv2d(ngf * (2**(n_downsampling-1)) * 2, ngf * (2**(n_downsampling-1)) * 2, kernel_size=3, stride=2, padding=1),
                         norm_layer(ngf * (2**(n_downsampling-1)) * 2), activation]
        self.down = nn.Sequential(*down)
        up = [nn.ConvTranspose2d(ngf * (2**(n_downsampling-1)) * 2, ngf * (2**(n_downsampling-1)) * 2, kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(ngf * (2**(n_downsampling-1)) * 2), activation]
        self.up = nn.Sequential(*up)

        self.MHCA= MultiHeadCrossAttention(ngf * (2**(n_downsampling-1)) * 2,num_heads=1,height=64,width=64)
        self.MHSA = MultiHeadSelfAttention(ngf * (2 ** (n_downsampling - 1)) * 2, num_heads=1, height=64, width=64)

        model = []

        ### resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        input1 = input[:,:self.input_nc,:,:]
        input2 = input[:,self.input_nc:,:,:]
        feat1 = self.encoder1(input1)
        feat2 = self.encoder2(input2)
        feat1 = self.MHSA(feat1)
        feat = self.MHCA(feat2,feat1)
        return self.model(feat)

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out