import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from torch.nn.functional import normalize


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.ConvTranspose2d) and m.in_channels == m.out_channels:
        initial_weight = get_upsampling_weight(
            m.in_channels, m.out_channels, m.kernel_size[0])
        m.weight.data.copy_(initial_weight)


class gen_conv2(nn.Conv2d):
    def __init__(self, cin, cout, ksize, stride=1, rate=1, activation=nn.ELU(),icnr = False):
        """Define conv for generator

        Args:
            cin: Input Channel number.
            cout: output Channel number.
            ksize: Kernel size.
            Stride: Convolution stride.
            rate: Rate for or dilated conv.
            activation: Activation function after convolution.
        """
        p = int(rate*(ksize-1)/2)
        if cout == 3 or activation is None:
          if icnr:
            super(gen_conv2, self).__init__(in_channels=cin, out_channels=cout, kernel_size=ksize, stride=stride, padding=p, dilation=rate, groups=1, bias=True)
            # kernel = ICNR(self.weight)
            # self.weight.data.copy_(kernel)
          else:
            super(gen_conv2, self).__init__(in_channels=cin, out_channels=cout, kernel_size=ksize, stride=stride, padding=p, dilation=rate, groups=1, bias=True)
        else:
          if icnr:
            super(gen_conv2, self).__init__(in_channels=cin, out_channels=int(cout/2), kernel_size=ksize, stride=stride, padding=p, dilation=rate, groups=1, bias=True)
            self.mask_conv2d = nn.Conv2d(in_channels=cin, out_channels=int(cout/2), kernel_size=1, stride=stride, padding=0, dilation=rate, groups=1, bias=True)
            # kernel = ICNR(self.weight)

            # self.weight.data.copy_(kernel)
  
          else:
            super(gen_conv2, self).__init__(in_channels=cin, out_channels=int(cout/2), kernel_size=ksize, stride=stride, padding=p, dilation=rate, groups=1, bias=True)
            # self.mask_conv2d = nn.Conv2d(in_channels=cin, out_channels=cin, kernel_size=ksize, stride=stride, padding=p, dilation=rate, groups=cin, bias=True)
            self.mask_conv2d = nn.Conv2d(in_channels=cin, out_channels=int(cout/2), kernel_size=1, stride=stride, padding=0, dilation=rate, groups=1, bias=True)
        self.cout = cout
        self.activation = activation

    def forward(self, input):
        x = super(gen_conv2, self).forward(input)
        if self.cout == 3 or self.activation is None:
            return x
        mask = self.mask_conv2d(input)
        # mask = self.mask_conv2d2(mask)
        # x, y = torch.split(x, int(self.out_channels/2), dim=1)
        x = self.activation(x)
        y = torch.sigmoid(mask)
        x = x * y


        # x = self.conv2d(input)
        # mask = self.mask_conv2d(input)
        # if self.activation is not None:
        #     x = self.activation(x) * self.gated(mask)
        # else:
        #     x = x * self.gated(mask)
        # if self.batch_norm:
        #     return self.batch_norm2d(x)
        # else:
        #     return x

        
        return x
        
class gen_conv1(nn.Conv2d):
    def __init__(self, cin, cout, ksize, stride=1, rate=1, activation=nn.ELU(),icnr = False):
        """Define conv for generator

        Args:
            cin: Input Channel number.
            cout: output Channel number.
            ksize: Kernel size.
            Stride: Convolution stride.
            rate: Rate for or dilated conv.
            activation: Activation function after convolution.
        """
        p = int(rate*(ksize-1)/2)
        if cout == 3 or activation is None:
            if icnr == True:
              super(gen_conv1, self).__init__(in_channels=cin, out_channels=cout, kernel_size=ksize, stride=stride, padding=p, dilation=rate, groups=1, bias=True)
              # kernel = ICNR(self.weight)
              # self.weight.data.copy_(kernel)
            else:
              super(gen_conv1, self).__init__(in_channels=cin, out_channels=cout, kernel_size=ksize, stride=stride, padding=p, dilation=rate, groups=1, bias=True)
        else:
            if icnr == True:
              super(gen_conv1, self).__init__(in_channels=cin, out_channels=int(cout/2), kernel_size=ksize, stride=stride, padding=p, dilation=rate, groups=1, bias=True)
              self.mask_conv2d = nn.Conv2d(in_channels=cin, out_channels=1, kernel_size=ksize, stride=stride, padding=p, dilation=rate, groups=1, bias=True)
             
              # kernel = ICNR(self.weight)
         
               
              # self.weight.data.copy_(kernel)
          
            else:
              super(gen_conv1, self).__init__(in_channels=cin, out_channels=int(cout/2), kernel_size=ksize, stride=stride, padding=p, dilation=rate, groups=1, bias=True)
              # self.mask_conv2d = nn.Conv2d(in_channels=cin, out_channels=cin, kernel_size=ksize, stride=stride, padding=p, dilation=rate, groups=cin, bias=True)
              self.mask_conv2d = nn.Conv2d(in_channels=cin, out_channels=1, kernel_size=ksize, stride=stride, padding=p, dilation=rate, groups=1, bias=True)
        self.cout = cout
        self.activation = activation

    def forward(self, input):
        x = super(gen_conv1, self).forward(input)
        if self.cout == 3 or self.activation is None:
            return x
        mask = self.mask_conv2d(input)
        # mask = self.mask_conv2d2(mask)
        # x, y = torch.split(x, int(self.out_channels/2), dim=1)
        x = self.activation(x)
        y = torch.sigmoid(mask)
        x = x * y


        # x = self.conv2d(input)
        # mask = self.mask_conv2d(input)
        # if self.activation is not None:
        #     x = self.activation(x) * self.gated(mask)
        # else:
        #     x = x * self.gated(mask)
        # if self.batch_norm:
        #     return self.batch_norm2d(x)
        # else:
        #     return x

        
        return x

class depth_to_space(nn.Module):
    def __init__(self):
        super(depth_to_space, self).__init__()
    """
    Implementation of depth to space using numpy

    Parameters:
    -----------
    tensor: input tensor, shape is [N, C, H, W]
    scale_factor: scale factor to up-sample tensor

    Returns:
    --------
    tensor: tensor after pixel shuffle, shape is [N, C/(s*s), s*H, s*W],
        where s refers to scale factor
    """
    def __call__(self,tensor, scale_factor):
        num, ch, height, width = tensor.shape
        if ch % (scale_factor * scale_factor) != 0:
            raise ValueError('channel of tensor must be divisible by '
                            '(scale_factor * scale_factor).')

        new_ch = ch // (scale_factor * scale_factor)
        new_height = height * scale_factor
        new_width = width * scale_factor

        tensor = tensor.reshape(
            [num, scale_factor, scale_factor, new_ch, height, width])
        # new axis: [num, new_ch, height, scale_factor, width, scale_factor]
        tensor = tensor.permute(0, 3, 4, 1, 5, 2)
        tensor = tensor.reshape([num, new_ch, new_height, new_width])
        return tensor

class gen_deconv1(gen_conv1):
    def __init__(self, cin, cout):
        """Define deconv for generator.
        The deconv is defined to be a x2 resize_nearest_neighbor operation with
        additional gen_conv operation.

        Args:
            cin: Input Channel number.
            cout: output Channel number.
            ksize: Kernel size.
        """
        super(gen_deconv1, self).__init__(cin, cout*4, ksize=3,icnr = True)
        

    def forward(self, x):
        # x = nn.functional.interpolate(x, scale_factor=2)
        x = super(gen_deconv1, self).forward(x)
        x = depth_to_space()(x,2)

        return x

# def ICNR(tensor, upscale_factor=2, inizializer=nn.init.kaiming_normal):
#     """Fills the input Tensor or Variable with values according to the method
#     described in "Checkerboard artifact free sub-pixel convolution"
#     - Andrew Aitken et al. (2017), this inizialization should be used in the
#     last convolutional layer before a PixelShuffle operation
#     Args:
#         tensor: an n-dimensional torch.Tensor or autograd.Variable
#         upscale_factor: factor to increase spatial resolution by
#         inizializer: inizializer to be used for sub_kernel inizialization
#     Examples:
#         >>> upscale = 8
#         >>> num_classes = 10
#         >>> previous_layer_features = Variable(torch.Tensor(8, 64, 32, 32))
#         >>> conv_shuffle = Conv2d(64, num_classes * (upscale ** 2), 3, padding=1, bias=0)
#         >>> ps = PixelShuffle(upscale)
#         >>> kernel = ICNR(conv_shuffle.weight, scale_factor=upscale)
#         >>> conv_shuffle.weight.data.copy_(kernel)
#         >>> output = ps(conv_shuffle(previous_layer_features))
#         >>> print(output.shape)
#         torch.Size([8, 10, 256, 256])
#     .. _Checkerboard artifact free sub-pixel convolution:
#         https://arxiv.org/abs/1707.02937
#     """
#     new_shape = [int(tensor.shape[0] / (upscale_factor ** 2))] + list(tensor.shape[1:])
#     subkernel = torch.empty(new_shape)
#     subkernel = inizializer(subkernel)
#     subkernel = subkernel.transpose(0, 1)

#     subkernel = subkernel.contiguous().view(subkernel.shape[0],
#                                             subkernel.shape[1], -1)

#     kernel = subkernel.repeat(1, 1, upscale_factor ** 2)

#     transposed_shape = [tensor.shape[1]] + [tensor.shape[0]] + list(tensor.shape[2:])
#     kernel = kernel.contiguous().view(transposed_shape)

#     kernel = kernel.transpose(0, 1)

#     return kernel

def ICNR(tensor, scale_factor=2, initializer=nn.init.kaiming_normal_):
    OUT, IN, H, W = tensor.shape
    sub = torch.zeros(OUT//scale_factor**2, IN, H, W)
    sub = initializer(sub)
    
    kernel = torch.zeros_like(tensor)
    for i in range(OUT):
        kernel[i] = sub[i//scale_factor**2]
        
    return kernel

class gen_deconv1_origin(gen_conv1):
    def __init__(self, cin, cout):
        """Define deconv for generator.
        The deconv is defined to be a x2 resize_nearest_neighbor operation with
        additional gen_conv operation.

        Args:
            cin: Input Channel number.
            cout: output Channel number.
            ksize: Kernel size.
        """
        super(gen_deconv1_origin, self).__init__(cin, cout, ksize=3)
        

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2)
        x = super(gen_deconv1_origin, self).forward(x)
        # x = depth_to_space()(x,2)

        return x

class gen_deconv2(gen_conv2):
    def __init__(self, cin, cout):
        """Define deconv for generator.
        The deconv is defined to be a x2 resize_nearest_neighbor operation with
        additional gen_conv operation.

        Args:
            cin: Input Channel number.
            cout: output Channel number.
            ksize: Kernel size.
        """
        super(gen_deconv2, self).__init__(cin, cout*4, ksize=3,icnr = True)

    def forward(self, x):
        # x = nn.functional.interpolate(x, scale_factor=2)
        x = super(gen_deconv2, self).forward(x)
        x = depth_to_space()(x,2)

        return x

class gen_deconv2_origin(gen_conv2):
    def __init__(self, cin, cout):
        """Define deconv for generator.
        The deconv is defined to be a x2 resize_nearest_neighbor operation with
        additional gen_conv operation.

        Args:
            cin: Input Channel number.
            cout: output Channel number.
            ksize: Kernel size.
        """
        super(gen_deconv2_origin, self).__init__(cin, cout, ksize=3)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2)
        x = super(gen_deconv2_origin, self).forward(x)
        # x = depth_to_space()(x,2)

        return x




class dis_conv(nn.Conv2d):
    def __init__(self, cin, cout, ksize=5, stride=2):
        """Define conv for discriminator.
        Activation is set to leaky_relu.

        Args:
            cin: Input Channel number.
            cout: output Channel number.
            ksize: Kernel size.
            Stride: Convolution stride.
        """
        p = int((ksize-1)/2)
        super(dis_conv, self).__init__(in_channels=cin, out_channels=cout, kernel_size=ksize, stride=stride, padding=p, dilation=1, groups=1, bias=True)

    def forward(self, x):
        x = super(dis_conv, self).forward(x)
        x = F.leaky_relu(x)
        return x

def batch_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    """Define batch convolution to use different conv. kernels in a batch.

    Args:
        x: input feature maps of shape (batch, channel, height, width)
        weight: conv.kernels of shape (batch, out_channel, in_channels, kernel_size, kernel_size)
    """
    if bias is None:
        assert x.shape[0] == weight.shape[0], "dim=0 of x must be equal in size to dim=0 of weight"
    else:
        assert x.shape[0] == weight.shape[0] and bias.shape[0] == weight.shape[
            0], "dim=0 of bias must be equal in size to dim=0 of weight"

    b_i, c, h, w = x.shape
    b_i, out_channels, in_channels, kernel_height_size, kernel_width_size = weight.shape

    out = x[None, ...].view(1, b_i * c, h, w)
    weight = weight.contiguous().view(b_i * out_channels, in_channels, kernel_height_size, kernel_width_size)

    out = F.conv2d(out, weight=weight, bias=None, stride=stride, dilation=dilation, groups=b_i,
                   padding=padding)

    out = out.view(b_i, out_channels, out.shape[-2], out.shape[-1])

    if bias is not None:
        out = out + bias.unsqueeze(2).unsqueeze(3)

    return out


def batch_transposeconv2d(x, weight, bias=None, stride=1, padding=0, output_padding=0, dilation=1):
    """Define batch transposed convolution to use different conv. kernels in a batch.

    Args:
        x: input feature maps of shape (batch, channel, height, width)
        weight: conv.kernels of shape (batch, in_channel, out_channels, kernel_size, kernel_size)
    """
    if bias is None:
        assert x.shape[0] == weight.shape[0], "dim=0 of x must be equal in size to dim=0 of weight"
    else:
        assert x.shape[0] == weight.shape[0] and bias.shape[0] == weight.shape[
            0], "dim=0 of bias must be equal in size to dim=0 of weight"

    b_i, c, h, w = x.shape
    b_i, in_channels, out_channels, kernel_height_size, kernel_width_size = weight.shape

    out = x[None, ...].view(1, b_i * c, h, w)
    weight = weight.contiguous().view(in_channels*b_i, out_channels, kernel_height_size, kernel_width_size)

    out = F.conv_transpose2d(out, weight=weight, bias=None, stride=stride, dilation=dilation, groups=b_i,
                   padding=padding, output_padding=output_padding)

    out = out.view(b_i, out_channels, out.shape[-2], out.shape[-1])

    if bias is not None:
        out = out + bias.unsqueeze(2).unsqueeze(3)
    return out



def hardmax(similar):
    val_max, id_max = torch.max(similar, 1)
    num = similar.size(1)
    sb = torch.Tensor(range(num)).long().to(similar.device)
    id_max = id_max[:, None, :, :]
    sb = sb[None, ..., None, None]
    similar = (sb==id_max).float().detach()
    return similar

class CP1(nn.Module):
    def __init__(self, bkg_patch_size=4, stride=1, ufstride=1, softmax_scale=10., nn_hard=False, pd=1,
                 fuse_k=3, is_fuse=False):
        super(CP1, self).__init__()
        self.bkg_patch_size = bkg_patch_size
        self.nn_hard = nn_hard
        self.stride = stride
        self.ufstride = ufstride
        self.softmax_scale = softmax_scale
        self.forward = self.forward_batch
        self.pd = pd
        self.fuse_k = fuse_k
        self.is_fuse = is_fuse

    def get_conv_kernel(self, x, mask=None):
        batch, c, h_small, w_small = x.shape
        x = x / torch.sqrt((x**2).sum(3, keepdim=True).sum(2, keepdim=True) + 1e-8)
        _x = F.pad(x, (self.pd,self.pd,self.pd,self.pd), mode='replicate')
        kernel = F.unfold(input=_x, kernel_size=(self.bkg_patch_size, self.bkg_patch_size), stride=self.ufstride)
        kernel = kernel.transpose(1, 2) \
            .view(batch, -1, c, self.bkg_patch_size, self.bkg_patch_size)
        # b*hw*c*k*c
        _mask = F.pad(mask, (self.pd,self.pd,self.pd,self.pd), mode='replicate')
        m = F.unfold(input=_mask, kernel_size=(self.bkg_patch_size, self.bkg_patch_size), stride=self.ufstride)
        m = m.transpose(1, 2).view(batch, -1, 1, self.bkg_patch_size, self.bkg_patch_size)
        m = m.squeeze(2)
        mm = (m.mean(3, keepdim=True).mean(2, keepdim=True)).float()
        #mm = (m.mean(3, keepdim=True).mean(2, keepdim=True)==1).float()
        return kernel, mm

    def forward_batch(self, f, b, mask=None):
        batch, c, h, w = b.shape
        batch, c, h_small, w_small = f.shape
        if mask is None:
            mask = torch.ones(batch, 1, h_small, w_small).to(f.device)
        else:
            mask = 1-mask
        # mask valid region
        softmax_scale = self.softmax_scale
        kernel, mmk = self.get_conv_kernel(b, mask)
        # mmk: valid ratio of each bkg patch
        _f = F.pad(f, (self.pd,self.pd,self.pd,self.pd), mode='replicate')
        cos_similar = batch_conv2d(_f, weight=kernel, stride=self.stride)
        _, cs, hs, ws = cos_similar.shape
        hb, wb = h//2, w//2

        if self.is_fuse:
            fuse_weight = torch.eye(self.fuse_k).to(f.device)
            fuse_weight = fuse_weight[None, None, ...]
            cos_similar = cos_similar.view(-1, cs, hs*ws)[:, None, ...]
            cos_similar = F.conv2d(cos_similar, fuse_weight, stride=1, padding=1)
            cos_similar = cos_similar.view(batch, 1, hb, wb, hs, ws)
            cos_similar = cos_similar.transpose(2, 3)
            cos_similar = cos_similar.transpose(4, 5)
            cos_similar = cos_similar.reshape(batch, 1, cs, hs*ws)
            cos_similar = F.conv2d(cos_similar, fuse_weight, stride=1, padding=1)
            cos_similar = cos_similar.view(batch, 1, hb, wb, hs, ws)
            cos_similar = cos_similar.transpose(2, 3)
            cos_similar = cos_similar.transpose(4, 5)
            cos_similar = cos_similar.squeeze(1)
            cos_similar = cos_similar.reshape(batch, cs, hs, ws)

        _mask = F.pad(mask, (self.pd,self.pd,self.pd,self.pd), mode='replicate')
        m = F.unfold(input=_mask, kernel_size=(self.bkg_patch_size, self.bkg_patch_size), \
                     stride=self.stride)
        m = m.transpose(1, 2).view(batch, -1, 1, self.bkg_patch_size, self.bkg_patch_size)
        m = m.squeeze(2)
        mmp = (m.mean(3).mean(2)).float()
        mmp = mmp.view(batch, 1, hs, ws) # mmp: valid ratio of fg patch
        mm = (mmk>mmp).float()  # replace with more valid
        ppp = (mmp>0.5).float() # ppp: mask of partial valid
        mm = mm*ppp # partial valid being replaced with more valid
        mm = mm + (mmk==1).float().expand_as(mm)  # and full valid
        mm = (mm>0).float()
        cos_similar = cos_similar * mm
        cos_similar = F.softmax(cos_similar*softmax_scale, dim=1)
        if self.nn_hard:
            cos_similar = hardmax(cos_similar)
        return cos_similar

class CP2(nn.Module):
    def __init__(self, bkg_patch_size=16, stride=8, ufstride=8, pd=4):
        super(CP2, self).__init__()
        self.stride = stride
        self.bkg_patch_size = bkg_patch_size
        self.forward = self.forward_batch
        self.ufstride = ufstride
        self.pd = pd
        #self.forward = self.forward_test


    def get_deconv_kernel(self, b, mask):
        batch, c, h, w = b.shape
        _mask = F.pad(mask, (self.pd,self.pd,self.pd,self.pd), mode='replicate')
        msk_kernel = F.unfold(input=_mask, kernel_size=(self.bkg_patch_size, self.bkg_patch_size),
                              stride=self.ufstride)
        msk_kernel = msk_kernel.transpose(1, 2).view(batch, -1, 1, self.bkg_patch_size, self.bkg_patch_size)
        _b = F.pad(b, (self.pd,self.pd,self.pd,self.pd), mode='replicate')
        bkg_kernel = F.unfold(input=_b, kernel_size=(self.bkg_patch_size, self.bkg_patch_size),
                              stride=self.ufstride)
        bkg_kernel = bkg_kernel.transpose(1, 2).view(batch, -1, c, self.bkg_patch_size, self.bkg_patch_size)
        bkg_kernel = bkg_kernel*(1-msk_kernel)

        return bkg_kernel, msk_kernel

    def forward_batch(self, cos_similar, b, mask):
        # use original background for reconstruction
        _, _, hs, ws = cos_similar.shape
        bkg_kernel, msk_kernel = self.get_deconv_kernel(b, mask)
        #hard_similar = hardmax(cos_similar.detach())
        output = batch_transposeconv2d(cos_similar,
                                       weight=bkg_kernel,stride=self.stride)
        ####
        # norm_kernel = torch.ones(1, 1, self.bkg_patch_size, self.bkg_patch_size).to(mask.device)
        # weight_map = torch.ones(1, 1, hs, ws).to(mask.device)
        # weight_map = F.conv_transpose2d(weight_map, norm_kernel, stride=self.stride)
        # mask_recon = batch_transposeconv2d(cos_similar,
        #                                    weight=msk_kernel,stride=self.stride)
        # mask_recon = mask_recon / weight_map
        ####
        output = output[:,:,self.pd:-self.pd,self.pd:-self.pd]
        #mask_recon = mask_recon[:,:,self.pd:-self.pd,self.pd:-self.pd]
        return output

import os
import sys
import time
import random
# import numpy as np
import matplotlib.pyplot as plt
from PIL import Image , ImageDraw
import logging
import math

import cv2


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def create_mask(width, height, mask_width, mask_height, x=None, y=None):
    mask = np.zeros((height, width))
    mask_x = x if x is not None else random.randint(0, width - mask_width)
    mask_y = y if y is not None else random.randint(0, height - mask_height)
    mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
    return mask

def random_bbox():
    """Generate a random tlhw.
    Returns:
        tuple: (top, left, height, width)
    """
    img_shape = [256, 256, 3]
    img_height = img_shape[0]
    img_width = img_shape[1]
    maxt = img_height - 128
    maxl = img_width - 128
    t=np.random.randint(low=0, high=maxt, size=None)
    l=np.random.randint(low=0, high=maxl, size=None)
    h=np.random.randint(low=30, high=80, size=None)
    w=np.random.randint(low=30, high=80, size=None)
    return (t, l, h, w)

def random_bbox_512():
    """Generate a random tlhw.
    Returns:
        tuple: (top, left, height, width)
    """
    img_shape = [512, 512, 3]
    img_height = img_shape[0]
    img_width = img_shape[1]
    maxt = img_height - 128
    maxl = img_width - 128
    t=np.random.randint(low=0, high=maxt, size=None)
    l=np.random.randint(low=0, high=maxl, size=None)
    h=np.random.randint(low=30, high=80, size=None)
    w=np.random.randint(low=30, high=80, size=None)
    return (t, l, h, w)

def bbox2mask_512(bbox, name='mask'):
    """Generate mask tensor from bbox.
    Args:
        bbox: tuple, (top, left, height, width)
    Returns:
        tf.Tensor: output with shape [1, H, W, 1]
    """
    def npmask(bbox, height, width, delta_h, delta_w):
        mask = np.zeros((1, height, width, 1), np.float32)
        h = np.random.randint(delta_h//2+1)
        w = np.random.randint(delta_w//2+1)
#         print("H",h)
#         print("W",w)
#         print("Box",bbox[0],bbox[1],bbox[2],bbox[3])
#         print("Box width ",bbox[0]+h," To ",bbox[0]+bbox[2]+h)
#         print("Box width ",bbox[1]+w," To ",bbox[1]+bbox[3]+w)
        mask[:, bbox[0]+h:bbox[0]+bbox[2]+h,
             bbox[1]+w:bbox[1]+bbox[3]+w, :] = 1.
        return mask
   
    img_shape = [512,512]
    height = img_shape[0]
    width = img_shape[1]
    mask = npmask(bbox,512,512,32,32)
#     mask.set_shape([1] + [height, width] + [1])
    return mask

def bbox2mask(bbox, name='mask'):
    """Generate mask tensor from bbox.
    Args:
        bbox: tuple, (top, left, height, width)
    Returns:
        tf.Tensor: output with shape [1, H, W, 1]
    """
    def npmask(bbox, height, width, delta_h, delta_w):
        mask = np.zeros((1, height, width, 1), np.float32)
        h = np.random.randint(delta_h//2+1)
        w = np.random.randint(delta_w//2+1)
#         print("H",h)
#         print("W",w)
#         print("Box",bbox[0],bbox[1],bbox[2],bbox[3])
#         print("Box width ",bbox[0]+h," To ",bbox[0]+bbox[2]+h)
#         print("Box width ",bbox[1]+w," To ",bbox[1]+bbox[3]+w)
        mask[:, bbox[0]+h:bbox[0]+bbox[2]+h,
             bbox[1]+w:bbox[1]+bbox[3]+w, :] = 1.
        return mask
   
    img_shape = [256,256]
    height = img_shape[0]
    width = img_shape[1]
    mask = npmask(bbox,256,256,32,32)
#     mask.set_shape([1] + [height, width] + [1])
    return mask


def brush_stroke_mask_512():
    """Generate mask tensor from bbox.
    Returns:
        tf.Tensor: output with shape [1, H, W, 1]
    """
    min_num_vertex = 4
    max_num_vertex = 12
    mean_angle = 2*math.pi / 5
    angle_range = 2*math.pi / 15
    min_width = 12
    max_width = 40
    H = 512
    W = 512
    def generate_mask(H, W):
        average_radius = math.sqrt(H*H+W*W) / 8
        mask = Image.new('L', (W, H), 0)

#         for _ in range(np.random.randint(1, 4)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                          v[1] - width//2,
                          v[0] + width//2,
                          v[1] + width//2),
                         fill=1)

        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
        mask = np.asarray(mask, np.float32)
        mask = np.reshape(mask, (1, H, W, 1))
        return mask
    img_shape = [512,512,3]
    height = img_shape[0]
    width = img_shape[1]
    mask = generate_mask(512,512)
    return mask

def brush_stroke_mask():
    """Generate mask tensor from bbox.
    Returns:
        tf.Tensor: output with shape [1, H, W, 1]
    """
    min_num_vertex = 4
    max_num_vertex = 12
    mean_angle = 2*math.pi / 5
    angle_range = 2*math.pi / 15
    min_width = 12
    max_width = 40
    H = 256
    W = 256
    def generate_mask(H, W):
        average_radius = math.sqrt(H*H+W*W) / 8
        mask = Image.new('L', (W, H), 0)

#         for _ in range(np.random.randint(1, 4)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                          v[1] - width//2,
                          v[0] + width//2,
                          v[1] + width//2),
                         fill=1)

        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
        mask = np.asarray(mask, np.float32)
        mask = np.reshape(mask, (1, H, W, 1))
        return mask
    img_shape = [256,256,3]
    height = img_shape[0]
    width = img_shape[1]
    mask = generate_mask(256,256)
    return mask



def stitch_images(inputs, *outputs, img_per_row=2):
    gap = 5
    columns = len(outputs) + 1

    width, height = inputs[0][:, :, 0].shape
    img = Image.new('RGB', (width * img_per_row * columns + gap * (img_per_row - 1), height * int(len(inputs) / img_per_row)))
    images = [inputs, *outputs]

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            im = np.array((images[cat][ix]).cpu()).astype(np.uint8).squeeze()
            im = Image.fromarray(im)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img


def imshow(img, title=''):
    fig = plt.gcf()
    fig.canvas.set_window_title(title)
    plt.axis('off')
    plt.imshow(img, interpolation='none')
    plt.show()


def imsave(img, path):
    if torch.is_tensor(img):
      im = Image.fromarray(img.detach().numpy().astype(np.uint8).squeeze())
      im.save(path)
    else:
      im = Image.fromarray(img.astype(np.uint8).squeeze())
      im.save(path)


class Progbar(object):
    """Displays a progress bar.

    Arguments:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=25, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules or
                                 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.

        Arguments:
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)

if __name__ == "__main__":
    pass
