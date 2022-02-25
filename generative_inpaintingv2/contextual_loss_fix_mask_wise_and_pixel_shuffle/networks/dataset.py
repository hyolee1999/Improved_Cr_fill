import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image  , ImageDraw
from scipy import misc
# from scipy.misc import imread
# from cv2 import imread
from imageio import imread
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from .utils import create_mask , brush_stroke_mask ,random_bbox,bbox2mask , brush_stroke_mask_512 ,random_bbox_512,bbox2mask_512
# from PIL import Image
# from PIL import Image, ImageDraw
import logging
import math

import cv2
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, edge_flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.edge_data = self.load_flist(edge_flist)
        self.mask_data = self.load_flist(mask_flist)

        self.input_size = config.INPUT_SIZE
        print(str(self.input_size))
        self.sigma = config.SIGMA
        self.edge = config.EDGE
        self.mask = config.MASK
        self.nms = config.NMS

        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random
        if config.MODE == 2:
            self.mask = 6

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
            # name  = self.load_name(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item 

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size

        # load image
        img = imread(self.data[index])
        # print("test")
        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)

        # resize/crop if needed
        # if size != 0:
        #     img = self.resize(img, size, size)

        if size != 256 and size != 768:
            # print("512")
            img = self.np_scale_to_shape(img,(256,256))
            img = self.np_random_crop(img,(256,256))
        elif size == 768:
            # print("768")
            img = self.np_scale_to_shape(img,(128,128))
            img = self.resize(img, 256, 256)
            # img = self.np_random_crop(img,(512,512))
        else:
            # print("256")
            img = self.resize(img, size, size)

        # create grayscale image
        # if self.mask == 6:
        #   img_gray = rgb2gray(img)

        # load mask
        mask = self.load_mask(img, index)

        # load edge
        # if self.mask == 6:
        #   edge = self.load_edge(img_gray, index, mask)

        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            # if self.mask == 6:
            #   img_gray = img_gray[:, ::-1, ...]
            #   edge = edge[:, ::-1, ...]
            mask = mask[:, ::-1, ...]
        # if self.mask == 6:
        #   return self.to_tensor(img), self.to_tensor(mask) ,  self.to_tensor(edge) , self.to_tensor(img_gray)
        return self.to_tensor(img), self.to_tensor(mask)

    def load_edge(self, img, index, mask):
        sigma = self.sigma

        # in test mode images are masked (with masked regions),
        # using 'mask' parameter prevents canny to detect edges for the masked regions
        mask = None if self.training else (1 - mask / 255).astype(np.bool)

        # canny
        if self.edge == 1:
            # no edge
            if sigma == -1:
                return np.zeros(img.shape).astype(np.float)

            # random sigma
            if sigma == 0:
                sigma = random.randint(1, 4)

            return canny(img, sigma=sigma, mask=mask).astype(np.float)

        # external
        else:
            imgh, imgw = img.shape[0:2]
            edge = imread(self.edge_data[index])
            edge = self.resize(edge, imgh, imgw)

            # non-max suppression
            if self.nms == 1:
                edge = edge * canny(img, sigma=sigma, mask=mask)

            return edge

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask

        # external + random block
        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # external + random block + half
        elif mask_type == 5:
            mask_type = np.random.randint(1, 4)

        # random block
        if mask_type == 1:
          my_box = random_bbox()
          mask_box = bbox2mask(my_box)
          mask_box = mask_box.reshape((256,256))
          mask_paint = brush_stroke_mask()
          mask_paint = mask_paint.reshape((256,256))      
          or_mask = np.logical_or(mask_paint,mask_box).astype(np.uint8)*255
          return or_mask
            # return create_mask(imgw, imgh, imgw // 2, imgh // 2)

        # half
        if mask_type == 2:
            # randomly choose right or left
          my_box = random_bbox_512()
          mask_box = bbox2mask_512(my_box)
          mask_box = mask_box.reshape((512,512))
          mask_paint = brush_stroke_mask_512()
          mask_paint = mask_paint.reshape((512,512))      
          or_mask = np.logical_or(mask_paint,mask_box).astype(np.uint8)*255
          return or_mask
            # return create_mask(imgw, imgh, imgw // 2, imgh, 0 if random.random() < 0.5 else imgw // 2, 0)

        # external
        if mask_type == 3:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imread(self.mask_data[mask_index])
            mask = rgb2gray(mask)
            mask = self.resize(mask, imgh, imgw)
            mask = (mask > 127.5).astype(np.uint8) * 255       # threshold due to interpolation
            return mask

        # test mode: load mask non random
        if mask_type == 6:
            mask = imread(self.mask_data[index])
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            mask = rgb2gray(mask)
            mask = (mask > 127.5).astype(np.uint8) * 255
            return mask

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        # img = scipy.misc.imresize(img, [height, width])
        img = np.array(Image.fromarray(img).resize([height, width]))

        return img
    
    def np_scale_to_shape(self,image, shape, align=False):
      """Scale the image.
      The minimum side of height or width will be scaled to or
      larger than shape.
      Args:
          image: numpy image, 2d or 3d
          shape: (height, width)
      Returns:
          numpy image
      """
      height, width = shape
      imgh, imgw = image.shape[0:2]
      if imgh < height or imgw < width or align:
          scale = np.maximum(height/imgh, width/imgw)
          # image = cv2.resize(
          #     image,
          #     (math.ceil(imgw*scale), math.ceil(imgh*scale)))
          image = np.array(Image.fromarray(image).resize([math.ceil(imgh*scale), math.ceil(imgw*scale)]))
      return image


    def np_random_crop(self,image, shape, random_h=None, random_w=None):
      """Random crop.
      Shape from image.
      Args:
          image: Numpy image, 2d or 3d.
          shape: (height, width).
          random_h: A random int.
          random_w: A random int.
      Returns:
          numpy image
          int: random_h
          int: random_w
      """
      height, width = shape

      imgh, imgw = image.shape[0:2]
      if random_h is None:
          random_h = np.random.randint(imgh-height+1)
      if random_w is None:
          random_w = np.random.randint(imgw-width+1)
      # print(image.shape)
      return image[random_h:random_h+height, random_w:random_w+width]
         

    def load_flist(self, flist):

        
        if isinstance(flist, list):
            return flist
        
        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            # print(flist)
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                # print(len(flist))
                flist.sort()
                return flist

            # print(flist)
            if os.path.isfile(flist):
               
                try:
                    # print(np.genfromtxt(flist, dtype=None))
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                    # return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]
      
        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item
