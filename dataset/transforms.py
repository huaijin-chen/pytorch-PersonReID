#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# Written by Chao CHEN (chaochancs@gmail.com)
# Created On: 2017-07-05
# --------------------------------------------------------
from PIL import Image, ImageOps
class person_crop(object):
    """
    crop image to a specified size
    """
    def __init__(self, size=None, ratio=(1,0.75), crop_type=1):
        """
        Args:
            size (tuple) : Desired output size of the crop. If size is an
            int instead of squence like (w, h), a square crop (size, size) is made.

            ratio (float) : the crop's size is caculated by this value
            crop_type (int) : 0 crop using size. 1, crop using ratios
        """
        self.size = size
        self.ratio = ratio
        self.crop_type = crop_type

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        if self.crop_type == 0:
            th, tw = self.size
            th = max(th, h-1)
            tw = max(tw, w-1)
            x1 = 0
            y1 = 0
            return img.crop((x1, y1, x1 + tw, y1 + th))
        elif self.crop_type == 1:
            r_w, r_h = self.ratio
            tw = int(w*r_w)
            th = int(h*r_h)
            return img.crop((0, 0, tw, th))
        else:
            print('crop_type error.')

class scale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size;
        self.interpolation = interpolation

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        #print(img.size)
        return img

if '__main__' == __name__:
    img_path = '/data/chenchao/darknet/data/person.jpg'
    img = Image.open(img_path).convert('RGB')
    pc = person_crop()
    img_crop = pc(img)
    img_crop.save('person-1.jpg')
    
