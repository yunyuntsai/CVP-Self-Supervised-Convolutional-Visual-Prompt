# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random
import os
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
from scipy import rand
import torchvision.transforms as transforms
import numpy as np
import torch
from PIL import Image, ImageOps
import kornia
from math import pi
import uuid
from utils import *



## https://github.com/google-research/augmix

def _augmix_aug(x_orig, normalize=True):
    # x_orig = preaugment(x_orig)
    
    x_orig = Image.fromarray(np.transpose(np.uint8(x_orig.cpu().numpy()*255), (1,2,0)))
    # unique_str = str(uuid.uuid4())[:8]
    # x_orig.save(os.path.join('./output/test_gaussian', unique_str + '.jpg'))

    if normalize:
        x_processed = preprocess_norm(x_orig)
    else:
        x_processed = preprocess(x_orig)

    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(augmentations)(x_aug)
        if normalize:
            mix += w[i] * preprocess_norm(x_aug)
        else:
            mix += w[i] * preprocess(x_aug)

    mix = m * x_processed + (1 - m) * mix

    return mix

augmix = _augmix_aug


def autocontrast(pil_img, level=None):
    return ImageOps.autocontrast(pil_img)

def equalize(pil_img, level=None):
    return ImageOps.equalize(pil_img)

def rotate(pil_img, level):
    degrees = int_parameter(rand_lvl(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR, fillcolor=128)

def solarize(pil_img, level):
    level = int_parameter(rand_lvl(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)

def shear_x(pil_img, level):
    w, h = pil_img.size
    level = float_parameter(rand_lvl(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((w, h), Image.AFFINE, (1, level, 0, 0, 1, 0), resample=Image.BILINEAR, fillcolor=128)

def shear_y(pil_img, level):
    w, h = pil_img.size
    level = float_parameter(rand_lvl(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((w, h), Image.AFFINE, (1, 0, 0, level, 1, 0), resample=Image.BILINEAR, fillcolor=128)

def translate_x(pil_img, level):
    w, h = pil_img.size
    level = int_parameter(rand_lvl(level), w / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((w, h), Image.AFFINE, (1, 0, level, 0, 1, 0), resample=Image.BILINEAR, fillcolor=128)

def translate_y(pil_img, level):
    w, h = pil_img.size
    level = int_parameter(rand_lvl(level), w / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((w, h), Image.AFFINE, (1, 0, 0, 0, 1, level), resample=Image.BILINEAR, fillcolor=128)

def posterize(pil_img, level):
    level = int_parameter(rand_lvl(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.
    Returns:
    An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)

def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.
    Returns:
    A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.

def rand_lvl(n):
    return np.random.uniform(low=0.1, high=n)


augmentations = [
    autocontrast,
    equalize,
    lambda x: rotate(x, 1),
    lambda x: solarize(x, 1),
    lambda x: shear_x(x, 1),
    lambda x: shear_y(x, 1),
    lambda x: translate_x(x, 1),
    lambda x: translate_y(x, 1),
    lambda x: posterize(x, 1),
]

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
preprocess_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
preprocess = transforms.Compose([
    transforms.ToTensor()
])
preaugment = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
])


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = augment_list()

    def __call__(self, img):
        #ops = random.choices(self.augment_list, k=self.n)
        ops = random.choices(self.augment_list, k=16)
        augment = ops[self.n][0]
        minval = ops[self.n][1]
        maxval = ops[self.n][2]

        # for op, minval, maxval in ops:
        #     val = (float(self.m) / 30) * float(maxval - minval) + minval
        #     img = op(img, val)
        #trn1 = transforms.ToPILImage()
        val = (float(self.m) / 30) * float(maxval - minval) + minval
        
        img = augment(img, val)

        return img

def get_transAug_param(aug_name):
    if aug_name == 'contrast':
        rand_num = 1
        contrast_epsilon = torch.tensor((0.5, 2.5))
        # contrast_epsilon = torch.tensor((0.1, 3.3))
        #contrast_epsilon = torch.tensor((0.1, 5))
        #contrast_epsilon = torch.tensor((0.1, 10))
        #contrast_epsilon = torch.tensor((0.1, 20)) #0.5 0.9
        return contrast_epsilon
    elif aug_name == 'brightness':
        rand_num = 1
        brightness_epsilon = torch.tensor((-0.35, 0.3))
        #brightness_epsilon = torch.tensor((-0.25, 0.6))
        return brightness_epsilon
    elif aug_name == 'hue':
        rand_num = 1
        hue_epsilon = torch.tensor((-pi/3, pi/3))
        return hue_epsilon
    elif aug_name == 'saturation':
        rand_num = 1
        sat_epsilon = torch.tensor((0.5, 1))
        # sat_epsilon = torch.tensor((0.1, 2))
        return sat_epsilon
    elif aug_name == 'sharpness':
        sharp_epsilon = torch.tensor((0.5, 1)) 
        # sharp_epsilon = torch.tensor((0.5, 3)) 
        return sharp_epsilon
    elif aug_name == 'solarize':
        solar_epsilon = torch.tensor((0.5, 0.7))
        return solar_epsilon
    elif aug_name == 'equalize':
        rand_num = 1
        equal_epsilon = torch.tensor((20, 60))
        return equal_epsilon
    elif aug_name == 'pixelate':
        rand_num = 1
        pixelate_epsilon = torch.tensor((0.9, 0.6))
        return pixelate_epsilon
    elif aug_name == 'gaussian_noise':
        rand_num = 1
        gaussian_epsilon = torch.tensor((0.08, 0.38))
        return gaussian_epsilon

def trans_aug_list(aug_list, x, param_list):

    for i in range(len(aug_list)):
        aug_x = trans_aug(aug_list[i], x, param_list[i])
        x = aug_x
    return x

def trans_aug(aug_name, data, param):
    if aug_name == 'contrast':
        return kornia.enhance.adjust_contrast(data, param)  
    elif aug_name == 'brightness':
        return kornia.enhance.adjust_brightness(data, param)
    elif aug_name == 'hue':
        return kornia.enhance.adjust_hue(data, param)
    elif aug_name == 'saturation':
        return kornia.enhance.adjust_saturation(data, param)
    elif aug_name == 'sharpness':
        return kornia.enhance.sharpness(data, param)
    elif aug_name == 'solarize':
        return kornia.enhance.solarize(data, thresholds=param, additions=.1)
    elif aug_name == 'equalize':
        return kornia.enhance.equalize_clahe(data, clip_limit=param)
    elif aug_name == 'pixelate':
        return apply_pixelate(data, param)
    elif aug_name == 'gaussian_noise':
        return apply_pixelate(data, param)
