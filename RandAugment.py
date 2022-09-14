# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
from scipy import rand
import torchvision.transforms as transforms
import numpy as np
import torch
from PIL import Image
import kornia
from math import pi
from utils import *


def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def Identity(img, v):
    return img




def augment_list():  # 16 oeprations and their ranges
    # https://github.com/google-research/uda/blob/master/image/randaugment/policies.py#L57
    # l = [
    #     (Identity, 0., 1.0),
    #     (ShearX, 0., 0.3),  # 0
    #     (ShearY, 0., 0.3),  # 1
    #     (TranslateX, 0., 0.33),  # 2
    #     (TranslateY, 0., 0.33),  # 3
    #     (Rotate, 0, 30),  # 4
    #     (AutoContrast, 0, 1),  # 5
    #     (Invert, 0, 1),  # 6
    #     (Equalize, 0, 1),  # 7
    #     (Solarize, 0, 110),  # 8
    #     (Posterize, 4, 8),  # 9
    #     # (Contrast, 0.1, 1.9),  # 10
    #     (Color, 0.1, 1.9),  # 11
    #     (Brightness, 0.1, 1.9),  # 12
    #     (Sharpness, 0.1, 1.9),  # 13
    #     # (Cutout, 0, 0.2),  # 14
    #     # (SamplePairing(imgs), 0, 0.4),  # 15
    # ]

    # https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py#L505
    l = [
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Invert, 0, 1),
        (Rotate, 0, 30),
        (Posterize, 0, 4),
        (Solarize, 0, 256),
        (SolarizeAdd, 0, 110),
        (Color, 0.1, 1.9),
        (Contrast, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        (ShearX, 0., 0.3),
        (ShearY, 0., 0.3),
        (CutoutAbs, 0, 40),
        (TranslateXabs, 0., 100),
        (TranslateYabs, 0., 100),
    ]

    return l


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}

def get_augment(name):
    return augment_dict[name]


def apply_augment(img, name, level):
    augment_fn, low, high = get_augment(name)
    trn1 = transforms.ToPILImage()
    trn2 = transforms.ToTensor()
    trn_img = [trn2(augment_fn(trn1(each_img), level * (high - low) + low)) for each_img in img]
    trn_img = [torch.unsqueeze(item, 0) for item in trn_img]
    trn_img = torch.vstack(trn_img)
    
    return trn_img
    #return augment_fn(trn_img, level * (high - low) + low)

def apply_pixelate(img, param):
    x_shape = img.shape
    downscale_transform = torch.nn.Sequential(transforms.Resize(size=int(x_shape[1]*param), interpolation=Image.BILINEAR),)
    upscale_transform = torch.nn.Sequential(transforms.Resize(size=int(x_shape[1]), interpolation=Image.NEAREST),)
    out = downscale_transform(img)
    out = upscale_transform(out)
    return out

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
