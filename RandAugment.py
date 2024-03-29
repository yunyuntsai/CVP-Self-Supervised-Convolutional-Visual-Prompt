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
    
    x_orig = Image.fromarray(np.transpose(np.uint8(x_orig.cpu().detach().numpy()*255), (1,2,0)))
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

clip_mean = [0.48145466, 0.4578275, 0.40821073]
clip_std = [0.26862954, 0.26130258, 0.27577711]
preprocess_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(clip_mean, clip_std)
])
preprocess = transforms.Compose([
    transforms.ToTensor()
])
preaugment = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
])


def init_sharpness_random_kernel(input, kernel_size):
    return torch.as_tensor(torch.rand(kernel_size, kernel_size), dtype=input.dtype, device=input.device)

def init_sharpness_random_kernel_3by3(input):
    return torch.as_tensor(torch.rand(3, 3), dtype=input.dtype, device=input.device)

def init_sharpness_random_kernel_3by3_batch(input):
    return torch.as_tensor(torch.rand(input.shape[0], 3, 3), dtype=input.dtype, device=input.device)

def init_sharpness_random_kernel_5by5(input):
    return torch.as_tensor(torch.rand(5, 5), dtype=input.dtype, device=input.device)  

def init_sharpness_random_composite_kernel(input):
    return [torch.as_tensor(torch.rand(1, 5), dtype=input.dtype, device=input.device), torch.as_tensor(torch.rand(5, 1), dtype=input.dtype, device=input.device)]

def init_sharpness_3by3_kernel(input):
    return torch.as_tensor([[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=input.dtype, device=input.device)

def init_sharpness_5by5_kernel(input):
    return torch.as_tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 15, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=input.dtype, device=input.device)

def customized_sharpness(input: torch.Tensor, kernel_param: torch.Tensor, factor: torch.Union[float, torch.Tensor]) -> torch.Tensor:
    """Apply sharpness to the input tensor.

    .. image:: _static/img/sharpness.png

    Implemented Sharpness function from PIL using torch ops. This implementation refers to:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py#L326

    Args:
        input: image tensor with shape :math:`(*, C, H, W)` to sharpen.
        factor: factor of sharpness strength. Must be above 0.
            If float or one element tensor, input will be sharpened by the same factor across the whole batch.
            If 1-d tensor, input will be sharpened element-wisely, len(factor) == len(input).

    Returns:
        Sharpened image or images with shape :math:`(*, C, H, W)`.

    Example:
        >>> x = torch.rand(1, 1, 5, 5)
        >>> sharpness(x, 0.5).shape
        torch.Size([1, 1, 5, 5])
    """
    enable_interpolation = False

    if not isinstance(factor, torch.Tensor):
        factor = torch.as_tensor(factor, device=input.device, dtype=input.dtype)

    if len(factor.size()) != 0 and factor.shape != torch.Size([input.size(0)]):
        raise AssertionError(
            "Input batch size shall match with factor size if factor is not a 0-dim tensor. "
            f"Got {input.size(0)} and {factor.shape}"
        )
    

    if kernel_param.shape[0] == kernel_param.shape[1]:
        ksize = kernel_param.shape[0]
        if enable_interpolation:
            #reshape kernel ksize*ksize --> channel*1*ksize*ksize
            #interpolate kernel from ksize*ksize --> ksize+2 * ksize+2
            #normalize with sum of kernel param.
            kernel = kernel_param.view(1, 1, ksize, ksize).repeat(input.size(1), 1, 1, 1) 
            kernel = torch.nn.functional.interpolate(kernel, (7, 7), mode='bilinear')
            kernel = kernel / kernel.sum()
        else:
            kernel = (kernel_param.view(1, 1, ksize, ksize).repeat(input.size(1), 1, 1, 1)/ kernel_param.sum())
    if kernel_param.shape[0] == 1 and kernel_param.shape[1] == 5:
        kernel = (kernel_param.view(1, 1, 1, 5).repeat(input.size(1), 1, 1, 1)/ kernel_param.sum())
    elif kernel_param.shape[0] == 5 and kernel_param.shape[1] == 1:
        kernel = (kernel_param.view(1, 1, 5, 1).repeat(input.size(1), 1, 1, 1)/ kernel_param.sum())
    

    # This shall be equivalent to depthwise conv2d:
    # Ref: https://discuss.pytorch.org/t/depthwise-and-separable-convolutions-in-pytorch/7315/2

    if enable_interpolation:
        degenerate = torch.nn.functional.conv2d(input, kernel, bias=None, padding=2, stride=1, groups=input.size(1))
    else:
        if kernel_param.shape[0] == 1 or kernel_param.shape[1] == 1:
            degenerate = torch.nn.functional.conv2d(input, kernel, bias=None, stride=1, groups=input.size(1))
        elif kernel_param.shape[0] == kernel_param.shape[1]:
            degenerate = torch.nn.functional.conv2d(input, kernel, bias=None, stride=1, groups=input.size(1))

    degenerate = torch.clamp(degenerate, 0.0, 1.0)

    # For the borders of the resulting image, fill in the values of the original image.
    mask = torch.ones_like(degenerate)
    if kernel_param.shape[0] == kernel_param.shape[1]:
        if enable_interpolation:
            padded_mask = torch.nn.functional.pad(mask, [1, 1, 1, 1])
            padded_degenerate = torch.nn.functional.pad(degenerate, [1, 1, 1, 1])  
        else:       
            ps = ksize//2 
            padded_mask = torch.nn.functional.pad(mask, [ps, ps, ps, ps])
            padded_degenerate = torch.nn.functional.pad(degenerate, [ps, ps, ps, ps])
    elif kernel_param.shape[0] == 1 and kernel_param.shape[1] == 5:
        padded_mask = torch.nn.functional.pad(mask, [2, 2, 0, 0])
        padded_degenerate = torch.nn.functional.pad(degenerate, [2, 2, 0, 0])
    elif kernel_param.shape[0] == 5 and kernel_param.shape[1] == 1:
        padded_mask = torch.nn.functional.pad(mask, [0, 0, 2, 2])
        padded_degenerate = torch.nn.functional.pad(degenerate, [0, 0, 2, 2])
    
    result = torch.where(padded_mask == 1, padded_degenerate, input)

    if len(factor.size()) == 0:
        return _blend_one(result, input, factor)
    return torch.stack([_blend_one(result[i], input[i], factor[i]) for i in range(len(factor))])



def _blend_one(input1: torch.Tensor, input2: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    r"""Blend two images into one.

    Args:
        input1: image tensor with shapes like :math:`(H, W)` or :math:`(D, H, W)`.
        input2: image tensor with shapes like :math:`(H, W)` or :math:`(D, H, W)`.
        factor: factor 0-dim tensor.

    Returns:
        : image tensor with the batch in the zero position.
    """
    if not isinstance(input1, torch.Tensor):
        raise AssertionError(f"`input1` must be a tensor. Got {input1}.")
    if not isinstance(input2, torch.Tensor):
        raise AssertionError(f"`input1` must be a tensor. Got {input2}.")

    if isinstance(factor, torch.Tensor) and len(factor.size()) != 0:
        raise AssertionError(f"Factor shall be a float or single element tensor. Got {factor}.")
    if factor == 0.0:
        return input1
    if factor == 1.0:
        return input2
    diff = (input2 - input1) * factor
    res = input1 + diff
    if factor > 0.0 and factor < 1.0:
        return res
    return torch.clamp(res, 0, 1)



def get_transAug_param(aug_name, corr_type):
    
    noise_list = ['gaussian_noise', 'impulse_noise', 'shot_noise', 'brightness', 'snow', 'frost']
    if aug_name == 'contrast':
        contrast_epsilon = torch.tensor((0, 2))
        return contrast_epsilon
    elif aug_name == 'brightness':
        brightness_epsilon = torch.tensor((0, 1))
        return brightness_epsilon
    elif aug_name == 'hue':
        hue_epsilon = torch.tensor((-pi/3, pi/3))
        return hue_epsilon
    elif aug_name == 'saturation':
        sat_epsilon = torch.tensor((0, 1))
        return sat_epsilon
    elif aug_name == 'sharpness':
        if corr_type in noise_list:
            sharp_epsilon = torch.tensor((0.5, 1))
        else: sharp_epsilon = torch.tensor((0.5, 3)) 
        return sharp_epsilon
    elif aug_name == 'solarize':
        solar_epsilon = torch.tensor((0.5, 0.7))
        return solar_epsilon

def get_imgnet_transaug_param(aug_name):
  
    if aug_name == 'sharpness':
        sharp_epsilon = torch.tensor((0.5, 1))
        return sharp_epsilon



def trans_aug_list(aug_list, x, param_list):

    for i in range(len(aug_list)):
        aug_x = trans_aug(aug_list[i], x, param_list[i])
        x = aug_x
    return x

def trans_aug(aug_name, data, kernel_param, param):
    if aug_name == 'contrast':
        return kornia.enhance.adjust_contrast(data, param)  
    elif aug_name == 'brightness':
        return kornia.enhance.adjust_brightness(data, param)
    elif aug_name == 'hue':
        return kornia.enhance.adjust_hue(data, param)
    elif aug_name == 'saturation':
        return kornia.enhance.adjust_saturation(data, param)
    elif aug_name == 'sharpness':
        # return kornia.enhance.sharpness(data, param)
        return customized_sharpness(data, kernel_param, param)

def trans_aug_bybatch(aug_name, data, kernel_param, param):
    if aug_name == 'sharpness':
        new_data_list = []
        # return kornia.enhance.sharpness(data, param)
        for i in range(data.shape[0]):
            new_data = customized_sharpness(data[i].unsqueeze(0), kernel_param[i], param[i])
            new_data_list.append(new_data.squeeze(0))
        return torch.stack(new_data_list, dim = 0)
        


