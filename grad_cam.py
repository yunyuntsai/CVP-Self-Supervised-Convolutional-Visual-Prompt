#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-26

from collections import Sequence

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn import functional as F
from tqdm import tqdm
import re
import cv2
import matplotlib.cm as cm
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
import os
import matplotlib.pyplot as plt


class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        self.image_shape = image.shape[2:]
        self.logits, _ = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)  # ordered results

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class BackPropagation(_BaseWrapper):
    def forward(self, image):
        self.image = image.requires_grad_()
        return super(BackPropagation, self).forward(self.image)

    def generate(self):
        gradient = self.image.grad.clone()
        self.image.grad.zero_()
        return gradient


class GuidedBackPropagation(BackPropagation):
    """
    "Striving for Simplicity: the All Convolutional Net"
    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """

    def __init__(self, model):
        super(GuidedBackPropagation, self).__init__(model)

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (F.relu(grad_in[0]),)

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(backward_hook))


class Deconvnet(BackPropagation):
    """
    "Striving for Simplicity: the All Convolutional Net"
    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """

    def __init__(self, model):
        super(Deconvnet, self).__init__(model)

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients and ignore ReLU
            if isinstance(module, nn.ReLU):
                return (F.relu(grad_out[0]),)

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(backward_hook))


class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output[0].detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        # print('medium  gcam: ', gcam.shape)
        # print(gcam.shape)
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam


def occlusion_sensitivity(
    model, images, ids, mean=None, patch=35, stride=1, n_batches=128
):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure A5 on page 17
    Originally proposed in:
    "Visualizing and Understanding Convolutional Networks"
    https://arxiv.org/abs/1311.2901
    """

    torch.set_grad_enabled(False)
    model.eval()
    mean = mean if mean else 0
    patch_H, patch_W = patch if isinstance(patch, Sequence) else (patch, patch)
    pad_H, pad_W = patch_H // 2, patch_W // 2

    # Padded image
    images = F.pad(images, (pad_W, pad_W, pad_H, pad_H), value=mean)
    B, _, H, W = images.shape
    new_H = (H - patch_H) // stride + 1
    new_W = (W - patch_W) // stride + 1

    # Prepare sampling grids
    anchors = []
    grid_h = 0
    while grid_h <= H - patch_H:
        grid_w = 0
        while grid_w <= W - patch_W:
            grid_w += stride
            anchors.append((grid_h, grid_w))
        grid_h += stride

    # Baseline score without occlusion
    baseline = model(images).detach().gather(1, ids)

    # Compute per-pixel logits
    scoremaps = []
    for i in tqdm(range(0, len(anchors), n_batches), leave=False):
        batch_images = []
        batch_ids = []
        for grid_h, grid_w in anchors[i : i + n_batches]:
            images_ = images.clone()
            images_[..., grid_h : grid_h + patch_H, grid_w : grid_w + patch_W] = mean
            batch_images.append(images_)
            batch_ids.append(ids)
        batch_images = torch.cat(batch_images, dim=0)
        batch_ids = torch.cat(batch_ids, dim=0)
        scores = model(batch_images).detach().gather(1, batch_ids)
        scoremaps += list(torch.split(scores, B))

    diffmaps = torch.cat(scoremaps, dim=1) - baseline
    diffmaps = diffmaps.view(B, new_H, new_W)

    return diffmaps

def cal_similarity(gcam, mix_gcam, raw, diff, output_dir, idx, target):

    B, C, H, W = diff.shape
    diff = diff[0].cpu().detach().numpy().transpose(1, 2, 0)
    diff -= diff.min()
    diff /= diff.max() 
    # print(diff.min(), diff.max(), diff.mean())
    # gcam /= 255
    # diff_mask = diff_mask[..., 0]
    diff_mask = (diff > (diff.mean() + diff.std())) + 0


    lower_bound = np.array([0, 0, 255])
    upper_bound = np.array([0, 0, 255])

    diff_mask *= 255

    mask = cv2.inRange(diff_mask, lower_bound, upper_bound)
    inverted_mask = cv2.bitwise_not(mask)

    mask_blur = cv2.GaussianBlur(inverted_mask, (5, 5), 0)
    ret, mask_thresh = cv2.threshold(mask_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    diff_final_mask = cv2.bitwise_and(np.uint8(diff_mask), np.uint8(diff_mask), mask=mask_thresh)


    gcam_lower_bound = np.array([255, 0, 0])
    gcam_upper_bound = np.array([255, 255, 0])

    gcam_mask = cv2.inRange(gcam[0], gcam_lower_bound, gcam_upper_bound)
    # inverted_gcam_mask = cv2.bitwise_not(gcam_mask)

    gcam_mask_blur = cv2.GaussianBlur(gcam_mask, (5, 5), 0)
    gcam_ret, gcam_mask_thresh = cv2.threshold(gcam_mask_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    gcam_final_mask = cv2.bitwise_and(np.uint8(gcam[0]), np.uint8(gcam[0]), mask=gcam_mask_thresh)


    # gcam_mask = gcam[..., 0]  #extract only the red channel 
    # print(gcam_mask.shape)
    # print('gcam: ', gcam_mask)
    # calculate iou metric / cos similarity

    diff_all_mask =  (diff > (diff.mean()+diff.std())) + 0
    # diff_all_mask =  (diff > 0) + 0
    diff_all_mask *= 255
    ssim = cosine_similarity(diff_all_mask.flatten().reshape(1,-1), gcam[0].flatten().reshape(1,-1))
    # cos = ssim(torch.tensor(gcam[0]).to(device), diff[0])
    print('ssim score: ', ssim)

    # print(mask_thresh)
    # print(gcam_mask_thresh)
    # ssim = jaccard_score((mask_thresh/255).flatten().reshape(1,-1), (gcam_mask_thresh/255).flatten().reshape(1,-1), average='micro')
    # print('iou score: ', ssim)
    # enhance_diff = (diff_mask *5).cpu().detach().numpy()
    diff *= 255.0
    # gcam_mask = gcam_mask[0] * 255.0
    # file_name1 = os.path.join(output_dir, 'cls_' + str(target) + 'diff_' + str(idx) + '.png')
    file_name = os.path.join(output_dir, 'cls_' + str(target) + '_idx_' + str(idx) + '.png')
    if idx < 5 :
        fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
        ax[0].title.set_text('Adv. Image (class: {})'.format(str(target)))
        ax[0].imshow(np.uint8(raw[0]))
        ax[1].title.set_text('Noise perturbation')
        ax[1].imshow(np.uint8(diff_final_mask))
        ax[2].title.set_text('Attention GradCam')
        ax[2].imshow(np.uint8(gcam[0]))
        # ax[3].title.set_text('Attention heatmap (red region)')
        # ax[3].imshow(np.uint8(gcam_final_mask))
        ax[3].title.set_text('Adv. Image + GradCam \n Cos. Similarity: {:.3f}'.format(ssim[0][0]))
        ax[3].imshow(np.uint8(mix_gcam[0]))  
        plt.tight_layout()
        plt.savefig(file_name)
        # cv2.imwrite( file_name, np.uint8(diff))
        # cv2.imwrite( file_name2, cv2.cvtColor(np.uint8(gcam_mask), cv2.COLOR_BGR2RGB))
    return ssim
    



def save_gradcam(filename, output_dir, gcam, raw_image, diff_image, device, idx, target, paper_cmap=False):

    
    gcam = gcam.cpu().detach().numpy()
    raw_image = (raw_image * 255.0).permute(0,2,3,1).cpu().detach().numpy()

    # print(raw_image.max(), raw_image.min())

    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
        # gcam = cmap.astype(np.float) 

    # print(cmap.max(), cmap.min(), cmap.mean(), cmap.std())

    score = cal_similarity(cmap, gcam,  raw_image, diff_image, output_dir, idx, target)
    return score
    # if idx 
    # cv2.imwrite(filename, cv2.cvtColor(np.uint8(gcam[0]), cv2.COLOR_BGR2RGB))
    # cv2.imwrite(filename, np.uint8(gcam[0]))

