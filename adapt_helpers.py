import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from RandAugment import augmix 
from copy import deepcopy
from data_utils import *
import re

def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits

def marginal_entropy_batch(outputs, b_size):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    aug_size = logits.shape[0]//b_size
    avg_logits = [logits[i*aug_size:(i+1)*aug_size].logsumexp(dim=0) - np.log(logits[i*aug_size:(i+1)*aug_size].shape[0]) for i in range(b_size)]
    avg_logits = torch.vstack(avg_logits)
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    loss = -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)
    return loss.mean(0), avg_logits

def softmax_entropy(outputs):
    """Entropy of softmax distribution from logits."""
    return -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)

# https://github.com/bethgelab/robustness/blob/main/robusta/batchnorm/bn.py#L175
def _modified_bn_forward(self, input):
    est_mean = torch.zeros(self.running_mean.shape, device=self.running_mean.device)
    est_var = torch.ones(self.running_var.shape, device=self.running_var.device)
    nn.functional.batch_norm(input, est_mean, est_var, None, None, True, 1.0, self.eps)
    running_mean = self.prior * self.running_mean + (1 - self.prior) * est_mean
    running_var = self.prior * self.running_var + (1 - self.prior) * est_var
    return nn.functional.batch_norm(input, running_mean, running_var, self.weight, self.bias, False, 0, self.eps)

def config_model(model):

    adapt_prior_strength = 16
    model.eval().cuda()

    if (not adapt_prior_strength is None and adapt_prior_strength >= 0):
            print('modifying BN forward pass')
            nn.BatchNorm2d.prior = None
            nn.BatchNorm2d.forward = _modified_bn_forward
    
    return model

def config_finetune_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.eval()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    count_bn = 0
    # num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('num parameters = {}'.format(num_parameters))
    # for m in model.modules():
        # if m.requires_grad:
        # count_bn += 1
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        print(m)
        if isinstance(m, nn.BatchNorm2d):
            count_bn+=1
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            # m.track_running_stats = False
            # m.running_mean = None
            # m.running_var = None
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad==True)
    print('num parameters = {}'.format(num_parameters))
    # print("number of parameters count: {}".format(count_bn))
    return model

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    return model, optimizer

def adapt_multiple(model, inputs, optimizer, niter, batch_size, denormalize=None):
    

    prior_strength = 16
    tr_num = 4

    if prior_strength < 0:
        nn.BatchNorm2d.prior = 1
    else:
        nn.BatchNorm2d.prior = float(prior_strength) / float(prior_strength + 1)
    for iteration in range(niter):
        if denormalize!=None:
            aug_inputs = [augmix(denormalize(inputs[i]), normalize=True) for i in range(batch_size) for _ in range(tr_num)]
        else:
            aug_inputs = [augmix(inputs[i], normalize=False) for i in range(batch_size) for _ in range(tr_num)]

        aug_inputs = torch.stack(aug_inputs).cuda()
        optimizer.zero_grad()
        outputs, _ = model(aug_inputs)
        # outputs = outputs[:, imagenet_r_mask] ##For IMGNET-R, output should be 200 classes
        loss, _ = marginal_entropy(outputs)
        loss.backward()
        optimizer.step()
    nn.BatchNorm2d.prior = 1


def clip_adapt_multiple(model, inputs, text_tokens, optimizer, niter, batch_size, denormalize=None):
    
    prior_strength = 16
    tr_num = 8

    if prior_strength < 0:
        nn.BatchNorm2d.prior = 1
    else:
        nn.BatchNorm2d.prior = float(prior_strength) / float(prior_strength + 1)
    for iteration in range(niter):
        if denormalize!=None:
            aug_inputs = [augmix(denormalize(inputs[i]), normalize=True) for i in range(batch_size) for _ in range(tr_num)]
            # aug_inputs = [inputs[i] for i in range(batch_size) for _ in range(tr_num)]
        else:
            aug_inputs = [augmix(inputs[i], normalize=False) for i in range(batch_size) for _ in range(tr_num)]

        aug_inputs = torch.stack(aug_inputs).cuda()

        optimizer.zero_grad()
        outputs, _ = model(aug_inputs, text_tokens)

        # outputs = outputs[:, imagenet_r_mask] ##For IMGNET-R, output should be 200 classes
        loss, _ = marginal_entropy(outputs)
        loss.backward()
        optimizer.step()

    nn.BatchNorm2d.prior = 1

def test_single(model, inputs, label):

    # imagenet_r_mask = gen_mask()
    imagenet_a_mask = get_imgnet_a_mask()
    # print('imagenet A mask: ', len(imagenet_a_mask))
    prior_strength = 16

    if prior_strength < 0:
        nn.BatchNorm2d.prior = 1
    else:
        nn.BatchNorm2d.prior = float(prior_strength) / float(prior_strength + 1)

    with torch.no_grad():
        outputs, _ = model(inputs)
        # print(outputs.max(1)[1])
        # outputs = outputs[:, imagenet_a_mask]
    # print(label)
    
    correctness = (outputs.max(1)[1] == label).sum().item()

    nn.BatchNorm2d.prior = 1
    return correctness

def clip_test_single(model, inputs, text_tokens, label):

    imagenet_r_mask = gen_mask()

    prior_strength = 16

    if prior_strength < 0:
        nn.BatchNorm2d.prior = 1
    else:
        nn.BatchNorm2d.prior = float(prior_strength) / float(prior_strength + 1)

    with torch.no_grad():
        outputs, _ = model(inputs, text_tokens)
        # print(outputs.max(1)[1])
        # outputs = outputs[:, imagenet_r_mask]
    correctness = (outputs.max(1)[1] == label).sum().item()

    nn.BatchNorm2d.prior = 1
    return correctness


def adapt_multiple_tent(model, inputs, optimizer, niter, batch_size, denormalize=None):
    
    imagenet_r_mask = gen_mask()

    prior_strength = 16
    tr_num = 2

    if prior_strength < 0:
        nn.BatchNorm2d.prior = 1
    else:
        nn.BatchNorm2d.prior = float(prior_strength) / float(prior_strength + 1)
    
    for iteration in range(niter):
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        # outputs = outputs[:, imagenet_r_mask] ##For IMGNET-R, output should be 200 classes
        loss  = softmax_entropy(outputs).mean(0)
        # loss, _ = marginal_entropy(outputs)
        loss.backward()
        optimizer.step()

    nn.BatchNorm2d.prior = 1


def test_adapt_multiple(model, inputs, label):

    imagenet_r_mask = gen_mask()

    prior_strength = 32

    if prior_strength < 0:
        nn.BatchNorm2d.prior = 1
    else:
        nn.BatchNorm2d.prior = float(prior_strength) / float(prior_strength + 1)

    with torch.no_grad():
        outputs, _ = model(inputs)
        # print(outputs.max(1)[1])
        # outputs = outputs[:, imagenet_r_mask]

    correctness = (outputs.max(1)[1] == label).sum().item()

    nn.BatchNorm2d.prior = 1
    return correctness



def test_time_augmentation_baseline(model, inputs, batch_size, label, denormalize=None):

    prior_strength = 16
    niter = 1
    tr_num = 16
    if prior_strength < 0:
        nn.BatchNorm2d.prior = 1
    else:
        nn.BatchNorm2d.prior = float(prior_strength) / float(prior_strength + 1)
    for iteration in range(niter):

        if denormalize!=None:
            aug_inputs = [augmix(denormalize(inputs[i]), normalize=True) for i in range(batch_size) for _ in range(tr_num)]
        else:
            aug_inputs = [augmix(inputs[i], normalize=False) for i in range(batch_size) for _ in range(tr_num)]
        
        aug_inputs = torch.stack(aug_inputs).cuda()
        with torch.no_grad():
            outputs, _ = model(aug_inputs)

    marginal_output = [outputs[i*tr_num: (i+1)*tr_num].mean(0) for i in range(batch_size)]
    marginal_output = torch.stack(marginal_output)
    print(marginal_output.max(1)[1])

    correctness = (marginal_output.max(1)[1] == label).sum().item()

    nn.BatchNorm2d.prior = 1
    
    return correctness