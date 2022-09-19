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

def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits


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
    tr_num = 32

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
        # print('before adapt:', outputs.max(1)[1])
        loss, logits = marginal_entropy(outputs)
        loss.backward()
        optimizer.step()
    nn.BatchNorm2d.prior = 1

def test_single(model, inputs, label):

    prior_strength = 16

    if prior_strength < 0:
        nn.BatchNorm2d.prior = 1
    else:
        nn.BatchNorm2d.prior = float(prior_strength) / float(prior_strength + 1)

    with torch.no_grad():
        outputs, _ = model(inputs)

    correctness = (outputs.max(1)[1] == label).sum().item()

    nn.BatchNorm2d.prior = 1
    return correctness