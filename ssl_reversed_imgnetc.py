"""
TODO: Check how the new delta, generated after test-time augmentation, differs as compared to the previous delta. One
TODO: way to compare would be to compare the CE-SSL-loss. Maybe plot the difference.
"""

import argparse
from ast import Param
from audioop import reverse
import os
import time
import csv
from pathlib import Path
from torch.utils.data import Subset
import torch.utils.data as data
from sklearn.random_projection import johnson_lindenstrauss_min_dim
import torch
from tqdm import tqdm
import torch.nn as nn
from learning.resnet import resnet50
from learning.wideresnet import WRN34_rot_out_branch,WRN34_rot_out_branch2
from utils import *
import torch.nn.functional as F
import torchvision.transforms as transforms
from RandAugment import trans_aug, get_imgnet_transaug_param, init_sharpness_3by3_kernel, init_sharpness_5by5_kernel, init_sharpness_random_kernel, init_sharpness_random_composite_kernel
from robustbench.utils import load_model
from robustbench.model_zoo.enums import ThreatModel
from robustbench.data import load_cifar10c
import kornia
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot
from adapt_helpers import adapt_multiple, test_single, copy_model_and_optimizer, load_model_and_optimizer, config_model, adapt_multiple_tent, test_time_augmentation_baseline
from tent_adapt_helper import setup_tent
from grad_cam import GradCAM, save_gradcam, BackPropagation
import tent_adapt_helper as tent
import norm_adapt_helper as norm
from swd import swd
import time


upper_limit, lower_limit = 1, 0
imgnet_mean=(0.485, 0.456, 0.406)
imgnet_std=(0.229, 0.224, 0.225)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


class Batches:
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle,
            drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return ({'input': x.to(device).float(), 'target': y.to(device).long()} for (x, y) in self.data_loader)

    def __len__(self):
        return len(self.data_loader)


def (model, model_ssl, criterion, X, epsilon, alpha, attack_iters, norm):
    """Reverse algorithm that optimize the SSL loss via PGD"""
    transform_num = 3
    contrast_batch = Contrastive_Transform(X.shape[2])

    delta = torch.unsqueeze(torch.zeros_like(X[0]).cuda(), 0)
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    elif norm == 'l_1':
        pass
    else:
        raise ValueError
    #delta = clamp(delta, lower_limit - torch.mean(X, dim=0), upper_limit - torch.mean(X, dim=0))
    delta.requires_grad = True
    for _ in range(attack_iters):

        new_x = X + delta
        # import pdb; pdb.set_trace()
        loss = -SslTrainer.compute_ssl_contrastive_loss(contrast_batch(new_x, transform_num), criterion, model, model_ssl, X.shape[0], transform_num, no_grad=False)[0]
        loss.backward()
        grad = delta.grad.detach()

        d = delta
        g = grad

        x = X
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        elif norm == "l_1":
            g_norm = torch.sum(torch.abs(g.view(g.shape[0], -1)), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=1, dim=0, maxnorm=epsilon).view_as(d)

        # d = clamp(d, lower_limit - x, upper_limit - x)

        delta.data = d
        delta.grad.zero_()
    max_delta = delta.detach()
    return max_delta

def reset_transform_param(X, aug_name, norm, epsilon, update_kernel, ksize):
    
    eps = get_imgnet_transaug_param(aug_name)

    if update_kernel == 'comp':
        print('init with composite random kernel')
        init_kernel_param = init_sharpness_random_composite_kernel(X)
    elif update_kernel == 'rand':
        init_kernel_param = init_sharpness_random_kernel(X, ksize)

    init_param = torch.rand(1) * (eps[1] - eps[0]) + eps[0]

    delta = torch.unsqueeze(torch.zeros_like(X[0]).cuda(), 0)
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    elif norm == 'l_1':
        pass
    else:
        raise ValueError
    return init_param, init_kernel_param

     
def compute_reverse_transformation(model, model_ssl, criterion, X, epsilon, alpha, attack_iters, norm, aug_name, denormalize, normalize, update_kernel, ksize):

    """Reverse algorithm that optimize the SSL loss via PGD"""
    # import pdb; pdb.set_trace()
    isRandom = True
    isIter = True
    transform_num = 3
    contrast_batch = Contrastive_Transform(X.shape[2])

    if isRandom and isIter:

        eps = get_imgnet_transaug_param(aug_name)
        step_size = 1.15 * ((eps[1] - eps[0]) / 2) / attack_iters


        init_factor_param, init_kernel_param = reset_transform_param(X, aug_name, norm, epsilon, update_kernel, ksize)
        param = init_factor_param
        if update_kernel == 'comp':
            kernel_param1, kernel_param2 = init_kernel_param
        else:
            kernel_param1 = init_kernel_param
        # delta = init_delta

        before_loss = SslTrainer.compute_ssl_contrastive_loss(contrast_batch(X, transform_num), criterion, model, model_ssl, X.shape[0], transform_num, no_grad=True)[0]
        
        param.requires_grad = True
        kernel_param1.requires_grad = True
        if update_kernel == 'comp':
            kernel_param2.requires_grad = True
        # delta.requires_grad = True

        for _ in range(attack_iters):

            factor_param_all = param.repeat(X.size(0))  
            if update_kernel == '':
                kernel_param1 = init_kernel_param
                new_x = normalize(trans_aug(aug_name, denormalize(X), kernel_param1, factor_param_all)) 
            elif update_kernel == 'rand':
                new_x = normalize(trans_aug(aug_name, denormalize(X), kernel_param1, factor_param_all))  
            elif update_kernel == 'comp':
                tmp_x = normalize(trans_aug(aug_name, denormalize(X), kernel_param1, factor_param_all))
                new_x = normalize(trans_aug(aug_name, denormalize(tmp_x), kernel_param2, factor_param_all))  
    
            loss = -SslTrainer.compute_ssl_contrastive_loss(contrast_batch(new_x, transform_num), criterion, model, model_ssl, X.shape[0], transform_num, no_grad=False)[0]

            print(loss)
            loss.backward()
            param_grad = param.grad.detach()
            
            kernel_param_grad1 = kernel_param1.grad.detach()
            if update_kernel == 'comp':
                kernel_param_grad2 = kernel_param2.grad.detach()
            # delta_grad = delta.grad.detach()


            p = param
            g = param_grad

            k = kernel_param1
            g1 = kernel_param_grad1

            if update_kernel == 'comp':
                k2 = kernel_param2
                g3 = kernel_param_grad2

            # d = delta
            # g2 = delta_grad

            x = X
            
            if g == 0: g = g - (torch.rand(1) / 1e5)

            p = torch.clamp(p + torch.sign(g) * step_size, eps[0], eps[1])
            k = torch.clamp(k + torch.sign(g1) * 0.1, torch.tensor(1).to(device), torch.tensor(5).to(device))
            if update_kernel == 'comp':
                k2 = torch.clamp(k2 + torch.sign(g3) * 0.1, torch.tensor(1).to(device), torch.tensor(5).to(device))

            # if norm == "l_inf":
            #     d = torch.clamp(d + alpha * torch.sign(g2), min=-epsilon, max=epsilon)
            # elif norm == "l_2":
            #     g_norm = torch.norm(g2.view(g2.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            #     scaled_g = g2 / (g_norm + 1e-10)
            #     d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
            # elif norm == "l_1":
            #     g_norm = torch.sum(torch.abs(g2.view(g2.shape[0], -1)), dim=1).view(-1, 1, 1, 1)
            #     scaled_g = g2 / (g_norm + 1e-10)
            #     d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=1, dim=0, maxnorm=epsilon).view_as(d)
            
            
            param.data = p
            param.grad.zero_()
            
            kernel_param1.data = k
            kernel_param1.grad.zero_()

            if update_kernel == 'comp':
                kernel_param2.data = k2
                kernel_param2.grad.zero_()

            # delta.data = d
            # delta.grad.zero_()
            #print('update param: {}'.format(param))
        
        final_loss = -1 * loss.item()
        if final_loss > before_loss:
            print('use initial kernel!!')
            if update_kernel == 'comp':
                max_kernel = [init_kernel_param[0].detach(), init_kernel_param[1].detach()]
            else:
                max_kernel = init_kernel_param.detach()
        else: 
            if update_kernel == 'comp':
                max_kernel = [kernel_param1.detach(), kernel_param2.detach()]
            else:
                max_kernel = kernel_param1.detach()
        max_param = param.detach()
        # max_delta = delta.detach()
        #param = param.detach()

        # loss = loss.item()
        #param = param.detach()
            
        return max_kernel, max_param, before_loss.item(), final_loss
   
               
def test_acc_reverse_vector(model, model_ssl, test_batches_orig, test_batches, criterion, attack_iters, args, normalize, denormalize):
    epsilon = (64 / 255.)
    pgd_alpha = (16 / 255.)

    test_n, clean_acc, acc = 0., 0., 0.
    n_proj=128
    batch_size = args.test_batch
    n_batches = math.ceil(5000 / batch_size)

    before_loss_list = []
    final_loss_list = []
    before_swd_list = []
    after_swd_list = []
    before_swd_score = []
    after_swd_score = []

    x_test, y_test  = next(iter(test_batches))
    x_test, y_test = x_test.cuda(), y_test.cuda()

    # x_orig, y_orig  = next(iter(test_batches_orig))
    # x_orig, y_orig = x_orig.cuda(), y_orig.cuda()
    # raw_dataloader_iterater = iter(test_batches_orig)
    # for i, batch in enumerate(test_batches):
       
    for counter in range(n_batches):
        start_time = time.time()
        # # if test_n <= 1600:
        x = x_test[counter * batch_size:(counter + 1) * batch_size]
        y = y_test[counter * batch_size:(counter + 1) * batch_size]

        # orig_x = x_orig[counter * batch_size:(counter + 1) * batch_size]
        # x = batch[0].cuda()
        # y = batch[1].cuda()
        test_n += y.shape[0]
            

        # orig_batch = next(raw_dataloader_iterater)
        # orig_x = orig_batch['input']

        with torch.no_grad():
            clean_out, _ = model(x)

        clean_acc += (clean_out.max(1)[1] == y).sum().item()

        if args.aug_name is None:
            delta = compute_reverse_attack(model, model_ssl, criterion, x,
                                                        epsilon, pgd_alpha, attack_iters, 'l_inf')
            new_x = x + delta

        else: 
    
            new_kernel, new_param, before_loss, final_loss = compute_reverse_transformation(model, model_ssl, criterion, x,
                                                        epsilon, pgd_alpha, attack_iters, 'l_2', args.aug_name,
                                                        denormalize, normalize, args.update_kernel, args.ksize)
            param_all = new_param.repeat(x.shape[0])

            if args.update_kernel == 'comp':
                tmp_x = normalize(trans_aug(args.aug_name, denormalize(x), new_kernel[0], param_all))
                new_x = normalize(trans_aug(args.aug_name, denormalize(tmp_x), new_kernel[1], param_all))              
            else:
                new_x = normalize(trans_aug(args.aug_name, denormalize(x), new_kernel, param_all))  
            
        end_time =  time.time()   
        with torch.no_grad():
            out, _ = model(new_x)

        # before_loss_list.append(before_loss)
        # final_loss_list.append(final_loss)

        # for i in range(1):
        #     before_swd_score.append(swd(x, orig_x , proj_per_repeat=n_proj, n_repeat_projection=512 // n_proj))
        #     after_swd_score.append(swd(new_x, orig_x, proj_per_repeat=n_proj, n_repeat_projection=512 // n_proj))
        # print('swd score: {}, {}'.format(len(before_swd_score), len(after_swd_score)))
        # before_swd_list.append(before_swd_score)
        # after_swd_list.append(after_swd_score)

        if args.allow_gcam :
            # with open('./loss_histogram/imnetc_' + args.corruption + '_s' + str(args.severity) +'.npy', 'wb') as f:
                # np.save(f, np.array([before_loss_list, final_loss_list]))
            for i in range(x.shape[0]):
                # if counter*batch_size+i==59:
                if out.max(1)[1][i] == y[i] and clean_out.max(1)[1][i] != y[i]:
                    print('label: ', y[i], ', reverse class: ',  out.max(1)[1][i], ', original class: ', clean_out.max(1)[1][i])
                    target_layer = 'layer4.2'
                    bp = BackPropagation(model=model)
                    gcam = GradCAM(model=model) ##initialize grad_cam funciton
                    print('computer grad cam..')
                    compute_gcam(counter*batch_size+i, gcam, bp, orig_x[i], x[i], new_x[i], y[i], target_layer, args, denormalize)



        acc += (out.max(1)[1] == y).sum().item()
        print("test number: {} time: {} before reverse: {} after reverse: {}".format(test_n, end_time-start_time, clean_acc/test_n, acc/test_n))
    # print(np.array(before_loss_list).mean(), np.array(final_loss_list).mean(), np.array(final_loss_list).mean()-np.array(before_loss_list).mean())
    print('Accuracy after SSL training: {}'.format(acc / test_n))
    # with open('./output/' + args.output_fn, 'wb') as f:
        # np.save(f, [np.array(before_swd_list), np.array(after_swd_list)])
    
    return clean_acc/test_n, acc/test_n


def test_acc_reverse_vector_memo_adapt(model, model_ssl, opt, test_batches, criterion, attack_iters, args, normalize, denormalize):
    
    epsilon = (16 / 255.)
    pgd_alpha = (4 / 255.)

    test_n, clean_acc, acc = 0., 0., 0.
    batch_size = args.test_batch
    n_batches = math.ceil(5000 / batch_size)

    before_loss_list = []
    final_loss_list = []
    correct = []

    model = config_model(model)

    x_test, y_test, _  = next(iter(test_batches))
    x_test, y_test = x_test.cuda(), y_test.cuda()


    # for i, batch in enumerate(test_batches):
    for counter in range(n_batches):
        
        model_state, opt_state = copy_model_and_optimizer(model, opt)

        x = x_test[counter * batch_size:(counter + 1) * batch_size]
        y = y_test[counter * batch_size:(counter + 1) * batch_size]
        test_n += y.shape[0]
        
        nn.BatchNorm2d.prior = 1
        with torch.no_grad():
            clean_out, _ = model(x)

        # clean_out = clean_out[: , :10]
        clean_acc += (clean_out.max(1)[1] == y).sum().item()

        if args.aug_name is None:
            delta = compute_reverse_attack(model, model_ssl, criterion, x,
                                                    epsilon, pgd_alpha, attack_iters, 'l_inf')
            new_x = x + delta

        else: 
            aug_name = args.aug_name
            delta, param, before_loss, final_loss = compute_reverse_transformation(model, model_ssl, criterion, x,
                                                    epsilon, pgd_alpha, attack_iters, 'l_2', args.aug_name,  denormalize, normalize)
            param_all = param.repeat(x.shape[0])
            # print(delta.shape)
            # print(param)
            new_x = normalize(trans_aug(args.aug_name, denormalize(x), param_all))  + delta
        
            before_loss_list.append(before_loss)
            final_loss_list.append(final_loss)

        if args.adapt_only:
            print('adapt_only!!')
            new_x = x

        # adapt_multiple(model, x, opt, 1, y.shape[0], denormalize)
        ## Evaluate baseline MEMO-single
        for i in range(new_x.shape[0]):
            adapt_x = new_x[i].unsqueeze(0)
            
            adapt_multiple(model, adapt_x, opt, 1, adapt_x.shape[0], denormalize)
            correctness = test_single(model, adapt_x, y[i])
            acc += correctness

        ## Evaluate baseline TENT
        # adapt_multiple_tent(model, x, opt, 1, y.shape[0])
        # correctness = test_single(model, x, y)

        # correctness = test_time_augmentation_baseline(model, new_x, y.shape[0], y, denormalize)
        # acc += correctness
            #reset model
            model, opt = load_model_and_optimizer(model, opt, model_state, opt_state)
        if counter % 1 == 0:
            print("test number: {} before reverse: {} after reverse: {}".format(test_n, clean_acc/test_n, acc/test_n))
    print('Accuracy after SSL training: {}'.format(acc / test_n))
    
    return clean_acc/test_n, acc/test_n


def test_acc_reverse_vector_tent_adapt(base_model, model_ssl, opt, test_batches, criterion, attack_iters, args, normalize, denormalize):
    
    epsilon = (16 / 255.)
    pgd_alpha = (4 / 255.)

    test_n, clean_acc, acc = 0., 0., 0.
    batch_size = args.test_batch
    n_batches = math.ceil(5000 / batch_size)

    before_loss_list = []
    final_loss_list = []
    correct = []

    old_model_state, old_opt = copy_model_and_optimizer(base_model, opt)
    old_backbone_model = resnet50(pretrained=True)
    old_backbone_model.cuda().eval()
    load_model_and_optimizer(old_backbone_model, opt, old_model_state, old_opt)
    
    x_test, y_test, _  = next(iter(test_batches))
    x_test, y_test = x_test.cuda(), y_test.cuda()

    if args.allow_adapt == 'tent':
        print('adapt with tent')
        model = tent.setup_tent(base_model)
    elif args.allow_adapt == 'norm':
        print('adapt with norm')
        model = norm.setup_norm(base_model)

    # for i, batch in enumerate(test_batches):
    for counter in range(n_batches):
        
        # x, y = batch['input'], batch['target']

        x = x_test[counter * batch_size:(counter + 1) * batch_size]
        y = y_test[counter * batch_size:(counter + 1) * batch_size]
        test_n += y.shape[0]
        
        with torch.no_grad():
            clean_out, _ = old_backbone_model(x)

        # clean_out = clean_out[: , :10]
        clean_acc += (clean_out.max(1)[1] == y).sum().item()

        if args.aug_name == 'linf':
            delta = compute_reverse_attack(old_backbone_model, model_ssl, criterion, x,
                                                    epsilon, pgd_alpha, attack_iters, 'l_inf')
            new_x = x + delta

        elif args.aug_name == 'sharpness': 
            
            delta, param, before_loss, final_loss = compute_reverse_transformation(old_backbone_model, model_ssl, criterion, x,
                                                    epsilon, pgd_alpha, attack_iters, 'l_2', args.aug_name,  denormalize, normalize)
            param_all = param.repeat(x.shape[0])
            # print(delta.shape)
            # print(param)
            new_x = normalize(trans_aug(args.aug_name, denormalize(x), param_all))  + delta
        
            before_loss_list.append(before_loss)
            final_loss_list.append(final_loss)

        ## adapt with baseline TENT
        if args.adapt_only:
            new_x = x
        out = model(new_x)
        acc += (out.max(1)[1] == y).sum().item()

        #reset model
        if counter % 1 == 0:
            print("test number: {} before reverse: {} after reverse: {}".format(test_n, clean_acc/test_n, acc/test_n))
    print('Accuracy after SSL training: {}'.format(acc / test_n))
    
    return clean_acc/test_n, acc/test_n

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=1024, type=int)
    parser.add_argument('--test_batch', default=200, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--save_freq', default=1, type=int)
    parser.add_argument('--data-dir', default='../ImageNet-Data', type=str)
    parser.add_argument('--corr-dir', default='./data/ImageNetC-customize/', type=str)
    parser.add_argument('--output_dir', default='./imgnetc_result', type=str)
    parser.add_argument('--output_fn', default='train_ssl', type=str)
    parser.add_argument('--save_root_path', default='data/ckpts/', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--md_path', default='./resnet50.pth', type=str)
    parser.add_argument('--ckpt', default='./data/ckpts/imagenetc_1/ssl_dist_199.pth', type=str)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=0.75)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--allow_adapt', default='', type=str)
    parser.add_argument('--update_kernel', default='', type=str)
    parser.add_argument('--ksize', default=3, type=int)
    parser.add_argument('--adapt_only', action='store_true')
    parser.add_argument('--allow_gcam', action='store_true')
    parser.add_argument('--use_subclass', action='store_true')
    parser.add_argument('--aug_name', default=None, type=str)
    parser.add_argument('--corruption', default='all', type=str)
    parser.add_argument('--severity', default=1, type=int)
    parser.add_argument('--attack_iters', help='list of attack iterations', type=lambda s: [int(item) for item in s.split(',')])
    return parser.parse_args()


class SslTrainer:

    def __init__(self):

        # self.normalize = torch.jit.script(transforms)
        # self.denormalize = torch.jit.script(inv_transforms)

        self.contrast_transform = Contrastive_Transform(size=224) #before downscale is 224

    def train_one_epoch(self, model, contrast_head, train_batches, opt, lr_schedule_values, start_steps, num_training_steps_per_epoch, criterion, debug=False):
        train_loss, train_n, matches = 0.0, 0, 0.0
        update_freq = 1

        for data_iter_step, batch in enumerate(tqdm(train_batches)):

            step = data_iter_step // update_freq

            if step >= num_training_steps_per_epoch:
                continue
            it = start_steps + step  # global training iteration
            # Update LR & WD for the first acc
            if lr_schedule_values is not None and data_iter_step % update_freq == 0:
                for i, param_group in enumerate(opt.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] 

            batch_train_loss, batch_matches, batch_size = self.step(model, contrast_head, batch['input'], opt, criterion)
            train_loss += batch_train_loss * batch_size
            matches += batch_matches
            train_n += batch_size
            if debug:
                break
        return train_loss, train_n, matches

    def step(self, model, contrast_head, x, opt, criterion):

        transforms_num = 3

        x_const = self.contrast_transform(x, transforms_num)

        opt.zero_grad()
        c_loss, correct = self.compute_ssl_contrastive_loss(x_const, criterion, model, contrast_head, x.shape[0], transforms_num)

    
        c_loss.backward()
        opt.step()

        return c_loss.item(), correct, x_const.shape[0]


    @staticmethod
    def compute_ssl_contrastive_loss(x, criterion, model, contrast_head, bs, transform_num, no_grad=True):
        if no_grad:
            with torch.no_grad():
                _, out = model(x)
        else:
            _, out = model(x)

        output = contrast_head(out)
     
        c_loss, correct = SslTrainer.constrastive_loss_func(output, criterion, bs, transform_num+1) #if using contrastive loss

        return c_loss, correct

    @staticmethod
    def constrastive_loss_func(contrastive_head, criterion, bs, n_views):
            features = F.normalize(contrastive_head, dim=1)

            labels = torch.cat([torch.arange(bs) for i in range(n_views)], dim=0)
            labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            labels = labels.cuda()
            
            similarity_matrix = torch.matmul(features, features.T)
            # print("sim matrix: {}".format(similarity_matrix))

            mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
            labels = labels[~mask].view(labels.shape[0], -1)
            similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

            # select and combine multiple positives
            positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
            # select only the negatives the negatives
            negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

            positives = torch.mean(positives, dim=1, keepdim=True)

            logits = torch.cat([positives, negatives], dim=1)
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
          
            temperature = 0.2
            logits = logits / temperature

            xcontrast_loss = criterion(logits, labels)

            correct = (logits.max(1)[1] == labels).sum().item()
            return xcontrast_loss, correct

    def test_contrast_head(self, criterion, model, contrast_head, batches):
            matches = 0
            test_n = 0
            test_loss = 0
            transform_num = 3

            contrast_tranform = Contrastive_Transform()
            for b in tqdm(batches):
                batch_size = b['target'].size(0)
                x = b['input']
                x_const = contrast_tranform(x, transform_num)

                transform_num = 3
                c_loss, batch_correct = self.compute_ssl_contrastive_loss(x_const, criterion, model, contrast_head, x.shape[0], transform_num)
                test_loss += c_loss * x_const.shape[0]
                matches += batch_correct
                test_n += x_const.shape[0]

            return test_loss, matches, test_n 

    @staticmethod
    def test_(pred, target):
        matches = (pred.max(1)[1] == target).sum().item()
        return  matches


    @staticmethod
    def save_state_dict(dist_head, optimizer, epoch, args, isbest):
        state_dict = {
            'epoch': epoch,
            'contrast_head_state_dict': dist_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        if isbest:
            torch.save(state_dict, os.path.join(args.output_dir, "ssl_contrast_best.pth"))
        else:
            torch.save(state_dict, os.path.join(args.output_dir, f'ssl_contrast_{epoch}.pth'))

def main():
    args = get_args()
    import uuid
    import datetime
    unique_str = str(uuid.uuid4())[:8]
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    transforms = [Crop(224, 224)]

    corruption_type = ['gaussian_noise', 'shot_noise','impulse_noise',
                      'defocus_blur', 'motion_blur' , 'glass_blur','zoom_blur','snow', 
                      'frost', 'fog', 'brightness', 'contrast', 'elastic_transform',
                      'pixelate', 'jpeg_compression', 'orig']


    print('load ImageNet-1K data....')
    train_dataset, test_dataset = imagenet1000()

    if args.use_subclass:

        class_keys = ['n01443537', 'n01494475', 'n01496331', 'n01498041', 
                    'n01514859' , 'n01518878', 'n01531178', 'n01532829', 'n01534433', 'n01537544']
    
        train_class_idx = [train_dataset.class_to_idx[k] for k in class_keys]
        test_class_idx = [test_dataset.class_to_idx[k] for k in class_keys]


        train_idx = [torch.tensor(train_dataset.targets) == i for i in train_class_idx]
        test_idx = [torch.tensor(test_dataset.targets) == i for i in test_class_idx]


        train_mask = train_idx[0]
        for i in range(1, len(train_idx)): train_mask = train_mask | train_idx[i]
        test_mask = test_idx[0]
        for i in range(1, len(test_idx)): test_mask = test_mask | test_idx[i]

        train_indices = train_mask.nonzero().reshape(-1)   
        test_indices = test_mask.nonzero().reshape(-1)

        train_subdataset = Subset(train_dataset, train_indices)
        test_subdataset = Subset(test_dataset, test_indices)
        

        train_batches = Batches(train_subdataset, args.batch_size, shuffle=True, num_workers=2)

        relabel_testsets = []
        for i in range(len(test_subdataset)):
            relabel_testsets.append([test_subdataset[i][0], test_class_idx.index(test_subdataset[i][1])])
        test_batches = Batches(relabel_testsets, args.test_batch, shuffle=False, num_workers=2)

    else: 
        train_batches = Batches(train_dataset, args.batch_size, shuffle=True, num_workers=2)
        test_batches = Batches(test_dataset, args.test_batch, shuffle=False, num_workers=2)

    h = []
    idx = []
    if args.use_subclass:
        relabel_testsets = []
        for i in range(len(test_subdataset)):
            h.append(test_class_idx.index(test_subdataset[i][1]))
    else: 
        for i in range(len(test_dataset)):
            if i % 50 < 5 : #1-K classes, 5 images/class
                # if test_dataset.targets[i] < 100: #10 classes 50 images/class
                idx.append(i)
                h.append(test_dataset.targets[i])

    
    mask = np.full(len(test_dataset), 0, dtype=int)
    mask[idx] = 1
    test_indices = list(mask.nonzero())[0]
    test_subdataset = Subset(test_dataset, test_indices)
    print('Number of original sub-testdataset : ', len(test_subdataset))
    test_batches_orig = Batches(test_subdataset, args.test_batch, shuffle=False, num_workers=2)

    
    from learning.resnet import resnet50
    from learning.ssl import Ssl_model_contrast2

    if args.use_subclass:
        model = resnet50(pretrained=True, num_classes=10)
        print(model)
    else: 
        model = resnet50(pretrained=True)
        print(model)

    ssl_head = Ssl_model_contrast2(2048)
    print(ssl_head)

    # model = nn.DataParallel(model).cuda()
    # dist_head = nn.DataParallel(dist_head).cuda()

    ssl_head.eval()
    decay, no_decay = [], []
    for name, param in ssl_head.named_parameters():
        if 'bn' not in name and 'bias' not in name:
            decay.append(param)
        else:
            no_decay.append(param)

    params = [{'params': decay, 'weight_decay': 0},
              {'params': no_decay, 'weight_decay': 0}]

    learning_rate = args.lr
    # backbone_opt = torch.optim.AdamW(model.parameters(), lr=0.00025)
    backbone_opt = torch.optim.SGD(model.parameters(), lr=0.00025, momentum=0.9)
    ssl_opt = torch.optim.Adam(params, lr=learning_rate)


    num_training_steps_per_epoch = len(train_dataset) // args.batch_size
    print("Use step level LR scheduler!")
    lr_schedule_values = cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    if args.md_path:
        if os.path.isfile(args.md_path):
            #checkpoint = torch.load(args.md_path)['state_dict']
            if args.use_subclass:
                checkpoint = torch.load(args.md_path)['model']
                model.load_state_dict(checkpoint)
            else:
                checkpoint = torch.load(args.md_path)
                model.load_state_dict(checkpoint)  #['state_dict']
            
            #model.load_state_dict(checkpoint)
            print("=> load chechpoint found at {}".format(args.md_path))
        else:
            print("=> no checkpoint found at '{}'".format(args.md_path))
    model = model.eval().cuda() 

    restart_epoch = 0
    if os.path.exists(args.ckpt):
        print("Loading checkpoint: {}".format(args.ckpt))
        ckpt = torch.load(args.ckpt)
        ssl_head.load_state_dict(ckpt['contrast_head_state_dict'])
        ssl_opt.load_state_dict(ckpt['optimizer_state_dict'])
        restart_epoch = ckpt['epoch'] + 1
    print('loaded model epoch: {}'.format(restart_epoch))
    ssl_head.eval().cuda()


    # defines transformation for SSL contrastive learning.
    size = 224
    imgnet_mean=(0.485, 0.456, 0.406)
    imgnet_std=(0.229, 0.224, 0.225)

    from torchvision.transforms import transforms

    script_transforms = torch.nn.Sequential(
        transforms.Normalize(imgnet_mean, imgnet_std)
    )
    normalize = torch.jit.script(script_transforms)

    inv_transforms = torch.nn.Sequential(
            transforms.Normalize(mean=[ 0., 0., 0. ], std=[ 1/0.229, 1/0.224, 1/0.225 ]),
            transforms.Normalize(mean=[ -0.485, -0.456, -0.406 ], std=[ 1., 1., 1.] )
    )
    denormalize = torch.jit.script(inv_transforms)

    criterion = torch.nn.CrossEntropyLoss().cuda()

    if args.eval:

        attack_iters = args.attack_iters
        severity= args.severity
        
        if args.corruption == 'all':
            c_idx = np.arange(15)
        else:
            c_idx = [np.array(corruption_type.index(args.corruption))]
            
        for i in c_idx:
            if i < 15:
                # if args.severity == 1:
                #     x_test = np.load(args.corr_dir + '/' + str(args.severity) + '/' + str(corruption_type[i])+'.npy')[0]
                # else: 
                #     x_test = np.load(args.corr_dir + '/' + str(args.severity) + '/' + str(corruption_type[i])+'.npy')                  
                # print('load ' + args.corr_dir + str(args.severity) + '/' + str(corruption_type[i])+'.npy')
                # if len(x_test) > 5000:
                #     subset_xtest = x_test[idx] 
                corr_dataset, orig_dataset = load_imagenetC(str(corruption_type[i]), args.severity)    
                print('load corruption data  --> {} {}'.format(corruption_type[i], args.severity))
            # else:
            #     x_test = np.load(args.corr_dir + 'original.npy')
            #     print(x_test.shape)
            #     print('load original.npy')

           
            # if len(x_test) > 5000 and i < 15:
            #     test_set = list(zip(transpose(normalise(x_test[idx], imgnet_mean, imgnet_std)), h))
            # else:
            #     test_set = list(zip(transpose(normalise(x_test, imgnet_mean, imgnet_std)), h))

            # if args.use_subclass:
            #     test_set = list(zip(transpose(normalise(x_test, imgnet_mean, imgnet_std)), h))
            print('Number of OOD test samples: ', len(corr_dataset))
            # test_batches_ood = Batches(test_set, args.test_batch, shuffle=False, num_workers=2)
            test_batches_orig = torch.utils.data.DataLoader(orig_dataset, batch_size=5000, shuffle=False, num_workers=2)
            test_batches_ood = torch.utils.data.DataLoader(corr_dataset, batch_size=5000, shuffle=False, num_workers=2)
        
            trainer = SslTrainer()
            
            robust_acc = []
            clean_acc = []
            for attack_iter in attack_iters:

                print('aug name: ', args.aug_name)
                print('corruption type: ', corruption_type[i:i+1][0])
                # acc1, acc2 = test_acc_reverse_vector(model, ssl_head, test_batches_orig, criterion, attack_iter, args.aug_name, normalize, denormalize)
                if args.allow_adapt == 'memo':
                    acc1, acc2 = test_acc_reverse_vector_memo_adapt(model, ssl_head, backbone_opt, test_batches_ood, 
                                        criterion, attack_iter, args, normalize, denormalize)
                elif args.allow_adapt == 'tent' or args.allow_adapt == 'norm':
                    acc1, acc2 = test_acc_reverse_vector_tent_adapt(model, ssl_head, backbone_opt, test_batches_ood, criterion, attack_iter, 
                                                                                            args, normalize, denormalize)
                else:
                    acc1, acc2 = test_acc_reverse_vector(model, ssl_head, test_batches_orig, test_batches_ood,
                                        criterion, attack_iter, args, normalize, denormalize)


                print("Reverse with cross, acc before reversed: {} acc after reversed: {} ".format(acc1, acc2))

                with open(os.path.join(args.output_dir, args.output_fn), 'a') as f: 
                    writer = csv.writer(f)
                    writer.writerow(['l_2 ', args.aug_name, 'reverse_iter: ', args.attack_iters, 'corruption: ', corruption_type[i:i+1], 'severity: '+ str(args.severity), 'batch-size: '+str(args.test_batch), acc1, acc2])
        
            
    else:
        trainer = SslTrainer()
        model.eval().cuda()
        ssl_head.train().cuda()
        best_matches = 0
        with open(os.path.join(args.output_dir, "train_log.csv"), 'a') as f: 
            writer = csv.writer(f)
            writer.writerow(['train epoch', 'train loss', 'train match acc.'])
            
        for epoch in range(restart_epoch, 201):

            print("Epoch number: {}".format(epoch))
            start_step = epoch * num_training_steps_per_epoch
            train_loss, train_n, train_matches = trainer.train_one_epoch(model, ssl_head, train_batches, ssl_opt,
                                                lr_schedule_values, start_step, num_training_steps_per_epoch,
                                                   criterion, args.debug)
            
            print('Epoch: %d, Train accuracy: %.4f, Train loss:  %.4f' % (epoch, train_matches / train_n, train_loss / train_n))
            
            if train_matches / train_n > best_matches:
                trainer.save_state_dict(ssl_head, ssl_opt, epoch, args, isbest=True)
                best_matches = train_matches / train_n
            if epoch % args.save_freq == 0 :
                trainer.save_state_dict(ssl_head, ssl_opt, epoch, args, isbest=False)
                        
            with open(os.path.join(args.output_dir, "train_log.csv"), 'a') as f: 
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss / train_n, train_matches / train_n])
               


if __name__ == "__main__":
    main()