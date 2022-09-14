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

from sklearn.random_projection import johnson_lindenstrauss_min_dim
import torch
from tqdm import tqdm
import torch.nn as nn
from timm.data.mixup import Mixup

from cifar_10_1_dataset import load_new_test_data
from learning.wideresnet import WRN34_rot_out_branch,WRN34_rot_out_branch2
from utils import *
import torchvision.transforms as transforms
import torch.nn.functional as F
from RandAugment import trans_aug_list, get_transAug_param
from robustbench.utils import load_model
from robustbench.data import load_cifar10c
import kornia
from scipy import stats
mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot
upper_limit, lower_limit = 1, 0


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


def compute_universal_reverse_attack(model, model_ssl, criterion, X, epsilon, alpha, attack_iters, norm):
    
    transform_num = 3
    contrast_batch = Contrastive_Transform()
    
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

    delta.requires_grad = True
    for _ in range(attack_iters):
        delta_all = delta.repeat(X.size(0), 1, 1, 1)
        new_x = X + delta_all
    
        loss = -SslTrainer.compute_ssl_contrastive_loss(contrast_batch(new_x, transform_num), criterion, model, model_ssl, X.shape[0], transform_num, no_grad=False)[0]
        # loss = -SslTrainer.compute_ssl_rot_loss(new_x, angles, criterion, model, model_ssl, no_grad=False)[0]
        loss.backward()
        grad = delta.grad.detach()
        d = delta
        g = grad
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha)
            d_norm = torch.norm(d)
            d = d / (d_norm + 1e-10)
        delta.data = d
        delta.grad.zero_()
    max_delta = delta.detach()
    return max_delta


def compute_reverse_attack(model, model_ssl, criterion, X, epsilon, alpha, attack_iters, norm):
    """Reverse algorithm that optimize the SSL loss via PGD"""

    transform_num = 3
    contrast_batch = Contrastive_Transform(X.shape[2])
    
    # delta = torch.zeros_like(X[0]).cuda()
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

        # TODO: here the neg sample is fixed, we can also try random neg sample to enlarge and diversify
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

        #d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data = d
        delta.grad.zero_()
    max_delta = delta.detach()
    return max_delta

def compute_reverse_transformation(model, model_ssl, criterion, X, epsilon, alpha, attack_iters, norm, aug_name):

    """Reverse algorithm that optimize the SSL loss via PGD"""
    # import pdb; pdb.set_trace()
    isRandom = True
    isIter = True
    transform_num = 3
    contrast_batch = Contrastive_Transform(X.shape[2])

    if isRandom and isIter:


        eps = get_transAug_param(aug_name[0])

        step_size = 1.15 * ((eps[1] - eps[0]) / 2) / attack_iters

        # delta = torch.zeros_like(X[0]).cuda()
        # delta.uniform_(-8/255, 8/255)
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

        # param = torch.rand((X.shape[0])) * (eps[1] - eps[0]) + eps[0]
        param = torch.rand(1) * (eps[1] - eps[0]) + eps[0]
        # delta = trans_aug(aug_name, X, param) - X 
        param.requires_grad = True
        delta.requires_grad = True

        before_loss = SslTrainer.compute_ssl_contrastive_loss(contrast_batch(X, transform_num), criterion, model, model_ssl, X.shape[0], transform_num, no_grad=True)[0]
        for _ in range(attack_iters):

            param_all = param.repeat(X.size(0))    
            new_x = trans_aug(aug_name[0], X, param_all)  + delta
            loss = -SslTrainer.compute_ssl_contrastive_loss(contrast_batch(new_x, transform_num), criterion, model, model_ssl, X.shape[0], transform_num, no_grad=False)[0]


            loss.backward()
            param_grad = param.grad.detach()
            delta_grad = delta.grad.detach()

            p = param
            g = param_grad

            d = delta
            g2 = delta_grad

            x = X
            
            p = torch.clamp(p + torch.sign(g) * step_size, eps[0], eps[1])

            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g2), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g2.view(g2.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                scaled_g = g2 / (g_norm + 1e-10)
                d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
            elif norm == "l_1":
                g_norm = torch.sum(torch.abs(g2.view(g2.shape[0], -1)), dim=1).view(-1, 1, 1, 1)
                scaled_g = g2 / (g_norm + 1e-10)
                d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=1, dim=0, maxnorm=epsilon).view_as(d)
            
            
            param.data = p
            param.grad.zero_()
            
            delta.data = d
            delta.grad.zero_()

            #print('update param: {}'.format(param))
        final_loss = -1 * loss.item()
        max_param = param.detach()
        max_delta = delta.detach()
        #param = param.detach()

        # loss = loss.item()
        #param = param.detach()
            
        return max_delta, max_param, before_loss.item(), final_loss

    elif isRandom and isIter==False:
        
        if attack_iters==0:
            best_loss = 2000
            augloss_list = []

            eps_list = [get_transAug_param(aug) for aug in aug_name]
            param_list = [ (torch.arange(40)/40) * (eps_list[i][1] - eps_list[i][0]) + eps_list[i][0] for i in range(len(eps_list))]
            
            orig_loss = SslTrainer.compute_ssl_contrastive_loss(contrast_batch(X, transform_num), criterion, model, model_ssl, X.shape[0], transform_num, no_grad=False)[0]
            print('orig. loss: {}'.format(orig_loss))

            for i in range(len(param_list)):
                for p in param_list[i]:
                    new_x = trans_aug(aug_name[i],  X , p) 
                    loss = SslTrainer.compute_ssl_contrastive_loss(contrast_batch(new_x, transform_num), criterion, model, model_ssl, X.shape[0], transform_num, no_grad=False)[0]
                    if loss.item() < best_loss:
                        best_aug = aug_name[i]
                        best_loss = loss.item()
                        best_param = p
            print('reverse loss: {} best: {} {}'.format(best_loss, best_aug, best_param))
                
            return [best_aug], [best_param]
        else:

            eps_list = [get_transAug_param(aug) for aug in aug_name]
            step_size_list = [1.15 * ((eps[1] - eps[0]) / 2) / attack_iters for eps in eps_list]
            
            param1 = torch.rand((X.shape[0])) * (eps_list[0][1] - eps_list[0][0]) + eps_list[0][0]
            # param2 = torch.rand((X.shape[0])) * (eps_list[1][1] - eps_list[1][0]) + eps_list[1][0]
            # param3 = torch.rand((X.shape[0])) * (eps_list[1][1] - eps_list[2][0]) + eps_list[2][0]

            param1.requires_grad = True
            # param2.requires_grad = True
            # param3.requires_grad = True

            for _ in range(attack_iters):
                
                new_x = trans_aug_list(aug_name, X, [param1])

                loss = -SslTrainer.compute_ssl_contrastive_loss(contrast_batch(new_x, transform_num), criterion, model, model_ssl, X.shape[0], transform_num, no_grad=False)[0]


                loss.backward()
                param1_grad = param1.grad.detach() 
                # param2_grad = param2.grad.detach() 
                # param3_grad = param3.grad.detach() 

                p1 = param1
                # p2 = param2
                # p3 = param3

                g1 = param1_grad
                # g2 = param2_grad
                # g3 = param3_grad

                x = X

                # for i in range(len(param_list)):
                p1 = torch.clamp(p1 + torch.sign(g1) * step_size_list[0], eps_list[0][0], eps_list[0][1]) 
                # p2 = torch.clamp(p2 + torch.sign(g2) * step_size_list[1], eps_list[1][0], eps_list[1][1]) 
                # p3 = torch.clamp(p3 + torch.sign(g3) * step_size_list[2], eps_list[2][0], eps_list[2][1]) 
                
                param1.data = p1
                param1.grad.zero_()

                # param2.data = p2
                # param2.grad.zero_()

                
                # param3.data = p3
                # param3.grad.zero_()

                #print('update param: {}'.format(param))
            max_param_list = [param1.detach()]
            # , param2.detach()]
            #param = param.detach()
            
            return aug_name, max_param_list


def compute_reverse_transformation_dist(model, model_ssl, criterion, X, attack_iters, aug_name):

    # import pdb; pdb.set_trace()
    torch.cuda.empty_cache()
    isRandom = False
    isIter = True
    if isRandom==False and isIter:

        if attack_iters==0:
            best_sloss = 2000
            augloss_list = []
          
            for aug in aug_name:
                dloss_list = []
                sloss_list = []
                jloss_list = []

                eps = get_transAug_param(aug)
                param = (torch.arange(40)/40) * (eps[1] - eps[0]) + eps[0]

                for p in param:
                    new_x = trans_aug(aug,  X , p)
                    real_label = torch.full((new_x.shape[0],), 0, device=device,  dtype=torch.int64)
                    l2 = SslTrainer.compute_ssl_dist_loss(new_x, real_label, criterion, model, model_ssl, no_grad=True)[0]
                    jloss_item = l2.item()
                    if jloss_item < best_sloss:
                        best_aug = aug
                        best_sloss = jloss_item
                        best_param = p

                    print('best: {} {}'.format(best_aug, best_param))
                    aug_newX = trans_aug(best_aug, X, best_param)
                    rot_loss, pred, target = SslTrainer.compute_ssl_dist_loss(aug_newX, torch.full((X.shape[0],), 0, device=device,  dtype=torch.int64), criterion, model, model_ssl, no_grad=True)
                    matches = (pred.max(1)[1] == target).sum().item()
               
                    return best_aug, best_param
                    


def test_acc(model, test_batches):
    acc = 0
    for batch in test_batches:
        x, y = batch['input'], batch['target']
        out, _ = model(x)
        acc += (out.max(1)[1] == y).sum().item()
    print('Accuracy before SSL training: {}'.format(acc / (100 * len(test_batches))))


def test_acc_reverse_vector(model, model_ssl, test_batches, criterion, attack_iters, aug_name):

    acc = 0
    epsilon = (8 / 255.)
    pgd_alpha = (2 / 255.)
    test_n = 0
    clean_acc = 0
    
    before_loss_list = []
    final_loss_list = []

    for i, batch in enumerate(test_batches):
        x, y = batch['input'], batch['target']
        test_n += y.shape[0]


        clean_out, _ = model(x)
        clean_acc += (clean_out.max(1)[1] == y).sum().item()

        if aug_name is None:
            delta = compute_reverse_attack(model, model_ssl, criterion, x,
                                                    epsilon, pgd_alpha, attack_iters, 'l_2')
            out, _ = model(x + delta)
        else: 
            delta, param, before_loss, final_loss = compute_reverse_transformation(model, model_ssl, criterion, x,
                                                    epsilon, pgd_alpha, attack_iters, 'l_2', aug_name)
            param_all = param.repeat(x.shape[0])
            # print(delta.shape)
            # print(param)
            new_x = trans_aug(aug_name[0], x, param_all)  + delta
            out, _ = model(new_x)
            
            before_loss_list.append(before_loss)
            final_loss_list.append(final_loss)

        acc += (out.max(1)[1] == y).sum().item()
        print("test number: {} before reverse: {} after reverse: {}".format(test_n, clean_acc/test_n, acc/test_n))
    print('Accuracy after SSL training: {}'.format(acc / test_n))

    with open('./loss_histogram/cifar10c_orig.npy', 'wb') as f:
        np.save(f, np.array([before_loss_list, final_loss_list]))
    return clean_acc/test_n, acc/test_n

def test_dvalue(gan_D, test_batches):

    transform = torch.nn.Sequential(transforms.Resize(size=64),)
    clean_dlist = []
    for i, batch in enumerate(test_batches):
        x, y = batch['input'], batch['target']
        with torch.no_grad():
            d_outputs = gan_D(transform(x)).view(-1)   
    
    print('d list avg. : {}'.format(torch.mean(d_outputs)))
    return torch.mean(d_outputs)

def red(text):
    return '\033[31m' + text +'\033[0m'

def green(text):
    return '\033[32m' + text + '\033[0m'

def test_acc_reverse_vector_aug(model, dist_head, test_batches, rot_criterion, attack_iters, aug_name, scripted_transforms):
    reverse_acc, clean_acc = 0, 0
    epsilon = (8 / 255.)
    pgd_alpha = (2 / 255.)
    test_n = 0
    loss_list = []
    loss_list1 = []
    reverse_loss_list = []
    before_loss_list = []
    reverse_img_list = []
    reverse_img_list1 = []
    orig_success_reverse_success = 0
    orig_fail_reverse_success = 0
    orig_success_reverse_fail = 0
    orig_fail_reverse_fail = 0
    matches = 0
    reverse_match = 0
    dloss_fail = []
    dloss_success = []

    if aug_name=='mix':
        aug_list = ['sharpness', 'saturation']
    else:
        aug_list = [aug_name]

    for i, batch in enumerate(test_batches):
        idx = i
        x, y = batch['input'], batch['target']
        
        #rot_tranform = Rotate_Batch()
        dist_batch =  Dist_Batch()
        # dist_transform = DistTransform_Test()

        # x_transformed, angles = rot_tranform(x) 
        x_dist, labels = dist_batch(x, 'corrupt')
        # x_dist = dist_transform(x)
        # print(x_dist.shape)
        # xr = np.array(torch.squeeze(x_transformed[0], 0).detach().cpu().numpy()*255., dtype=np.uint8)
        # xr = np.transpose(xr, (1,2,0))
 
        
        test_n += y.shape[0]

                
   
        real_label = torch.full((x_dist.shape[0],), 0, device=device,  dtype=torch.int64)
        before_loss, pred, target = SslTrainer.compute_ssl_dist_loss(x, torch.full((x.shape[0],), 0, device=device,  dtype=torch.int64), rot_criterion, model, dist_head, no_grad=True)
        matches += (pred.max(1)[1] == target).sum().item()
        print('before reverse match: {} / {}'.format(matches, test_n))
        new_X, rematch, reverse_loss, best_aug, best_param = compute_reverse_transformation_dist(model, dist_head, rot_criterion, x_dist, attack_iters, aug_list)
        before_loss_list.append(before_loss.item())
        reverse_loss_list.append(reverse_loss)
        reverse_match += rematch
        print('after reverse match: {} / {}'.format(reverse_match, test_n))
        print('before loss: {} reverse loss: {}'.format(before_loss, reverse_loss))
        
                    
        clean_out, fc_out = model(x)
        reverse_out, fc_out = model(new_X)

        reverse_acc += (reverse_out.max(1)[1] == y).sum().item()
        clean_acc += (clean_out.max(1)[1] == y).sum().item()
        print('before reverse acc: {} / {}'.format(clean_acc, test_n))
        print('after reverse acc {} / {}'.format(reverse_acc, test_n))
        x = np.array(torch.squeeze(x, 0).detach().cpu().numpy()*255., dtype=np.uint8)
        rx = np.array(torch.squeeze(new_X, 0).detach().cpu().numpy()*255., dtype=np.uint8)
        # with open('pixelate_test_compare.npy', 'wb') as f:
        #       np.save(f, [idx, x, rx, aug_method, aug_param]) 
        # print('y: ', y)

        # if test_n % 1000 == 0:
        #     print("Accuracy after batch %d: %.4f" % (i,  100 * (acc / test_n)))
        #     print("Clean accuracy after batch %d: %.4f" % (i, 100 * (clean_acc / test_n)))

    return 100*(reverse_acc / test_n), 100*(clean_acc / test_n), reverse_loss_list, before_loss_list



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=1024, type=int)
    parser.add_argument('--test_batch', default=200, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--save_freq', default=1, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--corr-dir', default='./data/CIFAR-10-C-customize/', type=str)
    parser.add_argument('--fname', default='train_ssl', type=str)
    parser.add_argument('--save_root_path', default='data/ckpts/', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--md_path', default='./cifar10_standard.pth', type=str)
    parser.add_argument('--ckpt2', default='./data/ckpts/cifar10c_3/ssl_contrast_199.pth', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--eval', action='store_true')

    # parser.add_argument('--aug_name', default=None, type=str)
    parser.add_argument('--aug_name', nargs='+', type=str)
    parser.add_argument('--corruption', default='all', type=str)
    parser.add_argument('--severity', default=1, type=int)
    parser.add_argument('--attack_iters', default=1, type=int)

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


    return parser.parse_args()

def confusion_matrix(predscore, target):
    l = len(predscore)
    fpr_list = []
    tpr_list = []
    threshold_list =  [0.1, 0.3, 0.5, 0.7, 0.9]
    # print(l)
    # print(len(drift_pos))
    for threshold in threshold_list:
        t_p = sum([1 if predscore[i] > threshold and target[i] == 1 else 0 for i in range(0,l)]) #model said it's fake and actually it's fake
        f_p = sum([1 if predscore[i] < threshold and target[i] == 0 else 0 for i in range(0,l)]) #model said it's fake and actually it's real 
        f_n = sum([1 if predscore[i] < threshold and target[i] == 1 else 0 for i in range(0,l)]) #model said it's real and actually it's fake
        t_n = sum([1 if predscore[i] > threshold and target[i] == 0 else 0 for i in range(0,l)]) #model said it's real and actually it's real
        FPR = f_p if f_p + t_n == 0 else f_p / (f_p + t_n)
        TPR = t_p if t_p + f_n == 0 else t_p / (t_p + f_n)
        tpr_list.append(TPR)
        fpr_list.append(FPR)
        print('threshold: ', threshold, 'TPR: ', TPR, 'FPR: ', FPR, 'tp: ', t_p, 'fp: ', f_p, 'fn: ', f_n, 'tn: ', t_n)
    return tpr_list, fpr_list

class SslTrainer:
    def __init__(self):
        self.rot_transform = RotationTransform()
        self.dist_transform = DistTransform_Train()
        self.contrast_transform = Contrastive_Transform(224)

    def train_one_epoch(self, model, dist_head, train_batches, opt, lr_schedule_values, start_steps, num_training_steps_per_epoch, aug_transforms, criterion, debug=False):
        
        train_loss, train_n, matches = 0.0, 0, 0.0
        update_freq = 1

        for data_iter_step, batch in enumerate(tqdm(train_batches)):

            target = batch['target']

            step = data_iter_step // update_freq

            if step >= num_training_steps_per_epoch:
                continue
            it = start_steps + step  # global training iteration
            # Update LR & WD for the first acc
            if lr_schedule_values is not None and data_iter_step % update_freq == 0:
                for i, param_group in enumerate(opt.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] 

            batch_train_loss, batch_matches, batch_size = self.step(model, dist_head, batch['input'], target, opt, criterion)
            train_loss += batch_train_loss * batch_size
            matches += batch_matches
            train_n += batch_size

            if debug:
                break
        return train_loss, train_n, matches

    def step(self, model, contrast_head, x, target, opt, criterion):
        #x_rotated, angles = self.rotate_batch_input(x)

        transforms_num = 3
        x_const = self.contrast_transform(x, transforms_num)

        # rot_loss, pred, target = self.compute_ssl_rot_loss(x_rotated, angles, criterion, model, rot_head, no_grad=False)
        # matches = self.test_rot(pred, target)

        c_loss, correct = self.compute_ssl_contrastive_loss(x_const, criterion, model, contrast_head, x.shape[0], transforms_num)
        # dist_loss, pred, target = self.compute_ssl_dist_loss(x_dist, labels, criterion, model, dist_head, no_grad=False)
        # matches = self.test_(pred, target)
        opt.zero_grad()
        c_loss.backward()
        opt.step()
     
        return c_loss.item(), correct, x_const.shape[0]

    def rotate_batch_input(self, batch_input):
        x_rotated, angles = list(zip(*[self.rot_transform(sample_x) for sample_x in batch_input]))
        x_rotated = [x.unsqueeze(0) for x in x_rotated]
        x_rotated = torch.cat(x_rotated)
        return x_rotated, angles

    def dist_batch_input(self, batch_input):
        x_dist, labels = list(zip(*[self.dist_transform(sample_x) for sample_x in batch_input]))
        x_dist = [x.unsqueeze(0) for x in x_dist]
        x_dist = torch.cat(x_dist)
        return x_dist, labels

    def test_rot_head(self, criterion, model, rot_head, batches):
        matches = 0
        test_n = 0
        loss = 0
        rot_tranform = Rotate_Batch()
        for b in tqdm(batches):
            batch_size = b['target'].size(0)
            x = b['input']
            x_rotated, angles = rot_tranform(x)
            rot_loss, pred, target = self.compute_ssl_rot_loss(x_rotated, angles, criterion, model, rot_head, no_grad=True)
            matches += (pred.max(1)[1] == target).sum().item()
            test_n += batch_size
            loss += rot_loss*batch_size
            print(matches)
        return matches, loss, test_n


    @staticmethod
    def compute_ssl_contrastive_loss(x, criterion, model, contrast_head, bs, transform_num, no_grad=True):
        if no_grad:
            with torch.no_grad():
                _, out = model(x)
        else:
            _, out = model(x)

        #fc_output, output = contrast_head(out)
        output = contrast_head(out)
     
        c_loss, correct = SslTrainer.constrastive_loss_func(output, criterion, bs, transform_num+1) #if using contrastive loss
        #m_loss = SslTrainer.xent_loss_func(output, criterion, bs, transform_num+1)
        #s_loss = SslTrainer.softmax_entropy_loss_func(fc_output).mean(0)
        
        #mix_loss = s_loss + c_loss
        return c_loss, correct
        #return mix_loss
        
    @staticmethod
    def compute_ssl_rot_loss(x, angles, criterion, model, rot_head, no_grad=False):
        if no_grad:
            with torch.no_grad():
                out, fc_out = model(x)
                pred = rot_head(fc_out)
        else:
            out, fc_out = model(x)
            pred = rot_head(fc_out)

        #softmax_ent = -(out.softmax(1) * out.log_softmax(1)).sum(1)

        # mean_softmax_ent = torch.mean(softmax_ent)
        target = torch.tensor([angle / 90 for angle in angles], dtype=torch.int64).cuda()
        return criterion(pred, target), pred, target

    @staticmethod
    def compute_ssl_dist_loss(x, label, criterion, model, dist_head, no_grad=True):
        if no_grad:
            with torch.no_grad():
                out, fc_out = model(x)
                pred = dist_head(fc_out)
      
        else:
            out, fc_out = model(x)
            pred = dist_head(fc_out)

        # if label==1:
        #     target = torch.full((x.shape[0],), 0, device=device, dtype=torch.int64).cuda()
        # else:
        #     target = torch.full((x.shape[0],), 1, device=device, dtype=torch.int64).cuda()
        target = torch.tensor([l for l in label], dtype=torch.int64).cuda()

        return criterion(pred, target), pred, target

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

    def test_dist_head(self, criterion, model, dist_head, batches, x_type):
        matches = 0
        matches_real=0
        matches_fake=0
        test_n = 0
        loss = 0
        loss_list = []
        predscore_list = []
        target_list = []
        dist_tranform = Dist_Batch()
        for b in tqdm(batches):
            batch_size = b['target'].size(0)
            x = b['input']
            # if x.shape[0] == 1:
            #     x = x.repeat(4,1,1,1)
            #     print(x.shape)
            if x_type == 'orig' or x_type=='corrupt' or x_type=='none': 
                dist_x, labels = dist_tranform(x, x_type)
            elif x_type == 'mixed':
                dist_x = x
                labels = b['target']

            
            rot_loss, pred, target = self.compute_ssl_dist_loss(dist_x, labels, criterion, model, dist_head, no_grad=True)
            matches += (pred.max(1)[1] == target).sum().item()
            for i in range(len(target)): 
                if target[i] == 0 and pred.max(1)[1][i] == target[i]:
                    matches_real += 1
                    # print('target: {} pred score: {}'.format(target[i], pred_score))
                elif target[i] == 1 and pred.max(1)[1][i] == target[i]:
                    matches_fake += 1
                    # print('target: {} pred score: {}'.format(target[i], pred_score))
                predscore_list.append(pred.softmax(1)[i][1].item())
                target_list.append(target[i].item())
            test_n += batch_size
            loss += rot_loss*batch_size
            loss_list.append(rot_loss.item())
        # lr_auc = roc_auc_score(target_list, predscore_list)
        # print('ROC AUC=%.3f' % (lr_auc))
        # lr_fpr, lr_tpr, _ = roc_curve(target_list, predscore_list)
        # tpr_list, fpr_list = confusion_matrix(predscore_list, target_list)

        return matches, matches_real, matches_fake, loss, test_n, loss_list, target_list, predscore_list


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
            # lr_auc = roc_auc_score(target_list, predscore_list)
            # print('ROC AUC=%.3f' % (lr_auc))
            # lr_fpr, lr_tpr, _ = roc_curve(target_list, predscore_list)
            # tpr_list, fpr_list = confusion_matrix(predscore_list, target_list)

            return test_loss, matches, test_n 


    @staticmethod
    def test_(pred, target):
        with torch.no_grad():
            matches = (pred.max(1)[1] == target).sum().item()
        return matches

    # @staticmethod
    # def save_state_dict(rot_head, opt, epoch):
    #     state_dict = {
    #         'epoch': epoch,
    #         'rot_head_state_dict': rot_head.state_dict(),
    #         'optimizer_state_dict': opt.state_dict()
    #     }
    #     torch.save(state_dict, f'data/ckpts/cifar10c/ssl_rot_{epoch}.pth')

    @staticmethod
    def save_state_dict(dist_head, opt, epoch, isbest):
        state_dict = {
            'epoch': epoch,
            'contrast_head_state_dict': dist_head.state_dict(),
            'optimizer_state_dict': opt.state_dict()
        }
        if isbest:
            torch.save(state_dict, f'data/ckpts/cifar10c_4/ssl_contrast_best.pth')
        else:
            torch.save(state_dict, f'data/ckpts/cifar10c_4/ssl_contrast_{epoch}.pth')

def main():
    args = get_args()
    import uuid
    import datetime
    unique_str = str(uuid.uuid4())[:8]
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

    args.fname = os.path.join(args.save_root_path, args.fname, timestamp + unique_str)
    if not os.path.exists(args.fname):
        os.makedirs(args.fname)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    transforms = [Crop(32, 32)]
    dataset = cifar10(args.data_dir)

    corruption_type = ['gaussian_noise', 'shot_noise','impulse_noise',
                      'defocus_blur', 'motion_blur' , 'glass_blur','zoom_blur','snow', 
                      'frost', 'fog', 'brightness', 'contrast', 'elastic_transform',
                      'pixelate', 'jpeg_compression', 'orig']
    
    train_set = list(zip(transpose(pad(dataset['train']['data'], 4) / 255.),
                         dataset['train']['labels']))
    train_set_x = Transform(train_set, transforms)
    train_batches = Batches(train_set_x, args.batch_size, shuffle=True, set_random_choices=True, num_workers=4)

    test_set = list(zip(transpose(dataset['test']['data'] / 255.), dataset['test']['labels']))
    print(len(test_set))
    test_batches = Batches(test_set, args.test_batch, shuffle=False, num_workers=2)


    # mixup_fn = None
    # mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    # if mixup_active:
    #     print("Mixup is activated!")
    #     mixup_fn = Mixup(
    #         mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
    #         prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
    #         label_smoothing=0.0, num_classes=10)


    #from learning.unlabel_WRN import WideResNet_2
    from learning.wideresnet import WideResNet
    from learning.ssl import Ssl_model_contrast
    #model = WideResNet_2(depth=28, widen_factor=10)
    model = WideResNet(depth=28, num_classes=10, widen_factor=10)
    dist_head = Ssl_model_contrast(640)

    #model = nn.DataParallel(model).cuda()
    # rot_head = nn.DataParallel(rot_head).cuda()
    # rot_head.train()
    # dist_head = nn.DataParallel(dist_head).cuda()
    dist_head.eval()
    decay, no_decay = [], []
    for name, param in dist_head.named_parameters():
        if 'bn' not in name and 'bias' not in name:
            decay.append(param)
        else:
            no_decay.append(param)

    params = [{'params': decay, 'weight_decay': 0},
              {'params': no_decay, 'weight_decay': 0}]

    learning_rate = args.lr
    opt = torch.optim.Adam(params, lr=learning_rate)

    num_training_steps_per_epoch = len(train_set) // args.batch_size
    print("Use step level LR scheduler!")
    lr_schedule_values = cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    if args.md_path:
        if os.path.isfile(args.md_path):
            #checkpoint = torch.load(args.md_path)['state_dict']
            checkpoint = torch.load(args.md_path)['state_dict']
            model.load_state_dict(checkpoint)  #['state_dict']
            #model.load_state_dict(checkpoint)
            print("=> load chechpoint found at {}".format(args.md_path))
        else:
            print("=> no checkpoint found at '{}'".format(args.md_path))
    model = model.eval().cuda() 

    restart_epoch = 0
    if os.path.exists(args.ckpt2):
        print("Loading checkpoint: {}".format(args.ckpt2))
        ckpt2 = torch.load(args.ckpt2)
        dist_head.load_state_dict(ckpt2['contrast_head_state_dict'])
        opt.load_state_dict(ckpt2['optimizer_state_dict'])
        restart_epoch = ckpt2['epoch'] + 1
    dist_head.eval().cuda()

    # defines transformation for SSL contrastive learning.
    s = 1
    size = 32
    from torchvision.transforms import transforms
    transforms = torch.nn.Sequential(
        transforms.RandomResizedCrop(size=size),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s),
        # transforms.RandomGrayscale(p=0.2),
    )
    scripted_transforms = torch.jit.script(transforms)
    rot_transform = RotationTransform()
    #rot_transform = RotAug()

    criterion = torch.nn.CrossEntropyLoss().cuda()
    #gan_criterion = torch.nn.BCELoss().cuda()


    if args.eval:
        # test_acc(model, test_batches)
        # test_acc(model, test_batches_ood)
        ALL_robust_acc = []
        ALL_clean_acc = []
        all_tp_list = []
        attack_iters = [args.attack_iters]
        severity= args.severity
        # test_acc_reverse_vector(model, rot_head, test_batches, rot_transform, rot_criterion)
        if args.corruption == 'all':
            c_idx = np.arange(15)
        else:
            c_idx = [np.array(corruption_type.index(args.corruption))]
            
        for i in c_idx:
            #x_test, y_test = load_cifar10c(1000, severity, './data', False, corruption_type[i:i+1])
            orig_x_test = np.load('./data/' + 'original.npy')
            if i < 15:
                x_test = np.load(args.corr_dir + str(corruption_type[i])+'.npy')[severity-1]
            else:
                x_test = np.load('./data/' + 'original.npy')

            orig_test_set = list(zip(transpose(orig_x_test / 255.), dataset['test']['labels']))
            test_set = list(zip(transpose(x_test/ 255.), dataset['test']['labels']))

            
            test_batches_orig = Batches(orig_test_set, args.test_batch, shuffle=False, num_workers=2)
            test_batches_ood = Batches(test_set, args.test_batch, shuffle=False, num_workers=2)

            trainer = SslTrainer()
            # orig_test_matches, rot_loss1, test_n1 = trainer.test_rot_head(rot_criterion, model, rot_head, test_batches_orig)        
            # corr_test_matches, rot_loss2, test_n2 = trainer.test_rot_head(rot_criterion, model, rot_head, test_batches_ood)
            # print('ori matching acc.: ' , orig_test_matches/test_n1, 'rot loss: ', rot_loss1/test_n1)
            # print('corr matching acc.: ' , corr_test_matches/test_n2, 'rot loss: ',rot_loss2/test_n2)

            # orig_test_matches, real, fake, rot_loss1, test_n, orig_loss_list = trainer.test_dist_head(criterion, model, dist_head, train_batches, 'none')                  
            # print('matching acc.: ' , orig_test_matches/test_n, 'real match: ', real/test_n, 'synt. match: ', fake/test_n)
            # orig_test_matches, real, fake, rot_loss1, test_n1, orig_loss_list, target_list, predscore_list = trainer.test_dist_head(criterion, model, dist_head, mixed_batches, 'mixed')
            # all_tp_list.append(list(zip(target_list, predscore_list)))

            
            # test_loss, matches, test_n  = trainer.test_contrast_head(criterion, model, dist_head, test_batches_orig)
            # print('orig test sample: ', test_n, 'ori matching acc.: ' , matches/ test_n, 'contrastive loss: ', test_loss/test_n)

                        
            # test_loss, matches, test_n  = trainer.test_contrast_head(criterion, model, dist_head, test_batches_ood)
            # print('ood test sample: ', test_n, 'ori matching acc.: ' , matches/ test_n, 'contrastive loss: ', test_loss/test_n)
            
            

            robust_acc = []
            clean_acc = []
            for attack_iter in attack_iters:
                print('aug name: ', args.aug_name)
                print('corruption type: ',corruption_type[i:i+1][0])
                acc1, acc2 = test_acc_reverse_vector(model, dist_head, test_batches_orig, criterion, attack_iter, args.aug_name)
                # acc1, acc2, reverse_loss_list, corr_loss_list = test_acc_reverse_vector_aug(model, dist_head, gan_D, test_batches_ood, rot_transform, criterion, gan_criterion, 0, rm, args.aug_name, scripted_transforms)
                    #print(reverse_loss)
                print("Reverse with cross, acc before reversed: {} acc after reversed: {} ".format(acc1, acc2))
                # stat3, pvalue3 = stats.ks_2samp(orig_loss, reverse_loss) #amazon --> 
                # print("ks: statistic: {}, pvalue: {}".format(stat3, pvalue3))
                # print("------------------------------")
                with open('./log/cifar10c_new_result_contrastive_reverse_l2sharp.csv', 'a') as f: 
                        writer = csv.writer(f)
                        writer.writerow(['l_2 + ', args.aug_name, ' reverse_iter: ', args.attack_iters, ' corruption: ', corruption_type[i:i+1], ' severity: '+ str(args.severity), 'batch-size: '+str(args.test_batch), acc1, acc2])
                        # writer.writerow(['aug_name : ', args.aug_name, ' (iter: ', args.attack_iters, ') corruption: ', corruption_type[i:i+1], ' severity: '+ str(args.severity), 'batch-size: '+str(args.test_batch), acc1, acc2])
                # with open('loss_histogram/pixelate_v2_loss_hitogram.npy', 'wb') as f:
                #      np.save(f, list(zip(orig_loss_list, corr_loss_list,  reverse_loss_list))) 
            # ALL_robust_acc.append(robust_acc)
            # ALL_clean_acc.append(clean_acc)
            
            #visualization(ALL_robust_acc, ALL_clean_acc)
            # with open('./log/cifar10c_trans_mincontrast.csv', 'a') as f: 
            #     writer = csv.writer(f)
            #     writer.writerow([min_contrast])
        # with open('loss_histogram/cifar10_1_mixed_all_tpr_fpr_s5.npy', 'wb') as f:
        #     np.save(f, all_tp_list)     
            
    else:
        trainer = SslTrainer()
        dist_head.train().cuda()
        with open('./log/cifar10c_3_contrast_trainlog.csv', 'a') as f: 
            writer = csv.writer(f)
            writer.writerow(['train epoch', 'train loss', 'train match acc.', 'test loss', 'test match acc.', 'test match real acc.', 'test match fake acc.'])
        
        for epoch in range(restart_epoch, args.epochs):
            print("Epoch number: {}".format(epoch))
            start_step = epoch*num_training_steps_per_epoch
            train_loss, train_n, train_matches = trainer.train_one_epoch(model, dist_head, train_batches, 
                                                                         opt, lr_schedule_values, start_step, num_training_steps_per_epoch,
                                                                         scripted_transforms, criterion, args.debug)
            
            print('Epoch: %d, Train accuracy: %.4f, Train loss:  %.4f' % (epoch, train_matches / train_n, train_loss / train_n))
            # test_matches, real, fake, test_loss, test_n, orig_loss_list, target_list, predscore_list = trainer.test_dist_head(criterion, model, dist_head, test_batches, 'none')
            # print('Test matching acc.: ' , test_matches/test_n, 'contastive loss: ', test_loss.item()/test_n)
            # print('match real: ', real/test_n, 'match fake: ', fake/test_n)
            if train_matches / train_n > best_matches:
                trainer.save_state_dict(dist_head, opt, epoch, isbest=True)

            if epoch % args.save_freq == 0 and epoch > 0:
                trainer.save_state_dict(dist_head, opt, epoch, isbest=False)
                        
            with open('./log/cifar10c_3_contrast_trainlog.csv', 'a') as f: 
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss / train_n, train_matches / train_n])
               


if __name__ == "__main__":
    main()
