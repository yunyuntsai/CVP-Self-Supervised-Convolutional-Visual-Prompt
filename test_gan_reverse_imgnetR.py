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

import glob
import cv2
from sklearn.random_projection import johnson_lindenstrauss_min_dim
import torch
from tqdm import tqdm
import torch.nn as nn

from cifar_10_1_dataset import load_new_test_data
from learning.wideresnet import WRN34_rot_out_branch,WRN34_rot_out_branch2
from utils import *
from data_utils import *
import torchvision.transforms as transforms
from RandAugment import trans_aug, get_transAug_param
from robustbench.utils import load_model
from robustbench.data import load_cifar10c
import kornia
from scipy import stats
mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot
from robustbench.data import load_imagenetc
upper_limit, lower_limit = 1, 0

imgnet_mean=(0.485, 0.456, 0.406)
imgnet_std=(0.229, 0.224, 0.225)
inv_transforms = torch.nn.Sequential(
            transforms.Normalize(mean=[ 0., 0., 0. ], std=[ 1/0.229, 1/0.224, 1/0.225 ]),
            transforms.Normalize(mean=[ -0.485, -0.456, -0.406 ], std=[ 1., 1., 1.] )
)
transforms = torch.nn.Sequential(
    transforms.Normalize(imgnet_mean, imgnet_std)
)
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


def compute_universal_reverse_attack(model, model_ssl, criterion, X, angles, epsilon, alpha, attack_iters, norm):
    delta = torch.unsqueeze(torch.zeros_like(X[0]).cuda(), 0)
    # if norm == "l_inf":
    #     delta.uniform_(-epsilon, epsilon)
    # elif norm == "l_2":
    #     delta.normal_()
    #     d_flat = delta.view(delta.size(0), -1)
    #     n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
    #     r = torch.zeros_like(n).uniform_(0, 1)
    #     delta *= r / n * epsilon
    # elif norm == 'l_1':
    #     pass
    # else:
    #     raise ValueError

    delta.requires_grad = True
    for _ in range(attack_iters):
        delta_all = delta.repeat(X.size(0), 1, 1, 1)
        new_x = X + delta_all

        loss = -SslTrainer.compute_ssl_rot_loss(new_x, angles, criterion, model, model_ssl, no_grad=False)[0]
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
        print(X.shape)

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

        #d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data = d
        delta.grad.zero_()
    max_delta = delta.detach()
    return max_delta

     
def compute_reverse_transformation(model, model_ssl, criterion, X, epsilon, alpha, attack_iters, norm, aug_name, denormalize, normalize):

    """Reverse algorithm that optimize the SSL loss via PGD"""
    # import pdb; pdb.set_trace()
    isRandom = True
    isIter = True
    transform_num = 3
    contrast_batch = Contrastive_Transform(X.shape[2])

    if isRandom and isIter:


        eps = get_transAug_param(aug_name)

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
            new_x = normalize(trans_aug(aug_name, denormalize(X), param_all))  + delta
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



def test_acc(model, test_batches, normalize, imagenet_r_mask):
    acc = 0
    test_size  = 0
    for batch in test_batches:
        x, y = batch['input'], batch['target']
        test_size += batch['target'].size(0)
        with torch.no_grad():
            out, _ = model(x)
            out = out[:,imagenet_r_mask]
            acc += (out.max(1)[1] == y).sum().item()
        print('test size: {} test acc: {}'.format(test_size, acc / test_size))
    return acc / test_size


def test_acc_reverse_vector(model, model_ssl, test_batches, criterion, attack_iters, aug_name, normalize, denormalize):
    epsilon = (8 / 255.)
    pgd_alpha = (4 / 255.)
    test_n = 0
    clean_acc = 0
    acc = 0
    size = 128
    before_loss_list = []
    final_loss_list = []
    imagenet_r_mask = gen_mask()

    # downscaling = DownScale_Transform(size)

    for i, batch in enumerate(test_batches):

        x, y = batch['input'], batch['target']

        # x = downscaling(x)
        test_n += y.shape[0]

        with torch.no_grad():
            clean_out, _ = model(x)

        # clean_out = clean_out[:, imagenet_r_mask]
        clean_acc += (clean_out.max(1)[1] == y).sum().item()

        if aug_name is None:
            delta = compute_reverse_attack(model, model_ssl, criterion, x,
                                                    epsilon, pgd_alpha, attack_iters, 'l_inf')
            with torch.no_grad():                                       
                out, _ = model(x + delta)
                # out = out[:, imagenet_r_mask]
        else: 
            delta, param, before_loss, final_loss = compute_reverse_transformation(model, model_ssl, criterion, x,
                                                    epsilon, pgd_alpha, attack_iters, 'l_2', aug_name,  denormalize, normalize)
            param_all = param.repeat(x.shape[0])
            # print(delta.shape)
            # print(param)
            new_x = normalize(trans_aug(aug_name, denormalize(x), param_all)) + delta
            with torch.no_grad():
                out, _ = model(new_x)
                # out = out[:, imagenet_r_mask]

            before_loss_list.append(before_loss)
            final_loss_list.append(final_loss)
            print('before loss: {} final_loss: {}'.format(before_loss, final_loss))

        # out = out[: , :10]
        acc += (out.max(1)[1] == y).sum().item()
        print("test number: {} before reverse: {} after reverse: {}".format(test_n, clean_acc/test_n, acc/test_n))
    print('Accuracy after SSL training: {}'.format(acc / test_n))
    
    # with open('./loss_histogram/imnetc_orig.npy', 'wb') as f:
    #     np.save(f, np.array([before_loss_list, final_loss_list]))
    
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

def test_acc_reverse_vector_aug(model, dist_head,  test_batches, rot_transform, rot_criterion, attack_iters, aug_name, normalize, denormalize, imagenet_r_mask):
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
    reverse_method = ['other']
    # transform = torch.nn.Sequential(transforms.Resize(size=64),)
    if aug_name=='mix':
        aug_list = ['sharpness', 'saturation', 'pixelate']
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
        if reverse_method == 'l_inf' or reverse_method == 'l_2':
            delta = compute_universal_reverse_attack(model, rot_head, rot_criterion, x_transformed, angles,
                                                            epsilon, pgd_alpha, attack_iters, reverse_method)
            out, _ = model(x + delta)
                
        else:
            real_label = torch.full((x.shape[0],), 0, device=device,  dtype=torch.int64)
            before_loss, pred, target = SslTrainer.compute_ssl_dist_loss(x, real_label, rot_criterion, model, dist_head, no_grad=True)
            matches += (pred.max(1)[1] == target).sum().item()
            print('before reverse match: {} / {}'.format(matches, test_n))
            new_X, rematch, reverse_loss, best_aug, best_param = compute_reverse_transformation(model, dist_head, rot_criterion,  x_dist, x, attack_iters, aug_list, normalize, denormalize)
            before_loss_list.append(before_loss.item())
            reverse_loss_list.append(reverse_loss)
            reverse_match += rematch
            print('after reverse match: {} / {}'.format(reverse_match, test_n))
            print('before loss: {} reverse loss: {}'.format(before_loss, reverse_loss))

            if attack_iters==0:
                '''
                with torch.no_grad():
                    orig_d_outputs = gan_D(transform(x)).view(-1)    
                    print('orig d output: {}'.format(orig_d_outputs.item()))
                print('orig. x prediction: {} score: {}'.format(torch.argmax(clean_out.softmax(1)), torch.max(clean_out.softmax(1))))
                # visualize_losslandscape(model, i, x, y, aug_name, loss)
                # new_x = trans_aug(final_param[1], x, final_param[0])
                # reverse_out, _ = model(new_x)
                # reverse_x = np.array(torch.squeeze(new_x, 0).detach().cpu().numpy()*255., dtype=np.uint8)
                # reverse_img = np.transpose(reverse_x, (1,2,0))
                # acc += (reverse_out.max(1)[1] == y).sum().item()
                reverse_out_labels = []
                reverse_out_scores = []
                reverse_dout_list = []
                # reverse_out_labels.append(torch.argmax(clean_out.softmax(1))
                # reverse_out_scores.append(torch.max(clean_out.softmax(1))))
                #reverse_out_min_ent = []
                for j in range(len(param_list)):
                    #print(param_list[j])
                    method_idx, param, d_loss  = param_list[j]
                    new_x = trans_aug(aug_list[int(method_idx)], x, param)
                    tmp_reverse_out, _ = model(new_x)
                    tmp_reverse_dout = gan_D(transform(new_x)).view(-1) 
                    reverse_out_labels.append(torch.argmax(tmp_reverse_out.softmax(1)).item())
                    # reverse_out_scores.append(torch.max(reverse_out.softmax(1).item())
                    reverse_dout_list.append(dout.item())
                    reverse_x = np.array(torch.squeeze(new_x, 0).detach().cpu().numpy()*255., dtype=np.uint8)  
                    #print('reverse. x pred label: {} score: {} d_score: {}'.format(torch.argmax(reverse_out.softmax(1)), torch.max(reverse_out.softmax(1)), tmp_reverse_dout))
                    reverse_img = np.transpose(reverse_x, (1,2,0))
                    # if reverse_out.max(1)[1] == y:
                    #     acc += (reverse_out.max(1)[1] == y).sum().item()  
                    #     print('final choice: {} param: {}'.format(aug_list[j], param_list[j][0]))
                    #     break
                    # else:
                    #     continue
                
                reverse_out_labels=torch.tensor(reverse_out_labels)
                reverse_dout_lists=torch.tensor(reverse_dout_list)
                print('reverse all: ', reverse_out_labels)
                reverse_out = torch.mode(reverse_out_labels,0)[0].item()
                
                print((reverse_out_labels==reverse_out).nonzero())
                print(reverse_dout_lists)
                reverse_d_outputs = torch.mean(reverse_dout_lists[(reverse_out_labels==reverse_out).nonzero()])
                print('majority vote result: ', reverse_out)
                acc += (reverse_out==y).sum().item()
                
                    
                # reverse_out_scores=torch.tensor(reverse_out_scores)
                # reverse_dout_list=torch.tensor(reverse_dout_list)
                '''
            else:
                new_x = trans_aug(aug_name, x, param)
                reverse_out, _ = model(new_x)
                reverse_x = np.array(torch.squeeze(new_x, 0).detach().cpu().numpy()*255., dtype=np.uint8)
                reverse_img = np.transpose(reverse_x, (1,2,0))
                # loss_list.append(loss)
        with torch.no_grad():
            clean_out, fc_out = model(x)
            reverse_out, fc_out = model(new_X)
            clean_out = clean_out[:, imagenet_r_mask]
            reverse_out = reverse_out[:, imagenet_r_mask]

        reverse_acc += (reverse_out.max(1)[1] == y).sum().item()
        clean_acc += (clean_out.max(1)[1] == y).sum().item()
        print('before reverse acc: {} / {}'.format(clean_acc, test_n))
        print('after reverse acc {} / {}'.format(reverse_acc, test_n))
        x = np.array(torch.squeeze(x, 0).detach().cpu().numpy()*255., dtype=np.uint8)
        rx = np.array(torch.squeeze(new_X, 0).detach().cpu().numpy()*255., dtype=np.uint8)
        # with open('pixelate_test_compare.npy', 'wb') as f:
        #       np.save(f, [idx, x, rx, aug_method, aug_param]) 
        # print('y: ', y)
        #clean_acc += (clean_out.max(1)[1] == y).sum().item()
        # clean_acc += (torch.topk(clean_out,1)[1] == y).sum().item()
        # with torch.no_grad():
        #     orig_d_outputs = gan_D(transform(x)).view(-1)    
        #     # print('orig d output: {}'.format(orig_d_outputs.item()))
        #     label = torch.full((x.shape[0],), 1, device=device, dtype = orig_d_outputs.dtype)
        #     orig_d_loss = gan_criterion(orig_d_outputs, label)
        # orig_s_loss = loss
        '''      
        if (torch.topk(clean_out,1)[1] == y).sum().item()==0 and reverse_out.max(1)[1] == y: 
            #loss_list.append([loss,i])
            #reverse_img_list.append([i, reverse_img, final_param, loss, param_list])
            orig_fail_reverse_success +=1
            print("index: ", idx, 'orig: ', red('fail'), 'reverse: ', green('success'))
            # dloss_fail.append(orig_s_loss)
            #save_image(new_x, x, c_type, s_level, param.item(), i, reverse_method, clean_out.max(1)[1].item(), reverse_out.max(1)[1].item(), loss)
        elif (torch.topk(clean_out,1)[1] == y).sum().item()==1 and reverse_out.max(1)[1] == y: 
            #loss_list.append([loss,i])
            #reverse_img_list.append([i, reverse_img, final_param, loss, param_list])
            orig_success_reverse_success +=1
            print("index: ", idx, 'orig: ', green('success'), 'reverse: ', green('success'))
            # dloss_success.append(orig_s_loss)
        elif (torch.topk(clean_out,1)[1] == y).sum().item()==1 and reverse_out.max(1)[1] == y: 
            #loss_list1.append([loss,i])
            #reverse_img_list1.append([i, reverse_img, final_param, loss, param_list])
            orig_success_reverse_fail +=1
            print("index: ", idx, 'orig: ', green('success'), 'reverse: ', red('fail'))
            # dloss_success.append(orig_s_loss)
            #save_image(new_x, x, c_type, s_level, param.item(), i, reverse_method, clean_out.max(1)[1].item(), reverse_out.max(1)[1].item(), loss)
        elif (torch.topk(clean_out,1)[1] == y).sum().item()==0 and reverse_out.max(1)[1] == y:
            ##loss_list1.append([loss,i])
            #reverse_img_list1.append([i, reverse_img, final_param, loss, param_list])
            orig_fail_reverse_fail +=1
            print("index: ", idx,'orig: ', red('fail'), 'reverse: ', red('fail'))
            # dloss_fail.append(orig_s_loss)
       
        print('-------------')
        '''
        # if test_n % 1000 == 0:
        #     print("Accuracy after batch %d: %.4f" % (i,  100 * (acc / test_n)))
        #     print("Clean accuracy after batch %d: %.4f" % (i, 100 * (clean_acc / test_n)))
    # print('Accuracy after SSL training: {}'.format(acc / test_n))
    # print('Clean accuracy after SSL training: {}'.format(clean_acc / test_n))
    # if attack_iters==0:
        # file_path1 = './data/CIFAR-10-C-customize/reversed/reversed_s5_success_gan_joint_' + str(c_type[0]) + '.npy'
        # file_path2 = './data/CIFAR-10-C-customize/reversed/reversed_s5_success_allloss_' + str(c_type[0]) + '.npy'
        # file_path3 = './data/CIFAR-10-C-customize/reversed/reversed_s5_fail_gan_joint_' + str(c_type[0]) + '.npy'
        # file_path4 = './data/CIFAR-10-C-customize/reversed/reversed_s5_fail_allloss_' + str(c_type[0]) + '.npy'
        # with open(file_path1, 'wb') as f:
        #     # print('save corrupted image to : {}'.format(file_path))
        #     np.save(f, np.array(reverse_img_list))
        # with open(file_path2, 'wb') as f:
        #      np.save(f, np.array(dloss_success)) 
        # with open(file_path3, 'wb') as f:
        #     # print('save corrupted image to : {}'.format(file_path))
        #     np.save(f, np.array(reverse_img_list1))
        # with open(file_path4, 'wb') as f:
        #      np.save(f, np.array(dloss_fail)) 
    print("reverse success: {} {}".format(orig_success_reverse_success, orig_fail_reverse_success))
    print("reverse fail:    {} {}".format(orig_success_reverse_fail, orig_fail_reverse_fail))
    return 100*(reverse_acc / test_n), 100*(clean_acc / test_n), reverse_loss_list, before_loss_list



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=1024, type=int)
    parser.add_argument('--test_batch', default=200, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--save_freq', default=5, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--fname', default='train_ssl', type=str)
    parser.add_argument('--save_root_path', default='data/ckpts/', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--md_path', default='./resnet50.pth', type=str)
    parser.add_argument('--ckpt', default='./data/ckpts/imagenetr_1/ssl_contrast_best.pth', type=str)


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
    parser.add_argument('--aug_name', default='contrast', type=str)
    parser.add_argument('--attack_iters', default=5, type=int)
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

        self.normalize = torch.jit.script(transforms)
        self.denormalize = torch.jit.script(inv_transforms)

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

 
    def dist_batch_input(self, batch_input):
        x_dist, labels = list(zip(*[self.dist_transform(sample_x) for sample_x in batch_input]))
        x_dist = [x.unsqueeze(0) for x in x_dist]
        x_dist = torch.cat(x_dist)
        return x_dist, labels

    @staticmethod
    def compute_ssl_dist_loss(x, label, criterion, model, dist_head, no_grad=True):
   
        if no_grad:
            with torch.no_grad():
                out, out_features = model(x)
                pred = dist_head(out_features)

        else:
            out, out_features = model(x)
            pred = dist_head(out_features)

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

    def test_dist_head(self, criterion, model, dist_head, batches, x_type, normalize, denormalize):
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
                dist_x, labels = dist_tranform(denormalize(x), x_type)
            elif x_type == 'mixed':
                dist_x = x
                labels = b['target']

            
            rot_loss, pred, target = self.compute_ssl_dist_loss(normalize(dist_x), labels, criterion, model, dist_head, no_grad=True)
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

    @staticmethod
    def test_(pred, target):
        matches = (pred.max(1)[1] == target).sum().item()
        return  matches

    @staticmethod
    def save_state_dict(dist_head, opt, epoch, isbest):
        state_dict = {
            'epoch': epoch,
            'contrast_head_state_dict': dist_head.state_dict(),
            'optimizer_state_dict': opt.state_dict()
        }
        if isbest:
            torch.save(state_dict, f'data/ckpts/imagenetr_1/ssl_contrast_best.pth')
        else:
            torch.save(state_dict, f'data/ckpts/imagenetr_1/ssl_contrast_{epoch}.pth')

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

    transforms = [Crop(224, 224)]

    # test_set = load_new_test_data('../CIFAR-10.1/', version_string='v6')
    # test_set = list(zip(transpose(test_set['data'] / 255.), test_set['labels']))
    # test_batches_ood = Batches(test_set, 1, shuffle=False, num_workers=2)
    # test_x, test_y = transpose(test_set['data'] / 255.), test_set['labels']
    # corruption_type = ['gaussian_noise', 'shot_noise','impulse_noise',
    #                   'defocus_blur', 'motion_blur' , 'glass_blur','zoom_blur','snow', 
    #                   'frost', 'fog', 'brightness', 'contrast', 'elastic_transform',
    #                   'pixelate', 'jpeg_compression', 'orig']
    

    print('load data....')
    train_dataset, test_dataset = imagenet1000()
    train_batches = Batches(train_dataset, args.batch_size, shuffle=True, num_workers=2)
    test_batches = Batches(test_dataset, args.test_batch, shuffle=False, num_workers=2)

    
    h = []
    idx = []
    for i in range(len(test_dataset)):
        if i % 50 <5:
            idx.append(i)
            h.append(test_dataset.targets[i])


    from learning.resnet import resnet50
    import pretrainedmodels
    from learning.ssl import Ssl_model_contrast2
    #model = pretrainedmodels.__dict__['fbresnet152'](num_classes=1000, pretrained='imagenet')
    model = resnet50(pretrained=True)
    contrast_head = Ssl_model_contrast2(2048)

    # model = nn.DataParallel(model).cuda()

    # dist_head = nn.DataParallel(dist_head).cuda()

    decay, no_decay = [], []
    for name, param in contrast_head.named_parameters():
        if 'bn' not in name and 'bias' not in name:
            decay.append(param)
        else:
            no_decay.append(param)

    params = [{'params': decay, 'weight_decay': 0},
              {'params': no_decay, 'weight_decay': 0}]

    learning_rate = args.lr
    opt = torch.optim.Adam(params, lr=learning_rate)


    num_training_steps_per_epoch = len(train_dataset) // args.batch_size
    print("Use step level LR scheduler!")
    lr_schedule_values = cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    if args.md_path:
        if os.path.isfile(args.md_path):
            #checkpoint = torch.load(args.md_path)['state_dict']
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
        contrast_head.load_state_dict(ckpt['contrast_head_state_dict'])
        # opt.load_state_dict(ckpt2['optimizer_state_dict'])
        restart_epoch = ckpt['epoch'] + 1
    contrast_head.eval().cuda()
    print(opt)

    # defines transformation for SSL contrastive learning.
    s = 1
    size = 224
    imgnet_mean=(0.485, 0.456, 0.406)
    imgnet_std=(0.229, 0.224, 0.225)

    from torchvision.transforms import transforms
    script_transforms = torch.nn.Sequential(
        # transforms.RandomResizedCrop(size=size),
        # transforms.RandomHorizontalFlip(),
        transforms.Normalize(imgnet_mean, imgnet_std)
        # transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s),
        # transforms.RandomGrayscale(p=0.2),
    )

    normalize = torch.jit.script(script_transforms)

    inv_transforms = torch.nn.Sequential(
            transforms.Normalize(mean=[ 0., 0., 0. ], std=[ 1/0.229, 1/0.224, 1/0.225 ]),
            transforms.Normalize(mean=[ -0.485, -0.456, -0.406 ], std=[ 1., 1., 1.] )
        )
    denormalize = torch.jit.script(inv_transforms)

    rot_transform = RotationTransform()

    criterion = torch.nn.CrossEntropyLoss().cuda()

    if args.eval:
        # test_acc(model, test_batches)
        # test_acc(model, test_batches_ood)
        ALL_robust_acc = []
        ALL_clean_acc = []
        all_tp_list = []
        attack_iters = [args.attack_iters]
        reverse_method =  ['aaa'] 
        
        ood_dataset = load_imagenetS() 
 
        test_batches_ood = Batches(ood_dataset, args.test_batch, shuffle=False, num_workers=2)
        trainer = SslTrainer()



         
        for rm in reverse_method:
            robust_acc = []
            clean_acc = []
            for attack_iter in attack_iters:
                # acc1, acc2, reverse_loss_list, corr_loss_list = test_acc_reverse_vector_aug(model, dist_head, test_batches_ood, rot_transform, criterion, 0,  args.aug_name, normalize, denormalize, imagenet_r_mask)
                test_acc_reverse_vector(model, contrast_head, test_batches_ood, criterion, attack_iter, args.aug_name, normalize, denormalize)
                # print("Reverse with cross, acc before reversed: {} acc after reversed: {} ".format(acc2, acc1))

                with open('./log/imagenetR_trans_result_dist_reverse.csv', 'a') as f: 
                        writer = csv.writer(f)
                        writer.writerow(['batch-size: ', str(args.test_batch), 'before reverse: ', acc2, 'after reverse: ', acc1])
        
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
        model.eval().cuda()
        contrast_head.train().cuda()
        best_matches = 0

        with open('./log/imagenetR_dist_trainlog.csv', 'a') as f: 
            writer = csv.writer(f)
            writer.writerow(['train epoch', 'train loss', 'train match acc.'])
        
        #orig_acc = test_acc(model, test_batches, 'orig')
        #print('imagenet ori. acc: {} %'.format(orig_acc*100))
        for epoch in range(restart_epoch, 201):
            print("Epoch number: {}".format(epoch))
            start_step = epoch * num_training_steps_per_epoch

            train_loss, train_n, train_matches = trainer.train_one_epoch(model, contrast_head, test_batches, opt,
                                                lr_schedule_values, start_step, num_training_steps_per_epoch,
                                                   criterion, args.debug)
            
            print('Epoch: %d, Train accuracy: %.4f, Train loss:  %.4f' % (epoch, train_matches / train_n, train_loss / train_n))
       
            if train_matches / train_n > best_matches:
                trainer.save_state_dict(contrast_head, opt, epoch, isbest=True)
                best_matches = train_matches / train_n
            if epoch % args.save_freq == 0 :
                trainer.save_state_dict(contrast_head, opt, epoch, isbest=False)

            with open('./log/imagenetR_contrast_trainlog.csv', 'a') as f: 
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss / train_n, train_matches / train_n])
               


if __name__ == "__main__":
    main()
