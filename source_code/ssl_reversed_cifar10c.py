import argparse
from ast import Param
from audioop import reverse
import os
import time
import csv
from pathlib import Path
from sklearn.random_projection import johnson_lindenstrauss_min_dim
import torch
from tqdm import tqdm
import torch.nn as nn
from timm.data.mixup import Mixup
from learning.wideresnet import WideResNet, WRN34_rot_out_branch,WRN34_rot_out_branch2
from utils import *
import torchvision.transforms as transforms
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from robustbench.utils import load_model
from robustbench.data import load_cifar10c
import kornia
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot
from grad_cam import GradCAM, save_gradcam, BackPropagation
from RandAugment import trans_aug_list, get_transAug_param, init_3by3_kernel, init_5by5_kernel, init_random_kernel_3by3, init_random_kernel_5by5, init_random_composite_kernel
from swd import swd

mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()
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
    
    mask = torch.nn.functional.pad(torch.as_tensor(torch.zeros(30,30)), [1,1,1,1], value=1).to(device)

    before_loss = SslTrainer.compute_ssl_contrastive_loss(contrast_batch(X, transform_num), criterion, model, model_ssl, X.shape[0], transform_num, no_grad=True)[0]
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
    return max_delta, before_loss.item(), -1* loss.item()

def compute_reverse_transformation(model, model_ssl, criterion, X, epsilon, alpha, attack_iters, norm, aug_name, update_kernel, corr_type):

    # import pdb; pdb.set_trace()

    transform_num = 3
    contrast_batch = Contrastive_Transform(X.shape[2])

    eps = get_transAug_param(aug_name[0], corr_type)

    step_size = 1.15 * ((eps[1] - eps[0]) / 2) / attack_iters

    if update_kernel == 'comp':
        print('init with composite random kernel')
        init_kernel_param = init_random_composite_kernel(X)
        kernel_param1, kernel_param2 = init_kernel_param
    elif update_kernel == 'fixed3' or update_kernel == 'fixed3_wo_update':
        init_kernel_param = init_3by3_kernel(X)
        kernel_param1 = init_kernel_param
    elif update_kernel == 'rand3' or update_kernel == 'rand3_wo_update': 
        init_kernel_param = init_random_kernel_3by3(X)
        kernel_param1 = init_kernel_param
    elif update_kernel == 'rand5':
        init_kernel_param = init_random_kernel_5by5(X)
        kernel_param1 = init_kernel_param
    init_param = torch.rand(1) * (eps[1] - eps[0]) + eps[0]
    param = init_param

    param.requires_grad = True
    kernel_param1.requires_grad = True
    if update_kernel == 'comp':
        kernel_param2.requires_grad = True

    before_loss = SslTrainer.compute_ssl_contrastive_loss(contrast_batch(X, transform_num), criterion, model, model_ssl, X.shape[0], transform_num, no_grad=True)[0]
    for _ in range(attack_iters):

        factor_param_all = param.repeat(X.size(0))  
        if update_kernel == 'fixed3_wo_update' or update_kernel == 'rand3_wo_update':
            kernel_param1 = init_kernel_param
            new_x = trans_aug(aug_name[0], X, kernel_param1, factor_param_all)
        elif update_kernel == 'rand3' or update_kernel == 'fixed3':
            new_x = trans_aug(aug_name[0], X, kernel_param1, factor_param_all)
        elif update_kernel == 'comp':
            new_x = trans_aug(aug_name[0], trans_aug(aug_name[0], X, kernel_param1, factor_param_all), kernel_param2, factor_param_all) 
        
        loss = -SslTrainer.compute_ssl_contrastive_loss(contrast_batch(new_x, transform_num), criterion, model, model_ssl, X.shape[0], transform_num, no_grad=False)[0]

        loss.backward()
        param_grad = param.grad.detach()
        kernel_param1_grad = kernel_param1.grad.detach()
        if update_kernel == 'comp':
            kernel_param2_grad = kernel_param2.grad.detach()

        p = param
        g = param_grad

        k1 = kernel_param1
        g1 = kernel_param1_grad

        if update_kernel == 'comp':
            k2 = kernel_param2
            g3 = kernel_param2_grad

        x = X
        
        p = torch.clamp(p + torch.sign(g) * step_size, eps[0], eps[1])
        k1 = k1 + torch.sign(g1) * 0.1
        
        if update_kernel == 'comp':
            k2 = k2 + torch.sign(g3) * 0.1
        
        param.data = p
        param.grad.zero_()
        
        kernel_param1.data = k1
        kernel_param1.grad.zero_()
        if update_kernel == 'comp':
            kernel_param2.data = k2
            kernel_param2.grad.zero_()

    final_loss = -1 * loss.item()

    if final_loss > before_loss:
        max_kernel = init_kernel_param
    else: 
        if update_kernel == 'comp':
            max_kernel = [kernel_param1.detach(), kernel_param2.detach()]
        else:
            max_kernel = kernel_param1
    max_param = param.detach()

    return max_kernel, max_param, before_loss.item(), final_loss



def test_acc_reverse_vector(model, model_ssl, test_batches_orig, test_batches_ood, criterion, attack_iters, args):

    acc = 0
    epsilon = (8 / 255.)
    pgd_alpha = (2 / 255.)
    test_n = 0
    clean_acc = 0
    
    before_loss_list = []
    final_loss_list = []
    before_swd_list = []
    after_swd_list = []

    raw_dataloader_iterater = iter(test_batches_orig)
    aug_name = args.aug_name
    if args.norm == None: norm = 'none'
    else: norm = args.norm[0]
    allow_gcam = args.allow_gcam
    n_proj = 128

    for i, batch in enumerate(test_batches_ood):
        if test_n < 1000:
            before_swd_score = []
            after_swd_score = []
            x, y = batch['input'], batch['target']
            test_n += y.shape[0]

            orig_batch = next(raw_dataloader_iterater)
            orig_x = orig_batch['input']

            clean_out, _ = model(x)
            clean_acc += (clean_out.max(1)[1] == y).sum().item()

            if aug_name is None:
                delta, before_loss, final_loss = compute_reverse_attack(model, model_ssl, criterion, x,
                                                            epsilon, pgd_alpha, attack_iters, norm)

                out, _ = model(x + delta)
                acc += (out.max(1)[1] == y).sum().item()
                before_loss_list.append(before_loss)
                final_loss_list.append(final_loss)
            else: 
                if norm == 'l_2' and aug_name!= None:
                    print('reverse with l2 and {}'.format(aug_name[0]))
                    kernel, param, before_loss, final_loss = compute_reverse_transformation(model, model_ssl, criterion, x,
                                                                epsilon, pgd_alpha, attack_iters, norm, aug_name, args.update_kernel, args.corruption)        
                    param_all = param.repeat(x.shape[0])
                    if args.update_kernel == 'comp':
                        new_x = trans_aug(aug_name[0], trans_aug(aug_name[0], x, kernel[0], param_all), kernel[1], param_all)
                    else:
                        new_x = trans_aug(aug_name[0], x, kernel, param_all) 
                elif norm == 'none' and aug_name != None:
                    print('reverse with {}'.format(aug_name[0]))
                    param, before_loss, final_loss = compute_reverse_semantic_only(model, model_ssl, criterion, x,
                                                                epsilon, pgd_alpha, attack_iters, norm, aug_name, arge.update_kernel)        
                    param_all = param.repeat(x.shape[0])
                    new_x = trans_aug(aug_name[0], x,  param_all) 

                out, _ = model(new_x)
                acc += (out.max(1)[1] == y).sum().item()
                before_loss_list.append(before_loss)
                final_loss_list.append(final_loss)


            if args.allow_gcam:
                for j in range(x.shape[0]):
                    if out.max(1)[1][j] == y[j] and clean_out.max(1)[1][j] != y[j]:
                        target_layer = 'block3.layer.3'
                        bp = BackPropagation(model=model)
                        gcam = GradCAM(model=model) ##initialize grad_cam funciton
                        compute_gcam(i*args.test_batch+j, gcam, bp, orig_x[j], x[j], new_x[j], y[j], target_layer, args, denormalize=False)

            print("test number: {} before reverse: {} after reverse: {}".format(test_n, clean_acc/test_n, acc/test_n))

    print('Accuracy after SSL training: {}'.format(acc / test_n))
    print('Loss: {} {}'.format(np.array(before_loss_list).mean(), np.array(final_loss_list).mean()))

    return clean_acc/test_n, acc/test_n, np.array(before_loss_list).mean(), np.array(final_loss_list).mean()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=1024, type=int)
    parser.add_argument('--test_batch', default=200, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--save_freq', default=1, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--corr-dir', default='./data/CIFAR-10-C-customize/', type=str)
    parser.add_argument('--output_dir', default='./cifarc_result/', type=str)
    parser.add_argument('--output_fn', default='', type=str)
    parser.add_argument('--save_root_path', default='data/ckpts/', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--md_path', default='./cifar10_standard.pth', type=str)
    parser.add_argument('--ckpt', default='./data/ckpts/cifar10c_3/ssl_contrast_199.pth', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--allow_adapt', default='', type=str)
    parser.add_argument('--adapt_only', action='store_true')
    parser.add_argument('--allow_gcam', action='store_true')
    parser.add_argument('--update_kernel', default='', type=str)
    parser.add_argument('--aug_name', nargs='+', type=str)
    parser.add_argument('--norm', nargs='+', type=str)
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


class SslTrainer:
    def __init__(self):

        self.contrast_transform = Contrastive_Transform(32)

    def train_one_epoch(self, model, ssl_head, train_batches, opt, lr_schedule_values, start_steps, num_training_steps_per_epoch, aug_transforms, criterion, debug=False):
        
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

            batch_train_loss, batch_matches, batch_size = self.step(model, ssl_head, batch['input'], target, opt, criterion)
            train_loss += batch_train_loss * batch_size
            matches += batch_matches
            train_n += batch_size

            if debug:
                break
        return train_loss, train_n, matches

    def step(self, model, contrast_head, x, target, opt, criterion):

        transforms_num = 3
        x_const = self.contrast_transform(x, transforms_num)

        c_loss, correct = self.compute_ssl_contrastive_loss(x_const, criterion, model, contrast_head, x.shape[0], transforms_num)

        opt.zero_grad()
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
        with torch.no_grad():
            matches = (pred.max(1)[1] == target).sum().item()
        return matches

    @staticmethod
    def save_state_dict(ssl_head, opt, epoch, args, isbest):
        state_dict = {
            'epoch': epoch,
            'contrast_head_state_dict': ssl_head.state_dict(),
            'optimizer_state_dict': opt.state_dict()
        }
        if isbest:
            torch.save(state_dict, os.path.join(args.output_dir, f'ssl_contrast_best.pth'))
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

    test_batches = Batches(test_set, args.test_batch, shuffle=False, num_workers=2)

    from learning.wideresnet import WideResNet
    from learning.ssl import Ssl_model_contrast

    model = WideResNet(depth=28, num_classes=10, widen_factor=10)
    ssl_head = Ssl_model_contrast(640)

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
    backbone_opt = torch.optim.AdamW(model.parameters(), lr=0.00025)
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
    for name, module in model.named_modules():
        print(name)

    restart_epoch = 0
    if os.path.exists(args.ckpt):
        print("Loading checkpoint: {}".format(args.ckpt))
        ckpt = torch.load(args.ckpt)
        ssl_head.load_state_dict(ckpt['contrast_head_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        restart_epoch = ckpt['epoch'] + 1
    ssl_head.eval().cuda()

    # defines transformation for SSL contrastive learning.
    s = 1
    size = 32
    from torchvision.transforms import transforms
    transforms = torch.nn.Sequential(
        transforms.RandomResizedCrop(size=size),
        transforms.RandomHorizontalFlip(),
    )
    scripted_transforms = torch.jit.script(transforms)

    criterion = torch.nn.CrossEntropyLoss().cuda()


    if args.eval:

        attack_iters = [args.attack_iters]
        severity= args.severity
        if args.corruption == 'all':
            c_idx = np.arange(15)
        else:
            c_idx = [np.array(corruption_type.index(args.corruption))]
            
        for i in c_idx:
            if i < 15:
                x_test = np.load(args.corr_dir + str(corruption_type[i])+'.npy')[severity-1]

            ood_test_set = list(zip(transpose(x_test/ 255.), dataset['test']['labels']))
            test_batches_ood = Batches(ood_test_set, args.test_batch, shuffle=False, num_workers=2)
            trainer = SslTrainer()
   
            robust_acc = []
            clean_acc = []
            for attack_iter in attack_iters:
                print('Aug name: ', args.aug_name)
                print('Corruption type: ',  corruption_type[i:i+1][0])
                if args.allow_adapt == 'memo':
                    acc1, acc2 = test_acc_reverse_vector_adapt(model, ssl_head, backbone_opt, test_batches_ood, criterion, attack_iter, args.aug_name, args.adapt_only, args.update_kernel, args.corruption)
                elif args.allow_adapt == 'tent' or args.allow_adapt == 'norm':
                    acc1, acc2 = test_acc_reverse_vector_tent_adapt(model, ssl_head, backbone_opt, test_batches_ood, criterion, attack_iter, args)
                elif args.allow_adapt == 'finetune':
                    acc1, acc2 = test_acc_reverse_vector_finetune(model, ssl_head, backbone_opt, test_batches_ood, criterion, attack_iter, args.aug_name, args.adapt_only, args.update_kernel, args.corruption)
                    with open(os.path.join(args.output_dir, args.output_fn), 'a') as f: 
                            writer = csv.writer(f)
                            writer.writerow([args.update_kernel, args.norm, args.aug_name, ' reverse_iter: ', args.attack_iters, ' corruption: ', corruption_type[i:i+1], ' severity: '+ str(args.severity), 'batch-size: '+str(args.test_batch), acc1, acc2])
                else:
                    acc1, acc2, loss1, loss2 = test_acc_reverse_vector(model, ssl_head, test_batches, test_batches_ood, criterion, attack_iter, args)

                    print("Reverse with cross, acc before reversed: {} acc after reversed: {} ".format(acc1, acc2))

                    with open(os.path.join(args.output_dir, args.output_fn), 'a') as f: 
                            writer = csv.writer(f)
                            writer.writerow([args.update_kernel, args.norm, args.aug_name, ' reverse_iter: ', args.attack_iters, ' corruption: ', corruption_type[i:i+1], ' severity: '+ str(args.severity), 'batch-size: '+str(args.test_batch), acc1, acc2, loss1, loss2])
    else:
        trainer = SslTrainer()
        ssl_head.train().cuda()
        with open(os.path.join(args.output_dir, 'train_log.csv'), 'a') as f: 
            writer = csv.writer(f)
            writer.writerow(['train epoch', ' train loss', ' train match acc.'])
        
        best_matches = 0.
        for epoch in range(restart_epoch, args.epochs):
            print("Epoch number: {}".format(epoch))
            start_step = epoch*num_training_steps_per_epoch
            train_loss, train_n, train_matches = trainer.train_one_epoch(model, ssl_head, train_batches, 
                                                                         opt, lr_schedule_values, start_step, num_training_steps_per_epoch,
                                                                         scripted_transforms, criterion, args.debug)
            
            print('Epoch: %d, Train accuracy: %.4f, Train loss:  %.4f' % (epoch, train_matches / train_n, train_loss / train_n))

            if train_matches / train_n > best_matches:
                trainer.save_state_dict(ssl_head, opt, epoch, args, isbest=True)
                best_matches = train_matches / train_n

            if epoch % args.save_freq == 0 and epoch > 0:
                trainer.save_state_dict(ssl_head, opt, epoch, args, isbest=False)
                        
            with open(os.path.join(args.output_dir, 'train_log.csv'), 'a') as f: 
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss / train_n, train_matches / train_n])
               


if __name__ == "__main__":
    main()
