from __future__ import print_function

import argparse
import os
from tqdm import tqdm
import time
import random
import wandb

import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, CIFAR10
import torch.nn as nn
import csv
import clip
# from models import prompters
from utils import *
import data_utils
from RandAugment import trans_aug_list, get_transAug_param, get_imgnet_transaug_param, init_sharpness_3by3_kernel, init_sharpness_5by5_kernel, init_sharpness_random_kernel_3by3, init_sharpness_random_kernel_5by5, init_sharpness_random_composite_kernel
from clip_utils import Accuracy, AverageMeter, ProgressMeter, save_checkpoint
from clip_utils import cosine_lr, convert_models_to_fp32, refine_classname, zeroshot_classifier, imagenet_classes, imagenet_templates
from adapt_helpers import clip_adapt_multiple, clip_test_single, copy_model_and_optimizer, load_model_and_optimizer, config_model
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

def parse_option():
    parser = argparse.ArgumentParser('Visual Prompting for CLIP')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epoch5s')

    # optimization
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=0.04,
                        help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="weight decay")
    parser.add_argument("--warmup", type=int, default=1000,
                        help="number of steps to warmup for")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--patience', type=int, default=1000)

    # model
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--arch', type=str, default='vit_b32')
    parser.add_argument('--method', type=str, default='padding',
                        choices=['padding', 'random_patch', 'fixed_patch'],
                        help='choose visual prompting method')
    parser.add_argument('--prompt_size', type=int, default=30,
                        help='size for visual prompts')
    parser.add_argument('--attack_iters', type=int, default=1,
                        help='update iteration')
    parser.add_argument('--corruption', type=str, default='gaussian_noise')
    parser.add_argument('--severity', type=int, default=1)
    parser.add_argument('--update_kernel', type=str, default='rand')
    parser.add_argument('--allow_adapt', type=str, default='')
    parser.add_argument('--adapt_only', action='store_true')

    # dataset
    parser.add_argument('--root', type=str, default='./output',
                        help='dataset')
    parser.add_argument('--output_fn', type=str, default='./output',
                        help='dataset')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='dataset')
    parser.add_argument('--image_size', type=int, default=224,
                        help='image size')

    # other
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for initializing training')
    parser.add_argument('--model_dir', type=str, default='./save/models',
                        help='path to save models')
    parser.add_argument('--image_dir', type=str, default='./save/images',
                        help='path to save images')
    parser.add_argument('--corr_dir', type=str, default='./data/',
                        help='path to save images')
    parser.add_argument('--filename', type=str, default=None,
                        help='filename to save')
    parser.add_argument('--trial', type=int, default=1,
                        help='number of trials')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to resume from checkpoint')
    parser.add_argument('--evaluate', default=False,
                        action="store_true",
                        help='evaluate model test set')
    parser.add_argument('--gpu', type=int, default=None,
                        help='gpu to use')
    parser.add_argument('--use_wandb', default=False,
                        action="store_true",
                        help='whether to use wandb')

    args = parser.parse_args()

    args.filename = '{}_{}_{}_{}_{}_{}_lr_{}_decay_{}_bsz_{}_warmup_{}_trial_{}'. \
        format(args.method, args.prompt_size, args.dataset, args.model, args.arch,
               args.optim, args.learning_rate, args.weight_decay, args.batch_size, args.warmup, args.trial)

    return args

best_acc1 = 0
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    global best_acc1, device

    args = parse_option()
    print (args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # create model
    model, preprocess = clip.load('ViT-B/32', device, jit=False)
    convert_models_to_fp32(model)
    model.eval()

    #create ssl model
    from learning.ssl import Ssl_model_contrast
    ssl_head = Ssl_model_contrast(512)
    ssl_head.eval()
   

    # prompter = prompters.__dict__[args.method](args).to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            ssl_head.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # create data
    template = 'This is a photo of a {}'
    # print(f'template: {template}')

    if args.dataset == 'cifar':
        train_dataset = CIFAR10(args.root, transform=preprocess,
                                download=True, train=True)

        val_dataset = CIFAR10(args.root, transform=preprocess,
                            download=True, train=False)
    elif args.dataset == 'imagenet' or args.dataset == 'imagenetC' or args.dataset == 'imagenetR' or args.dataset == 'imagenetS' or args.dataset == 'imagenetA':

        traindir='/local/rcs/yunyun/ImageNet-Data/train'
        valdir='/local/rcs/yunyun/ImageNet-Data/val'

        print('load ImageNet-1K data....')

        train_dataset = datasets.ImageFolder(traindir, transform=preprocess)
        val_dataset = datasets.ImageFolder(valdir, transform=preprocess)
    

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size, pin_memory=True,
                              num_workers=args.num_workers, shuffle=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size, pin_memory=True,
                            num_workers=args.num_workers, shuffle=False)

    class_names = train_dataset.classes
    class_names = refine_classname(class_names)
    texts = [template.format(label) for label in class_names]

    # define criterion and optimizer
    # optimizer = torch.optim.SGD(prompter.parameters(),
                                # lr=args.learning_rate,
                                # momentum=args.momentum,
                                # weight_decay=args.weight_decay)

    # criterion = torch.nn.CrossEntropyLoss().to(device)
    # scaler = GradScaler()
    # total_steps = len(train_loader) * args.epochs
    # scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)

    decay, no_decay = [], []
    for name, param in ssl_head.named_parameters():
        if 'bn' not in name and 'bias' not in name:
            decay.append(param)
        else:
            no_decay.append(param)

    params = [{'params': decay, 'weight_decay': 0},
              {'params': no_decay, 'weight_decay': 0}]

    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    #for adaptation only
    backbone_opt = torch.optim.Adam(model.parameters(), lr=0.00015)

    num_training_steps_per_epoch = len(train_dataset) // args.batch_size
    print("Use step level LR scheduler!")
    scaler = GradScaler()
    total_steps = len(train_loader) * args.epochs
    lr_schedule_values = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    cudnn.benchmark = True

    # make dir
    refined_template = template.lower().replace(' ', '_')
    args.filename = f'{args.filename}_template_{refined_template}'

    args.model_folder = os.path.join(args.model_dir, args.filename)
    if not os.path.isdir(args.model_folder):
        os.makedirs(args.model_folder)

    # wandb
    if args.use_wandb:
        wandb.init(project='Visual Prompting')
        wandb.config.update(args)
        wandb.run.name = args.filename
        wandb.watch(prompter, criterion, log='all', log_freq=10)

    restart_epoch = 0
    # if args.resume:
    #     print("Loading checkpoint: {}".format(args.ssl_ckpt))
    #     ckpt = torch.load(args.ssl_ckpt)
    #     ssl_head.load_state_dict(ckpt['contrast_head_state_dict'])
    #     opt.load_state_dict(ckpt['optimizer_state_dict'])
    #     restart_epoch = ckpt['epoch'] + 1
    ssl_head.eval().cuda()


    if args.evaluate:


        if args.dataset == 'cifar':
            corruption = args.corruption 
            severity = args.severity
            x_test = np.load(args.corr_dir + '/' + corruption + '.npy')[severity-1]
            ood_test_set = list(zip(transpose(x_test/ 255.), val_dataset.targets))
            print('load cifar corruption data: ', len(ood_test_set))
         
        elif args.dataset == 'imagenetC':        
            corruption = args.corruption 
            severity = args.severity
            ood_test_set, orig_dataset = load_imagenetC(corruption, severity)  
            print('load imagenet corruption data: ', len(ood_test_set))

        elif args.dataset == 'imagenetR': 
            ood_test_set = load_imagenetR() 
            print('load imagenet Rendition data: ', len(ood_test_set))
        
        elif args.dataset == 'imagenetS':
            ood_test_set = load_imagenetS() 
            print('load imagenet Sketch data: ', len(ood_test_set))
    
        elif args.dataset == 'imagenetA':
            ood_test_set = load_imagenetA() 
            print('load imagenet Adversarial data: ', len(ood_test_set))
        
        ood_val_loader = DataLoader(ood_test_set,
                            batch_size=args.batch_size, pin_memory=True,
                            num_workers=args.num_workers, shuffle=False)
    
        acc1 = validate(ood_val_loader, texts, model, ssl_head, backbone_opt, lr_schedule_values, criterion, scaler, args) 
        return

    epochs_since_improvement = 0

    for epoch in range(args.epochs):

        # train for one epoch
        train_loss, train_matches, train_n = train(val_loader, texts, model, ssl_head, optimizer, lr_schedule_values, criterion, scaler, epoch, args)
        print('Epoch: %d, Train accuracy: %.4f, Train loss:  %.4f' % (epoch, train_matches / train_n, train_loss / train_n))
        # evaluate on validation set
        # acc1 = validate(val_loader, texts, model, prompter, criterion, args)

        # remember best acc@1 and save checkpoint
        acc1 = train_matches / train_n
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': ssl_head.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, args, is_best=is_best)

        if is_best:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            print(f"There's no improvement for {epochs_since_improvement} epochs.")

            if epochs_since_improvement >= args.patience:
                print("The training halted by early stopping criterion.")
                break

    wandb.run.finish()

def compute_reverse_l2(model, model_ssl, optimizer, scheduler, criterion, scalar, X, epsilon, alpha, attack_iters, norm, args):


    """Reverse algorithm that optimize the SSL loss via PGD"""
    # import pdb; pdb.set_trace()
    isRandom = True
    isIter = True
    transform_num = 3
    contrast_batch = Contrastive_Transform(X.shape[2])

    if isRandom and isIter:

        
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
        # print('init delta: ', delta.max(), delta.min())
        before_loss = SslTrainer.compute_ssl_contrastive_loss(contrast_batch(X, transform_num), criterion, model, model_ssl, X.shape[0], transform_num, no_grad=True)[0]
        trainer = SslTrainer()
        
        # if args.dataset == 'imagenetC':
        #     mask = torch.nn.functional.pad(torch.as_tensor(torch.zeros(194,194)), [15,15,15,15], value=1).to(device)
        # else:
        #     print('here')
        mask = torch.nn.functional.pad(torch.as_tensor(torch.zeros(194,194)), [15,15,15,15], value=1).to(device)
        
        for _ in range(attack_iters):
            
            new_x = X + (mask*delta)
            # loss, _, _ = trainer.train_one_step(model, model_ssl, new_x, optimizer, scalar, criterion)
            loss = -trainer.compute_ssl_contrastive_loss(contrast_batch(new_x, transform_num), criterion, model, model_ssl, X.shape[0], transform_num, no_grad=False)[0]
            loss.backward()

        
            delta_grad = delta.grad.detach()
            
    
            d = delta
            g2 = delta_grad

        
            x = X
            
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
            
            
            delta.data = d
            delta.grad.zero_()

            #print('update param: {}'.format(param))
        final_loss = -1 * loss.item()
    
        max_delta = delta.detach()
        #param = param.detach()

        # loss = loss.item()
        #param = param.detach()
        # print('max_delta: ', max_delta.max(), max_delta.min())
        # print('max_kernel: ', max_kernel)
        # print('max_param:', max_param)
        return max_delta, before_loss.item(), final_loss


def compute_reverse_transformation(model, model_ssl, optimizer, scheduler, criterion, scalar, X, epsilon, alpha, attack_iters, norm, aug_name, update_kernel, normalize, denormalize, args):


    """Reverse algorithm that optimize the SSL loss via PGD"""
    # import pdb; pdb.set_trace()
    isRandom = True
    isIter = True
    transform_num = 3
    contrast_batch = Contrastive_Transform(X.shape[2])

    if isRandom and isIter:

        eps = get_imgnet_transaug_param(aug_name)

        step_size = 1.15 * ((eps[1] - eps[0]) / 2) / attack_iters

        # delta = torch.unsqueeze(torch.zeros_like(X[0]).cuda(), 0)
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

        param = torch.rand((X.shape[0])) * (eps[1] - eps[0]) + eps[0]
        if update_kernel == 'fixed3' or update_kernel == 'fixed3_wo_update':
            init_kernel_param = init_sharpness_3by3_kernel(X)
            kernel_param1 = init_kernel_param
        elif update_kernel == 'fixed5' or update_kernel == 'fixed5_wo_update':
            init_kernel_param = init_sharpness_5by5_kernel(X)
            kernel_param1 = init_kernel_param
        elif update_kernel == 'comp':
            # print('init with composite random kernel')
            init_kernel_param = init_sharpness_random_composite_kernel(X)
            kernel_param1, kernel_param2 = init_kernel_param
        elif update_kernel == 'rand3' or update_kernel == 'rand3_wo_update':
            init_kernel_param = init_sharpness_random_kernel_3by3(X)
            kernel_param1 = init_kernel_param
        elif update_kernel == 'rand5' or update_kernel == 'rand5_wo_update':
            init_kernel_param = init_sharpness_random_kernel_5by5(X)
            kernel_param1 = init_kernel_param
        init_param = torch.rand(1) * (eps[1] - eps[0]) + eps[0]
        init_param = torch.tensor(0.5)

        param = init_param
        param.requires_grad = True
        kernel_param1.requires_grad = True
        if update_kernel == 'comp':
            kernel_param2.requires_grad = True
        # delta.requires_grad = True
        # print('init delta: ', delta.max(), delta.min())
        before_loss = SslTrainer.compute_ssl_contrastive_loss(contrast_batch(X, transform_num), criterion, model, model_ssl, X.shape[0], transform_num, no_grad=True)[0]
        trainer = SslTrainer()
        for _ in range(attack_iters):

            factor_param_all = param.repeat(X.size(0))  
            if update_kernel == 'rand3_wo_update' or update_kernel == 'rand5_wo_update' or update_kernel == 'fixed3_wo_update' or update_kernel == 'fixed5_wo_update':
                kernel_param1 = init_kernel_param
                if args.dataset == 'cifar': new_x = trans_aug(aug_name, X, kernel_param1, factor_param_all)  
                else: new_x = normalize(trans_aug(aug_name, denormalize(X), kernel_param1, factor_param_all))  
            elif update_kernel == 'rand3' or update_kernel == 'rand5' or update_kernel == 'fixed3' or update_kernel == 'fixed5':
                if args.dataset == 'cifar': new_x = trans_aug(aug_name, X, kernel_param1, factor_param_all)  
                else: new_x = normalize(trans_aug(aug_name, denormalize(X), kernel_param1, factor_param_all))  
            elif update_kernel == 'comp':
                if args.dataset == 'cifar': 
                    new_x = trans_aug(aug_name, trans_aug(aug_name, X, kernel_param1, factor_param_all), kernel_param2, factor_param_all) 
                else: 
                    tmp_x = normalize(trans_aug(aug_name, denormalize(X), kernel_param1, factor_param_all))
                    new_x = normalize(trans_aug(aug_name, denormalize(tmp_x), kernel_param2, factor_param_all)) 
                
            
            # loss, _, _ = trainer.train_one_step(model, model_ssl, new_x, optimizer, scalar, criterion)
            loss = -trainer.compute_ssl_contrastive_loss(contrast_batch(new_x, transform_num), criterion, model, model_ssl, X.shape[0], transform_num, no_grad=False)[0]
            loss.backward()

            param_grad = param.grad.detach()
            kernel_param1_grad = kernel_param1.grad.detach()
            if update_kernel == 'comp':
                kernel_param2_grad = kernel_param2.grad.detach()
            # delta_grad = delta.grad.detach()
            
            # print('delta grad: ', delta_grad.max(), delta_grad.min())
            p = param
            g = param_grad

            k1 = kernel_param1
            g1 = kernel_param1_grad

            # d = delta
            # g2 = delta_grad

            if update_kernel == 'comp':
                k2 = kernel_param2
                g3 = kernel_param2_grad

            x = X
            
            p = torch.clamp(p + torch.sign(g) * step_size, eps[0], eps[1])
            # k = torch.clamp(k + torch.sign(g1) * 0.1, torch.tensor(0).to(device), torch.tensor(5).to(device))
            k1 = k1 + torch.sign(g1) * 0.1
            if update_kernel == 'comp':
                k2 = k2 + torch.sign(g3) * 0.1
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
            
            kernel_param1.data = k1
            kernel_param1.grad.zero_()
            if update_kernel == 'comp':
                kernel_param2.data = k2
                kernel_param2.grad.zero_()

            # delta.data = d
            # delta.grad.zero_()

            #print('update param: {}'.format(param))
        final_loss = -1 * loss.item()
        # print('before loss: ', before_loss, 'final loss: ', final_loss)
        if final_loss > before_loss:
            # print('use initial kernel!!')
            # max_kernel = [init_kernel_param[0].detach(), init_kernel_param[1].detach()]
            max_kernel = init_kernel_param
        else: 
            if update_kernel == 'comp':
                max_kernel = [kernel_param1.detach(), kernel_param2.detach()]
            else:
                max_kernel = kernel_param1
        max_param = param.detach()
        # max_delta = delta.detach()
        #param = param.detach()

        # loss = loss.item()
        #param = param.detach()
        # print('max_delta: ', max_delta.max(), max_delta.min())
        # print('max_kernel: ', max_kernel)
        # print('max_param:', max_param)
        return max_kernel, max_param, before_loss.item(), final_loss



def train(train_loader, texts, model, ssl_head, optimizer, scheduler, criterion, scalar, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    ssl_head.train()
    trainer = SslTrainer()

    num_batches_per_epoch = len(train_loader)

    end = time.time()
    train_loss, matches, train_n = 0,0,0


    for i, (images, target) in enumerate(tqdm(train_loader)):

        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        images = images.to(device)
        target = target.to(device)
        text_tokens = clip.tokenize(texts).to(device)

        # with automatic mixed precision
     
        batch_train_loss, batch_matches, batch_size = trainer.train_one_step(model, ssl_head, images, optimizer, scalar, criterion)

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)


        train_loss += batch_train_loss * batch_size
        matches += batch_matches
        train_n += batch_size

        # measure accuracy
        # acc1 = accuracy(output, target, topk=(1,))
        # losses.update(loss.item(), images.size(0))
        # top1.update(acc1[0].item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)

        #     if args.use_wandb:
        #         wandb.log({
        #             'training_loss': losses.avg,
        #             'training_acc': top1.avg
        #              })

        if i % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': ssl_head.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, args)

    # return losses.avg, top1.avg
    return train_loss, matches, train_n

def validate(val_loader, texts, model, ssl_head, optimizer, scheduler, criterion, scalar, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1_org = AverageMeter('Original Acc@1', ':6.2f')
    top1_adapt = AverageMeter('Adapted Acc@1', ':6.2f')
    # top1_prompt = AverageMeter('Prompt Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1_adapt, top1_org],
        prefix='Validate: ')

    # switch to evaluation mode
    # prompter.eval()
    correct, test_n = 0., 0.
    epsilon = (8 / 255.)
    pgd_alpha = (2 / 255.)
    attack_iters = args.attack_iters

    trsf = torch.nn.Sequential(
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    )

    script_transforms = torch.nn.Sequential(
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    )
    normalize = torch.jit.script(script_transforms)

    inv_transforms = torch.nn.Sequential(
        transforms.Normalize(mean=[ 0., 0., 0. ], std=[ 1/0.26862954, 1/0.26130258, 1/0.27577711 ]),
        transforms.Normalize(mean=[ -0.48145466, -0.4578275, -0.40821073 ], std=[ 1., 1., 1.] )
    )
    denormalize = torch.jit.script(inv_transforms)
    if args.dataset == 'imagenetC' or args.dataset == 'imagenetS':
        zeroshot_weights = zeroshot_classifier(model, imagenet_classes, imagenet_templates)
    elif args.dataset == 'imagenetR':
        imagenetr_mask = data_utils.gen_mask()
        imgnetr_index = [i for i, x in enumerate(imagenetr_mask) if x]
        imagenet_r_classes = [imagenet_classes[i] for i in imgnetr_index]
        print(imagenet_r_classes)
        zeroshot_weights = zeroshot_classifier(model, imagenet_r_classes, imagenet_templates)
    elif args.dataset == 'imagenetA':
        imagenetA_mask = data_utils.get_imgnet_a_mask()
        imgneta_index = [i for i, x in enumerate(imagenetA_mask) if x]
        imagenet_a_classes = [imagenet_classes[i] for i in imgneta_index]
        print(imagenet_a_classes)
        zeroshot_weights = zeroshot_classifier(model, imagenet_a_classes, imagenet_templates)

    end = time.time()

    if args.allow_adapt:
        model = config_model(model)

    for i, (images, target) in enumerate(tqdm(val_loader)):

        if args.allow_adapt:
            # print('copy model state')
            model_state, opt_state = copy_model_and_optimizer(model, optimizer)
            prior_strength = 16
            nn.BatchNorm2d.prior = 1

        images = images.to(device)
        target = target.to(device)
        if args.dataset == 'cifar':
            images = trsf(images)
            text_tokens = clip.tokenize(texts).to(device)
        # print('orig range: ', images.max(), images.min())
       # predict
        with torch.no_grad():
            if args.dataset == 'cifar':
                output_org, _ = model(images, text_tokens)
               
            else:
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                output_org = 100. * image_features @ zeroshot_weights
          
        # measure accuracy   
        acc1 = Accuracy(output_org, target, topk=(1,))
        
        top1_org.update(acc1[0].item(), images.size(0))

        if args.allow_adapt == '':
            if args.update_kernel != 'none':
                kernel, param, before_loss, final_loss = compute_reverse_transformation(model, ssl_head, optimizer, scheduler, criterion, scalar, images,
                                                            epsilon, pgd_alpha, attack_iters, 'l_2', 'sharpness', args.update_kernel, normalize, denormalize, args)  
                param_all = param.repeat(images.shape[0])


                if args.update_kernel == 'comp':
                    if args.dataset == 'cifar':
                        tmp_x = trans_aug('sharpness', images, kernel[1], param_all)
                        new_x = trans_aug('sharpness', tmp_x, kernel[0], param_all) 
                    else:
                        tmp_x = normalize(trans_aug('sharpness', denormalize(images), kernel[1], param_all))
                        new_x = normalize(trans_aug('sharpness', denormalize(tmp_x), kernel[0], param_all)) 
                
                else:
                    if args.dataset == 'cifar':
                        new_x = trans_aug('sharpness', images, kernel, param_all)
                    else:
                        new_x = normalize(trans_aug('sharpness', denormalize(images), kernel, param_all))
            else:
                # print('compute reverse vector w/ l2 norm only!')
                delta, before_loss, final_loss = compute_reverse_l2(model, ssl_head, optimizer, scheduler, criterion, scalar, images,
                                                            epsilon, pgd_alpha, attack_iters, 'l_inf', args)  
        
                mask = torch.nn.functional.pad(torch.as_tensor(torch.zeros(194,194)), [15,15,15,15], value=1).to(device)
                new_x = images + (mask*delta)

            with torch.no_grad():
                if args.dataset == 'cifar':
                    output_ada, _ = model(new_x, text_tokens)
                else:
                    image_features = model.encode_image(new_x)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    output_ada = 100. * image_features @ zeroshot_weights

            # measure accuracy 
            acc2 = Accuracy(output_ada, target, topk=(1,))
        
            top1_adapt.update(acc2[0].item(), images.size(0))

      
        elif args.allow_adapt == 'finetune':
            # if args.adapt_only:
            #     new_x = images
            # for i in range(new_x.shape[0]):
            #     adapt_x = new_x[i].unsqueeze(0)
               
            #     clip_adapt_multiple(model, adapt_x, text_tokens, optimizer, 1, adapt_x.shape[0], denormalize=denormalize)
            #     correctness = clip_test_single(model, adapt_x, text_tokens, target[i])
            #     acc += correctness
            #     model, optimizer = load_model_and_optimizer(model, optimizer, model_state, opt_state)
            # acc2 = (acc / images.size(0)) *200
            contrast_batch = Contrastive_Transform(images.shape[2])
            transform_num = 3
            
            if prior_strength < 0:
                nn.BatchNorm2d.prior = 1
            else:
                nn.BatchNorm2d.prior = float(prior_strength) / float(prior_strength + 1)

            optimizer.zero_grad()
            ssl_loss = SslTrainer.compute_ssl_contrastive_loss(contrast_batch(images, transform_num), criterion, model, ssl_head, images.shape[0], transform_num, no_grad=False)[0]   
                
            ssl_loss.backward()
            optimizer.step()
            nn.BatchNorm2d.prior = 1

            with torch.no_grad():
                if args.dataset == 'cifar':
                    correctness = clip_test_single(model, images, text_tokens, target)
                    acc2 = (correctness / images.size(0)) *100
                    top1_adapt.update(acc2, images.size(0))
                else:
                    prior_strength = 16

                    if prior_strength < 0:
                        nn.BatchNorm2d.prior = 1
                    else:
                        nn.BatchNorm2d.prior = float(prior_strength) / float(prior_strength + 1)

                    image_features = model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    output_ada = 100. * image_features @ zeroshot_weights
                    acc2 = Accuracy(output_ada, target, topk=(1,))
                    top1_adapt.update(acc2[0].item(), images.size(0))
            model, optimizer = load_model_and_optimizer(model, optimizer, model_state, opt_state)
            
            
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            progress.display(i)
            print(' * Adapt Acc@1 {top1_adapt.avg:.3f} Original Acc@1 {top1_org.avg:.3f}'
                    .format(top1_adapt=top1_adapt, top1_org=top1_org))

    with open(os.path.join('./imgnetc_clip_result', args.output_fn), 'a') as f: 
        writer = csv.writer(f)
        writer.writerow([args.update_kernel, 'sharpness', 'corruption: '+ args.corruption, ' severity: '+ str(args.severity), 'batch-size: '+ str(args.batch_size), top1_org.avg, top1_adapt.avg])

    return top1_org.avg, top1_adapt.avg



class SslTrainer:

    def __init__(self):

        # self.normalize = torch.jit.script(transforms)
        # self.denormalize = torch.jit.script(inv_transforms)

        self.contrast_transform = Contrastive_Transform(size=224) #before downscale is 224

    def train_one_step(self, model, contrast_head, x, optimizer, scalar, criterion):

        transforms_num = 3
        x_const = self.contrast_transform(x, transforms_num)
        optimizer.zero_grad()
        with autocast():
            c_loss, correct = self.compute_ssl_contrastive_loss(x_const, criterion, model, contrast_head, x.shape[0], transforms_num)
       
            scalar.scale(c_loss).backward()
            scalar.step(optimizer)
        scalar.update()
        
        return c_loss.item(), correct, x_const.shape[0]

    @staticmethod
    def compute_ssl_contrastive_loss(x, criterion, model, contrast_head, bs, transform_num, no_grad=True):
        if no_grad:
            with torch.no_grad():
                x_ftrs = model.encode_image(x).float()
        else:
            x_ftrs = model.encode_image(x).float()
        

        output = contrast_head(x_ftrs)
        
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



if __name__ == '__main__':
    main()