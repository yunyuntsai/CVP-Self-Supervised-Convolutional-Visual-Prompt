import random
from collections import namedtuple

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import RandAugment  
from RandAugment import get_transAug_param, trans_aug  
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import scipy.ndimage as ndi
import uuid
import datetime
import os
import base64
import socket
import math
import cv2
from PIL import Image
import matplotlib.cm as cm
import pathlib
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################
## Components from https://github.com/davidcpage/cifar10-fast ##
################################################################

#####################
## data preprocessing
#####################

cifar10_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255

imgnet_mean=(0.485, 0.456, 0.406)
imgnet_std=(0.229, 0.224, 0.225)

def getDictImageNetClasses(path_imagenet_classes_name='imagenet_list.txt'):
    '''
    Returns dictionary of classname --> classid. Eg - {n02119789: 'kit_fox'}
    '''

    count = 0
    dict_imagenet_classname2id = {}
    list_imagenet_classname = []
    with open(path_imagenet_classes_name) as f:
        line = f.readline()
        print(line)
        while line:
            split_name = line.strip().split()
            cat_name = split_name[2]
            id = split_name[0]
            if cat_name in dict_imagenet_classname2id.keys():
                print(cat_name)
            dict_imagenet_classname2id[id] = cat_name.lower()
            count += 1
            list_imagenet_classname.append(id)
            # print(cat_name, id)
            line = f.readline()
    # print("Total categories categories", count)

    return dict_imagenet_classname2id, list_imagenet_classname


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()


def set_bn_train(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.train()


# model.apply(set_bn_eval)


def normalise(x, mean, std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean * 255
    x *= 1.0 / (255 * std)
    return x


def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border), (0, 0)], mode='reflect')


def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target])

def resize(x, size):
    factor = size / x.shape[1] 
    x_out = ndi.zoom(x, (1, factor, factor, 1), order=2)
    print(x_out.shape)
    return x_out


#####################
# data augmentation
#####################

class RotAug:
    """Rotate in all angles and augment."""
    def __init__(self):
        self.angles = [0, 90, 180, 270]
        #self.angles = [0]
        from torchvision.transforms import transforms
        self.transforms = torch.nn.Sequential(
            transforms.Resize(size=32),
            transforms.CenterCrop(size=32),
            transforms.ColorJitter(0.8, 0.8, 0.8 , 0.2),
            transforms.RandomGrayscale(p=0.2),
        )

    def __call__(self, x):
        out, labels = [], []
        for angle in self.angles:
            x_angle = TF.rotate(x, angle)
            # out += [x_angle] + [self.transforms(x_angle)] * 3
            # labels.append([angle] * 4)
            out += [x_angle]
            labels.append([angle])
        out = [torch.unsqueeze(item, 0) for item in out]
        out = torch.vstack(out)
        labels = [item for angle in labels for item in angle]
        return out, labels

class Rotate_Batch:
    
    def __init__(self):
        self.rot_transform = RotationTransform()

    def __call__(self, batch_input):
        x_rotated, angles = list(zip(*[self.rot_transform(sample_x) for sample_x in batch_input]))
        x_rotated = [x.unsqueeze(0) for x in x_rotated]
        x_rotated = torch.cat(x_rotated)
        return x_rotated, angles


class Dist_Batch:
    
    def __init__(self):
        
        self.dist_transformwithlabel = DistTransformWithLabel()
        self.dist_transform = DistTransform_Train()

        
    def __call__(self, batch_input, name='none'):
        if name=='corrupt':
            label=1
            x_dist, labels = list(zip(*[self.dist_transformwithlabel(sample_x, label) for sample_x in batch_input]))
            x_dist = [x.unsqueeze(0) for x in x_dist]
            x_dist = torch.cat(x_dist)
        elif name=='orig':
            label=0
            x_dist, labels = list(zip(*[self.dist_transformwithlabel(sample_x, label) for sample_x in batch_input]))
            x_dist = [x.unsqueeze(0) for x in x_dist]
            x_dist = torch.cat(x_dist)
        else:
            x_dist, labels = list(zip(*[self.dist_transform(sample_x) for sample_x in batch_input]))
            x_dist = [x.unsqueeze(0) for x in x_dist]
            x_dist = torch.cat(x_dist)
        return x_dist, labels

class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self):
        self.angles = [0, 90, 180, 270]
        #self.angles = [0]
    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle), angle

class DistTransform_Train:
    """Rotate by one of the given angles."""
 
    def __init__(self):
        from torchvision.transforms import transforms
        self.choice = [0, 1]
        self.aug = ['contrast', 'saturation', 'sharpness', 'brightness', 'hue']
    def __call__(self, x):
        label = random.choice(self.choice)
        aug_name = random.choice(self.aug)
        # print('random aug_name: ', aug_name)
        if label:
            eps, rand_num  = get_transAug_param(aug_name, x.shape[0])
    
            param = torch.rand((1)) * (eps[1] - eps[0]) + eps[0]

            x_transform = trans_aug(aug_name, x, param.item())

            return  x_transform , label
        else: 

            return x , label

class DistTransformWithLabel:
    """Rotate by one of the given angles."""
 
    def __init__(self):
        from torchvision.transforms import transforms
    def __call__(self, x, label):
        return x, label


class Contrastive_Transform:
    """Contrastive learning with given tranformations"""

    def __init__(self, size):
      from torchvision.transforms import transforms

      self.transforms = torch.nn.Sequential(
            #transforms.Resize(size=256),
            transforms.RandomResizedCrop(size=size),
            transforms.ColorJitter(0.8, 0.8, 0.8 , 0.2),
            # transforms.RandomGrayscale(p=0.2),
            # transforms.RandomRotation((-90, 90)),
            transforms.RandomHorizontalFlip(),)  
            
    
    def __call__(self, x, transforms_num):
        out = []
        out += [x]
        if transforms_num > 0:
            for i in range(transforms_num):
                x_transform = self.transforms(x)
                out += [x_transform.to(device)]
            out = [torch.unsqueeze(item, 0) for item in out] ## add
        out = torch.vstack(out)

        x_concat = out[0]
        for i in range(1, out.shape[0]):

            x_concat = torch.cat((x_concat, out[i]), axis=0)

        return x_concat


class DownScale_Transform:
    """Contrastive learning with given tranformations"""

    def __init__(self, size):
        from torchvision.transforms import transforms
        self.transforms = torch.nn.Sequential(
            transforms.Resize(size=size),)     
    
    def __call__(self, x):
        return self.transforms(x)
        

class Crop(namedtuple('Crop', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        return x[:, y0:y0 + self.h, x0:x0 + self.w]

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W + 1 - self.w), 'y0': range(H + 1 - self.h)}

    def output_shape(self, x_shape):
        C, H, W = x_shape
        return (C, self.h, self.w)


class FlipLR(namedtuple('FlipLR', ())):
    def __call__(self, x, choice):
        return x[:, :, ::-1].copy() if choice else x

    def options(self, x_shape):
        return {'choice': [True, False]}


class Cutout(namedtuple('Cutout', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        x = x.copy()
        x[:, y0:y0 + self.h, x0:x0 + self.w].fill(0.0)
        return x

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W + 1 - self.w), 'y0': range(H + 1 - self.h)}


class Transform():
    def __init__(self, dataset, transforms):
        self.dataset, self.transforms = dataset, transforms
        self.choices = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, labels = self.dataset[index]
        for choices, f in zip(self.choices, self.transforms):
            args = {k: v[index] for (k, v) in choices.items()}
            data = f(data, **args)
        return data, labels

    def set_random_choices(self):
        self.choices = []
        x_shape = self.dataset[0][0].shape
        N = len(self)
        for t in self.transforms:
            options = t.options(x_shape)
            x_shape = t.output_shape(x_shape) if hasattr(t, 'output_shape') else x_shape
            self.choices.append({k: np.random.choice(v, size=N) for (k, v) in options.items()})


#####################
## dataset
#####################

def cifar10(root):
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    return {
        'train': {'data': train_set.data, 'labels': train_set.targets},
        'test': {'data': test_set.data, 'labels': test_set.targets}
    }

def load_new_test_data(version_string='', load_tinyimage_indices=False):
    data_path = os.path.join(os.path.dirname(__file__), '../CIFAR-10.1/datasets/')
    filename = 'cifar10.1'
    if version_string == '':
        version_string = 'v7'
    if version_string in ['v4', 'v6', 'v7']:
        filename += '_' + version_string
    else:
        raise ValueError('Unknown dataset version "{}".'.format(version_string))
    label_filename = filename + '_labels.npy'
    imagedata_filename = filename + '_data.npy'
    label_filepath = os.path.abspath(os.path.join(data_path, label_filename))
    imagedata_filepath = os.path.abspath(os.path.join(data_path, imagedata_filename))
    print('Loading labels from file {}'.format(label_filepath))
    assert pathlib.Path(label_filepath).is_file()
    labels = np.load(label_filepath)
    print('Loading image data from file {}'.format(imagedata_filepath))
    assert pathlib.Path(imagedata_filepath).is_file()
    imagedata = np.load(imagedata_filepath)
    assert len(labels.shape) == 1
    assert len(imagedata.shape) == 4
    assert labels.shape[0] == imagedata.shape[0]
    assert imagedata.shape[1] == 32
    assert imagedata.shape[2] == 32
    assert imagedata.shape[3] == 3
    if version_string == 'v6' or version_string == 'v7':
        assert labels.shape[0] == 2000
    elif version_string == 'v4':
        assert labels.shape[0] == 2021

    if not load_tinyimage_indices:
        return imagedata, labels
    else:
        ti_indices_data_path = os.path.join(os.path.dirname(__file__), '../other_data/')
        ti_indices_filename = 'cifar10.1_' + version_string + '_ti_indices.json'
        ti_indices_filepath = os.path.abspath(os.path.join(ti_indices_data_path, ti_indices_filename))
        print('Loading Tiny Image indices from file {}'.format(ti_indices_filepath))
        assert pathlib.Path(ti_indices_filepath).is_file()
        with open(ti_indices_filepath, 'r') as f:
            tinyimage_indices = json.load(f)
        assert type(tinyimage_indices) is list
        assert len(tinyimage_indices) == labels.shape[0]
        return imagedata, labels, tinyimage_indices

def imagenet1000():

    if 'thor' in socket.gethostname():
        traindir='/local/rcs/yunyun/ImageNet-Data/train'
        valdir='/local/rcs/yunyun/ImageNet-Data/val'
    
    imgnet_mean=(0.485, 0.456, 0.406)
    imgnet_std=(0.229, 0.224, 0.225)


    mu = torch.tensor(imgnet_mean).view(3,1,1).cuda()
    std = torch.tensor(imgnet_std).view(3,1,1).cuda()

    train_set = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224), #256 #224
                transforms.ToTensor(),
                transforms.Normalize(imgnet_mean, imgnet_std),]))

    test_set = datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(256), transforms.CenterCrop(224),
                     transforms.ToTensor(),
                    transforms.Normalize(imgnet_mean, imgnet_std),]))

    return train_set, test_set

def load_imagenetR():

    if 'thor' in socket.gethostname():
        data_dir='/local/rcs/yunyun/imagenet-r'
    
    imgnet_mean=(0.485, 0.456, 0.406)
    imgnet_std=(0.229, 0.224, 0.225)

    dataset = datasets.ImageFolder(
            data_dir,
            transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(imgnet_mean, imgnet_std),]))

    return dataset

def load_imagenetS():

    if 'thor' in socket.gethostname():
        data_dir='/local/rcs/yunyun/sketch'
    elif 'ironman' in socket.gethostname():
        data_dir='/local/rcs/yl4888/sketch'
    
    imgnet_mean=(0.485, 0.456, 0.406)
    imgnet_std=(0.229, 0.224, 0.225)

    dataset = datasets.ImageFolder(
            data_dir,
            transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(imgnet_mean, imgnet_std),]))

    return dataset

def load_imagenetA():
    
    data_dir = ''
    if 'ironman' in socket.gethostname():
        data_dir='/local/rcs/yl4888/imagenet-a'
    
    imgnet_mean=(0.485, 0.456, 0.406)
    imgnet_std=(0.229, 0.224, 0.225)

    dataset = datasets.ImageFolder(
            data_dir,
            transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(imgnet_mean, imgnet_std),]))

    return dataset

#####################
## data loading
#####################

class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle,
            drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return ({'input': x.to(device).half(), 'target': y.to(device).long()} for (x, y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def objectnet_accuracy_B(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    cnt = 0
    top1_cnt = 0
    top5_cnt = 0

    with torch.no_grad():
        maxk = max(topk)
        batch_size = output.size(0)

        output = output.data
        _, pred = torch.topk(output, k=maxk, dim=1, largest=True, sorted=True)  # Oh forget to sorted it

        # _, pred = output.topk(maxk, 1, True, True)
        # pred = pred.t()
        # correct = pred.eq(target.view(1, -1).expand_as(pred))
        pred = pred.cpu().numpy()

        # target = target # Labels are splited by -1

        target_num = len(target)
        target = [each.numpy() for each in target]

        for jj in range(batch_size):
            # label_gt_all = target[jj]
            label_gt_list = []

            for kk in range(target_num):
                label_gt_list.append(target[kk][jj])

            cnt += 1

            pred_index = pred[jj]

            for label_gt in label_gt_list:
                if pred_index[0] == label_gt:
                    top1_cnt += 1
                    break

            flag = True
            for nnn in range(5):  # If each batch is from same category, can make it in matrix to speed up
                if flag == False:
                    break
                for label_gt in label_gt_list:
                    if pred_index[nnn] == label_gt:
                        top5_cnt += 1
                        flag = False
                        break

        return top1_cnt * 100. / cnt, top5_cnt * 100. / cnt


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        print(pred)
        print(target)

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        # print('correct size', correct.size())
        # print('correct size', correct[:5].size())
        # print('correct size', correct[:5].reshape(-1).size())
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def visualization(robust_acc, clean_acc):
    print(robust_acc)
    print(clean_acc)
    plt.title('Acc v.s test batchsize')
    plt.xlabel('test batch size')
    plt.ylabel('Acc (%)')
    robust_arr = np.array(robust_acc)
    clean_arr = np.array(clean_acc)
    bar_width=0.15
    x = np.arange(15)

    plt.bar(x, clean_arr,bar_width, color='blue',edgecolor='black', label = 'Clean Acc.')
    plt.bar(x+bar_width,robust_arr,bar_width,color='green',edgecolor='black', label = 'Robust Acc.')
    #plt.plot(x, robust_arr, linewidth = 2, label = 'Robust Acc.',marker = '.')
    #plt.plot(x, clean_arr, linewidth = 2, label = 'Clean Acc.',marker = '.')
    # for j in len(bs):
    #     plt.text(bs[j], robust_arr[j], '{:0.2f}'.format(robust_arr[j]))
    #     plt.text(bs[j], clean_arr[j], '{:0.2f}'.format(clean_arr[j]))
    plt.xticks(x)
    plt.legend()
    plt.show()
    plt.savefig('acc_cifar10c.png')


def show_image(robust_img, ori_img, c_type, s_level, param):
    row = 1 
    col = 2
    fig =  plt.figure(figsize=(10, 7))
    ORI_img = ori_img[0].permute(1,2,0).cpu().detach()
    ROB_img = robust_img[0].permute(1,2,0).cpu().detach()
    diff = torch.mean(torch.abs(ROB_img-ORI_img))
    # diff1 = torch.linalg.norm(torch.abs(ROB_img), ord=float('inf'))
    # diff2 = torch.linalg.norm(torch.abs(ORI_img), ord=float('inf'))
    #fig.suptitle("c_type: {}  severity: {}".format(c_type[0], str(s_level)), fontsize=16)
    fig.add_subplot(row, col, 1)
    plt.imshow(ori_img[0].permute(1,2,0).cpu().detach())
    plt.title('original, c_type: {} severity: {}'.format(c_type[0],s_level))
    fig.add_subplot(row, col, 2)
    plt.imshow(robust_img[0].permute(1,2,0).cpu().detach())
    plt.title('reversed, diff: {:0.5f}, param: {:0.5f}'.format(diff, param))
    print("difference= {}".format(diff))
    unique_str = str(uuid.uuid4())[:3]
    file_path = os.path.join('./log/img/trans_ori', c_type[0] + '_' + str(s_level) + '_' + unique_str + '_img.png')
    # plt.show()
    plt.tight_layout()
    plt.savefig(file_path)

def save_image(idx, reverse_x, x, c_type, s_level, param, reverse_method, clean_pred, reverse_pred):
    file_root_path = './data/ImageNetC-customize/reversed/'
    param = np.round(param.item(), decimals=2)
    # reverse_x = np.array(torch.squeeze(reverse_x, 0).detach().cpu().numpy()*255., dtype=np.uint8)
    #image_path = reverse_method + '/' + c_type[0].replace("_", "") + '/' + c_type[0].replace("_", "") + '_' + str(s_level) + '_img_' + str(idx) + '_pred_' + str(clean_pred) + '-' + str(reverse_pred) + '_param_' + str(param) + '.npy'
    image_path = reverse_method + '/'  + c_type.replace("_", "") + '_' + str(s_level) + '_img_' + str(idx) + '_pred_' + str(clean_pred) + '-' + str(reverse_pred) + '_param_' + str(param) + '.npy'
    # save_img = np.transpose(reverse_x, (1,2,0))
    x = np.transpose(x, (1,2,0))
    reverse_x = np.transpose(reverse_x, (1,2,0))
    # print("loss track: ",loss_list)
    
    with open(os.path.join(file_root_path, image_path), 'ab') as f:
        print(image_path)
        np.save(f, [x, reverse_x]) 


def visualize_losslandscape(model, idx, x, y, aug_name, loss):

    eps, rand_num = RandAugment.get_transAug_param(aug_name, x.shape[0])  
    param = (torch.arange(50)/50) * (eps[1] - eps[0]) + eps[0]
    
    accs = []
    imgs = []
    for p in param:
        new_x = RandAugment.trans_aug(aug_name, x, p)
        reverse_x = np.array(torch.squeeze(new_x, 0).detach().cpu().numpy()*255., dtype=np.uint8)
        reverse_img = np.transpose(reverse_x, (1,2,0))
        reverse_out, _ = model(new_x)
        #save_img = np.transpose(reverse_x, (1,2,0))
        #image_path = './outputs/losslanscape_' + str(idx) + '_' + str(param) + '.npy'
        accs.append((reverse_out.max(1)[1] == y).sum().item())
        imgs.append(reverse_img)
        #with open(image_path, 'ab') as f:
        #    np.save(f, save_img) 
    print(accs)
    accs_x = np.arange(len(accs))
    save_path = './outputs/losslanscape_c2c_s1_' + str(idx) + '.npy'
    with open(save_path, 'ab') as f:
        print(save_path)
        np.save(f, np.array([imgs, accs, loss])) 

    # fig =plt.figure(figsize=(10,6))
    # #fig.subplots_adjust(bottom=0.025, left=0.025, top = 0.975, right=0.975)
    # X = []
    # for i in range (1,11):
    #     X.append((2,10,i))
    # X.append((2,10,(11,20)))
    # for nrows, ncols, plot_number in X: 
    #     sub = fig.add_subplot(nrows, ncols, plot_number)
    #     print(plot_number)
    #     if type(plot_number) is tuple:
    #         sub.plot(param, loss, label='rotations task loss')
    #         sub.fill_between(param, np.min(np.array(loss)), np.max(np.array(loss)), where=np.array(accs) > 0,
    #                     color='green', alpha=0.3)
    #         sub.legend()
    #     else:
    #         sub.imshow(imgs[(plot_number-1)*(len(loss)//10)])
    #         sub.set_xticks([])
    #         sub.set_yticks([])
    # plt.tight_layout()
    # plt.savefig('./outputs/losslanscape_stog_s1_' + str(idx) + '.png')

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def compute_gcam(gcam, bp, orig_images, corrupt_images, reverse_images, 
                    targets, target_layer, denormalize=True):
    ##visualize gradcam for one image

    target_class = np.arange(10)

    with torch.cuda.amp.autocast():
        if targets.item() in target_class:
            orig_img = orig_images.unsqueeze(0)
            corr_img = corrupt_images.unsqueeze(0)
            re_img = reverse_images.unsqueeze(0)

            orig_probs, orig_ids = bp.forward(orig_img)  
            probs, corr_ids = bp.forward(corr_img)  # sorted
            re_probs, re_ids = bp.forward(re_img)

            # Remove all the hook function in the "model"
            bp.remove_hook()

            _ = gcam.forward(orig_img)
            gcam.backward(ids=orig_ids[:, [0]])
            orig_regions = gcam.generate(target_layer=target_layer)

            _ = gcam.forward(corr_img)
            gcam.backward(ids=corr_ids[:, [0]])
            corr_regions = gcam.generate(target_layer=target_layer)

            _ = gcam.forward(re_img)
            gcam.backward(ids=re_ids[:, [0]])
            re_regions = gcam.generate(target_layer=target_layer)

            # if ids[0][0].item() != targets[i].item() and re_ids[0][0].item() == targets[i].item():
            myuuid = uuid.uuid4()
            save_gradcam('./output/gradcam/{}-cls-{}-{}.JPEG'.format(str(myuuid).split("-")[0], corr_ids[0][0].item(), re_ids[0][0].item()), 
                                    orig_regions[0], corr_regions[0], re_regions[0], orig_img, corr_img, re_img, denormalize, False)
                
            gcam.remove_hook()
        torch.cuda.empty_cache()


def save_gradcam(filename, orig_gcam, corr_gcam, re_gcam, orig_img, corr_image, reverse_image, denormalize=True, paper_cmap=False):

    orig_gcam = orig_gcam.cpu().numpy()
    if denormalize:
        orig_img = denormalize(orig_img)
        corr_image = denormalize(corr_image)
        reverse_image = denormalize(reverse_image)

    orig_img = np.transpose(orig_img.cpu().detach().numpy(), (0,2,3,1)) * 255.0

    corr_gcam = corr_gcam.cpu().numpy()
    corr_image = np.transpose(corr_image.cpu().detach().numpy(), (0,2,3,1)) * 255.0

    re_gcam = re_gcam.cpu().numpy()
    reverse_image = np.transpose(reverse_image.cpu().detach().numpy(), (0,2,3,1)) * 255.0
    
    orig_cmap = cm.jet(orig_gcam)[..., :3] * 255.0 
    corr_cmap = cm.jet(corr_gcam)[..., :3] * 255.0
    re_cmap = cm.jet(re_gcam)[..., :3] * 255.0

    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        orig_gcam = (orig_cmap.astype(np.float) + orig_img.astype(np.float)) / 2
        corr_gcam = (corr_cmap.astype(np.float) + corr_image.astype(np.float)) / 2
        re_gcam = (re_cmap.astype(np.float) + reverse_image.astype(np.float)) / 2

    # cv2.imwrite(filename, Image.fromarray(np.uint8(gcam))).save()
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    # fig.suptitle("Snow", fontsize=18)
    plt.setp(ax, xticks=[], yticks=[])
    ax[0,0].set_title('Original', fontsize=30)
    ax[0,0].set_ylabel("Input", fontsize=30)
    ax[0,0].imshow(np.uint8(orig_img[0]))
    ax[1,0].set_ylabel("Grad-CAM", fontsize=30)
    ax[1,0].imshow(np.uint8(orig_gcam[0]))
    # ax[0,1].set_ylabel("corrupted input", fontsize=16)
    ax[0,1].set_title('Corrupted', fontsize=30)
    ax[0,1].imshow(np.uint8(corr_image[0]))
    ax[1,1].imshow(np.uint8(corr_gcam[0]))
    # ax[0,2].set_ylabel("recalibrated input", fontsize=16)
    ax[0,2].set_title('Adapted', fontsize=30)
    ax[0,2].imshow(np.uint8(reverse_image[0]))
    ax[1,2].imshow(np.uint8(re_gcam[0]))
    
    # im = Image.fromarray(np.uint8(gcam[0]))
    print('save figs to {}'.format(filename))
    fig.tight_layout()
    fig.savefig(filename, dpi=300)