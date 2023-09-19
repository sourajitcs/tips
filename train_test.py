#Reference codebases:
# https://github.com/achaman2/truly_shift_invariant_cnns/tree/main/cifar10_training
# https://github.com/raymondyeh07/learnable_polyphase_sampling
# https://github.com/adobe/antialiased-cnns/blob/master/antialiased_cnns/blurpool.py
# https://github.com/HobbitLong/SupContrast/blob/master/util.py
# https://chat.openai.com/ was used to generate a small fraction of this code which we then further adopted and modified


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys, argparse
import os
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch import linalg as LA
import torch.optim as optim
from torch import einsum
from torch.autograd import Variable
from einops import rearrange
import shutil
import time
import warnings
from csv import reader
from functools import reduce
import torchvision
import torch._utils
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import math
from torchvision import models
import progressbar
from time import sleep
from pathlib import Path
import csv
import itertools
import random
import gc
import statistics
import pandas as pd
import logging
import functools
import requests
from io import BytesIO
import scipy
from scipy import stats
import seaborn as sns
import cv2
from mpl_toolkits.mplot3d import Axes3D
from torchsummary import summary
from sklearn.manifold import TSNE
from torchviz import make_dot
from collections import defaultdict
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor

from blurpool import BlurPool
from aps import APS
from tips import TIPS
from resnet import resnet18, resnet34, resnet50, resnet101
from util import Index_Regularize, set_optimizer, adjust_learning_rate, warmup_learning_rate, accuracy, AverageMeter, CircularShiftTransform, N_CropTransform
from util import compute_class_distribution, compute_mean_std


warnings.filterwarnings("ignore")

seed_value = 42
torch.manual_seed(seed_value)


def downstream_parse_option(dataset_path=None, num_epochs=None, batch_size=None, patience=None, 
                            earlyStopping_patience=None, learning_rate=None):
    
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--dataset', type=str, default='tiny-imagenet',
                        help='dataset name')
    parser.add_argument('--MODEL_PATH', type=str, default='',
                        help='model weights storing / loading path')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='GPU/CPU name, no multi GPU training facilitated with this version of the code')
    parser.add_argument('--wake_up_at', type=int, default=192,
                        help='loss_ti weight')
    parser.add_argument('--alpha', type=float, default=0.35,
                        help='loss_ti weight')
    parser.add_argument('--num_down', type=int, default=5,
                        help='num of downsample layers')
    parser.add_argument('--num_classes', type=int, default=200,
                        help='num of classes')
    parser.add_argument('--tips', action='store_true',
                        help='tips status')
    parser.add_argument('--avgpool', action='store_false',
                        help='avgpool status')
    parser.add_argument('--maxpool', action='store_false',
                        help='maxpool status')
    parser.add_argument('--aps', action='store_false',
                        help='aps status')
    parser.add_argument('--blurpool', action='store_false',
                        help='blurpool status')
    parser.add_argument('--model_name', type=str, default=resnet50,
                        help='model name')
    parser.add_argument('--H', type=int, default=64,
                        help='input image height')
    parser.add_argument('--W', type=int, default=64,
                        help='input image width')
    parser.add_argument('--data_path', type=str, default="",
                        help='dataset loader path')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='number of training epochs')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--patience', type=int, default=128,
                        help='number of training epochs')
    parser.add_argument('--earlyStopping_patience', type=int, default=10,
                        help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    
    
    parser.add_argument('-f')
    opt = parser.parse_args()
    if not(dataset_path is None):
        opt.data_path = dataset_path
    if not(num_epochs is None):
        opt.epochs = num_epochs
    if not(batch_size is None):
        opt.batch_size = batch_size
    if not(patience is None):
        opt.patience = patience
    if not(earlyStopping_patience is None):
        opt.earlyStopping_patience = earlyStopping_patience
    if not(learning_rate is None):
        opt.learning_rate = learning_rate
        
    if opt.batch_size >= opt.patience:
        opt.warm = True
        opt.cosine = True
        
    if opt.warm:
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate
    
    return  opt



def get_dataloader(dataset_path, batch_size=8, dataset='cifar10', 
                   H=32, W=32, max_shift_h=4, max_shift_w=4, 
                   n_aug_views=2, standard_cnt=0, circular_cnt=0, 
                   datatype='train', pin_memory=False, 
                   shuffle=True, num_workers=4, normalization=True, 
                   worker_init_fn=None, data_loader_seed=None, contrast=True):
    
    if dataset == 'mnist':
        dataset = datasets.MNIST
        mean = (0.1307)
        std = (0.3081)
    elif dataset == 'cifar10':
        dataset = datasets.CIFAR10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'cifar100':
        dataset = datasets.CIFAR100
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif dataset == 'food101':
        dataset = datasets.Food101
        mean = (0.5450, 0.4435, 0.3436)
        std = (0.2302, 0.2409, 0.2387) 
    elif dataset == 'tiny-imagenet':
        if (dataset_path is None):
            raise NotImplementedError(
                'disk location not provided for: {}'.format(dataset))
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225) 
    else:
        raise NotImplementedError(
                'dataset not supported: {}'.format(dataset))
    
    if (normalization):
        normalize = transforms.Normalize(mean=mean, std=std)
    else:
        if (dataset==datasets.MNIST):
            normalize = transforms.Normalize((0.), (1.0))
        else:
            normalize = transforms.Normalize((0., 0., 0.), (1.0, 1.0, 1.0))
    
    train_transform_original = transforms.Compose([
        transforms.Resize(size=(H,W)), 
        transforms.ToTensor(),
        normalize,
    ])
    train_transform_standard_shift = transforms.Compose([
        transforms.Resize(size=(H,W)), 
        transforms.RandomCrop(size=(H,W), padding=(max_shift_h, max_shift_w), 
                              pad_if_needed=True, fill=0, padding_mode='constant'), 
        transforms.ToTensor(),
        normalize,
    ])
    train_transform_circular_shift = transforms.Compose([
        transforms.Resize(size=(H,W)), 
        CircularShiftTransform(torch_function=torch.roll, H=H, W=W, 
                                    max_shift_h=max_shift_h, max_shift_w=max_shift_w), 
        normalize,
    ])
    
    test_transform_original = transforms.Compose([
        transforms.Resize(size=(H,W)), 
        transforms.ToTensor(),
        normalize,
    ])
    test_transform_standard_shift = transforms.Compose([
        transforms.Resize(size=(H,W)), 
        transforms.RandomCrop(size=(H,W), padding=(max_shift_h, max_shift_w), 
                              pad_if_needed=True, fill=0, padding_mode='constant'), 
        transforms.ToTensor(),
        normalize,
    ])
    test_transform_circular_shift = transforms.Compose([
        transforms.Resize(size=(H,W)), 
        CircularShiftTransform(torch_function=torch.roll, H=H, W=W, 
                                    max_shift_h=max_shift_h, max_shift_w=max_shift_w), 
        normalize,
    ])
    
    if (standard_cnt==0 and circular_cnt==0):
        circular_cnt = n_aug_views//2
        standard_cnt = n_aug_views - circular_cnt
    elif (standard_cnt!=0 and circular_cnt==0):
        standard_cnt = n_aug_views
    elif (standard_cnt==0 and circular_cnt!=0):
        circular_cnt = n_aug_views
        
    if (datatype=='train'):
        train_status = True
        data_transform_original = train_transform_original
        data_transform_standard_shift = train_transform_standard_shift
        data_transform_circular_shift = train_transform_circular_shift
    elif (datatype=='test' or datatype=='val'):
        train_status = False
        data_transform_original = test_transform_original
        data_transform_standard_shift = test_transform_standard_shift
        data_transform_circular_shift = test_transform_circular_shift
    else:
        raise NotImplementedError(
                'Datatype: datatype is not supported: {}'.format(datatype))
        
    data_transforms = [data_transform_original] + \
                      [data_transform_standard_shift] * standard_cnt + \
                      [data_transform_circular_shift] * circular_cnt
    aug_label = [0] + [1] * standard_cnt + [2] * circular_cnt
    
    if contrast:
        data_transforms = [data_transform_original] + \
                          [data_transform_standard_shift] * standard_cnt + \
                          [data_transform_circular_shift] * circular_cnt
        aug_label = [0] + [1] * standard_cnt + [2] * circular_cnt
        if dataset == 'tiny-imagenet':
            if train_status:
                data_set = datasets.ImageFolder(root=dataset_path+"/train", 
                                                transform=N_CropTransform(data_transforms, H, W))
            else:
                data_set = datasets.ImageFolder(root=dataset_path+"/val", 
                                                transform=N_CropTransform(data_transforms, H, W))
        else:
            data_set = dataset(root=dataset_path, train=train_status, download=True, 
                               transform=N_CropTransform(data_transforms, H, W))
    else:
        data_transforms = data_transform_original
        aug_label = [0]
        if dataset == 'tiny-imagenet':
            if train_status:
                data_set = datasets.ImageFolder(root=dataset_path+"/train", 
                                                transform=data_transforms)
            else:
                data_set = datasets.ImageFolder(root=dataset_path+"/val", 
                                                transform=data_transforms)
        else:
            data_set = dataset(root=dataset_path, train=train_status, 
                               download=True, transform=data_transforms)
    
    data_loader = torch.utils.data.DataLoader(
        data_set, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory, 
        worker_init_fn=worker_init_fn)

    return data_loader, aug_label



def evaluate_module(opt, data_loader, model, criterion, optimizer, epoch, device, 
                    test_max_shift_h=4, test_max_shift_w=4):
    """one epoch run"""
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    running_s_cons = 0.0
    running_s_stab = 0.0
    running_c_cons = 0.0
    running_c_stab = 0.0
    end = time.time()
    
    model.eval()
    with torch.no_grad():
    
        for idx, (x, y) in enumerate(data_loader):
            data_time.update(time.time() - end)
            bsz = y.shape[0]
            x = x.to(device)
            y = y.to(device)
            test_shift_h = np.random.randint(-test_max_shift_h, test_max_shift_h+1)
            test_shift_w = np.random.randint(-test_max_shift_w, test_max_shift_w+1)
            x_ss_1 = feat_transform(max_shift_h=test_max_shift_h, 
                                    max_shift_w=test_max_shift_w, 
                                    transform_type='standard')(x)
            x_ss_2 = feat_transform(max_shift_h=test_max_shift_h, 
                                    max_shift_w=test_max_shift_w, 
                                    transform_type='standard')(x)
            x_cs_1 = feat_transform(max_shift_h=test_max_shift_h, 
                                    max_shift_w=test_max_shift_w, 
                                    transform_type='circular')(x)
            x_cs_2 = feat_transform(max_shift_h=test_max_shift_h, 
                                    max_shift_w=test_max_shift_w, 
                                    transform_type='circular')(x)
        
            if (model.tips):
                if (model.return_soft_polyphase_indices):
                    y_hat,      _, _, _ = model(x)
                    y_hat_ss_1, _, _, _ = model(x_ss_1)
                    y_hat_ss_2, _, _, _ = model(x_ss_2)
                    y_hat_cs_1, _, _, _ = model(x_cs_1)
                    y_hat_cs_2, _, _, _ = model(x_cs_2)
                else:
                    y_hat,      _, _ = model(x)
                    y_hat_ss_1, _, _ = model(x_ss_1)
                    y_hat_ss_2, _, _ = model(x_ss_2)
                    y_hat_cs_1, _, _ = model(x_cs_1)
                    y_hat_cs_2, _, _ = model(x_cs_2)
            else:
                y_hat = model(x)
                y_hat_ss_1 = model(x_ss_1)
                y_hat_ss_2 = model(x_ss_2)
                y_hat_cs_1 = model(x_cs_1)
                y_hat_cs_2 = model(x_cs_2)

                
            s_cons = (torch.argmax(y_hat_ss_1, dim=1)==\
                      torch.argmax(y_hat_ss_2, dim=1)).tolist()
            s_cons = np.sum(s_cons)/len(s_cons)
            running_s_cons += s_cons
            
            s_stab = (torch.argmax(y_hat, dim=1)==\
                      torch.argmax(y_hat_ss_1, dim=1)==\
                      torch.argmax(y_hat_ss_2, dim=1)).tolist()
            s_stab = np.sum(s_stab)/len(s_stab)
            running_s_stab += s_stab
            
            c_cons = (torch.argmax(y_hat_cs_1, dim=1)==\
                      torch.argmax(y_hat_cs_2, dim=1)).tolist()
            c_cons = np.sum(c_cons)/len(c_cons)
            running_c_cons += c_cons
            
            c_stab = (torch.argmax(y_hat, dim=1)==\
                      torch.argmax(y_hat_cs_1, dim=1)==\
                      torch.argmax(y_hat_cs_2, dim=1)).tolist()
            c_stab = np.sum(c_stab)/len(c_stab)
            running_c_stab += c_stab

            loss = criterion(y_hat, y)
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(y_hat, y, topk=(1, 5))
            top1.update(acc1[0], bsz)
            top5.update(acc5[0], bsz)
            batch_time.update(time.time() - end)
            end = time.time()
                
        running_s_cons = 100 * (running_s_cons / len(data_loader))
        running_s_stab = 100 * (running_s_stab / len(data_loader))
        running_c_cons = 100 * (running_c_cons / len(data_loader))
        running_c_stab = 100 * (running_c_stab / len(data_loader))
        
        return losses.avg, top1.avg, top5.avg, \
        running_s_cons, running_s_stab, running_c_cons, running_c_stab
    

def train_module(opt, data_loader, model, criterion, aux_criterion, regularizer, 
                 optimizer, epoch, device, wake_up_at=4, aux_weights=[], 
                 alpha=0.35):
    """one epoch run"""
    ## wake_up_at is refered to as t_epsilon in our paper
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cls_losses = AverageMeter()
    reg_losses = AverageMeter()
    aux_losses = [AverageMeter() for _ in range(model.num_down)]
    top1 = AverageMeter()
    top5 = AverageMeter()
    cls_weight = 1.0
    end = time.time()
    aux_weights = [1/float(model.num_down)]*int(model.num_down)
    
    
    model.train()
    for idx, (images, labels) in enumerate(data_loader):
        data_time.update(time.time() - end)
        bsz = labels.shape[0]
        images = images.to(device)
        labels = labels.to(device)
        if (model.tips):
            if (model.return_soft_polyphase_indices):
                output, psi_x, x_t, tau = model(images)
                loss_reg = regularizer(tau)
            else:
                output, psi_x, x_t = model(images)
                loss_reg = torch.tensor(0.0)
            
            loss_cls = criterion(output, labels)
            
            loss_aux = [None]*model.num_down
            if (epoch<wake_up_at):
                cls_weight = 0.0
                aux_weights = [0.0 for _ in aux_weights]
            for i in range(model.num_down):
                aux = aux_criterion(psi_x[i], x_t[i])
                aux_losses[i].update(aux.item(), bsz)
                loss_aux[i] = aux_weights[i] * aux
            
            loss = loss_reg + (1-alpha)*loss_cls + alpha*torch.sum(torch.stack(loss_aux))
            #loss = (1-float(sum(aux_weights))) *  loss_cls + \
            #       torch.sum(torch.stack(loss_aux))
            cls_losses.update(loss_cls.item(), bsz)
            reg_losses.update(loss_reg.item(), bsz)
            losses.update(loss.item(), bsz)
        else:
            output = model(images)
            loss = criterion(output, labels)
            loss_reg = torch.tensor(0.0)
            [aux_losses[i].update(0.0, bsz) for i in range(model.num_down)]
            cls_losses.update(loss.item(), bsz)
            losses.update(loss.item(), bsz)
        
        warmup_learning_rate(opt, epoch, idx, len(data_loader), optimizer)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)
        top5.update(acc5[0], bsz)
        
        optimizer.zero_grad()
        loss.backward()
        max_norm_value = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                       max_norm_value, norm_type=2.0)
        optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()
        
    aux_loss_ret = [aux_losses[i].avg for i in range(model.num_down)]
    return losses.avg, cls_losses.avg, reg_losses.avg, aux_loss_ret, \
           top1.avg, top5.avg


def train(opt, train_loader, test_loader, model, criterion, aux_criterion, regularizer, 
          optimizer, test_max_shift_h=4, test_max_shift_w=4, wake_up_at=4, aux_weights=[], 
          device='cuda:0', MODEL_PATH=None):
    print ("---------------Training Started---------------")
    best_val_loss = float('inf')
    early_stopping_counter = 0
    num_epochs = opt.epochs
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    MODEL_STORAGE_PATH = MODEL_PATH + '.pth'
    model.to(device)
    
    for epoch in tqdm(range(1, num_epochs + 1)):
        adjust_learning_rate(opt, optimizer, epoch)
        
        time1 = time.time()
        train_loss, cls_loss, reg_loss, aux_loss, acc1, acc5 = train_module(opt, train_loader, \
                            model, criterion, aux_criterion, regularizer, optimizer, 
                            epoch, device, wake_up_at=wake_up_at, aux_weights=aux_weights)
        time2 = time.time()
        print('>>> Train: $Epoch {}/{}, Avg loss {:1.3f}, ACC@1 {:3.2f} %, ACC@5 {:3.2f} %, train time {:.2f}' \
              .format(epoch, num_epochs, train_loss, acc1, acc5, (time2 - time1)))
        print(">>> Train: $Cls loss: ", cls_loss, "Reg loss: ", reg_loss, \
              " Aux loss: ", aux_loss)
        running_s_cons, running_s_stab, running_c_cons, running_c_stab
        time1 = time.time()
        val_loss, val_acc1, val_acc5, s_cons, s_stab, c_cons, c_stab = \
        evaluate_module(opt, test_loader, model, 
                        criterion, optimizer, 
                        epoch, device, 
                        test_max_shift_h=test_max_shift_h, 
                        test_max_shift_w=test_max_shift_w)
        time2 = time.time()
        print('>>>$ Test: $Epoch {}/{}, Avg loss {:1.3f}, ACC@1 {:3.2f} %, ACC@5 {:3.2f} %, test time {:.2f}' \
              .format(epoch, num_epochs, val_loss, val_acc1, val_acc5, (time2 - time1)))
        print('>>>$ Test: $Epoch {}/{}, s_cons {:3.2f} %, s_stab {:3.2f} %, c_cons {:3.2f} %, c_stab {:3.2f} %' \
              .format(epoch, num_epochs, ss_ss, s_stab, c_cons, c_stab))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= opt.earlyStopping_patience:
            print("Early stopping triggered. Stopping training.")
            break
            
        torch.save({
            'model_state_dict': model.state_dict(), 
            'optim_state_dict': optimizer.state_dict(),
        }, MODEL_STORAGE_PATH)


model_dict = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
}


if __name__ == "__main__":

    opt = downstream_parse_option()

    model = model_dict[opt.model_name](zero_init_residual=True, 
                            in_channel=3, global_pool_kernerl=1, num_down=opt.num_down, num_classes=opt.num_classes, 
                            tips=opt.tips, avgpool=opt.avgpool, maxpool=opt.maxpool, aps=opt.aps, blurpool=opt.blurpool, 
                            blur_kernel_size=3, soft_kernel=3, stride=2, aps_p=2, pad_type='reflect', 
                            max_shift_h=opt.H//4, max_shift_w=opt.W//4, 
                            transform_type='circular', return_soft_polyphase_indices=True)

    criterion = torch.nn.CrossEntropyLoss()
    aux_criterion = torch.nn.MSELoss()
    regularizer = Index_Regularize()
    optimizer = set_optimizer(opt, model)
    
    train_loader, _ = get_dataloader(opt.data_path, batch_size=opt.batch_size, 
                                    dataset=opt.dataset, H=opt.H, W=opt.W, 
                                    max_shift_h=opt.H//4, max_shift_w=opt.W//4, 
                                    shuffle=True, datatype='train', 
                                    contrast=False, normalization=True)
    test_loader, _ = get_dataloader(opt.data_path, batch_size=opt.batch_size, 
                                    dataset=opt.dataset, H=opt.H, W=opt.W, 
                                    max_shift_h=opt.H//4, max_shift_w=opt.W//4, 
                                    shuffle=True, datatype='test', 
                                    contrast=False, normalization=True)


    torch.autograd.set_detect_anomaly(True)
    train(opt, train_loader, test_loader, model, criterion, aux_criterion, regularizer, 
        optimizer, test_max_shift_h=opt.H//4, test_max_shift_w=opt.W//4,
        wake_up_at=opt.wake_up_at, 
        device=opt.device, MODEL_PATH=opt.MODEL_PATH)
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
   








