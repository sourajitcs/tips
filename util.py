#Reference codebases:
# https://github.com/HobbitLong/SupContrast/blob/master/util.py
# https://chat.openai.com/ was used to generate a small fraction of this code which we then further adopted and modified

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import linalg as LA

def show_batch(data_loader, aug_label, dataset_name='cifar10', dataset_path='/.data', batch_size=8, contrast=False):
    
    if (dataset=='mnist'):
        data = datasets.MNIST
        mean = (0.1307)
        std = (0.3081)
    elif (dataset=='cifar10'):
        data = datasets.CIFAR10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif (dataset=='cifar100'):
        data = datasets.CIFAR100
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif (dataset=='food101'):
        data = datasets.Food101
        mean = (0.5450, 0.4435, 0.3436)
        std = (0.2302, 0.2409, 0.2387) 
    elif (dataset=='oxford102'):
        data = datasets.Flowers102
        mean = (0.5450, 0.4435, 0.3436)
        std = (0.2302, 0.2409, 0.2387) 
    elif (dataset=='tiny-imagenet'):
        label_path = dataset_path + "/tiny-imagenet-200/classes.txt"
        with open(label_path, "r") as file:
            for idx, line in enumerate(file):
                id2class[idx] = str(line)[:-1]
    else:
        raise NotImplementedError(
                'dataset not supported: {}'.format(dataset))






    examples = next(iter(data_loader))
    inputs, labels = examples
    
    if (contrast):
        n_views = len(inputs)
        print ("Dataset Length: ", len(data_loader), "Augmented Length: ", n_views, 
               inputs[0].shape, inputs[0].dtype, labels.shape, labels.dtype)
        named_labels = [id2class[label.item()] for label in labels]
        aug_labels = [id2aug_class[cur_aug_label] for cur_aug_label in aug_label]
        
        fig,ax=plt.subplots(nrows=n_views, ncols=batch_size, 
                            figsize=(4*batch_size,n_views*4))
        
        for current_view in range(n_views):
            for current_batch in range(batch_size):
                ax[current_view][current_batch].imshow(transforms.ToPILImage()(inputs[current_view][current_batch].type(torch.FloatTensor)))
                if(current_view==0):
                    ax[current_view][current_batch].set_title(named_labels[current_batch], 
                                                             fontsize=20)
                if(current_batch==0):
                    ax[current_view][current_batch].set_ylabel(aug_labels[current_view], 
                                                              fontsize=20)
                ax[current_view][current_batch].set_xticks([])
                ax[current_view][current_batch].set_yticks([]) 
                
    else:
        print ("Dataset Length: ", len(data_loader), 
               inputs.shape, inputs[0].dtype, labels.shape, labels.dtype)
        named_labels = [id2class[label.item()] for label in labels]
        aug_labels = [id2aug_class[cur_aug_label] for cur_aug_label in aug_label]
        
        fig,ax=plt.subplots(ncols=batch_size, figsize=(4*batch_size,4))
        for current_batch in range(batch_size):
            ax[current_batch].imshow(transforms.ToPILImage()(inputs[current_batch].type(torch.FloatTensor)))
            ax[current_batch].set_title(named_labels[current_batch], 
                                       fontsize=20)
            if(current_batch==0):
                ax[current_batch].set_ylabel(aug_labels[current_batch], fontsize=20)
            ax[current_batch].set_xticks([])
            ax[current_batch].set_yticks([])
    
    plt.tight_layout()
    


class Index_Regularize(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(Index_Regularize, self).__init__()
        self.ignore_label = ignore_label
    
    def forward(self, Tau):
        reg = torch.mean(1-LA.vector_norm(Tau, dim = (-1), ord = 2))
        return reg
    

def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(np.greater(epoch, np.array([args.lr_decay_epochs.split(',')], 
                                                  dtype=np.int64)))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = torch.flatten(correct[:k]).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

    
class CircularShiftTransform(object):
    
    def __init__(self, torch_function=torch.roll, 
                 H=32, W=32, max_shift_h=4, max_shift_w=4):
        
        self.torch_function = torch_function
        self.H = H
        self.W = W
        self.max_shift_h = max_shift_h
        self.max_shift_w = max_shift_w
        
        self.h_shift = np.random.randint(-self.max_shift_h, self.max_shift_h+1)
        self.w_shift = np.random.randint(-self.max_shift_w, self.max_shift_w+1)
        

    def __call__(self, sample):
        h_shift = np.random.randint(-self.max_shift_h, self.max_shift_h+1)
        w_shift = np.random.randint(-self.max_shift_w, self.max_shift_w+1)
        
        tensor_sample = transforms.ToTensor()(sample)
        transformed_sample = self.torch_function(tensor_sample, 
                                                 shifts = (h_shift, w_shift), 
                                                 dims = (-2, -1))
        
        return transformed_sample
    

class N_CropTransform:
    """Creates N crops of the same image"""
    """Adopted from https://github.com/HobbitLong/SupContrast/blob/master/util.py"""
    def __init__(self, transforms, H, W):
        self.transforms = transforms
        self.H = H
        self.W = W

    def __call__(self, sample):
        sample = transforms.RandomResizedCrop(size=(self.H, self.W), 
                                              scale=(0.7, 1.))(sample)
        #sample = transforms.RandomHorizontalFlip()(sample)
        return [transform(sample) for transform in self.transforms]
    

def compute_class_distribution(dataset, batch_size=256, num_workers=8):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                              shuffle=False, num_workers=num_workers)
    class_counts = {}

    for _, labels in tqdm(data_loader):
        # Count the samples per class
        for label in labels:
            if label.item() in class_counts:
                class_counts[label.item()] += 1
            else:
                class_counts[label.item()] = 1

    return class_counts


def compute_mean_std(dataset, batch_size=256, num_workers=8):
    # Create a data loader with batch size 1 to load images one by one
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                              shuffle=False, num_workers=num_workers)
    mean_sum = torch.zeros(3)
    std_sum = torch.zeros(3)

    num_samples = len(dataset)
    for images, _ in tqdm(data_loader):
        # Flatten the image tensor and calculate mean and std per channel
        images = images.view(images.size(0), images.size(1), -1)
        mean_sum += images.mean(dim=2).sum(dim=0)
        std_sum += images.std(dim=2).sum(dim=0)

    mean = mean_sum / num_samples
    std = std_sum / num_samples

    return mean, std

