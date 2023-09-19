import numpy as np
import torch
from torchvision import transforms, datasets

from util import CircularShiftTransform, N_CropTransform, compute_mean_std



def get_dataloader(dataset_path='./data', batch_size=8, dataset='cifar10', H=32, W=32, max_shift_h=4, max_shift_w=4, 
                   contrast=False, n_aug_views=2, standard_cnt=0, circular_cnt=0, datatype='train', normalization=True, 
                   shuffle=True, num_workers=4, pin_memory=False, worker_init_fn=None, data_loader_seed=None):
    
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
        if (dataset=='mnist'):
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
        if (dataset=='tiny-imagenet'):
            if train_status:
                data_set = datasets.ImageFolder(root=dataset_path+"/train", 
                                                transform=N_CropTransform(data_transforms, H, W))
            else:
                data_set = datasets.ImageFolder(root=dataset_path+"/val", 
                                                transform=N_CropTransform(data_transforms, H, W))
        elif (dataset=='food101') or (dataset=='oxford102'):
            data_set = data(root=dataset_path, split=datatype, 
                            download=True, transform=N_CropTransform(data_transforms, H, W))
        else:
            data_set = data(root=dataset_path, train=train_status, 
                            download=True, transform=N_CropTransform(data_transforms, H, W))
    else:
        data_transforms = data_transform_original
        aug_label = [0]
        if (dataset=='tiny-imagenet'):
            if train_status:
                data_set = datasets.ImageFolder(root=dataset_path+"/train", 
                                                transform=data_transforms)
            else:
                data_set = datasets.ImageFolder(root=dataset_path+"/val", 
                                                transform=data_transforms)
        elif (dataset=='food101') or (dataset=='oxford102'):
            data_set = data(root=dataset_path, split=datatype, 
                            download=True, transform=data_transforms)
        else:
            data_set = data(root=dataset_path, train=train_status, 
                            download=True, transform=data_transforms)
    
    print(data_set.targets)

    data_loader = torch.utils.data.DataLoader(
        data_set, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory, 
        worker_init_fn=worker_init_fn)

    return data_loader, aug_label
