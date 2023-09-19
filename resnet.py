#Reference codebases:
# https://github.com/bearpaw/pytorch-classification
# https://github.com/HobbitLong/SupContrast/blob/master/networks/resnet_big.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from blurpool import BlurPool
from aps import APS
from tips import TIPS


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, 
                 soft_kernel=3, pad_type='reflect', is_last=False, 
                 return_soft_polyphase_indices=False, 
                 blurpool=False, blur_kernel_size=3, 
                 tips=False, avgpool=False, maxpool=False, aps=False, aps_p=2, 
                 max_shift_h=4, max_shift_w=4, transform_type='standard'):
        super(BasicBlock, self).__init__()

        self.is_last = is_last
        self.blurpool = blurpool
        self.blur_kernel_size = blur_kernel_size
        self.tips = tips
        self.avgpool = avgpool
        self.maxpool = maxpool
        self.aps = aps
        self.aps_p = aps_p
        self.soft_kernel = soft_kernel
        self.stride = stride
        self.pad_type = pad_type
        self.max_shift_h = max_shift_h
        self.max_shift_w = max_shift_w
        self.transform_type = transform_type
        self.return_soft_polyphase_indices = return_soft_polyphase_indices
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if self.stride != 1:
            if (self.tips):
                self.downsample = TIPS(in_channels=planes, pad_type=self.pad_type, 
                                       kernel=self.soft_kernel, stride=self.stride, 
                                       max_shift_h=self.max_shift_h, 
                                       max_shift_w=self.max_shift_w, 
                                       transform_type=self.transform_type, 
                                       return_soft_polyphase_indices= \
                                       self.return_soft_polyphase_indices)
            elif (self.maxpool):
                self.downsample = nn.MaxPool2d(kernel_size=self.stride)
            elif (self.avgpool):
                self.downsample = nn.AvgPool2d(kernel_size=self.stride) 
            elif (self.blurpool):
                self.downsample = BlurPool(in_channels=planes, 
                                           pad_type=self.pad_type, 
                                           kernel_size=self.blur_kernel_size,
                                           stride=self.stride)
            elif (self.aps):
                self.downsample = APS(pad_type=self.pad_type, stride=self.stride, 
                                      p=self.aps_p)
            else:
                raise ValueError('Pooling operator not supported')
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if self.stride != 1 or in_planes != self.expansion * planes:
            if (self.tips):
                self.shortcut = nn.Sequential(
                    TIPS(in_channels=in_planes, pad_type=self.pad_type, 
                         kernel=self.soft_kernel, stride=self.stride, 
                         max_shift_h=self.max_shift_h, max_shift_w=self.max_shift_w, 
                         transform_type=self.transform_type, 
                         return_soft_polyphase_indices= \
                         self.return_soft_polyphase_indices), 
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, 
                              stride=1, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
            elif (self.maxpool):
                self.shortcut = nn.Sequential(
                    nn.MaxPool2d(kernel_size=self.stride), 
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, 
                              stride=1, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
            elif (self.avgpool):
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=self.stride), 
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, 
                              stride=1, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
            elif (self.blurpool):
                self.shortcut = nn.Sequential(
                    BlurPool(in_channels=in_planes, 
                                           pad_type=self.pad_type, 
                                           kernel_size=self.blur_kernel_size,
                                           stride=self.stride), 
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, 
                              stride=1, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
            elif (self.aps):
                self.shortcut = nn.Sequential(
                    APS(pad_type=self.pad_type, stride=self.stride, 
                                      p=self.aps_p), 
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, 
                              stride=1, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
            else:
                raise ValueError('Pooling operator not supported')
                
    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        if self.stride != 1:
            if self.tips:
                if (self.return_soft_polyphase_indices):
                    out, psi_x1, x_t1, tau1 = self.downsample(out)
                else:
                    out, psi_x1, x_t1 = self.downsample(out)
            else:
                out = self.downsample(out)
        out = self.bn2(self.conv2(out))
        if (len(self.shortcut) == 3):
            if (self.tips):
                if (self.return_soft_polyphase_indices):
                    x, psi_x2, x_t2, tau2 = self.shortcut[0](x)
                else:
                    x, psi_x2, x_t2 = self.shortcut[0](x)
            else:
                x = self.shortcut[0](x)
            x = self.shortcut[1](x)
            x = self.shortcut[2](x)
            out += x
        else:
            out += self.shortcut(x)
        preact = out
        out = self.relu(out)
        if (self.stride != 1 and self.tips):
            psi_x = torch.cat([psi_x1, psi_x2], dim=1)
            x_t = torch.cat([x_t1, x_t2], dim=1)
        if self.is_last:
            return out, preact
        else:
            if (self.tips):
                if (self.return_soft_polyphase_indices):
                    return out, psi_x, x_t, tau1, tau2
                else:
                    return out, psi_x, x_t
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, 
                 soft_kernel=3, pad_type='reflect', is_last=False, 
                 return_soft_polyphase_indices=False, 
                 blurpool=False, blur_kernel_size=3, 
                 tips=False, avgpool=False, maxpool=False, aps=False, aps_p=2, 
                 max_shift_h=4, max_shift_w=4, transform_type='standard'):
        super(Bottleneck, self).__init__()
        
        self.is_last = is_last
        self.blurpool = blurpool
        self.blur_kernel_size = blur_kernel_size
        self.tips = tips
        self.avgpool = avgpool
        self.maxpool = maxpool
        self.aps = aps
        self.aps_p = aps_p
        self.soft_kernel = soft_kernel
        self.stride = stride
        self.pad_type = pad_type
        self.max_shift_h = max_shift_h
        self.max_shift_w = max_shift_w
        self.transform_type = transform_type
        self.return_soft_polyphase_indices = return_soft_polyphase_indices
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if self.stride != 1:
            if (self.tips):
                self.downsample = TIPS(in_channels=planes, pad_type=self.pad_type, 
                                       kernel=self.soft_kernel, stride=self.stride, 
                                       max_shift_h=self.max_shift_h, 
                                       max_shift_w=self.max_shift_w, 
                                       transform_type=self.transform_type, 
                                       return_soft_polyphase_indices= \
                                       self.return_soft_polyphase_indices)
            elif (self.maxpool):
                self.downsample = nn.MaxPool2d(kernel_size=self.stride)
            elif (self.avgpool):
                self.downsample = nn.AvgPool2d(kernel_size=self.stride) 
            elif (self.blurpool):
                self.downsample = BlurPool(in_channels=planes, 
                                           pad_type=self.pad_type, 
                                           kernel_size=self.blur_kernel_size,
                                           stride=self.stride)
            elif (self.aps):
                self.downsample = APS(pad_type=self.pad_type, stride=self.stride, 
                                      p=self.aps_p)
            else:
                raise ValueError('Pooling operator not supported')
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, 
                               bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu = nn.ReLU()
        
        self.shortcut = nn.Sequential()
        if self.stride != 1 or in_planes != self.expansion * planes:
            if (self.tips):
                self.shortcut = nn.Sequential(
                    TIPS(in_channels=in_planes, pad_type=self.pad_type, 
                         kernel=self.soft_kernel, stride=self.stride, 
                         max_shift_h=self.max_shift_h, max_shift_w=self.max_shift_w, 
                         transform_type=self.transform_type, 
                         return_soft_polyphase_indices= \
                         self.return_soft_polyphase_indices), 
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, 
                              stride=1, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
            elif (self.maxpool):
                self.shortcut = nn.Sequential(
                    nn.MaxPool2d(kernel_size=self.stride), 
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, 
                              stride=1, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
            elif (self.avgpool):
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=self.stride), 
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, 
                              stride=1, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
            elif (self.blurpool):
                self.shortcut = nn.Sequential(
                    BlurPool(in_channels=in_planes, 
                                           pad_type=self.pad_type, 
                                           kernel_size=self.blur_kernel_size,
                                           stride=self.stride), 
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, 
                              stride=1, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
            elif (self.aps):
                self.shortcut = nn.Sequential(
                    APS(pad_type=self.pad_type, stride=self.stride, 
                                      p=self.aps_p), 
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, 
                              stride=1, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
            else:
                raise ValueError('Pooling operator not supported')
                
        
    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        out = self.relu(out)
        if self.stride != 1:
            if self.tips:
                if (self.return_soft_polyphase_indices):
                    out, psi_x1, x_t1, tau1 = self.downsample(out)
                else:
                    out, psi_x1, x_t1 = self.downsample(out)
            else:
                out = self.downsample(out)
        out = self.bn3(self.conv3(out))
        if (len(self.shortcut) == 3):
            if (self.tips):
                if (self.return_soft_polyphase_indices):
                    x, psi_x2, x_t2, tau2 = self.shortcut[0](x)
                else:
                    x, psi_x2, x_t2 = self.shortcut[0](x)
            else:
                x = self.shortcut[0](x)
            x = self.shortcut[1](x)
            x = self.shortcut[2](x)
            out += x
        else:
            out += self.shortcut(x)
        preact = out
        out = self.relu(out)
        if (self.stride != 1 and self.tips):
            psi_x = torch.cat([psi_x1, psi_x2], dim=1)
            x_t = torch.cat([x_t1, x_t2], dim=1)
        if self.is_last:
            return out, preact
        else:
            if (self.tips):
                if (self.return_soft_polyphase_indices):
                    return out, psi_x, x_t, tau1, tau2
                else:
                    return out, psi_x, x_t
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, global_pool_kernerl=1, 
                 zero_init_residual=False, num_down=3, num_classes=10, 
                 blurpool=False, blur_kernel_size=3, 
                 tips=False, avgpool=False, maxpool=False, aps=False, aps_p=2, 
                 soft_kernel=3, stride=2, pad_type='reflect', 
                 max_shift_h=4, max_shift_w=4, transform_type='standard', 
                 return_soft_polyphase_indices=False, sanity_check=False):
        super(ResNet, self).__init__()
        
        self.in_planes = 64
        self.global_pool_kernerl = global_pool_kernerl
        self.num_down = num_down
        self.num_classes = num_classes
        if (block.__name__=='Bottleneck'):
            self.fc_len = 2048*self.global_pool_kernerl**2
        elif (block.__name__=='BasicBlock'):
            self.fc_len = 512*self.global_pool_kernerl**2
        self.blurpool = blurpool
        self.blur_kernel_size = blur_kernel_size
        self.tips = tips
        self.avgpool = avgpool
        self.maxpool = maxpool
        self.aps = aps
        self.aps_p = aps_p
        self.soft_kernel = soft_kernel
        self.stride = stride
        self.pad_type = pad_type
        self.max_shift_h = max_shift_h
        self.max_shift_w = max_shift_w
        self.transform_type = transform_type
        self.return_soft_polyphase_indices = return_soft_polyphase_indices
        self.sanity_check = sanity_check
        if (self.stride % 2==1):
            raise ValueError('Odd strides are NOT supported, found {}'.\
                             format(self.stride))
        if (self.num_down>5):
            raise ValueError('Max 5 downsampling are supported, found {}'.\
                             format(self.num_down))
        if (block.__name__=='BasicBlock' and self.num_down<3):
            raise ValueError('Min 3 downsampling are supported for block type {}, found {}'.\
                             format(block.__name__, self.num_down))
        if (block.__name__=='Bottleneck' and self.num_down<4):
            raise ValueError('Min 4 downsampling are supported for block type {}, found {}'. \
                             format(block.__name__, self.num_down))
        
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, 
                                   stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        if (self.num_down==5):
            if (self.tips):
                self.downsample = TIPS(in_channels=64, pad_type=self.pad_type, 
                                       kernel=self.soft_kernel, stride=self.stride, 
                                       max_shift_h=self.max_shift_h, 
                                       max_shift_w=self.max_shift_w, 
                                       transform_type=self.transform_type, 
                                       return_soft_polyphase_indices= \
                                       self.return_soft_polyphase_indices)
            elif (self.maxpool):
                self.downsample = nn.MaxPool2d(kernel_size=self.stride)
            elif (self.avgpool):
                self.downsample = nn.AvgPool2d(kernel_size=self.stride)
            elif (self.blurpool):
                self.downsample = BlurPool(in_channels=64, 
                                           pad_type=self.pad_type, 
                                           kernel_size=self.blur_kernel_size,
                                           stride=self.stride)
            elif (self.aps):
                self.downsample = APS(pad_type=self.pad_type, stride=self.stride, 
                                      p=self.aps_p)
            else:
                raise ValueError('Pooling operator NOT supported')
        if (self.num_down>=4):
            self.layer1 = self._make_layer(block, 64, num_blocks[0],self.soft_kernel, 
                                       self.stride, self.pad_type, 
                                       self.max_shift_h, self.max_shift_w, 
                                       self.transform_type, self.tips, self.avgpool, 
                                       self.maxpool, self.aps, self.aps_p, 
                                       self.blurpool, self.blur_kernel_size, 
                                       self.return_soft_polyphase_indices)
        else:
            self.layer1 = self._make_layer(block, 64, num_blocks[0],self.soft_kernel, 
                                       1, self.pad_type, 
                                       self.max_shift_h, self.max_shift_w, 
                                       self.transform_type, False, False, 
                                       False, False, self.aps_p, False, 
                                       self.blur_kernel_size, False)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], self.soft_kernel, 
                                       self.stride, self.pad_type, 
                                       self.max_shift_h, self.max_shift_w, 
                                       self.transform_type, self.tips, self.avgpool, 
                                       self.maxpool, self.aps, self.aps_p, 
                                       self.blurpool, self.blur_kernel_size, 
                                       self.return_soft_polyphase_indices)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], self.soft_kernel, 
                                       self.stride, self.pad_type, 
                                       self.max_shift_h, self.max_shift_w, 
                                       self.transform_type, self.tips, self.avgpool, 
                                       self.maxpool, self.aps, self.aps_p, 
                                       self.blurpool, self.blur_kernel_size, 
                                       self.return_soft_polyphase_indices)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], self.soft_kernel, 
                                       self.stride, self.pad_type, 
                                       self.max_shift_h, self.max_shift_w, 
                                       self.transform_type, self.tips, self.avgpool, 
                                       self.maxpool, self.aps, self.aps_p, 
                                       self.blurpool, self.blur_kernel_size, 
                                       self.return_soft_polyphase_indices)
        self.avgpool = nn.AdaptiveAvgPool2d(self.global_pool_kernerl)
        self.fc = nn.Linear(self.fc_len, self.num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, soft_kernel, stride, pad_type, 
                    max_shift_h, max_shift_w, transform_type, tips, avgpool, maxpool, 
                    aps, aps_p, blurpool, blur_kernel_size, 
                    return_soft_polyphase_indices):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            if (stride!=1):
                layers.append(block(self.in_planes, planes, stride=stride, 
                                    soft_kernel=soft_kernel, pad_type=pad_type, 
                                    max_shift_h=max_shift_h, 
                                    max_shift_w=max_shift_w, 
                                    transform_type=transform_type,
                                    tips=tips, avgpool=avgpool, maxpool=maxpool, 
                                    aps=aps, aps_p=aps_p, blurpool=blurpool, 
                                    blur_kernel_size=blur_kernel_size, 
                                    return_soft_polyphase_indices= \
                                    return_soft_polyphase_indices))
                self.in_planes = planes * block.expansion
            else:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        Tau = []
        PSI_X = []
        X_T = []
        if (self.sanity_check):
            print ("$$$     inp: ", x.shape)
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        if (self.num_down==5):
            if (self.tips):
                if (self.return_soft_polyphase_indices):
                    out, psi_x, x_t, tau = self.downsample(out)
                    PSI_X.append(psi_x)
                    X_T.append(x_t)
                    Tau.append(tau)
                else:
                    out, psi_x, x_t = self.downsample(out)
                    PSI_X.append(psi_x)
                    X_T.append(x_t)
            else:
                out = self.downsample(out)
        if (self.sanity_check):
            print ("$$$ layer 0: ", out.shape)
        if (self.num_down>=4):
            for layer_number, layer in enumerate(self.layer1):
                if (layer_number==0):
                    if (self.tips):
                        if (self.return_soft_polyphase_indices):
                            out, psi_x, x_t, tau1, tau2 = layer(out)
                            PSI_X.append(psi_x)
                            X_T.append(x_t)
                            Tau.append(tau1)
                            Tau.append(tau2)
                        else:
                            out, psi_x, x_t = layer(out)
                            PSI_X.append(psi_x)
                            X_T.append(x_t)
                    else:
                        out = layer(out)
                else:
                    out = layer(out)
        else:
            out = self.layer1(out)
        if (self.sanity_check):
            print ("$$$ layer 1: ", out.shape)
        for layer_number, layer in enumerate(self.layer2):
            if (layer_number==0):
                if (self.tips):
                    if (self.return_soft_polyphase_indices):
                        out, psi_x, x_t, tau1, tau2 = layer(out)
                        PSI_X.append(psi_x)
                        X_T.append(x_t)
                        Tau.append(tau1)
                        Tau.append(tau2)
                    else:
                        out, psi_x, x_t = layer(out)
                        PSI_X.append(psi_x)
                        X_T.append(x_t)
                else:
                    out = layer(out)
            else:
                out = layer(out)
        if (self.sanity_check):
            print ("$$$ layer 2: ", out.shape)
        for layer_number, layer in enumerate(self.layer3):
            if (layer_number==0):
                if (self.tips):
                    if (self.return_soft_polyphase_indices):
                        out, psi_x, x_t, tau1, tau2 = layer(out)
                        PSI_X.append(psi_x)
                        X_T.append(x_t)
                        Tau.append(tau1)
                        Tau.append(tau2)
                    else:
                        out, psi_x, x_t = layer(out)
                        PSI_X.append(psi_x)
                        X_T.append(x_t)
                else:
                    out = layer(out)
            else:
                out = layer(out)
        if (self.sanity_check):
            print ("$$$ layer 3: ", out.shape)
        for layer_number, layer in enumerate(self.layer4):
            if (layer_number==0):
                if (self.tips):
                    if (self.return_soft_polyphase_indices):
                        out, psi_x, x_t, tau1, tau2 = layer(out)
                        PSI_X.append(psi_x)
                        X_T.append(x_t)
                        Tau.append(tau1)
                        Tau.append(tau2)
                    else:
                        out, psi_x, x_t = layer(out)
                        PSI_X.append(psi_x)
                        X_T.append(x_t)
                else:
                    out = layer(out)
            else:
                out = layer(out)
        if (self.sanity_check):
            print ("$$$ layer 4: ", out.shape)
        out = self.avgpool(out)
        if (self.sanity_check):
            print ("$$$ avgpool: ", out.shape)
        out = torch.flatten(out, 1)
        if (self.sanity_check):
            print ("$$$ flatten: ", out.shape)
        out = self.fc(out)
        if (self.sanity_check):
            print ("$$$     cls: ", out.shape)
        
        if (self.tips):
            if (self.return_soft_polyphase_indices):
                tau = torch.cat(Tau, dim=1)
                return out, PSI_X, X_T, tau
            else:
                return out, PSI_X, X_T
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)



