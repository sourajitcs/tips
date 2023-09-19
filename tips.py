#Reference codebases:
# https://github.com/adobe/antialiased-cnns/blob/master/antialiased_cnns/blurpool.py
# https://github.com/achaman2/truly_shift_invariant_cnns/blob/main/models/aps_models/apspool.py
# https://chat.openai.com/ was used to generate a small fraction of this code which we then further adopted and modified

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer


class feat_transform(nn.Module):
    def __init__(self, max_shift_h=2, max_shift_w=2, 
                 transform_type='standard', 
                 shift_direction_is_rand=True, 
                 shift_amount_is_rand=True):
        super(feat_transform, self).__init__()
        
        self.max_shift_h = max_shift_h
        self.max_shift_w = max_shift_w
        self.transform_type = transform_type
        self.shift_direction_is_rand = shift_direction_is_rand
        self.shift_amount_is_rand = shift_amount_is_rand

    def forward(self, feat):
        
        shift_h = np.random.randint(-self.max_shift_h, self.max_shift_h+1)
        shift_w = np.random.randint(-self.max_shift_w, self.max_shift_w+1)
        
        if (self.shift_direction_is_rand and not(self.shift_amount_is_rand)):
            shift_h = int(math.copysign(shift_h, self.max_shift_h))
            shift_w = int(math.copysign(shift_w, self.max_shift_w))
        elif (not(self.shift_direction_is_rand) and self.shift_amount_is_rand):
            shift_h = int(math.copysign(self.max_shift_h, shift_h))
            shift_w = int(math.copysign(self.max_shift_w, shift_w))
        elif (not(self.shift_direction_is_rand) and not(self.shift_amount_is_rand)):
            shift_h = self.max_shift_h
            shift_w = self.max_shift_w
        
        if (self.transform_type=='standard'):
            # vertical shift along y-axis
            shifted_feat = torch.roll(feat, shifts=shift_h, dims=-2)
            if shift_h > 0:
                shifted_feat[:, :, :shift_h, :] = 0
            elif shift_h < 0:
                shifted_feat[:, :, shift_h:, :] = 0
            # horizontal shift along x-axis
            shifted_feat = torch.roll(shifted_feat, shifts=shift_w, dims=-1)
            if shift_w > 0:
                shifted_feat[:, :, :, :shift_w] = 0
            elif shift_w < 0:
                shifted_feat[:, :, :, shift_w:] = 0
            return shifted_feat
        elif (self.transform_type=='circular'):
            shifted_feat = torch.roll(feat, shifts=(shift_h, shift_w), dims=(-2,-1))
            return shifted_feat
        else:
            raise ValueError('Unknown transform type: {}'.format(self.transform_type))


class TIPS(nn.Module):
    def __init__(self, in_channels, num_poly=4, 
                 pad_type='reflect', kernel=3, stride=2, 
                 max_shift_h=2, max_shift_w=2, 
                 transform_type='standard', 
                 return_soft_polyphase_indices=False, sanity_check=False):
        super(TIPS, self).__init__()
        
        self.pad_sizes_ee = [int(0), int(0), 
                             int(0), int(0)]
        self.pad_ee = get_pad_layer(pad_type)(self.pad_sizes_ee)
        self.pad_sizes_eo = [int(1), int(0), 
                             int(0), int(0)]
        self.pad_eo = get_pad_layer(pad_type)(self.pad_sizes_eo)
        self.pad_sizes_oe = [int(0), int(0), 
                             int(1), int(0)]
        self.pad_oe = get_pad_layer(pad_type)(self.pad_sizes_oe)
        self.pad_sizes_oo = [int(1), int(0), 
                             int(1), int(0)]
        self.pad_oo = get_pad_layer(pad_type)(self.pad_sizes_oo)
        
        self.in_channels = in_channels
        self.kernel = kernel
        self.stride = stride
        self.num_poly = num_poly
        if (self.num_poly is None or self.num_poly==0):
            self.num_poly = self.stride*self.stride
        self.max_shift_h = max_shift_h
        self.max_shift_w = max_shift_w
        self.transform_type = transform_type
        self.return_soft_polyphase_indices = return_soft_polyphase_indices
        self.sanity_check = sanity_check
        
        self.trans = feat_transform(max_shift_h=self.max_shift_h, 
                                    max_shift_w=self.max_shift_w, 
                                    transform_type=self.transform_type)
        
        self.conv1 = nn.Conv2d(self.in_channels, self.in_channels, 
                               kernel_size=self.kernel, groups=self.in_channels, 
                               padding = int((self.kernel-1)/2.), 
                               stride=1, bias=False)
        
        self.relu1 = nn.ReLU(inplace=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d((self.stride, self.stride))
        
        self.conv2 = nn.Conv2d(self.in_channels, self.in_channels, 
                               kernel_size=1, groups=self.in_channels, 
                               stride=1, bias=False)
        
        self.softmax = nn.Softmax(dim=-1)
        
        
    def forward(self, x):
        N, C, H, W = x.shape
        if (H%2==0 and W%2==0):
            x = self.pad_ee(x)
        elif (H%2==0 and W%2!=0):
            x = self.pad_eo(x)
        elif (H%2!=0 and W%2==0):
            x = self.pad_oe(x)
        elif (H%2!=0 and W%2!=0):
            x = self.pad_oo(x)
        N, C, H, W = x.shape
        
        # f_theta
        x_t = self.trans(x) ### x_t
        x_t.detach() ### Freeze 
        
        x_w = self.conv1(x)
        psi_x = self.relu1(x_w) ### psi(x)
        
        x_w = self.avgpool(psi_x)
        
        x_w = self.conv2(x_w)

        tau = x_w.view(N, C, -1)

        tau = self.softmax(tau)
        
        # polyphase decomposition
        xpoly_0 = x[:, :, 0::self.stride, 0::self.stride]
        xpoly_1 = x[:, :, 1::self.stride, 0::self.stride]
        xpoly_2 = x[:, :, 0::self.stride, 1::self.stride]
        xpoly_3 = x[:, :, 1::self.stride, 1::self.stride]
        if (self.sanity_check):
            print ("xpoly_0: ", xpoly_0.shape, '\n', xpoly_0)
            print ("xpoly_1: ", xpoly_1.shape, '\n',  xpoly_1)
            print ("xpoly_2: ", xpoly_2.shape, '\n',  xpoly_2)
            print ("xpoly_3: ", xpoly_3.shape, '\n',  xpoly_3)


        xpoly_stacks = torch.stack([xpoly_0, xpoly_1, xpoly_2, xpoly_3], dim=2)
        xpoly_stacks = xpoly_stacks.view(N, C, self.num_poly, 
                                         H//self.stride*W//self.stride)
        
        # dot product [ tau, poly(x) ]
        soft_polyphase = (tau.reshape(N, C, self.num_poly, 1) \
                          * xpoly_stacks).view(N, C, self.num_poly, 
                                               H//self.stride, W//self.stride)
        soft_polyphase = torch.sum(soft_polyphase, dim = 2)
        
        if (self.return_soft_polyphase_indices):
            return soft_polyphase, psi_x, x_t 
        else:
            return soft_polyphase, psi_x, x_t 

