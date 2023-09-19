#Reference codebases:
# https://github.com/achaman2/truly_shift_invariant_cnns/blob/main/models/aps_models/apspool.py

import numpy as np
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


def get_polyphase_indices(x, p):
    """
    argument:
        x : has the form (B, C, 4, N_poly) & N_poly is 2d so x is 5d
            where 4 represents the number of polyphases
            and N_poly is the polyphase of the 2d feature maps
    returns:
        polyphase_indices
    """
    if (p=='infinity'):
        B = x.shape[0]
        C = x.shape[1]
        max_vals = torch.max(x.reshape(B, C, 4, -1).abs(), dim = -1).values
        polyphase_indices = torch.argmax(max_vals, dim = 2)
    elif (p=='non_abs_max'):
        B = x.shape[0]
        C = x.shape[1]
        max_vals = torch.max(x.reshape(B, C, 4, -1), dim = -1).values
        polyphase_indices = torch.argmax(max_vals, dim = 2)
    elif(isinstance(p, (int))):
        norms = torch.norm(x, dim = (-2, -1), p = p)
        if (p>=0):
            polyphase_indices = torch.argmax(norms, dim = 2)
        else:
            polyphase_indices = torch.argmin(norms, dim = 2)
    else:
        raise Exception('Unknown APS criterion')
        
    return polyphase_indices


class APS(nn.Module):
    def __init__(self, pad_type='reflect', num_poly=4, stride=2, p=2):
        super(APS, self).__init__()
        
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
        self.num_poly = num_poly
        self.stride = stride
        self.p = p
        if (self.num_poly is None or self.num_poly==0):
            self.num_poly = self.stride*self.stride
        if (self.stride>2):
            raise Exception('Stride>2 currently not supported')
    
    def forward(self, x):
        if (self.stride==1):
            return x
        
        N, C, H, W = x.shape
        if (H%2==0 and W%2==0):
            x = self.pad_ee(x)
        elif (H%2==0 and W%2!=0):
            x = self.pad_eo(x)
        elif (H%2!=0 and W%2==0):
            x = self.pad_oe(x)
        elif (H%2!=0 and W%2!=0):
            x = self.pad_oo(x)
        
        # polyphase decomposition
        xpoly_0 = x[:, :, 0::self.stride, 0::self.stride]
        xpoly_1 = x[:, :, 1::self.stride, 0::self.stride]
        xpoly_2 = x[:, :, 0::self.stride, 1::self.stride]
        xpoly_3 = x[:, :, 1::self.stride, 1::self.stride]
        x = torch.stack([xpoly_0, xpoly_1, xpoly_2, xpoly_3], dim=2)
        N, C, poly, h, w = x.shape
        
        # get the polyphase index selected with lp-norm criteria
        polyphase_indices = get_polyphase_indices(x, self.p)
        
        # return the polyphase at the selected polyphase index
        x = x.view(-1, poly, h, w)
        polyphase_indices = polyphase_indices.view(-1)
        x = torch.gather(x, 1, polyphase_indices.\
                         unsqueeze(1).unsqueeze(2).unsqueeze(2).\
                         repeat(1, 1, h, w)).squeeze(1).view(N, C, h, w)
        
        return x

