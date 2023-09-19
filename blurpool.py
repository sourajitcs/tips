#Reference codebases:
# https://github.com/adobe/antialiased-cnns/blob/master/antialiased_cnns/blurpool.py

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


class BlurPool(nn.Module):
    def __init__(self, in_channels, pad_type='reflect', kernel_size=3, stride=2):
        super(BlurPool, self).__init__()
        
        self.in_channels = in_channels
        self.pad_type = pad_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_sizes = [int(1.*(self.kernel_size-1)/2), 
                          int(np.ceil(1.*(self.kernel_size-1)/2)), 
                          int(1.*(self.kernel_size-1)/2), 
                          int(np.ceil(1.*(self.kernel_size-1)/2))]
        self.pad_sizes = [pad_size for pad_size in self.pad_sizes]
        self.pad = get_pad_layer(self.pad_type)(self.pad_sizes)
        
        if(self.kernel_size==1):
            a = np.array([1.,])
        elif(self.kernel_size==2):
            a = np.array([1., 1.])
        elif(self.kernel_size==3):
            a = np.array([1., 2., 1.])
        elif(self.kernel_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.kernel_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.kernel_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.kernel_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])
            
        kernel = torch.Tensor(a[:,None]*a[None,:])
        kernel = kernel/torch.sum(kernel)
        self.register_buffer('kernel', kernel[None,None,:,:]. \
                             repeat((self.in_channels,1,1,1)))
        
    def forward(self, x):
        if(self.kernel_size==1):
            return x[:,:,::self.stride,::self.stride]  
        else:
            return F.conv2d(self.pad(x), self.kernel, stride=self.stride, 
                            groups=x.shape[1])


