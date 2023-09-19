#Reference codebases:
# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from blurpool import BlurPool
from aps import APS
from tips import TIPS

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, down_type='TIPS'):
        super().__init__()
        self.down_type = down_type
        if (self.down_type=='maxpool'):
            self.down_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
            )
        elif (self.down_type=='avgpool'):
            self.down_conv = nn.Sequential(
                nn.AvgPool2d(2),
                DoubleConv(in_channels, out_channels)
            )
        elif (self.down_type=='TIPS'):
            self.down_conv = nn.Sequential(
                TIPS(in_channels, pad_type='reflect', 
                    kernel=3, stride=2, 
                    max_shift_h=50, 
                    max_shift_w=50, 
                    transform_type='standard', 
                    return_soft_polyphase_indices= False),
                DoubleConv(in_channels, out_channels)
            )
        elif (self.down_type=='APS'):
            self.down_conv = nn.Sequential(
                APS(pad_type='reflect', stride=2, p=2),
                DoubleConv(in_channels, out_channels)
            )
        elif (self.down_type=='blurpool'):
            self.down_conv = nn.Sequential(
                BlurPool(in_channels, pad_type='reflect', kernel_size=3, stride=2),
                DoubleConv(in_channels, out_channels)
            )
        

    def forward(self, x):
        if (self.down_type=='TIPS'):
            temp, _, _ = self.down_conv(x)
            return temp
        else:
            return self.down_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
