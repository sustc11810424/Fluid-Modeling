import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.nn.utils import spectral_norm, weight_norm
# from mmcv.ops import ModulatedDeformConv2dPack

class ResNeXt(nn.Module):
    def __init__(self, dim, cardinality, conv=nn.Conv2d):
        super(ResNeXt, self).__init__()
        D = 4
        self.layers = nn.Sequential(
            conv(dim, D*cardinality, 1),
            nn.PReLU(D*cardinality),
            conv(D*cardinality, D*cardinality, 3, padding=1, groups=cardinality),
            nn.GroupNorm(cardinality, D*cardinality),
            nn.PReLU(D*cardinality),
            conv(D*cardinality, dim, 1),
            nn.PReLU(dim),
        )
    def forward(self, x):
        return x + self.layers(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, conv=nn.Conv2d):
        super(Upsample, self).__init__()
        self.blocks = nn.Sequential(
            conv(in_channels, out_channels*4, kernel_size=5, padding=2, groups=groups) ,
            nn.PixelShuffle(2),
            nn.PReLU(out_channels),
        )
    def forward(self, x):
        return self.blocks(x)

class ExampleCNN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(ExampleCNN, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=7, stride=2, padding=3, padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2, padding=2, padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=2, padding=2, padding_mode='replicate'),
            nn.LeakyReLU(),
            # nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=2, padding=2, padding_mode='replicate'),
            # nn.LeakyReLU(),
            nn.InstanceNorm2d(64),
        ) # 8 * downsample
        self.hidden = nn.Sequential(
            ResNeXt(dim=64, cardinality=32),
            ResNeXt(dim=64, cardinality=32),
            ResNeXt(dim=64, cardinality=32),
        ) # at least 32*32
        self.decode = nn.Sequential(
            # Upsample(64, 64),
            Upsample(64, 32, groups=4),
            Upsample(32, 32, groups=4),
            Upsample(32, 32, groups=4),
            nn.InstanceNorm2d(32),
        )

        self.final = nn.Sequential(
            ResNeXt(dim=32, cardinality=8),
            ResNeXt(dim=32, cardinality=8),
            nn.Conv2d(32, out_dim, 1),
        )
        
    def forward(self, x: torch.Tensor):
        if len(x.size())<4: x.unsqueeze_(0)
        x = self.encode(x)
        x = self.hidden(x)
        x = self.decode(x)
        x = self.final(x)
        return x

class Dcn(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(Dcn, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(),
        ) # 8 * downsample
        self.hidden = nn.Sequential(
            nn.InstanceNorm2d(64),
            ResNeXt(dim=64, cardinality=32),
            # ResNeXt(dim=64, cardinality=32),
            # ResNeXt(dim=64, cardinality=32),
            # ResNeXt(dim=64, cardinality=32),
        ) # at least 32*32
        self.decode = nn.Sequential(
            # ModulatedDeformConv2dPack(64, 64, 3, 1, 1, groups=4),
            nn.PReLU(64),
            Upsample(64, 64, groups=4),
            # ModulatedDeformConv2dPack(64, 64, 3, 1, 1, groups=4),
            nn.PReLU(64),
            Upsample(64, 64, groups=4),
            # ModulatedDeformConv2dPack(64, 64, 3, 1, 1, groups=4),
            nn.PReLU(64),
            Upsample(64, 64, groups=4),
            
            nn.InstanceNorm2d(64),
            nn.PReLU(64),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(64 * 2, 32, 3, 1, 1, groups=4),
            
            nn.InstanceNorm2d(32),
            nn.PReLU(32),
        )
        self.final = nn.Sequential(
            ResNeXt(dim=32, cardinality=8),
            ResNeXt(dim=32, cardinality=8),
            nn.Conv2d(32, out_dim, 1),
        )
        
    def forward(self, x: torch.Tensor):
        if len(x.size())<4: x.unsqueeze_(0)
        z = x[:, 0].unsqueeze(1).repeat(1, 64, 1, 1)
        x = self.encode(x)
        x = self.hidden(x)
        x = self.decode(x)
        x = self.refine(torch.cat([x, z], dim=1))
        x = self.final(x)
        return x

class Gen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, padding=3) ,
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2, padding_mode='replicate'),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, padding_mode='replicate'),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(),
        ) # 8 * downsample

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2, padding_mode='replicate'),
            nn.InstanceNorm2d(32),
        )

        self.res1 = nn.Sequential(
            ResNeXt(dim=64, cardinality=16),
            ResNeXt(dim=64, cardinality=16),
            ResNeXt(dim=64, cardinality=16),
            nn.InstanceNorm2d(64),
        )

        self.convT1 = Upsample(64, 64)
        self.convT2 = Upsample(2*64, 64)    
        self.convT3 = Upsample(64, 64) 

        self.final = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32,kernel_size=3,stride=1,padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3,stride=1,padding=1, padding_mode='replicate'),
            nn.PReLU(),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1),
        )
    def forward(self, x):
        if len(x.size())<4: x.unsqueeze_(0)
        
        out2 = self.conv2(x)
        x = self.conv3(out2)
        x = self.res1(x)
        x = self.convT1(x)
        x = self.convT2(torch.cat([x, out2], dim=1))
        x = self.convT3(x)
        x = self.final(x)
        return x
