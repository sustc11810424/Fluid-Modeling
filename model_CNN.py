import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.nn.utils import spectral_norm, weight_norm

class ResNeXt(nn.Module):
    def __init__(self, dim, cardinality):
        super(ResNeXt, self).__init__()
        D = 4
        self.layers = nn.Sequential(
            nn.Conv2d(dim, D*cardinality, 1),
            nn.LeakyReLU(),
            nn.Conv2d(D*cardinality, D*cardinality, 3, padding=1, groups=cardinality),
            nn.LeakyReLU(),
            nn.Conv2d(D*cardinality, dim, 1),
            nn.LeakyReLU(),
        )
    def forward(self, x):
        return x + self.layers(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(Upsample, self).__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, out_channels*4, kernel_size=5, padding=2, padding_mode='replicate', groups=groups) ,
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
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2, padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2, padding_mode='replicate'),
            nn.LeakyReLU(),
        ) # 16 * downsample
        self.hidden = nn.Sequential(
            ResNeXt(dim=64, cardinality=32),
            ResNeXt(dim=64, cardinality=32),
            ResNeXt(dim=64, cardinality=32),
        ) # at least 32*32
        self.decode = nn.Sequential(
            Upsample(64, 64),
            Upsample(64, 32),
            Upsample(32, 32),
            Upsample(32, 16),
            nn.Conv2d(16, 16, 3, 1, 1, padding_mode='replicate'),
            nn.PReLU(16),
            nn.Conv2d(16, 16, 3, 1, 1, padding_mode='replicate'),
            nn.PReLU(16),
            nn.Conv2d(16, out_dim, 1),
        )
        
    def forward(self, x: torch.Tensor):
        if len(x.size())<4: x.unsqueeze_(0)
        x = self.encode(x)
        x = self.hidden(x)
        x = self.decode(x)
        return x

class Deformable(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(Deformable, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=7, stride=2, padding=3, padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2, padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2, padding_mode='replicate'),
            nn.LeakyReLU(),
        ) # 16 * downsample
        self.hidden = nn.Sequential(
            ResNeXt(dim=64, cardinality=32),
            ResNeXt(dim=64, cardinality=32),
            ResNeXt(dim=64, cardinality=32),
        ) # at least 32*32
        self.decode = nn.Sequential(
            Upsample(64, 64),
            Upsample(64, 32),
            Upsample(32, 32),
            Upsample(32, 16),
            nn.Conv2d(16, 16, 3, 1, 1, padding_mode='replicate'),
            nn.PReLU(16),
            nn.Conv2d(16, 16, 3, 1, 1, padding_mode='replicate'),
            nn.PReLU(16),
            nn.Conv2d(16, out_dim, 1),
        )
        
    def forward(self, x: torch.Tensor):
        if len(x.size())<4: x.unsqueeze_(0)
        x = self.encode(x)
        x = self.hidden(x)
        x = self.decode(x)
        return x

from torch.nn.utils import spectral_norm, weight_norm
class ResBlock(nn.Module):
    def __init__(self, num_channels, groups=1, bias=False):
        super(ResBlock, self).__init__()
        self.blocks = nn.Sequential(
            nn.LeakyReLU(),
            spectral_norm(nn.Conv2d(num_channels, num_channels,kernel_size=3,padding=1, bias=bias, groups=groups)) ,
            nn.LeakyReLU(),
            spectral_norm(nn.Conv2d(num_channels, num_channels,kernel_size=3,padding=1, bias=bias, groups=groups)) ,
        )

    def forward(self, x):
        out = self.blocks(x)
        return x + out

class ConvTr(nn.Module):
    def __init__(self, in_channels, out_channels, type="upsample", activ="lrelu", groups=1):
        super(ConvTr, self).__init__()
        if type=="upsample":
            self.blocks = nn.Sequential(
                nn.Upsample(scale_factor=2),
                spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, padding_mode='replicate', groups=groups)) ,
            )
        if activ=="lrelu":
            self.blocks.add_module('activ', nn.LeakyReLU())
    def forward(self, x):
        return self.blocks(x)

class GenBaseline(nn.Module):
    def __init__(self):
        super(GenBaseline, self).__init__()
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, padding=3)) ,
            nn.LeakyReLU(),
            spectral_norm(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2, padding_mode='replicate')) ,
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, padding_mode='replicate')) ,
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2, padding_mode='replicate')) ,
            nn.InstanceNorm2d(64),
        )

        self.res1 = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU()
        )

        self.convT1 = ConvTr(64, 64)
        self.convT2 = ConvTr(2*64, 64)    
        self.convT3 = ConvTr(64+32, 64) 

        self.final = nn.Sequential(
            nn.InstanceNorm2d(64),
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=32,kernel_size=3,stride=1,padding=1, padding_mode='replicate')),
            nn.LeakyReLU(),
            spectral_norm(nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3,stride=1,padding=1, padding_mode='replicate')),
            nn.LeakyReLU(),
            spectral_norm(nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1)),
            
        )
    def forward(self, x):
        if len(x.size())<4: x.unsqueeze_(0)
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        x = self.conv3(out2)
        x = self.res1(x)
        x = self.convT1(x)
        x = self.convT2(torch.cat([x, out2], dim=1))
        x = self.convT3(torch.cat([x, out1], dim=1))
        x = self.final(x)
        return x

class Gen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, padding=3)) ,
            nn.LeakyReLU(),
            spectral_norm(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2, padding_mode='replicate')) ,
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, padding_mode='replicate')) ,
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2, padding_mode='replicate')) ,
            nn.InstanceNorm2d(32),
        )

        self.res1 = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU()
        )

        self.convT1 = Upsample(64, 64)
        self.convT2 = Upsample(2*64, 64)    
        self.convT3 = Upsample(64+32, 64) 

        self.final = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=32,kernel_size=3,stride=1,padding=1, padding_mode='replicate')),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(),
            spectral_norm(nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3,stride=1,padding=1, padding_mode='replicate')),
            nn.LeakyReLU(),
            spectral_norm(nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1)),
            
        )
    def forward(self, x):
        if len(x.size())<4: x.unsqueeze_(0)
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        x = self.conv3(out2)
        x = self.res1(x)
        x = self.convT1(x)
        x = self.convT2(torch.cat([x, out2], dim=1))
        x = self.convT3(torch.cat([x, out1], dim=1))
        x = self.final(x)
        return x
