import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, GATConv
from torch_geometric.data import Data
from collections import OrderedDict
import pytorch_lightning as pl
import os
import numpy as np
from utils import get_mesh_graph, plot_scalar_field
from torch.nn.utils import spectral_norm, weight_norm

class AbstractModel(nn.Module):
    """
    Abstract class
    """
    def __init__(self):
        super(CNNModel, self).__init__()
        pass

class ExampleCNN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(ExampleCNN, self).__init__()
        self.layers = nn.ModuleList([nn.Conv2d(in_dim, 8, 5, padding=2), nn.LeakyReLU()])
        for i in range(2): 
            self.layers.append(nn.Conv2d(8, 8, 3, padding=1))
            self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Conv2d(8, out_dim, 3, padding=1))
        
    def forward(self, x: torch.Tensor):
        if len(x.size())<4: x.unsqueeze_(0)
        for layer in self.layers:
            x = layer(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

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
            self.blocks.add_module('activ', nn.LeakyReLU(inplace=True))
    def forward(self, x):
        return self.blocks(x)

class Gen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, padding=3)) ,
            nn.LeakyReLU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1, padding_mode='replicate')) ,
            nn.LeakyReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, padding_mode='replicate')) ,
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, padding_mode='replicate')) ,
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
            nn.LeakyReLU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3,stride=1,padding=1, padding_mode='replicate')),
            nn.LeakyReLU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1)),
            nn.Tanh(),
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

class ExampleGNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers=3):
        super(ExampleGNN, self).__init__()
        self.encoder = GCNConv(in_dim, hidden_dim)
        self.hidden = nn.Sequential(
            OrderedDict([(f'hidden block{i}', GCNConv(hidden_dim, hidden_dim)) for i in range(num_layers)])
        )
        self.decoder = GCNConv(hidden_dim, out_dim)
    
    def forward(self, data):
        x, edge_index, _ = data.x, data.edge_index, data.edge_attr
        x = self.encoder(x, edge_index)
        x = F.leaky_relu(x)
        for layer in self.hidden:
            x = F.leaky_relu(layer(x, edge_index))
        x = self.decoder(x, edge_index)
        return x

class GATGNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers=3, heads=8):
        super(GATGNN, self).__init__()
        self.encoder = GATConv(in_dim, hidden_dim, heads)
        self.hidden = nn.Sequential(
            OrderedDict([(f'hidden block{i}', GATConv(heads * hidden_dim, hidden_dim, heads)) for i in range(num_layers)])
        )
        self.decoder = GATConv(heads * hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, _ = data.x, data.edge_index, data.edge_attr
        x = self.encoder(x, edge_index)
        x = F.leaky_relu(x)
        for layer in self.hidden:
            x = layer(x, edge_index)
        x = self.decoder(x, edge_index)
        return x

class FeedForward(pl.LightningModule):
    def __init__(self, model):
        super(FeedForward, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y, afs = batch
        solution = self.model(x)
        
        loss = F.mse_loss(solution, y.x if isinstance(y, Data) else y)

        # TODO a better logger
        self.log('training_loss', loss)
        if batch_idx==0: 
            x, y, af = self.trainer.datamodule.get_example()
            x = x.to(self.device)
            with torch.no_grad():
                if isinstance(x, Data):
                    nodes, edges, elems, marker_dict = self.trainer.datamodule.mesh_dict[af]
                    figure = plot_scalar_field(torch.stack((self.model(x).detach().cpu().T, y.x.T)), X=nodes.T, tri=elems[0]) 
                else:
                    figure = plot_scalar_field(torch.cat((self.model(x).detach().cpu(), y[None])))
                self.logger.experiment.add_figure(
                    f'Epoch{self.current_epoch}:', 
                    figure,
                )
        return loss

    def test_step(self, batch, batch_idx):
        x, y, other = batch
        solution = self.model(x)
        loss = F.mse_loss(solution, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
            'monitor': 'training_loss',
        }

