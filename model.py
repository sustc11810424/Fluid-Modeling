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
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=8, stride=2, padding=3, padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=6, stride=2, padding=2, padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=6, stride=2, padding=2, padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=6, stride=2, padding=2, padding_mode='replicate'),
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
        self.learning_rate = 0.001

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y, afs = batch
     
        solution = self.model(x)
        
        loss = F.mse_loss(solution, y.x if isinstance(y, Data) else y)
        
        # TODO a better logger
        self.log('training_loss', loss)
 
        return loss 

    def on_train_epoch_end(self, outputs):
        x, y, af = self.trainer.datamodule.get_example()
        x = x.to(self.device)
        with torch.no_grad():
            if isinstance(x, Data):
                nodes, edges, elems, marker_dict = self.trainer.datamodule.mesh_dict[af]
                pred = self.model(x).detach().cpu()
                fields = torch.stack((
                    pred.T,
                    y.x.T,
                    (y.x.T-pred.T).abs()
                ))
                figure = plot_scalar_field(fields, X=nodes.T, tri=elems[0]) 
            else:
                pred = self.model(x).detach().cpu()
                fields = torch.cat((
                    pred,
                    y[None],
                    (y[None]-pred).abs()
                ))
                figure = plot_scalar_field(fields)
        
        self.log('learning rate', self.optimizers().state_dict()['param_groups'][0]['lr'])
        self.logger.experiment.add_figure(
            f'Epoch{self.current_epoch}:', 
            figure,
        )

    def test_step(self, batch, batch_idx):
        x, y, other = batch
        solution = self.model(x)
        loss = F.mse_loss(solution, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'training_loss',
            'interval': 'step',
            'frequency': 5,
            'strict': True
        }

