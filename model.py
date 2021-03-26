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
            x = layer(x, edge_index)
            x = F.leaky_relu(x)
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
            x = F.leaky_relu(x)
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
                    f'Epoch{self.current_epoch}: {af}', 
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

