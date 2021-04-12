import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, GATConv
from torch_geometric.data import Data

import os
import numpy as np
from utils import get_mesh_graph, plot_scalar_field

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
