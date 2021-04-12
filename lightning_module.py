import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np

import pytorch_lightning as pl
from utils import *

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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'training_loss',
            'interval': 'step',
            'frequency': 5,
            'strict': True
        }