import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torch_geometric.data import Dataset as GeoDataset, DataLoader as GeoDataloader, Data
from torch_geometric.utils import add_self_loops, to_undirected
from matplotlib.tri import Triangulation
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from utils import interp_scalar_field, get_mesh_graph
torch.set_default_tensor_type(torch.DoubleTensor)

# fields we care
Fields = ['Density', 'Momentum_x', 'Momentum_y', 'Energy', 'Pressure', 'Temperature', 'Mach', 'Pressure_Coefficient']

def interp_data(data_dir='./data', size=(512, 512), xlim=(-0.5, 1.5), ylim=(-1, 1)):
    """
    Interpolate fields on irregular grids to ndarrays of shape (1+#fields, size[0], size[1]) and save as .npy files.
    """
    x = np.linspace(xlim[0], xlim[1], size[0])
    y = np.linspace(ylim[0], ylim[1], size[1])
    XX, YY = np.meshgrid(x,  y)
    import os
    for airfoil in os.listdir(data_dir):
        path = os.path.join(data_dir, airfoil)
        nodes, edge_index, elems, marker_dict = get_mesh_graph(os.path.join(path, 'mesh.su2'))
        t = tqdm([f for f in os.listdir(path) if f.endswith('.csv')], desc=airfoil)
        for f in t:
            filename = os.path.join(path, f)
            data = pd.read_csv(filename)[Fields].to_numpy().T
            tri = Triangulation(nodes.T[0], nodes.T[1], elems[0])
            mask = tri.get_trifinder()(XX, -YY) != -1
            fields = np.stack([interp_scalar_field(tri, data[field], size=size) for field in range(len(data))])
            
            npy = np.concatenate([mask[None], fields])
            npy.dump(filename[:-4]+'.npy')

class UnstructuredMeshDataset(GeoDataset):
    """
    This dataset loads triangular mesh.
    """
    def __init__(self, file_list, mesh_dict, statistics):
        super(UnstructuredMeshDataset, self).__init__(None, None, None)
        self.file_list = file_list
        self.mesh_dict = mesh_dict
        self.max = statistics['max']
        self.min = statistics['min']

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        airfoil, file_path = self.file_list[idx]
        Mach, AoA = file_path[:-4].split('_')[-2:]
        Mach = float(Mach)
        AoA = float(AoA)
        data = pd.read_csv(file_path)[Fields] # discard the "ID" column

        # TODO add more transform
        data = data / self.max # normalize to (-1., 1.)
        nodes, edge_index, elems, marker_dict = self.mesh_dict[airfoil]

        edge_index = to_undirected(edge_index)
        free_stream = Mach * torch.ones(nodes.size(0), 2) * torch.tensor([np.cos(AoA* np.pi / 180.), np.sin(AoA* np.pi / 180.)])
        x = torch.hstack((nodes, free_stream))
        flow_field = torch.from_numpy(data.to_numpy())
        
        return Data(x=x, edge_index=edge_index), Data(x=flow_field, edge_index=edge_index), airfoil

class RegularGridDataset(Dataset):
    """

    """
    def __init__(self, file_list, mesh_dict=None, statistics=None):
        super(RegularGridDataset, self).__init__()
        self.file_list = file_list
        self.mesh_dict = mesh_dict
        # TODO
        self.max = statistics['max'].to_numpy().reshape(-1, 1, 1)
        self.min = statistics['min'].to_numpy().reshape(-1, 1, 1)
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        airfoil, file_path = self.file_list[idx]
        Mach, AoA = file_path[:-4].split('_')[-2:]
        Mach = float(Mach)
        AoA = float(AoA)
        
        data = np.load(file_path, allow_pickle=True) # discard the "ID" column
        mask = torch.tensor(data[0])
        fields = torch.tensor(data[1:])
        fields[:, mask==0] = 0.0 # TODO maybe a better way?
        
        # TODO add more appropriate transform
        fields = fields / self.max # normalize to (-1., 1.)

        free_stream = Mach * torch.ones(2, mask.shape[0], mask.shape[1]) * torch.tensor([np.cos(AoA* np.pi / 180.), np.sin(AoA* np.pi / 180.)]).reshape((2, 1, 1))
        x = torch.cat((mask[None], free_stream*mask[None]))
        
        return x, fields, airfoil

Data_postfix = {
    'unstructured': '.csv',
    'structured': '.quad.csv', # TODO
    'regular': '.npy',
}

Data_dataset = {
    'unstructured': (UnstructuredMeshDataset, GeoDataloader),
    'structured': None, # TODO
    'regular': (RegularGridDataset, DataLoader),
}

class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', discretization='unstructured', batch_size=32, statistics=None):
        super().__init__()
        assert discretization in {'unstructured', 'structured', 'regular'}
        self.postfix = Data_postfix[discretization]
        self.Dataset, self.DataLoader = Data_dataset[discretization]
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.statistics = statistics

    def prepare_data(self):
        data_list = []
        mesh_dict = {}
        
        for airfoil in os.listdir(self.data_dir):
            path = os.path.join(self.data_dir, airfoil)
            for f in os.listdir(path):
                if f.endswith(self.postfix):
                    data_list.append((airfoil, os.path.join(path, f)))
                elif f.endswith('.su2'):
                    mesh_dict[airfoil] = get_mesh_graph(os.path.join(path, f))
        train_len = int(len(data_list)*0.75) 
        self.mesh_dict = mesh_dict

        # TODO replace this with a better one!
        if self.statistics==None:
            print('collecting global statistics')
            df = data_list[0][1].replace('.npy', '.csv')
            max = pd.read_csv(df).max()
            min = pd.read_csv(df).min()
            for _, file_name in data_list[1:]:
                df = pd.read_csv(file_name.replace('.npy', '.csv'))
                max = pd.concat((max, df), axis=1).max(axis=1)
                min = pd.concat((min, df), axis=1).min(axis=1)
            
            self.statistics = {
                'max': max[Fields],
                'min': min[Fields],
            }
        
        print(self.statistics)
        
        self.train_set, self.test_set = torch.utils.data.random_split(self.Dataset(data_list, mesh_dict, self.statistics), [train_len, len(data_list)-train_len])

    # def setup(self, )

    def train_dataloader(self):
        return self.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return self.DataLoader(self.test_set, batch_size=self.batch_size)

    def get_example(self):
        idx = np.random.randint(0, len(self.test_set))
        return self.test_set[idx]

if __name__ == '__main__':
    interp_data(size=(128, 128))
