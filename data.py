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
from torchvision.transforms import Compose, Normalize
torch.set_default_tensor_type(torch.DoubleTensor)

# fields we care
Fields = [
    'Density', 
    'Momentum_x', 
    'Momentum_y', 
    'Energy', 
    'Pressure', 
    'Temperature', 
    # 'Mach', 
    # 'Pressure_Coefficient'
]

def interp_data(data_dir='./data', size=(512, 512), xlim=(-0.5, 1.5), ylim=(-1, 1), Fields=Fields):
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
        self.max = statistics['max'].to_numpy()
        self.min = statistics['min'].to_numpy()
        self.max_mag = np.abs(np.stack([self.max, self.min])).max(axis=0)

        assert len(self.max_mag) == len(self.max)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        airfoil, file_path = self.file_list[idx]
        Mach, AoA = file_path[:-4].split('_')[-2:]
        Mach = float(Mach)
        AoA = float(AoA)
        data = pd.read_csv(file_path)[Fields] # discard the "ID" column

        # TODO add more transform
        # data = data / self.max_mag # normalize to (-1., 1.)
        data = (data - self.min) / (self.max - self.min)

        nodes, edge_index, elems, marker_dict = self.mesh_dict[airfoil]
        node_type = torch.zeros(len(nodes), 1)
        for i, marker in enumerate(marker_dict.keys()):
            for item in marker_dict[marker]:
                node_type[item] = i + 1

        edge_index = to_undirected(edge_index)
        free_stream = Mach * torch.ones(nodes.size(0), 2) * torch.tensor([np.cos(AoA* np.pi / 180.), np.sin(AoA* np.pi / 180.)])
        x = torch.hstack((nodes, node_type, free_stream))
        flow_field = torch.from_numpy(data.to_numpy())
        
        return Data(x=x, edge_index=edge_index), Data(x=flow_field, edge_index=edge_index), airfoil

class RegularGridDataset(Dataset):
    """

    """
    def __init__(self, file_list, mesh_dict=None, stats=None):
        super(RegularGridDataset, self).__init__()
        self.file_list = file_list
        self.mesh_dict = mesh_dict
        # TODO
        self.stats = stats
        self.transforms = Compose([
            Normalize(mean=stats['mean'], std=stats['std'])
        ])
        # self.max_mag = np.abs(np.stack([self.max, self.min])).max(axis=0).reshape(-1, 1, 1)

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

        if self.transforms:
            fields = self.transforms(fields)
        
        free_stream = Mach * torch.ones(2, mask.shape[0], mask.shape[1]) * torch.tensor([np.cos(AoA* np.pi / 180.), np.sin(AoA* np.pi / 180.)]).reshape((2, 1, 1))
        pressure = torch.ones(1, mask.shape[0], mask.shape[1])
        x = torch.cat((
            mask[None], 
            free_stream*mask[None], 
            # pressure*mask[None]
        ))
        
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
    def __init__(self, data_dir='./data', discretization='unstructured', batch_size=32, statistics=None, interp=None, Fields=Fields):
        super().__init__()
        assert discretization in {'unstructured', 'structured', 'regular'}
        self.postfix = Data_postfix[discretization]
        self.Dataset, self.DataLoader = Data_dataset[discretization]
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.statistics = statistics
        self.interp = interp
        self.Fields = Fields

    def prepare_data(self):
        # TODO: generate data here

        # TODO: add more options
        if self.interp is not None:
            interp_data(
                data_dir=self.data_dir,
                size=(128,  128),
                Fields=self.Fields,
            )
        pass

    def read_data(self):
        data_list = [] # file names
        mesh_dict = {} # airfoil: mesh
        
        for airfoil in os.listdir(self.data_dir):
            path = os.path.join(self.data_dir, airfoil)
            for f in os.listdir(path):
                if f.endswith(self.postfix):
                    data_list.append((airfoil, os.path.join(path, f)))
                elif f.endswith('.su2'):
                    mesh_dict[airfoil] = get_mesh_graph(os.path.join(path, f))
        return data_list, mesh_dict
    
    def gather_stats(self):
        df = pd.read_csv(self.data_list[0][1].replace('.npy', '.csv'))
        stats = df.describe().T[['count', 'mean', 'std', 'min', 'max']]
        t = tqdm(self.data_list[1:], desc='collecting statistics')
        for _, file_name in t:
            df = pd.read_csv(file_name.replace('.npy', '.csv'))
            desc = df.describe().T[['count', 'mean', 'std', 'min', 'max']]
            stats['max'] = pd.concat((stats['max'], desc['max']), axis=1).max(axis=1)
            stats['min'] = pd.concat((stats['min'], desc['min']), axis=1).min(axis=1)
            stats['mean'] = stats['mean'] * stats['count'] / (stats['count'] + desc['count']) + desc['mean']*desc['count'] / (stats['count'] + desc['count'])
            stats['count'] = stats['count'] + desc['count']
            stats['std'] = np.sqrt(((stats['std']**2)*stats['count'] + (desc['std']**2)*desc['count']) / (stats['count'] + desc['count']))
        t.close()

        return stats

    def setup(self, stage=None):
        self.data_list, self.mesh_dict = self.read_data()
        
        # gather statistics if necessary
        if self.statistics is None:
            self.statistics = self.gather_stats()
            self.statistics.to_csv('stats.csv')
        else:
            print('use existing stats')

        self.statistics = self.statistics.loc[self.Fields]
        print(self.statistics)

        # data splitting
        train_len = int(len(self.data_list)*0.75) 
        self.train_set, self.test_set = torch.utils.data.random_split(
            dataset=self.Dataset(self.data_list, self.mesh_dict, self.statistics), 
            lengths=[train_len, len(self.data_list)-train_len]
        )

    def train_dataloader(self):
        return self.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return self.DataLoader(self.test_set, batch_size=self.batch_size)

    def get_example(self):
        idx = np.random.randint(0, len(self.test_set))
        return self.test_set[idx]

if __name__ == '__main__':
    interp_data(size=(128, 128))
