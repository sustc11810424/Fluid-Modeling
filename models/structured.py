import torch
import torch.nn as nn 
import torch.nn.functional as F

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, conv=nn.Conv2d):
        super(Upsample, self).__init__()
        self.blocks = nn.Sequential(
            conv(in_channels, out_channels*4, kernel_size=3, padding=1, groups=groups) ,
            nn.PixelShuffle(2),
            nn.PReLU(out_channels),
        )
    def forward(self, x):
        return self.blocks(x)

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1) 
        self.act = nn.PReLU()

    def forward(self, batch):
        return self.act(self.conv(batch))

class MLPEncoder(nn.Module):
    """
    Encode N*C into N*C'.
    """
    def __init__(self, in_dim, z_dim, **kwargs):
        super().__init__()
        self.input = nn.Linear(in_features=in_dim, out_features=32)
        self.hidden = nn.Sequential(
            nn.Linear(32, 32),
            nn.PReLU(),
        )
        self.output = nn.Linear(in_features=32, out_features=z_dim)

    def forward(self, batch):
        batch = F.leaky_relu(self.input(batch))
        batch = self.hidden(batch)
        batch = self.output(batch)
        return batch

class CNNDecoder(nn.Module):
    """
    Fully convolutional. Decode N*C*H*W features into target.
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.upsample = nn.Sequential(
            Upsample(in_channels, 32),
            Upsample(32, 32),
            Upsample(32, 32),
        )
        self.output = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, out_channels, 1),
        )

    def forward(self, batch):
        batch = self.upsample(batch)
        batch = self.output(batch)
        return batch

class CNNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2, padding_mode='replicate'),
            nn.PReLU(),
        )
        self.downsample = nn.Sequential(
            Downsample(32, 32),
            Downsample(32, 32),
            Downsample(32, 32),
        )
        self.output = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        )

class ConditionalNorm(nn.Module):
    """
    Condition a N*C*H*W query on a N*C latent code.
    """
    def __init__(self, num_features, **kwargs):
        super().__init__()
        self.num_features = num_features
        self.IN = nn.InstanceNorm2d(num_features)

    def forward(self, x, c):
        assert x.size()[:2]==c.size()[:2]
        assert c.size[2]==2
        x = self.IN(x)
        return x * c[..., 0, None, None] + c[..., 1, None, None]

class NaiveEnc(nn.Module):
    """
    This encoder uses a conv-encoder to encode geometry (structured mesh) and a mlp-encoder to encode flow parameters.
    """
    def __init__(self, geo_dim=2, param_dim=2, latent_dim=32, **kwargs):
        super().__init__()
        self.encoder_geo = CNNEncoder(geo_dim, latent_dim)
        self.encoder_param = MLPEncoder(param_dim, latent_dim)

    def forward(self, batch):
        mesh, params = batch[:2]
        return [self.encoder_geo(mesh), self.encoder_param(params)]

model_dict = {
    'NaiveEnc': NaiveEnc,
    'CNNDecoder': CNNDecoder,
}

def get_model(cfg: dict):
    model = model_dict[cfg.get('model')]
    kwargs = cfg.get('kwargs')
    
    return model(**kwargs)
    