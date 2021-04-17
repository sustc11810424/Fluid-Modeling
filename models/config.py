from models import structured, unstructured, image
from torch.nn import MSELoss

method_dict = {
    'structured': structured,
    'unstructured': unstructured,
    'image': image,
}

def get_model(cfg: dict):
    method = cfg.get('method')
    return method_dict[method].get_model(cfg)

def get_loss(cfg: dict):
    return MSELoss()