import os
import numpy as np
import torch
import random

from torch.utils.data import DataLoader
from monai.data import CacheDataset
from monai.utils import set_determinism

from utils.datasets import ImageDataset

def init_seeds(seed):
    # Setting seeds
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_determinism(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

        
def get_dataloader(cfg, phase='train'):
    m_params = cfg['model_params']
    t_params = cfg['train_params']
    t_params['phase'] = phase
    
    dataset = ImageDataset(**t_params)
    print(f'No. {phase} samples: {len(dataset)}')
    
#     cache_dataset = CacheDataset(dataset, num_workers=t_params['num_workers_to_cache'], cache_rate=1.0, transform=None)
    
    if phase == 'train':
        shuffle = True
    else:
        shuffle = False
        
    dataloader = DataLoader(dataset, batch_size=t_params['train_batch_size'], 
                                  num_workers=t_params['num_workers_from_cache'], 
                                  pin_memory=torch.cuda.is_available(), shuffle=shuffle)
    return dataloader

def set_devices(device_ids):
    """
    restrict visible device
    :param device_ids: device ids start at 0
    """
    assert len(device_ids) > 0, "there must be at least 1 id"

    os.environ['CUDA_VISIBLE_DEVICES'] = device_ids
    

def create_exp_dir(path, visual_folder=False):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        if visual_folder is True:
            os.mkdir(path + '/visual')  # for visual results
    else:
        print("DIR already existed.")
    print('Experiment dir : {}'.format(path))
