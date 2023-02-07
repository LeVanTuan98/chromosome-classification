# Standard library imports
import os
import time
import yaml
import shutil
import logging
import argparse
import numpy as np

# Third party imports
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Local application imports
from utils.utils import get_dataloader 
from utils.utils import init_seeds, create_exp_dir, set_devices
from models.classifier import PLImageClassifier

def train(cfg, debug=0):   
    # 2. Create the experimental folder and set the working environmental
    if debug == 1:
        cfg['train_params']['num_workers'] = 0
        # cfg['train_params']['train_batch_size'] = 8
        cfg['train_params']['n_epochs'] = 2
        cfg['warmup_epochs'] = 1
        cfg['debug_mode'] = 1
        print('DEBUG mode')
        save_dir = 'experiments/tunning/test-debug'
        create_exp_dir(save_dir, visual_folder=True)
    else:
        cfg['debug_mode'] = 0
        # If not debug, we create a folder to store the model weights and etc
        save_dir = f'experiments/tune_model_resize_5/{cfg["name"]}_{cfg["model_params"]["backbone"]}_{cfg["train_params"]["resize_mode"]}_{cfg["train_params"]["input_shape"][0]}_{cfg["train_params"]["num_random"]}'
        # save_dir = f'experiments_HCM_with_past/{cfg["name"]}-{cfg["train_params"]["n_incremental"]}-{time.strftime("%Y%m%d-%H%M%S")}'
        create_exp_dir(save_dir, visual_folder=True)
    cfg['train_params']['save_dir'] = save_dir

    # Copy the config file into the save dir
#     shutil.copy(nn_config_path, save_dir)
    
    with open(f'{save_dir}/base.yml', 'w') as file:
        yaml.safe_dump(cfg, file, allow_unicode=False)

    # Set random seeds for reproducibility
    init_seeds(cfg['seed'])
    set_devices(cfg['gpu_id'])
    # print(cfg['gpu_id'])
    # 3. Initialize a classification model

    model = PLImageClassifier(cfg)

    # 4. Get datalodaers
    train_loader = get_dataloader(cfg, phase='train')
    val_loader = get_dataloader(cfg, phase='valid')

    # 5. Generate a pytorch lightning trainer
    checkpoint_callback = ModelCheckpoint(dirpath=save_dir,
                                          filename='best_model',
                                          save_last=True,
                                          save_top_k=1,
                                          save_weights_only=True,
                                          verbose=True,
                                          monitor='val_acc', mode='max')
    checkpoint_callback.FILE_EXTENSION = ".pth"
    checkpoint_callback.CHECKPOINT_NAME_LAST = "last_model"
    logger = TensorBoardLogger("lightning_logs", name=os.path.basename(save_dir))
    trainer = Trainer(gpus=cfg['gpu_id'] if torch.cuda.is_available() else 0,
                      auto_select_gpus=True,
                      checkpoint_callback=checkpoint_callback,
                      max_epochs=cfg['train_params']['n_epochs'], profiler="simple",
                      logger=logger,
                      callbacks= [EarlyStopping(monitor="val_acc",
                                         patience=cfg['train_params']['early_stop'], 
                                         strict=False,
                                         verbose=False,
                                         mode='max')]
                     )

    # 6. Train the network
    trainer.fit(model, train_loader, val_loader)
    

if __name__ ==  "__main__": 
#     dict_params = {
#         'backbone': ['densenet121', 'densenet161', 'densenet169', 'densenet201', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b3'],
#         'input_shape': [[175, 135], [256, 256]],
#         'resize_mode': ['adapt']
#     }

#     dict_params = {
#         'backbone': ['densenet161'],
#         'input_shape': [[256, 256]],
#         'resize_mode': ['adapt']
#     }
    
#     dict_params = {
#         'backbone': ['efficientnet_b3'],
#         'input_shape': [[256, 200]],
#         'resize_mode': ['adapt'],
#         'num_random': [1]
#     }
    
    dict_params = {
        'backbone': ['efficientnet_b3'],
        'input_shape': [[256, 256]],
        'resize_mode': ['scale_up', 'padding'],
        'num_random': [1]
    }
    
#     dict_params = {
#         'backbone': ['efficientnet_b3'],
#         'input_shape': [[256, 256]],
#         'resize_mode': ['adapt'],
#         'num_random': [2, 3, 4, 5]
#     }
    
    idx = 0
    total = len(dict_params['backbone']) * len(dict_params['input_shape']) * len(dict_params['resize_mode']) * len(dict_params['num_random'])
#     1. Read experiment configurations
    nn_config_path = './configs/base.yml'
    with open(nn_config_path) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
        
    for backbone in dict_params['backbone']:
        for input_shape in dict_params['input_shape']:
            for resize_mode in dict_params['resize_mode']:
                for num_random in dict_params['num_random']:
                    idx += 1
    #                 if idx in np.arange(0, 13):
    #                     continue
                    cfg['model_params']['backbone'] = backbone
                    cfg['train_params']['input_shape'] = input_shape
                    cfg['train_params']['resize_mode'] = resize_mode
                    cfg['train_params']['num_random'] = num_random

                    print(f"{idx}/{total}: backbone: {backbone} - input_shape: {input_shape} - padding: {resize_mode} - num_random: {num_random}")

                    train(cfg)

            
    