
# Standard library imports
import os
import time
import yaml
import shutil
import logging
import argparse
import numpy as np
from torchinfo import summary

# Third party imports
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Local application imports
from utils.utils import init_seeds, create_exp_dir, set_devices
from models.classifier import PLImageClassifier
from utils.utils import get_dataloader


# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Main file')
    args.add_argument('--config', default='configs/base.yml', type=str,
                      help='config file path (default: None)')
    args.add_argument('--debug', default=0, type=int,
                      help='debug mode? (default: 0')
    cmd_args = args.parse_args()

    assert cmd_args.config is not None, "Please specify a config file"

    # 1. Read experiment configurations
    nn_config_path = cmd_args.config
    with open(nn_config_path) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    # cfg['m_params']['labels']
    # 2. Create the experimental folder and set the working environmental
    if cmd_args.debug == 1:
        cfg['train_params']['num_workers'] = 0
        # cfg['train_params']['train_batch_size'] = 8
        cfg['train_params']['n_epochs'] = 2
        cfg['warmup_epochs'] = 1
        cfg['debug_mode'] = 1
        print('DEBUG mode')
        save_dir = 'experiments/test-debug'
        create_exp_dir(save_dir, visual_folder=True)
    else:
        cfg['debug_mode'] = 0
        # If not debug, we create a folder to store the model weights and etc
        save_dir = f'experiments/{cfg["name"]}-{cfg["train_params"]["n_epochs"]}-{time.strftime("%Y%m%d-%H%M%S")}'
        # save_dir = f'experiments_HCM_with_past/{cfg["name"]}-{cfg["train_params"]["n_incremental"]}-{time.strftime("%Y%m%d-%H%M%S")}'
        create_exp_dir(save_dir, visual_folder=True)
    cfg['train_params']['save_dir'] = save_dir

    # Copy the config file into the save dir
    shutil.copy(nn_config_path, save_dir)

    # Set random seeds for reproducibility
    init_seeds(cfg['seed'])
    set_devices(cfg['gpu_id'])
    # print(cfg['gpu_id'])
    # 3. Initialize a classification model
    model = PLImageClassifier(cfg)

#     4. Get datalodaers
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
    logger = TensorBoardLogger("tb_logs", name=os.path.basename(save_dir))
    trainer = Trainer(gpus=cfg['gpu_id'] if torch.cuda.is_available() else 0,
                      auto_select_gpus=True,
                      checkpoint_callback=checkpoint_callback,
                      max_epochs=cfg['train_params']['n_epochs'], 
                      profiler="simple",
                      logger=logger,
                      callbacks= [EarlyStopping(monitor="val_acc",
                                         patience=cfg['train_params']['early_stop'],
                                         strict=False,
                                         verbose=False,
                                         mode='max')])

    # 6. Train the network
    trainer.fit(model, train_loader, val_loader)
#     summary(model, input_size=(2, 1, 175, 135))


    