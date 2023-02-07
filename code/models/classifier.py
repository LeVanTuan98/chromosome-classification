# -*- coding: utf-8 -*-
"""
Created on 7/09/2020 5:52 pm

@author: 
"""
# Standard library imports
import numpy as np
from collections import namedtuple
from typing import Dict, List, Union
# Third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Local application imports
from models.feature_extractor.densenet import densenet121, densenet161, densenet169, densenet201
from models.feature_extractor.efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7



# Define the classifier network
class ImageClassifier(nn.Module):
    def __init__(self, cfg):
        """
        Initialize an instance of the chest-notchest network
        Args:
            cfg: dictionary of the network parameters
        """
        super(ImageClassifier, self).__init__()
        # Convert cfg from dictionary to a class object
        cfg = namedtuple('cfg', cfg.keys())(*cfg.values())
        self.cfg = cfg

        # Set the backbone for the network
        self.avai_backbones = self.get_backbones()
        if self.cfg.backbone not in self.avai_backbones.keys():
            raise KeyError(
                'Invalid backbone name. Received "{}", but expected to be one of {}'.format(
                    self.cfg.backbone, self.avai_backbones.keys()))

        self.backbone = self.avai_backbones[cfg.backbone][0](cfg)
        self.backbone_type = self.avai_backbones[self.cfg.backbone][1]

        # Set to not train backbone's parameters if necessary
        if self.cfg.bb_freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Set globalbool for this class
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Set the number of classes, it is the number of network's output
        self.n_classes = cfg.num_labels

        self.n_maps = 1
        print('No. classes: ', self.n_classes)

        self.expand = 1
    
        # Get the number of output features of the backbone
        self.n_out_features = self.backbone.num_features
        
        # Set the classifier
        if cfg.conv_fc:
            fc = getattr(nn, 'Conv2d')
            self.fc = fc(self.n_out_features * self.expand,
                     self.n_classes * self.n_maps, kernel_size=1, stride=1, groups=1, bias=True)
        else:
            fc = getattr(nn, 'Linear')
            self.fc = fc(self.n_out_features * self.expand,
                     self.n_classes * self.n_maps, bias=True)

        # Initialize the classifier
        classifier = getattr(self, "fc")
        if isinstance(classifier, nn.Conv2d):
            classifier.weight.data.normal_(0, 0.01)
            classifier.bias.data.zero_()

        # Initialize the batchnorm for the output features
        self.bn = nn.BatchNorm2d(self.n_out_features * self.expand)


    def forward(self, x):
        """
        Args:
            x: Tensor of size (batchsize, n_channels, H, W)

        Returns: logit of

        """
        # Get the output of the backbone
        feature_map = self.backbone(x)          # of size (batchsize, n_out_features, 7, 7)
#         print(feature_map.shape)
        # Get the output of the global pool
        feat = self.global_pool(feature_map)    # of size (batchsize, n_out_features, 1, 1)
#         print(feat.shape)
        # Get the output of batchnorm if required
        if self.cfg.fc_bn:
            bn = getattr(self, "bn")
            feat = bn(feat)                     # of size (batchsize, n_out_features, 1, 1)

        # Get the output of the dropout, of size (batchsize, n_out_features, 1, 1)
        feat = F.dropout(feat, p=self.cfg.fc_drop, training=self.training)

        # Get the output of the classifier, of size (batchsize, n_classes)
        classifier = getattr(self, "fc")
        if self.cfg.conv_fc:
            if self.cfg.wildcat:
                logits = classifier(feat)
                logits = self.spatial_pooling(logits)
            else:
                logits = classifier(feat).squeeze(-1).squeeze(-1)

        else:
            logits = classifier(feat.view(feat.size(0), -1))
        return logits
    

    @staticmethod
    def get_backbones():
        """
        Returns: dictionary of famous networks
        """
        __factory = {'densenet121': [densenet121, 'densenet'],
                     'densenet161': [densenet161, 'densenet'],
                     'densenet169': [densenet169, 'densenet'],
                     'densenet201': [densenet201, 'densenet'], 
                     'efficientnet_b0': [efficientnet_b0, 'efficientnet'],
                     'efficientnet_b1': [efficientnet_b1, 'efficientnet'],
                     'efficientnet_b2': [efficientnet_b2, 'efficientnet'],
                     'efficientnet_b3': [efficientnet_b3, 'efficientnet'],
                     'efficientnet_b4': [efficientnet_b4, 'efficientnet'],
                     'efficientnet_b5': [efficientnet_b5, 'efficientnet'],
                     'efficientnet_b6': [efficientnet_b6, 'efficientnet'],
                     'efficientnet_b7': [efficientnet_b7, 'efficientnet'],
                    }
        return __factory


class PLImageClassifier(pl.LightningModule):
    def __init__(self, cfg, device='cpu'):
        super(PLImageClassifier, self).__init__()
        self.cfg = cfg
        
        self.classifier = ImageClassifier(cfg['model_params'])

        self.loss_fn = getattr(torch.nn, cfg['loss'])()

        if cfg['model_params']['num_labels'] == 2 and 'softmax' in cfg['model_params'].keys() and cfg['model_params']['softmax'] == False: 
            self.activation = nn.Sigmoid()
        else:
            self.activation = None

    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        # 1. Get a minibatch data for training
        x, y_gt = batch[0], batch[1]
        # x of size (batchsize, 3, H, W); y_gt of size (batchsize, 1)
        # print('\nTrain', x.shape, y_gt.shape)
        # 2. Compute the forward pass
        y_pr = self.classifier(x)   # of size (batchsize, n_classes)
        if self.activation != None:
            y_pr = self.activation(y_pr)
            y_gt = y_gt.view(-1)
            y_pr = y_pr.view(-1)
        else:
            y_gt = y_gt.type(torch.LongTensor).to(x.device)
            
        # 3. Compute loss, then update weights
        loss = self.loss_fn(y_pr, y_gt)
        
         # 4. Compute the accuracy
        if self.activation != None:
            y_pr = (y_pr > 0.5).to(torch.uint8)
        else:
            _, y_pr = torch.max(y_pr, dim=1)
#         print(y_pr, y_gt)
        
        y_pr, y_gt = y_pr.cpu(), y_gt.cpu()
        train_acc = torch.tensor(accuracy_score(y_pr, y_gt))
        train_f1 = torch.tensor(f1_score(y_pr, y_gt, average='macro', labels=self.cfg['model_params']['labels']))

        # 4. Logging the loss
        # self.log('train_loss', loss)
        tensorboard_logs = {'train_loss': loss, 'train_acc': train_acc, 'train_f1': train_f1}

        return {'loss': loss, 'log': tensorboard_logs}
#         return {'train_loss': loss, 'train_acc': train_acc, 'train_f1': train_f1} #, 'log': tensorboard_logs}


    def validation_step(self, batch, batch_idx):
        # 1. Get a minibatch data for training
        x, y_gt = batch[0], batch[1]
        
        # x of size (batchsize, 3, H, W); y_gt of size (batchsize, 1)
        # print('\nVal', x.shape, y_gt.shape)
        
        # 2. Compute the forward pass
        y_pr = self.classifier(x)   # of size (batchsize, n_classes)
        if self.activation != None:
            y_pr =  self.activation(y_pr)
            y_gt = y_gt.view(-1)
            y_pr = y_pr.view(-1)
        else:
            y_gt = y_gt.type(torch.LongTensor).to(x.device)
            
        #print(y_gt.shape, y_pr.shape)
        #print(y_pr.min(), y_pr.max(), y_gt.min(), y_gt.max())
        # 3. Compute loss, then update weights
        # print(y_pr.shape, y_gt.shape)
        loss = self.loss_fn(y_pr, y_gt)

        # 4. Compute the accuracy
        if self.activation != None:
            y_pr = (y_pr > 0.5).to(torch.uint8)
        else:
            _, y_pr = torch.max(y_pr, dim=1)
#         print(y_pr, y_gt)
        
        y_pr, y_gt = y_pr.cpu(), y_gt.cpu()
        val_acc = torch.tensor(accuracy_score(y_pr, y_gt))
        val_f1 = torch.tensor(f1_score(y_pr, y_gt, average='macro', labels=self.cfg['model_params']['labels']))
        val_pre = torch.tensor(precision_score(y_pr, y_gt, average='macro', labels=self.cfg['model_params']['labels']))
        val_rec = torch.tensor(recall_score(y_pr, y_gt, average='macro', labels=self.cfg['model_params']['labels']))
        
        # 4. Logging the loss
#         tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc,
#                             'val_f1': avg_f1, 'val_pre': avg_pre, 'val_rec': avg_rec}
        return {'val_loss': loss, 'val_acc': val_acc, 'val_f1': val_f1, 
                'val_pre': val_pre, 'val_rec': val_rec} #, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]])  -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['val_f1'] for x in outputs]).mean()
        avg_pre = torch.stack([x['val_pre'] for x in outputs]).mean()
        avg_rec = torch.stack([x['val_rec'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc,
                            'val_f1': avg_f1, 'val_pre': avg_pre, 'val_rec': avg_rec}
        self.log('val_acc', avg_acc)
        self.log('val_f1', avg_f1)
        self.log('val_pre', avg_pre)
        self.log('val_rec', avg_rec)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = init_obj(self.cfg['optimizer']['type'], self.cfg['optimizer']['args'],
                             torch.optim, self.classifier.parameters())
        return optimizer
    
    
# ==========================================================================================================
# Auxiliary function
# ==========================================================================================================
def init_obj(module_name, module_args, module, *args, **kwargs):
    """
    Finds a function handle with the name given as 'type' in config, and returns the
    instance initialized with corresponding arguments given.
    `object = config.init_obj('name', module, a, b=1)`
    is equivalent to
    `object = module.name(a, b=1)`
    """
    assert all([k not in module_args for k in
                kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    return getattr(module, module_name)(*args, **module_args)
    
# ==========================================================================================================
# Main function
# ==========================================================================================================
if __name__ == "__main__":
    import yaml
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader
    from torchinfo import summary
    # Local application imports
    from utils.datasets import ImageDataset

    # load the config file
    nn_config_path = '../configs/base.yml'
    with open(nn_config_path) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

#     model = PLImageClassifier(cfg)

#     m_params = cfg['model_params']
#     t_params = cfg['train_params']

#     # Generate the train dataloader
#     train_dataset = ImageDataset(t_params['dataset_dir'], t_params['train_txtfiles'],
#                                  m_params, mode='train', n_cutoff_imgs=t_params['n_cutoff_imgs'],
#                                  labels=m_params['labels'])
#     train_dataloader = DataLoader(train_dataset, batch_size=t_params['train_batch_size'],
#                                   pin_memory=True, num_workers=t_params['num_workers'])


#     trainer = Trainer(gpus=1 if torch.cuda.is_available() else 0,
#                       max_epochs=2, profiler=True)
#     trainer.fit(model)

    # # Generate a random input
    # x = torch.rand((2, 3, 224, 224))    # got error when batchsize = 1 at batchnorm
    # print(x.shape)
    #
    # Initialize the model
    model = ImageClassifier(cfg=cfg['model_params'])
    summary(model, input_size=(1, 1, 224, 224))
    #
    # # Compute the output
    # y = model(x)
    # print(y.shape)
    