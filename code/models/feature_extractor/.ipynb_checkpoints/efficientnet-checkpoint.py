# import torchvision.models as models
import timm
import torch.nn as nn
import torch
from collections import namedtuple


__all__ = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']


def _load_state_dict(model, in_channels=3):
    state_dict = model.state_dict()
    if in_channels != 3:
        param = list(state_dict)
        conv0 = state_dict[param[0]] 
#         print(conv0.shape)
        model.conv_stem = nn.Conv2d(in_channels, model.conv_stem.out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier = nn.Sequential()
        model.global_pool = nn.Sequential()
        state_dict[param[0]] = torch.mean(conv0, dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
#         print(state_dict['features.0.0.weight'].shape)
    else:
        model.classifier = nn.Sequential()
        model.global_pool = nn.Sequential()
    return model.load_state_dict(state_dict, strict=False)

def _efficientnet(cfg, progress=True, **kwargs):
#     cfg = namedtuple('cfg', cfg.keys())(*cfg.values())
    model_name = cfg.backbone
    model = timm.create_model(model_name, pretrained=cfg.pretrained)
    if cfg.pretrained:
        _load_state_dict(model, cfg.in_channels)
    else:
        model.conv_stem = nn.Conv2d(cfg.in_channels, model.conv_stem.out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier = nn.Sequential()
        model.global_pool = nn.Sequential()
    return model


def efficientnet_b0(cfg, progress=True, **kwargs):
    return _efficientnet(cfg)
    
def efficientnet_b1(cfg, progress=True, **kwargs):
    return _efficientnet(cfg)

def efficientnet_b2(cfg, progress=True, **kwargs):
    return _efficientnet(cfg)

def efficientnet_b3(cfg, progress=True, **kwargs):
    return _efficientnet(cfg)

def efficientnet_b4(cfg, progress=True, **kwargs):
    return _efficientnet(cfg)

def efficientnet_b5(cfg, progress=True, **kwargs):
    return _efficientnet(cfg)

def efficientnet_b6(cfg, progress=True, **kwargs):
    return _efficientnet(cfg)

def efficientnet_b7(cfg, progress=True, **kwargs):
    return _efficientnet(cfg)
        
if __name__ == '__main__':
    import yaml
    from torchinfo import summary
    from collections import namedtuple


    nn_config_path = '/workspace/tuanle/03-C_classification/Chromosome_classification/configs/base.yml'
    with open(nn_config_path) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    cfg['model_params']['backbone'] = 'efficientnet_b3'
#     print(cfg)

    model = efficientnet_b3(cfg['model_params'])
    print(model.num_features)
    summary(model, input_size=(1, 3, 175, 135))
    