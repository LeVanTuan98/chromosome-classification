import os
import yaml
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastprogress import progress_bar
import SimpleITK as sitk
import time
import nibabel as nib
from tqdm import tqdm
import cv2

import torch 
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import monai.transforms

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
from utils.datasets import ImageDataset
from models.classifier import PLImageClassifier


class Inference:
    def __init__(self, cfg, trained_model_path, input_shape=(224, 224), scale_intensity=[0, 1], device='cpu'):
        self.cfg = cfg
        self.trained_model_path = trained_model_path
        self.device = device
        self.input_shape = input_shape
        self.labels_dict = {0:'Non_Contrast', 1:'Arterial_Phase', 2:'Venous_Phase', 3:'Delayed_Phase'}

        # Set the augmentation
#         self.transforms = monai.transforms.Compose(
#             [
#                 monai.transforms.Resize(input_shape, size_mode='all'),
#                 monai.transforms.ToTensor(dtype=torch.float32)
#             ]
#         )
        self.transforms = T.Compose([
            T.ToPILImage(),
            T.Resize(self.input_shape),
            T.ToTensor()
        ])
        self.model = self.load_model()


    def load_model(self):
        print('Load the trained model in: ', self.trained_model_path)  
        model = PLImageClassifier(self.cfg)
        weights = torch.load(self.trained_model_path, map_location='cpu')['state_dict']
        model.load_state_dict(weights, strict=False)
        model = model.to(self.device)
        model.eval()

        return model

    def predict(self, img):
        x = self.transforms(img).unsqueeze(0).to(self.device)
        y = self.model(x)
        y = torch.argmax(y, dim=1).item()
        
        return y

#     def predict_from_nii(self, t_params, nii_file):
#         read_start = time.time()
#         # Read the data using simple itk to obtain the size of (n_slices x H x W)
#         data = sitk.GetArrayFromImage(sitk.ReadImage(nii_file)) 
#         data = data[np.linspace(0, data.shape[0] - 1, t_params['n_slices'], dtype=int), ...]
    
#         data = np.clip(data, t_params['hu_range'][0], t_params['hu_range'][1]) # clip the hu_range
        
#         # Transformn the data
#         data = self.transforms(data)
#         x = data.unsqueeze(0).to(self.device)
#         read_time = time.time() - read_start
        
#         infer_start = time.time()
#         y = self.model(x)
#         infer_time = time.time() - infer_start
#         y = torch.argmax(y, dim=1).item()

#         return y, self.labels_dict[y], read_time, infer_time


    def predict_from_csv(self, t_params, csv_files, root_dir='', csv_columns=['ct_liver_cropped', 'phase_name'], 
                            batch_size=128, num_workers=4, return_prob=False):

        t_params['csv_files'] = csv_files
        t_params['root_dir'] = root_dir
        t_params['csv_columns'] = csv_columns
        t_params['phase'] = 'test'

        test_set = ImageDataset(**t_params)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        
        y_preds = []
        outputs = []
        gt_labels = []

        for (img, label) in progress_bar(test_loader):
            x = img.to(self.device)
            y = self.model(x)

            y_preds += list(F.softmax(y, dim=1).detach().cpu().numpy())
            outputs += list(y.detach().cpu().numpy())
            gt_labels += list(label)

        if return_prob:
            return y_preds, gt_labels, outputs
        else:
            return y_preds, gt_labels
        
        
    def predict_from_file(self, t_params, img_path, return_prob=False):
        data = cv2.imread(img_path)

        index = np.where(data > 0)
        x_min, x_max = int(max(0, index[0].min())), int(index[0].max())
        y_min, y_max = int(max(0, index[1].min())), int(index[1].max())
        img = data[x_min:x_max+1, y_min:y_max+1, 0]
        cropped_img = np.expand_dims(img, axis=2)


        if t_params['resize_mode'] == 'scale_up':
#             img = data[x_min:x_max, y_min:y_max, :]
#             cropped_img = np.expand_dims(img[..., 0], axis=2)
            
            transforms = T.Compose([
                T.ToPILImage(),
                T.Resize(t_params['input_shape']),
                T.ToTensor()
            ])
            
            output_img = transforms(cropped_img)
        elif t_params['resize_mode'] == 'padding':
#             img = data[0, x_min:x_max, y_min:y_max]
#             cropped_img = np.expand_dims(img[..., 0], axis=2)
            shape = [t_params['input_shape'][0], t_params['input_shape'][1], 1]
            w, h = x_max-x_min, y_max-y_min
            xs = t_params['input_shape'][0]//2 - w//2
            ys = t_params['input_shape'][1]//2 - h//2
            
            transforms = T.Compose([
                T.ToPILImage(),
                T.ToTensor()
            ])
            
            output_img = np.zeros(shape, dtype=np.uint8)
            output_img[xs:xs+w+1, ys:ys+h+1, 0] = cropped_img[..., 0]
            output_img = transforms(output_img)
            
        elif t_params['resize_mode'] == 'adapt':
#             img = data[x_min:x_max, y_min:y_max, :]
#             cropped_img = np.expand_dims(img[..., 0], axis=2)

            h, w = cropped_img.shape[0:2]
    #             print('Shape of cropped img:', cropped_img.shape)
            input_size = [1, t_params['input_shape'][0], t_params['input_shape'][1]]
    #             print(input_size)
            h_ratio = input_size[1]/h
            w_ratio = input_size[2]/w
            # print(h_ratio, w_ratio)

            if h_ratio < w_ratio:
                transforms = T.Compose([
                    T.ToPILImage(),
                    T.Resize([input_size[1], round(w*h_ratio)]),
                    T.ToTensor()
                ])
                resize_img = transforms(cropped_img)
            else:
                transforms = T.Compose([
                    T.ToPILImage(),
                    T.Resize([int(h*w_ratio), input_size[2]]),
                    T.ToTensor()
                ])
                resize_img = transforms(cropped_img)

            new_h, new_w = resize_img.shape[1:]
    #         print('Shape of resize img:', resize_img.shape)
        # 1st method: random
    #         ys = np.random.randint(0, input_size[2]-new_w+1)   
    #         xs = np.random.randint(0, input_size[1]-new_h+1)

            # 2nd method: middle imgage
            xs = input_size[1]//2 - new_h//2
            ys = input_size[2]//2 - new_w//2
    #             print(xs, ys)
            output_img = torch.zeros(input_size)
            output_img[:, xs:xs+new_h, ys:ys+new_w] = resize_img

        x = output_img.unsqueeze(0).to(self.device)
        y = self.model(x)
        label = torch.argmax(y, dim=1).item()

        if return_prob:
            return label, y
        else:
            return label
            

def read_config(cfg_path):
    with open(cfg_path) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    return cfg



if __name__ == '__main__':
    
#     list_dir = [
# #             'chromosome_classification-200-20220511-050334', #0.943
# #             'chromosome_classification-200-20220520-045056', #0.936
# #         'chromosome_classification-200-20220520-073827', #0.940
# #         'chromosome_classification-200-20220520-152801', #0.96
#         'chromosome_classification-200-20220521-171320'
#     ]
    exp_dir = './experiments/tune_model_resize_5'
#     list_dir = glob.glob(f'{exp_dir}/chromosome_*')
    list_dir = [
        f'{exp_dir}/chromosome_classification_efficientnet_b3_scale_up_256_1',
#         f'{exp_dir}/chromosome_classification_efficientnet_b3_padding_256_1',

    ]
    
    df_results = []
    
    
    for dir_ckpt in list_dir:
        cfg_path = glob.glob(f'{dir_ckpt}/base*.yml')[-1]
        ckpt_path = f'{dir_ckpt}/best_model.pth'
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cfg = read_config(cfg_path)
        cfg['model_params']['pretrained'] = False
        t_params = cfg['train_params']

        
        label_list = [str(i) for i in range(cfg['model_params']['num_labels'])]

        model = Inference(cfg, ckpt_path, input_shape=t_params['input_shape'], device=device)
        np.random.seed(0)


#         2. predict from test set
        root_dir = ''
        csv_files = './data/test.csv'
#         y_preds, y_gts = model.predict_from_csv(t_params, csv_files, root_dir=root_dir, csv_columns=['image_dir', 'label'],
#                                                 batch_size=16, num_workers=4, return_prob=False)

#         y_gt_lbls = np.array(y_gts).astype(int)
#         y_pred_lbls = np.argmax(np.array(y_preds), axis=1)

        df = pd.read_csv(csv_files)
        y_gt_lbls = df['label']
        y_pred_lbls = []
        for k, r in tqdm(df.iterrows()):
            y, _ = model.predict_from_file(t_params, r['image_dir'], return_prob=True)
            y_pred_lbls.append(y)
        
        y_gt_lbls = np.array(y_gt_lbls).astype(int)
        y_pred_lbls = np.array(y_pred_lbls).astype(int)
        

        f1 = f1_score(y_gt_lbls, y_pred_lbls, average='macro', labels=label_list)
        pre = precision_score(y_pred_lbls, y_gt_lbls, average='macro', labels=label_list)
        rec = recall_score(y_pred_lbls, y_gt_lbls, average='macro', labels=label_list)
        acc = accuracy_score(y_pred_lbls, y_gt_lbls)
        report = classification_report(y_gt_lbls, y_pred_lbls)
    

        print('\n')
        print('F1: \t', f1)
        print('Precision: \t', pre)
        print('Recall: \t', rec)
        print('Accuracy: \t', acc)
        print('Classification Report: \n', report)
        print('\n')
        
        df_results.append({'exp': os.path.basename(dir_ckpt), 
                           'backbone': cfg['model_params']['backbone'],
                           'resize_mode': cfg['train_params']['resize_mode'],
                           'input_shape': cfg['train_params']['input_shape'],
                           'f1': f1, 
                           'precision': pre, 
                           'recall': rec, 
                           'accuracy': acc,
#                            'inference_time': f'{time_consume.mean()} +- {time_consume.std()}'
                          })
    pd.DataFrame(df_results).to_csv(f'{exp_dir}/results_for_tunning_model_padding_middle.csv', index=False)
        
#     from torchinfo import summary 
    
#     exp_path = '/workspace/tuanle/03-C_classification/Chromosome_classification/experiments/tune_model/chromosome_classification_efficientnet_b3'
#     yml_file = glob.glob(f'{exp_path}/*.yml')[0]
 
#     with open(yml_file) as file:
#         cfg = yaml.load(file, Loader=yaml.FullLoader)

#     # Initialize the model
#     model = PLImageClassifier(cfg)

#     # Compute the output
#     summary(model, input_size=[1, 3, 224, 224])
    
















