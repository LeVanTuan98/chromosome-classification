import os
import numpy as np
import pandas as pd

from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


class ImageDataset(Dataset):
    def __init__(self, csv_files, phase='train', input_shape=[224, 224], root_dir='', csv_cols=['image_dir', 'label'], crop_size=[200, 200], resize_mode='resize', num_random=1, **kwargs):
        self.phase = phase
        self.input_shape = input_shape
        self.root_dir = root_dir
        self.resize_mode = resize_mode
        print('Resize mode:', resize_mode)
#         print('Input shape:', input_shape)
        if phase == 'train':
            csv_files = csv_files['train']
        elif phase == 'valid':
            csv_files = csv_files['valid']
        
        # read data from csv file
        if not isinstance(csv_files, str):
            csv_files = csv_files[0]
        df = pd.read_csv(csv_files)
        df = pd.concat([df for i in range(num_random)])
        
        self.imgs = df.values.tolist()
        
#         self.augmentation = T.Compose([                
#             T.ToPILImage(),
#             T.RandomRotation(45), 
#             T.RandomHorizontalFlip(),
#             T.RandomVerticalFlip(),
#             T.ToTensor()
#         ])
            
    def __len__(self):
        return len(self.imgs)
            
    def __getitem__(self, idx):
        sample = self.imgs[idx]
        img_path = sample[1]
        label = np.float32(sample[2]) # 0-23
        
        data = cv2.imread(img_path)
#         data = self.augmentation(data)
        # Crop data
#         index = torch.where(data > 0)
#         x_min, x_max = int(max(0, index[2].min())), int(index[2].max())
#         y_min, y_max = int(max(0, index[1].min())), int(index[1].max())
#         img = data[0, y_min:y_max+1, x_min:x_max+1].numpy().transpose(1, 0)

        index = np.where(data > 0)
        x_min, x_max = int(max(0, index[0].min())), int(index[0].max())
        y_min, y_max = int(max(0, index[1].min())), int(index[1].max())
        img = data[x_min:x_max+1, y_min:y_max+1, 0]
        
        cropped_img = np.expand_dims(img, axis=2)

        
        if self.resize_mode == 'scale_up':
#             img = data[x_min:x_max, y_min:y_max, :]
#             cropped_img = np.expand_dims(img[..., 0], axis=2)
            
            transforms = T.Compose([
                T.ToPILImage(),
                T.Resize(self.input_shape),
                T.ToTensor()
            ])
            
            output_img = transforms(cropped_img)
        elif self.resize_mode == 'padding':
#             img = data[0, x_min:x_max, y_min:y_max]
#             cropped_img = np.expand_dims(img[..., 0], axis=2)
            shape = [self.input_shape[0], self.input_shape[1], 1]
            w, h = x_max-x_min, y_max-y_min
            xs = self.input_shape[0]//2 - w//2
            ys = self.input_shape[1]//2 - h//2
            
            transforms = T.Compose([
                T.ToPILImage(),
                T.ToTensor()
            ])
            
            output_img = np.zeros(shape, dtype=np.uint8)
            output_img[xs:xs+w, ys:ys+h, 0] = data[x_min:x_max, y_min:y_max, 0]
            output_img = transforms(output_img)
            
        elif self.resize_mode == 'adapt':
#             img = data[x_min:x_max, y_min:y_max, :]
#             cropped_img = np.expand_dims(img[..., 0], axis=2)
            h, w = cropped_img.shape[0:2]
#             print('Shape of cropped img:', cropped_img.shape)
            input_size = [1, self.input_shape[0], self.input_shape[1]]
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
#             print('Shape of resize img:', resize_img.shape)
            # 1st method: random
#             ys = np.random.randint(0, input_size[2]-new_w+1)   
#             xs = np.random.randint(0, input_size[1]-new_h+1)
            
            # 2nd method: middle imgage
            xs = input_size[1]//2 - new_h//2
            ys = input_size[2]//2 - new_w//2
#             print(xs, ys)
            output_img = torch.zeros(input_size)
            output_img[:, xs:xs+new_h, ys:ys+new_w] = resize_img
            
#             output_img = output_img / 255.0
        return output_img.float(), label
        