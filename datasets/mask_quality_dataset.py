import os
from os import path
import numpy as np
from PIL import Image
import pandas as pd
import ast

import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from datasets.range_transform import im_normalization



class MaskQualityDB(Dataset):
    """ Dataset to train the QNet."""

    def __init__(self, root, csv_set, resolution='224'):
        self.root = root
        self.resolution = resolution
        self.mask_dir = path.join(root, 'Annotations', self.resolution)
        self.image_dir = path.join(root, 'RGBFrames', self.resolution)

        
        df = pd.read_csv(csv_set).reset_index()
        self.data = []
        for idx, row in df.iterrows():
            # row keys: state_name (e.g. 4e016d5a__1_round_1), ious
            ious = np.array(ast.literal_eval(row['ious']))
            frames = np.arange(0, len(ious), dtype=int)

            # delete no object frames
            delete_frames = np.where(ious == 20) 
            ious = np.delete(ious, delete_frames)
            frames = np.delete(frames, delete_frames)
            
            states = [row['state_name']] * len(ious)
            curr_data = list(zip(states, ious, frames))
            self.data.extend(curr_data)
        
        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

        self.iou_bins = np.arange(0,1.01,0.05)


    def __len__(self):
        return len(self.data)
    

    def map_input_to_label(self, input_value, bins):
        for i in range(1,len(bins)):
            if input_value >= bins[i-1] and input_value <= bins[i]:
                return i-1

        raise ValueError(f'Invalid input: {input_value}')


    def __getitem__(self, index):
        state, iou, frame_num = self.data[index]
        mask_path = os.path.join(self.mask_dir, state, f'{frame_num:05d}.png')
        mask = Image.open(mask_path).convert('P')
        mask = np.array(mask, dtype=np.uint8)
        mask = torch.from_numpy(mask)/255

        video_name = state.split('__')[0]
        rgb_path = os.path.join(self.image_dir, video_name, f'{frame_num:05d}.png')
        img = Image.open(rgb_path).convert('RGB')
        img = self.im_transform(img) 
        label = self.map_input_to_label(iou, self.iou_bins)

        data = {
            'img': img,
            'mask': mask,
            'label': label
        }
        return data