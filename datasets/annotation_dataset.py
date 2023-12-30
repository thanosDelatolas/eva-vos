import os
from os import path
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from datasets.range_transform import im_normalization
from datasets.helpers import all_to_onehot

class AnnotationDataset(Dataset):
    """ Dataset for video annnotation.
        A video has N objects, then this dataset will create 
        one sample for each object (N four-dimensional videos)

        It can be used for MOSE and DAVIS datasets.
    """

    def __init__(self, root, 
            imset, 
            resolution='480p', 
            min_idx=None,
            max_idx=None,
            second_type_transform=None, # for ViT, dino, resnet
        ):
        super().__init__()

        self.root = root
        self.mask_dir = path.join(root, 'Annotations', resolution)
        self.image_dir = path.join(root, 'JPEGImages', resolution)
        self.resolution = resolution
        self.second_type_transform = second_type_transform

        #_imset_dir = path.join(root, 'ImageSets')
        _imset_f = imset #path.join(_imset_dir, imset)

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}

        ii = 0
        with open(path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
        
                if _video == '':
                    continue
                
                _mask = np.array(Image.open(path.join(self.mask_dir, _video, '00000.png')).convert("P"))
                n_objs = np.max(_mask)

                # 0 is background
                for obj_id in range(1,n_objs+1):
                    if min_idx is not None and max_idx is not None and (ii < min_idx or ii>max_idx):
                        ii+=1
                        continue
                    ii+=1

                    video_title = f'{_video}__{obj_id}'
                    n_frames = len(os.listdir(path.join(self.image_dir, _video)))

                    self.videos.append(video_title)
                    self.num_frames[video_title] = n_frames
                    self.shape[video_title] = np.shape(_mask)


        # transforms for mivos
        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])


    def __len__(self):
        return len(self.videos)


    def __getitem__(self, index):
        video_title = self.videos[index]
        info = {}
        info['name'] = video_title
        info['num_frames'] = self.num_frames[video_title] 
        info['shape'] = self.shape[video_title]

        images = []
        masks = []
        video, object_id = video_title.split('__')
        object_id = int(object_id)
        
        if self.second_type_transform:
            images_for_encoder = []

        for f in range(self.num_frames[video_title]):
            img_file = path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
            img = Image.open(img_file).convert('RGB')
            images.append(self.im_transform(img))
            
            if self.second_type_transform:
                images_for_encoder.append(self.second_type_transform(img))
            

            mask_file = path.join(self.mask_dir, video, '{:05d}.png'.format(f))
            mask = Image.open(mask_file).convert('P')
            masks.append(np.array(mask, dtype=np.uint8))


        images = torch.stack(images, 0)
        masks = np.stack(masks, 0)

        # object of interest
        label = [object_id]        
        masks = torch.from_numpy(all_to_onehot(masks, label)).float()

        if self.resolution != '480p':
            masks = self.mask_transform(masks)
        masks = masks.unsqueeze(2)

        info['labels'] = [object_id]

        data = {
            'rgb': images,
            'gt': masks,
            'info': info,
        }
        
        if self.second_type_transform:
            images_for_encoder = torch.stack(images_for_encoder, 0)
            data['images_for_encoder'] = images_for_encoder

        return data
