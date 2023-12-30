import os
import numpy as np
from PIL import Image
import pandas as pd

import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from datasets.range_transform import im_normalization
from datasets.helpers import all_to_onehot

class AnnotTypeDB(Dataset):
    """Dataset to train the PPO agent that selects the most suitable annotation type"""
    def __init__(self, root, imset, sample_size=None):
        super().__init__()
        self.root = root
        self.sample_size = sample_size
        self.image_dir = os.path.join(root, 'Images')
        self.mask_dir = os.path.join(root, 'Masks')
        self.embeddings_path = os.path.join(root, 'SAM_Embeddings')
        mose_root = root.replace('AnnotDB', 'MOSE')
        self.gt_annotation_dir = os.path.join(mose_root, 'Annotations', '480p')
        
        self._read_csv(imset)

        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
            transforms.Resize((480, 854), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((480, 854), interpolation=transforms.InterpolationMode.NEAREST),
        ])


    def _read_csv(self, imset):
        df = pd.read_csv(os.path.join(self.root, f'{imset}.csv'))
        self.frame_annotations_df = df

        invalid_idxs = []
        for index, row in self.frame_annotations_df.iterrows(): 
            state_id = row['id']
            img_file = os.path.join(self.image_dir, f'{state_id}.png')
            try:
                Image.open(img_file).convert('RGB')
            except OSError:
                invalid_idxs.append(index)
        self.frame_annotations_df.drop(invalid_idxs, inplace=True)
        self.frame_annotations_df.reset_index(drop=True, inplace=True)
        #print(f'Number of invalid images: {len(invalid_idxs)}')
        
        self.grouped_df = self.frame_annotations_df.groupby('video_name')
        self.sample_df()
    

    def __len__(self):
        if self.sample_size is not None:
            return len(self.sampled_df)
        else :
            return len(self.frame_annotations_df)
    
    def __random_sample(self, group):
        return group.sample(min(len(group), self.sample_size))

    def sample_df(self):
        if self.sample_size is not None:
            self.sampled_df = self.grouped_df.apply(self.__random_sample)


    def __getitem__(self, index):
        if self.sample_size is not None:
            row = self.sampled_df.iloc[index]
        else :
            row = self.frame_annotations_df.iloc[index]
        state_id = row['id']
        
        mask_file = os.path.join(self.mask_dir, f'{state_id}.png')
        mask = Image.open(mask_file).convert('P')
        mask = np.array(mask, dtype=np.uint8)
        mask = torch.from_numpy(mask)/255
        mask = self.mask_transform(mask.unsqueeze(0)).squeeze()
        embedding_file = os.path.join(self.embeddings_path, f'{state_id}.npy')
        sam_embedding = np.load(embedding_file)

        img_file = os.path.join(self.image_dir, f'{state_id}.png')
        img = Image.open(img_file).convert('RGB')
        img = self.im_transform(img)
 
        no_obj_name, label = row['video_name'].split('__')
        frame_num = row['frame_num']
        gt_mask_file = os.path.join(self.gt_annotation_dir, no_obj_name, f'{frame_num:05d}.png')
        gt_mask = Image.open(gt_mask_file).convert('P')
        
        gt_mask = np.array(gt_mask, dtype=np.uint8)
        gt_mask = all_to_onehot(gt_mask, [int(label)]).squeeze(0)
        gt_mask = torch.from_numpy(gt_mask)
        gt_mask = self.mask_transform(gt_mask).squeeze()

        data = {
            'sam_embedding': sam_embedding,
            'mask': mask,
            'img':img,
            'gt_mask': gt_mask
        }
        return data