import os 
from tqdm import tqdm
import numpy as np
import argparse
import pandas as pd

import torch
from torch.utils.data import DataLoader

# MiVOS
from mivos.model.propagation.prop_net import PropagationNetwork
from mivos.model.fusion_net import FusionNet
from mivos.inference_core import InferenceCore

# custom imports
from datasets import AnnotationDataset
from interactions.mask import oracle_mask_dataset
from util.fq_dataset import saver, save_frames
from util.mypath import Path

torch.autograd.set_grad_enabled(False)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fq_root = Path.db_root_path('FQ_DB')
os.makedirs(fq_root, exist_ok=True)
mose_root = Path.db_root_path('MOSE')

"""
Arguments loading
"""
parser = argparse.ArgumentParser()
parser.add_argument('--imset', type=str, default='subset_train_4', help='Generate for each training set indepedenlty')

args = parser.parse_args()
imset_str = args.imset
assert imset_str in {'subset_train_4', 'val'}
imset = os.path.join(mose_root, 'ImageSets', f'{imset_str}.txt')
"""
Datasets
"""
db = AnnotationDataset(mose_root, imset=imset)
dataloader = DataLoader(db , batch_size=1)

"""
Load models
"""
prop_saved = torch.load('./model_weights/mivos/stcn_yt_vos.pth')
prop_model = PropagationNetwork(top_k=50).to(DEVICE).eval()
prop_model.load_state_dict(prop_saved)

fusion_saved = torch.load('./model_weights/mivos/fusion_stcn_yt_vos.pth')
fusion_model = FusionNet().to(DEVICE).eval()
fusion_model.load_state_dict(fusion_saved)

saved_rgb = []
results_dict = {
    'state_name':[], # e.g. bear__1_round_1 (video__obj_round_1)
    'ious': [],
    'selected_frame':[]
}
for data in tqdm(dataloader, total=len(dataloader), desc=f'FQ for {imset_str}'):
    
    state_id = 1
    images = data['rgb'].to(DEVICE)
    video_name = data['info']['name'][0]
    no_obj_name = video_name.split('__')[0]   
    
    # oracle_mask states
    processor = InferenceCore(prop_model,fusion_model, images, 1)
    generated_masks_per_round, oracle_frame_list, ious_list,_ = oracle_mask_dataset(8, processor, data)
    assert len(generated_masks_per_round) == len(oracle_frame_list) == len(ious_list)
    dont_save = []
    for ii in range(len(ious_list)):
        iou = ious_list[ii]
        if np.argmin(iou) != oracle_frame_list[ii]:
            dont_save.append(ii)
   
    state_id, results_dict = saver(generated_masks_per_round,oracle_frame_list, ious_list, video_name, state_id, fq_root, results_dict, full_res=False, dont_save=dont_save)

    orig_video = video_name.split('__')[0]
    if orig_video not in saved_rgb:
        save_frames(images, orig_video, fq_root, full_res=False)
        saved_rgb.append(orig_video)
    

df = pd.DataFrame.from_dict(results_dict)
df.to_csv(os.path.join(fq_root, f'res_{imset_str}.csv'), index=False)