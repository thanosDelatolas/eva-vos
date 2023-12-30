import os
import argparse
import numpy as np
from copy import deepcopy
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mivos.model.propagation.prop_net import PropagationNetwork
from mivos.model.fusion_net import FusionNet
from mivos.inference_core import InferenceCore
from interactions.eval import eval_processor_metric
from interactions.mulitple_annotations import oracle_action

from datasets import AnnotationDataset
from datasets.helpers import *

from util.mypath import Path
from annotator import Annotator

torch.autograd.set_grad_enabled(False)

"""Argument Loading"""
parser = argparse.ArgumentParser()
parser.add_argument('--imset', type=str, default='subset_train_1')
parser.add_argument('--rounds', type=int, default=10)

args = parser.parse_args()
imset_f = args.imset
assert imset_f in {'subset_train_1', 'subset_train_2', 'subset_train_3', 'subset_train_4', 'val', 'test'}

"""Paths"""
mose_root = Path.db_root_path('MOSE')
imset = os.path.join(mose_root, 'ImageSets', f'{imset_f}.txt')

prop_path =  './model_weights/mivos/stcn_yt_vos.pth'
fusion_path = './model_weights/mivos/fusion_stcn_yt_vos.pth'

annotation_db = Path.db_root_path('AnnotDB')

images_dir = os.path.join(annotation_db, 'Images')
masks_dir = os.path.join(annotation_db, 'Masks')
sam_embeddings_dir = os.path.join(annotation_db, 'SAM_Embeddings')
os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)
os.makedirs(sam_embeddings_dir, exist_ok=True)

"""Dataset and dataloader"""
dataset = AnnotationDataset(mose_root, imset=imset)
db_loader = DataLoader(dataset, batch_size=1, shuffle=False)

"""Models"""
annotator = Annotator(prompt_type='c')

prop_model = PropagationNetwork(top_k=50).cuda().eval()
prop_model.load_state_dict(torch.load(prop_path))

fusion_model = FusionNet().cuda().eval()
fusion_model.load_state_dict(torch.load(fusion_path))

db_data = {
    'id': [],
    'frame_cost': [],
    'video_cost': [],
    'selected_annotation': [],
    'frame_num':[],
    'round':[],
    'video_name':[],

    'init_iou': [],
    #'click_iou': [],
    '3clicks_iou':[],
    'mask_iou':[],
}
annotation_types = ['3clicks', 'mask']
for data in tqdm(db_loader, desc=f'Creating annot db for {imset_f}'): 
    # B x T x C x H x W
    images = data['rgb'].cuda()
    masks = data['gt'][0].cuda()
    info = data['info']    
    k = len(info['labels'][0])
    num_frames = info['num_frames'].item()
    name = info['name'][0]
    processor = InferenceCore(prop_model,fusion_model, images, k)
        
    frames_list = [0]
    metric = None
    metric_no_zgt = None

    frame_iteraction_type = np.zeros((num_frames,))
    frame_iteraction_type[0] = 1 # mask
    masks_from_sam = {}
    frames_cost = np.zeros((num_frames,))

    frame_dict = {
        'annotations' : [],
        'click_labels': None,
        'click_coords': None,
        'bbox': None,
        'sam_logits': None,
        'metric': 0,
    }
    pf_annots = [deepcopy(frame_dict) for _ in range(num_frames)]

    for r in range(1,args.rounds+1):
        if metric_no_zgt is not None and np.min(metric_no_zgt) == 1.0:
            continue
        
        frame = frames_list[r-1]
        
        if r > 1:
            init_iou = metric[frame]
            sam_mask, cost, ann_action, sam_logits, prompt_clicks, prompt_labels, bbox, action_data = oracle_action(annotator=annotator, annotation_types=annotation_types,  frame_annots=pf_annots[frame],\
                gt_mask=masks[:,frame].squeeze(), mivos_mask=gen_masks[frame], im=images[:,frame].squeeze(), frame_num=frame, return_action_data=True
            )
            
            img_embedding = annotator.sam_controller.predictor.get_image_embedding()

            if ann_action == 'mask':
                frame_iteraction_type[frame] = 1
                mask_for_iteraction = masks[:,frame]
            else :
                mask_for_iteraction = sam_mask.float().unsqueeze(0).cuda()
                frame_iteraction_type[frame] = 2
                masks_from_sam[frame] = mask_for_iteraction.squeeze()
                pf_annots[frame]['click_labels'] = prompt_labels
                pf_annots[frame]['click_coords'] = prompt_clicks
                pf_annots[frame]['bbox'] = bbox
                pf_annots[frame]['sam_logits'] = sam_logits

        else :
            mask_for_iteraction = masks[:,frame]
            cost = 80
            ann_action = 'mask'
        
        pf_annots[frame]['annotations'].append(ann_action)
        frames_cost[frame] += cost
        processor.interact(mask_for_iteraction,frame)
        
        # evaluate
        _,gen_masks, metric_no_zgt, metric = eval_processor_metric(processor,data,frames_list, frame_iteraction_type, masks_from_sam, metric='j')
        gen_masks = torch.from_numpy(gen_masks).cuda().float()
        for ii, m in enumerate(metric):
            pf_annots[ii]['metric'] = m
        
        if r > 1:
            file_id = f'{name}_{r}_frame_{frame}'
            db_data['id'].append(file_id)
            db_data['frame_cost'].append(frames_cost[frame])
            db_data['video_cost'].append(np.sum(frames_cost))
            db_data['selected_annotation'].append(ann_action)
            db_data['frame_num'].append(frame)
            db_data['round'].append(r)
            db_data['video_name'].append(name)
            db_data['init_iou'].append(init_iou)
            for action in annotation_types:
                db_data[f'{action}_iou'].append(action_data[action]['iou'])
       
            # save masks
            ma = deepcopy(gen_masks[frame]).cpu()
            ma_img = tens2image(ma)
            canvas = (ma_img * 255).astype(np.uint8)
            img_E = Image.fromarray(canvas)
            img_E.save(os.path.join(masks_dir, f'{file_id}.png'))

            # save img
            rgb_frame = deepcopy(images[:,frame]).cpu()
            rgb_img = im_normalize(tens2image(rgb_frame))
            canvas = (rgb_img * 255).astype(np.uint8)
            img_E = Image.fromarray(canvas)
            img_E.save(os.path.join(images_dir,f'{file_id}.png'))

            np.save(os.path.join(sam_embeddings_dir, f'{file_id}.npy'), img_embedding.cpu().numpy().squeeze())
        selected_frame = np.argmin(metric)
        frames_list.append(selected_frame) 


df = pd.DataFrame.from_dict(db_data)
df.to_csv(os.path.join(annotation_db, f'{imset_f}.csv'), index=False)