import os
import argparse
import pandas as pd
import numpy as np
import re

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import AnnotationDataset
from util.mypath import Path

# interact policies
from interactions.mask import qnet_mask, rand_mask, oracle_mask, l2_mask, upper_bound_mask
from interactions.mulitple_annotations import oracle_oracle, rand_type, rand_rand, eva_vos

# EVA-VOS
from models.qnet import QualityNet
from ppo.ppo_agent import PPOAgent

# models
from feature_extractors import DINOFeatureExtractor, ViTFeatureExtractor, ResnetFeatureExtractor
from annotator import Annotator
from mivos.model.propagation.prop_net import PropagationNetwork
from mivos.model.fusion_net import FusionNet

""" Arguments loading """
parser = argparse.ArgumentParser()
parser.add_argument('--rounds', type=int, default=60, help='Rounds of interactions')
parser.add_argument('--policy', default='eva_vos', help='Policy for rounds')
parser.add_argument('--db', type=str, default='MOSE')
parser.add_argument('--encoder', type=str, default='resnet50', help='Only used with l2_mask policy')
parser.add_argument('--min-idx', type=int, help='From min-idx until max-idx')
parser.add_argument('--max-idx', type=int, help='From min-idx until max-idx')

# multiple annotations arguments
parser.add_argument('--types', nargs='+', help='Annotation types', default=['3clicks', 'mask']) # , 'click', 'bbox', 'mask', '3clicks', '10clicks', etc.

args = parser.parse_args()
assert args.db in {'MOSE', 'DAVIS_17'}
assert args.rounds >= 1 , "At least one round is required"

torch.autograd.set_grad_enabled(False)

"""Paths"""
if args.db =='MOSE':
    db_root_path = Path.db_root_path(args.db)
    imset = os.path.join(db_root_path, 'ImageSets', f'test.txt')
else:
    db_root_path = Path.db_root_path(args.db)
    imset = os.path.join(db_root_path, 'ImageSets/2017', f'val.txt')

prop_path =  './model_weights/mivos/stcn_yt_vos.pth'
fusion_path = './model_weights/mivos/fusion_stcn_yt_vos.pth'

"""Models"""
prop_saved = torch.load(prop_path)
prop_model = PropagationNetwork().cuda().eval()
prop_model.load_state_dict(prop_saved)

fusion_saved = torch.load(fusion_path)
fusion_model = FusionNet().cuda().eval()
fusion_model.load_state_dict(fusion_saved)

second_type_transform = None
policy_str = f'{args.policy}'

if args.policy == 'qnet_mask' or args.policy == 'eva_vos':
    qnet = QualityNet().cuda().eval()
    qnet.load_state_dict(torch.load('./model_weights/qnet/qnet.pth'))

    if args.policy == 'eva_vos':
        rl_agent = PPOAgent(2, 'resnet18', './model_weights/rl_agent/model.pth')
        annotator = Annotator()

elif args.policy == 'l2_mask':
    if args.encoder.__contains__('dino'):
        arch = args.encoder.split('_')[1] # e.g. dino_large
        encoder = DINOFeatureExtractor(arch)

    elif args.encoder.__contains__('vit'):
        arch = args.encoder.split('_')[1] # e.g. vit_large
        encoder = ViTFeatureExtractor(arch)

    elif args.encoder.__contains__('resnet'):
        # e.g. resnet50
        encoder = ResnetFeatureExtractor(args.encoder)
    else :
        raise AttributeError(f'{args.encoder} is invalid!')
    
    second_type_transform = encoder.transforms
    policy_str += f'_{args.encoder}'

elif args.policy in {'oracle_oracle', 'rand_type', 'rand_rand'}:
    avail_types = ['click', 'bbox', 'mask']
    annotation_types = args.types

    for annot_type in sorted(annotation_types):
        if annot_type not in avail_types:
            pattern = r'^\d+clicks$'
            if not re.match(pattern, annot_type):
                raise AttributeError('Invalid annotation type')
        policy_str += f'_{annot_type}'
    
    annotator = Annotator()

    if args.policy.__contains__('type'):
        assert len(annotation_types) == 1, f'Only one annotation type for {args.policy}'
    annotation_type = annotation_types[0]

"""Dataset and Dataloader"""
db = AnnotationDataset(db_root_path, imset=imset, \
    second_type_transform=second_type_transform, min_idx=args.min_idx, max_idx=args.max_idx
)
dataloader = DataLoader(db , batch_size=1)

if args.min_idx is not None and args.max_idx is not None:
    policy_str += f'from_{args.min_idx}_to_{args.max_idx}'
    
"""Inference"""
policy_results = {
    'video': [],
    'mu_metric': [],
    'annotation_time': [],
    'round':[]
}

if args.policy == 'eva_vos':
    policy_results['rl_values'] = []
    policy_results['round_metrics'] = []
    policy_results['annotated_frames'] = []

elif args.policy == 'oracle_oracle':
    policy_results['round_metrics'] = []
    policy_results['annotated_frames'] = []

if args.policy in {'oracle_oracle', 'rand_type', 'rand_rand', 'eva_vos'}:
    policy_results['annotation_actions'] = []

for data in tqdm(dataloader, desc=f'{policy_str}'):
    video_name = data['info']['name']
    ### mask only
    if args.policy == 'qnet_mask':
        mu_metrics, annotation_times = qnet_mask(qnet, args.rounds, prop_model, fusion_model, data, 'j_and_f')
    
    elif args.policy == 'rand_mask':
        mu_metrics, annotation_times = rand_mask(args.rounds, prop_model, fusion_model, data, 'j_and_f')
    
    elif args.policy == 'oracle_mask':
        mu_metrics, annotation_times = oracle_mask(args.rounds, prop_model, fusion_model, data, 'j_and_f')
    
    elif args.policy == 'l2_mask':
        mu_metrics, annotation_times = l2_mask(encoder, args.rounds, prop_model, fusion_model, data, 'j_and_f')
    
    elif args.policy == 'upper_bound_mask':
        mu_metrics, annotation_times = upper_bound_mask(args.rounds, prop_model, fusion_model, data, 'j_and_f')
   
    ### mulitple annotations
    elif args.policy == 'oracle_oracle':
        mu_metrics, annotation_times, annotations_actions, round_metrics, annotated_frames = oracle_oracle(args.rounds, prop_model, fusion_model, data, annotator, annotation_types, 'j_and_f')
        policy_results['round_metrics'].extend(round_metrics)
        policy_results['annotated_frames'].extend(annotated_frames)

    elif args.policy == 'rand_type':
        mu_metrics, annotation_times, annotations_actions = rand_type(args.rounds, prop_model, fusion_model, data, annotator, annotation_type, 'j_and_f')

    elif args.policy == 'rand_rand':
        mu_metrics, annotation_times, annotations_actions = rand_rand(args.rounds, prop_model, fusion_model, data, annotator, annotation_types, 'j_and_f')

    elif args.policy == 'eva_vos':
        mu_metrics, annotation_times, rl_values, annotations_actions, round_metrics, annotated_frames = eva_vos(qnet, rl_agent, args.rounds, prop_model, fusion_model, data, annotator, eval_metric='j_and_f')
        policy_results['rl_values'].extend(rl_values)
        policy_results['round_metrics'].extend(round_metrics)
        policy_results['annotated_frames'].extend(annotated_frames)

    else:
        raise AttributeError(f'Policy: {args.policy} is invalid!')
    
    policy_results['video'].extend(video_name*len(mu_metrics))
    policy_results['mu_metric'].extend(mu_metrics)
    policy_results['annotation_time'].extend(annotation_times)
    policy_results['round'].extend(np.arange(len(mu_metrics)))

    if args.policy in {'oracle_oracle', 'rand_type', 'rand_rand', 'eva_vos'}:
        policy_results['annotation_actions'].extend(annotations_actions)
    
os.makedirs(f'./Experiments/{args.db}', exist_ok=True)

df = pd.DataFrame.from_dict(policy_results)
df.to_csv(os.path.join(f'./Experiments/{args.db}', f'{policy_str}.csv'), index=False)