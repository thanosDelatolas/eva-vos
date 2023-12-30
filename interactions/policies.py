from copy import deepcopy
from einops import repeat
import numpy as np
import random
import gc

import torch
from torchvision import transforms

from .eval import eval_processor_metric

mask_to_224 = transforms.Compose([
    transforms.Resize((224, )*2, interpolation=transforms.InterpolationMode.NEAREST, antialias=True), 
])

im_to_224 = transforms.Compose([
    transforms.Resize((224, )*2, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
])


def get_min_l2_dist(interacted_features, curr_features):
    ''' Returns the minimum l2 distance between the features of the current frame and 
    the feautures of the frames that are stored in memory
    '''

    min_l2 = np.Inf
    curr_features = curr_features.cpu().numpy()
    interacted_features = interacted_features.cpu().numpy()
    
    for features in interacted_features:        
        dist = np.linalg.norm(curr_features - features)

        if dist < min_l2:
            min_l2 = dist
    
    return min_l2


@torch.no_grad()
def qnet_frame_selection(qnet, frames, masks, interacted_frames):
    num_frames = frames.shape[0]

    imgs_ = im_to_224(frames)
    masks_ = mask_to_224(masks.squeeze())    
    masks_ = repeat(masks_, 'f h w -> f c h w', c=3)

    features = qnet.extract_features(imgs_, masks_)
    interacted_features = features[interacted_frames]

    max_l2 = -np.Inf
    frame_num = None
    for ii in range(num_frames):
        curr_features = features[ii]
        min_l2 = get_min_l2_dist(interacted_features, curr_features)

        if min_l2 > max_l2:
            max_l2 = min_l2
            frame_num = ii
    
    return frame_num


def rand_frame_selection(num_frames, interacted_frames):
    frames = np.arange(num_frames)
    not_interacted_frames = list(set(frames) - set(interacted_frames))
    selected_frame = random.choice(not_interacted_frames)
    return selected_frame

@torch.no_grad()
def get_frame_l2(images, interacted_frames, encoder):

    num_frames = images.shape[0]    
    # num_frames x latent space
    features = encoder.extract_features(images)
    interacted_features = features[interacted_frames]
    
    max_l2 = -np.Inf
    frame_num = -1

    for ii in range(num_frames):
        curr_features = features[ii]
        min_l2 = get_min_l2_dist(interacted_features, curr_features)
        
        if min_l2 > max_l2:
            max_l2 = min_l2
            frame_num = ii
    
    return frame_num

@torch.no_grad()
def get_frame_upper_bound(processor, data, prev_interacted_frames, frame_iteraction_type, metric='j'):

    best_metric = -np.Inf
    best_frame = -1
    masks = data['gt'][0].cuda()

    # 1 x frames x3xhxw
    rgb = data['rgb'].cuda()
    _,frames,_,_,_ = rgb.shape
    
    for f in range(frames):
        if f not in prev_interacted_frames:
            p = deepcopy(processor)
            p.interact(masks[:,f],f)
            interacted_frames = deepcopy(prev_interacted_frames)
            interacted_frames.append(f)
            f_type = deepcopy(frame_iteraction_type)
            f_type[f] = 1
            mu_metric, _,_,_ = eval_processor_metric(p,data,interacted_frames, metric=metric, frame_iteraction_type=f_type)
            
            if mu_metric >= best_metric:
                best_metric = mu_metric
                best_frame = f
            gc.collect()
            torch.cuda.empty_cache()
            del p 
            
    return best_frame