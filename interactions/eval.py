import torch
from copy import deepcopy
import numpy as np

from .metrics import *
from mivos.inference_core import InferenceCore

def get_segmentations(processor, rgb, device='cuda'):    

    # Do unpad -> upsample to original size 
    out_masks = torch.zeros((processor.t, 1, *rgb.shape[-2:]), dtype=torch.float32, device=device)
    for ti in range(processor.t):
        prob = processor.prob[:,ti]

        if processor.pad[2]+processor.pad[3] > 0:
            prob = prob[:,:,processor.pad[2]:-processor.pad[3],:]
        if processor.pad[0]+processor.pad[1] > 0:
            prob = prob[:,:,:,processor.pad[0]:-processor.pad[1]]

        out_masks[ti] = torch.argmax(prob, dim=0)*255
    
    out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)

    return out_masks


def eval_processor_metric(processor, data, interacted_frames, frame_iteraction_type, masks_from_sam=None, metric='j', device='cuda'):
    """Evaluates MiVOS for different annotation types
    frame_iteraction_type: for each frame has one of the following values
        0: no interaction
        1: segmentation masks (gt masks)
        2: click or bbox
    """
    assert metric in {'j', 'j_and_f'}
    # B x T x C x H x W
    rgb = data['rgb'].to(device)
    msk = data['gt'][0].to(device)
    h, w = msk.shape[-2], msk.shape[-1]
    info = data['info']
    #k = len(info['labels'][0])

    out_masks = get_segmentations(processor, rgb)       
    gt_masks = data['gt'].squeeze()
    frame_quality = []
    frame_quality_all = []

    frame_num = 0
    # masks that are generate with the model and with the interactios
    gen_masks = out_masks.copy()/255 # T x H x W
    for pred_mask,gt_mask,img in zip(out_masks,gt_masks,data['rgb'].squeeze()):
        
        pred_mask = torch.from_numpy(np.expand_dims(pred_mask,axis=0))      
                    
        pred = pred_mask.to(torch.bool)
        gt_mask = gt_mask.to(torch.bool).unsqueeze(0)
        
        if frame_num in interacted_frames and frame_iteraction_type[frame_num] == 1:
            pred = gt_mask.clone()
            pred_mask = gt_mask.clone()
            gen_masks[frame_num,:,:] = gt_mask
        elif frame_num in interacted_frames and frame_iteraction_type[frame_num] == 2:
            pred = masks_from_sam[frame_num].cpu().bool().unsqueeze(0)
            pred_mask = pred.clone()
            gen_masks[frame_num,:,:] = pred_mask.squeeze()
        
        n_not_zero = torch.count_nonzero(gt_mask)

        if n_not_zero == 0:
            frame_quality_all.append(20)# 20 is a token for zero gt-masks!
        else :
            if metric == 'j':
                seg_quality = compute_iou(pred, gt_mask)
            else :
                seg_quality = get_j_and_f(pred, gt_mask)

            frame_quality.append(seg_quality)
            frame_quality_all.append(seg_quality)
            
        frame_num+=1

    return np.mean(np.array(frame_quality)), gen_masks.copy(), frame_quality, frame_quality_all


def not_avail_frames(ious, interacted_frames, num_frames):
    zgt = np.where(np.array(ious) == 20)[0].tolist()
    all_not_avail_frames = set(zgt + interacted_frames)
    avail_frames = list(set( set(list(range(0,num_frames))) - all_not_avail_frames) )

    return len(avail_frames) == 0


def initialize(prop_model, fusion_model, data, device):
    info = data['info']    
    k = len(info['labels'][0])
    num_frames = info['num_frames'].item()
    masks = data['gt'][0].to(device)
    images = data['rgb'].squeeze().to(device)

    processor = InferenceCore(prop_model,fusion_model, images.unsqueeze(0), k)
    frame_iteraction_type = np.zeros((num_frames,))
    frame_iteraction_type[0] = 1 # mask
    metric = None
    frames_list = [0]
    mu_metrics = []
    annotation_times = [80]

    frame_dict = {
        'annotations' : [],
        'click_labels': None,
        'click_coords': None,
        'bbox': None,
        'sam_logits': None,
        'metric': 0,
    }
    pf_annots = [deepcopy(frame_dict) for _ in range(num_frames)]

    return images, masks, num_frames, info, processor, frame_iteraction_type, metric, \
        frames_list, mu_metrics,annotation_times, pf_annots