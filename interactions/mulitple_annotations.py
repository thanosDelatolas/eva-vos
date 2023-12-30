import re
import random
from einops import repeat
from copy import deepcopy

import torch
from torchvision import transforms

from datasets.helpers import *
from .eval import compute_iou, not_avail_frames, initialize, eval_processor_metric
from .policies import qnet_frame_selection

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def reward_func(iou, cost, init_iou):
    return (iou-init_iou)/cost


def ann_type_to_annotator_input(annot_type):
    """Returns annot and conf num for the annotator class"""
    if annot_type == 'click':
        return 'click', 1
    elif annot_type == 'bbox':
        return 'bbox', 1
    elif re.match(r'^\d+clicks$', annot_type):
        num_clicks = int(annot_type.split('clicks')[0])
        return 'click', num_clicks

    elif annot_type == 'mask':
        return 'mask',1
    else :
        raise AttributeError(f'{annot_type} does not exist!')


def annotate(annotator, annot_type, gt_mask, im, mivos_mask=None, save_annotations_path=None, frame_annots=None):
    ann_type, num_prompts = ann_type_to_annotator_input(annot_type=annot_type)
    sam_mask, cost, curr_iou, sam_logits, prompt_clicks, prompt_labels, bbox  = annotator.get_mask(annotation_type=ann_type, num_prompts=num_prompts, gt_mask=gt_mask, im=im, mivos_mask=mivos_mask, save_annotations_path=save_annotations_path, prev_iter_data=frame_annots)
    return sam_mask, cost, curr_iou, sam_logits, prompt_clicks, prompt_labels, bbox
    

def oracle_action(annotator, annotation_types, gt_mask, mivos_mask, im, frame_annots, frame_num=-1, return_action_data=False):
    best_reward = -1e10
    best_action = None
    best_mask = None
    best_cost = 1e10
    best_logits = None
    best_click_prompts = None
    best_click_labels = None
    best_bbox = None 
    init_iou = compute_iou(gt_mask.unsqueeze(0).to(torch.bool), mivos_mask.unsqueeze(0).to(torch.bool))
    
    actions_data={
        'init_iou': init_iou,
        'frame_num': frame_num
    }
    for ann_type in annotation_types:
        if ann_type == 'bbox' and 'bbox' in frame_annots['annotations']:
            continue
        
        sam_mask, ann_cost, curr_iou, sam_logits, prompt_clicks, prompt_labels, bbox = annotate(annotator, ann_type, gt_mask, im, mivos_mask.unsqueeze(0).to(torch.bool), frame_annots=frame_annots)
        r = reward_func(curr_iou, ann_cost, init_iou)

        actions_data[ann_type]={
            'iou': curr_iou,
            'cost': ann_cost,
            'reward': r
        }
        if r >= best_reward:
            best_reward = deepcopy(r) 
            best_mask = deepcopy(sam_mask)
            best_action = deepcopy(ann_type)
            best_cost = deepcopy(ann_cost)
            best_logits = deepcopy(sam_logits)
            best_click_prompts = deepcopy(prompt_clicks)
            best_click_labels = deepcopy(prompt_labels)
            best_bbox = deepcopy(bbox)

    actions_data['selected_action'] = best_action


    if return_action_data:
        return best_mask, best_cost, best_action, best_logits, best_click_prompts, best_click_labels, best_bbox, actions_data
    else :
        return best_mask, best_cost, best_action, best_logits, best_click_prompts, best_click_labels, best_bbox


def store_action_data(frame, ann_action, frame_iteraction_type, masks, sam_mask, masks_from_sam, pf_annots, prompt_labels, prompt_clicks, bbox, sam_logits):
    if ann_action == 'mask':
        frame_iteraction_type[frame] = 1 # mask
        mask_for_iteraction = masks[:,frame]
    else :
        mask_for_iteraction = sam_mask.float().unsqueeze(0).cuda()
        frame_iteraction_type[frame] = 2 # click or bbox
        masks_from_sam[frame] = mask_for_iteraction.squeeze()
        pf_annots[frame]['click_labels'] = prompt_labels
        pf_annots[frame]['click_coords'] = prompt_clicks
        pf_annots[frame]['bbox'] = bbox
        pf_annots[frame]['sam_logits'] = sam_logits

    return frame_iteraction_type, pf_annots, mask_for_iteraction


def oracle_oracle(rounds, prop_model, fusion_model, data, annotator, annotation_types=['click', 'mask'], eval_metric='j'):
    """Oracle frame and annotation selections"""
    assert len(annotation_types) > 1, 'oracle_oracle requires more than one annotation types'

    images, masks, num_frames, info, processor, frame_iteraction_type, metric, \
        frames_list, mu_metrics, _, pf_annots = initialize(prop_model, fusion_model, data, DEVICE)
    annotation_times = []
    masks_from_sam = {}
    fully_annotated = False
    annotations_actions = []
    round_metrics = []
    for r in range(1, rounds+1):

        if (r >= num_frames and np.min(metric) == 1) or fully_annotated:
            # all frames are annotated
            continue

        if metric is not None and not_avail_frames(ious=metric, interacted_frames=frames_list, num_frames=num_frames):          
            # all frames 
            continue
        
        frame = frames_list[-1]
        if r > 1:
            sam_mask, cost, ann_action, sam_logits, prompt_clicks, prompt_labels, bbox, action_data = oracle_action(annotator=annotator, annotation_types=annotation_types,  frame_annots=pf_annots[frame],\
                gt_mask=masks[:,frame].squeeze(), mivos_mask=gen_masks[frame], im=images[frame].squeeze(), frame_num=frame, return_action_data=True
            )

            frame_iteraction_type, pf_annots, mask_for_iteraction = store_action_data(frame, ann_action, frame_iteraction_type, masks, sam_mask, masks_from_sam, pf_annots, prompt_labels, prompt_clicks, bbox, sam_logits)

        else :
            mask_for_iteraction = masks[:,frame]
            cost = 80
            ann_action = 'mask'
        
        pf_annots[frame]['annotations'].append(ann_action)
        processor.interact(mask_for_iteraction,frame)

        mu,gen_masks, _, metric = eval_processor_metric(processor,data,frames_list, frame_iteraction_type, masks_from_sam, metric=eval_metric)
        gen_masks = torch.from_numpy(gen_masks).cuda().float()
        for ii, m in enumerate(metric):
            pf_annots[ii]['metric'] = m
        
        not_annotated_with_mask_frames = np.where((frame_iteraction_type != 1))[0]       
        if len(not_annotated_with_mask_frames) == 0:
            fully_annotated = True

        selected_frame = np.argmin(metric)
        frames_list.append(selected_frame)
        
        mu_metrics.append(mu)
        annotation_times.append(cost)
        annotations_actions.append(ann_action)
        round_metrics.append(metric)

    return mu_metrics, annotation_times, annotations_actions, round_metrics, frames_list[:-1]


def rand_type(rounds, prop_model, fusion_model, data, annotator, annotation_type='3clicks', eval_metric='j'):
    """Random frame section and using only one annotation type"""
    assert type(annotation_type) == str, f'One annotation type is required'

    images, masks, num_frames, info, processor, frame_iteraction_type, metric, \
        frames_list, mu_metrics, _, pf_annots = initialize(prop_model, fusion_model, data, DEVICE)
    annotation_times = []
    masks_from_sam = {}
    fully_annotated = False
    annotations_actions = []
    for r in range(1, rounds+1):

        if (r >= num_frames and np.min(metric) == 1) or fully_annotated:
            # all frames are annotated
            continue

        if metric is not None and not_avail_frames(ious=metric, interacted_frames=frames_list, num_frames=num_frames):          
            # all frames 
            continue
        
        frame = frames_list[-1]
        if r > 1:
            mivos_mask = gen_masks[frame]
            sam_mask, cost, curr_iou, sam_logits, prompt_clicks, prompt_labels, bbox  = annotate(annotator=annotator, annot_type=annotation_type, gt_mask=masks[:,frame].squeeze(), \
                im=images[frame].squeeze(), mivos_mask=mivos_mask.unsqueeze(0).to(torch.bool), frame_annots=pf_annots[frame]
            )
            ann_action = annotation_type
            frame_iteraction_type, pf_annots, mask_for_iteraction = store_action_data(frame, ann_action, frame_iteraction_type, masks, sam_mask, masks_from_sam, pf_annots, prompt_labels, prompt_clicks, bbox, sam_logits)

        else :
            mask_for_iteraction = masks[:,frame]
            cost = 80
            ann_action = 'mask'
        
        if mask_for_iteraction.ndim == 3:
            mask_for_iteraction = mask_for_iteraction.unsqueeze(0)

        pf_annots[frame]['annotations'].append(ann_action)
        processor.interact(mask_for_iteraction,frame)

        mu,gen_masks, _, metric = eval_processor_metric(processor,data,frames_list, frame_iteraction_type, masks_from_sam, metric=eval_metric)
        gen_masks = torch.from_numpy(gen_masks).cuda().float()
        for ii, m in enumerate(metric):
            pf_annots[ii]['metric'] = m
        
        
        not_annotated_with_mask_frames = np.where((frame_iteraction_type != 1))[0]
        selected_frame =  np.random.choice(not_annotated_with_mask_frames, size=1)[0]
        
        if len(not_annotated_with_mask_frames) == 0:
            fully_annotated = True
        else :
            selected_frame = np.random.choice(not_annotated_with_mask_frames, size=1)[0]
            frames_list.append(selected_frame) 

        mu_metrics.append(mu)
        annotation_times.append(cost)
        annotations_actions.append(ann_action)
    
    return mu_metrics, annotation_times, annotations_actions


def rand_rand(rounds, prop_model, fusion_model, data, annotator, annotation_types=['3clicks', 'mask'], eval_metric='j'):
    """Random frame and annotation selections"""
    assert len(annotation_types) > 1, 'More than one annotation types are required'

    images, masks, num_frames, info, processor, frame_iteraction_type, metric, \
        frames_list, mu_metrics, _, pf_annots = initialize(prop_model, fusion_model, data, DEVICE)
    annotation_times = []
    masks_from_sam = {}
    fully_annotated = False
    annotations_actions = []
    for r in range(1, rounds+1):

        if (r >= num_frames and np.min(metric) == 1) or fully_annotated:
            # all frames are annotated
            continue

        if metric is not None and not_avail_frames(ious=metric, interacted_frames=frames_list, num_frames=num_frames):          
            # all frames 
            continue
        
        frame = frames_list[-1]
        if r > 1:
            mivos_mask = gen_masks[frame]
            ann_action = random.choice(annotation_types)
            sam_mask, cost, curr_iou, sam_logits, prompt_clicks, prompt_labels, bbox  = annotate(annotator=annotator, annot_type=ann_action, gt_mask=masks[:,frame].squeeze(), \
                im=images[frame].squeeze(), mivos_mask=mivos_mask.unsqueeze(0).to(torch.bool), frame_annots=pf_annots[frame]
            )
           
            frame_iteraction_type, pf_annots, mask_for_iteraction = store_action_data(frame, ann_action, frame_iteraction_type, masks, sam_mask, masks_from_sam, pf_annots, prompt_labels, prompt_clicks, bbox, sam_logits)

        else :
            mask_for_iteraction = masks[:,frame]
            cost = 80
            ann_action = 'mask'
        
        if mask_for_iteraction.ndim == 3:
            mask_for_iteraction = mask_for_iteraction.unsqueeze(0)

        pf_annots[frame]['annotations'].append(ann_action)
        processor.interact(mask_for_iteraction,frame)

        mu,gen_masks, _, metric = eval_processor_metric(processor,data,frames_list, frame_iteraction_type, masks_from_sam, metric=eval_metric)
        gen_masks = torch.from_numpy(gen_masks).cuda().float()
        for ii, m in enumerate(metric):
            pf_annots[ii]['metric'] = m
        
        
        not_annotated_with_mask_frames = np.where((frame_iteraction_type != 1))[0]
        selected_frame =  np.random.choice(not_annotated_with_mask_frames, size=1)[0]
        
        if len(not_annotated_with_mask_frames) == 0:
            fully_annotated = True
        else :
            selected_frame = np.random.choice(not_annotated_with_mask_frames, size=1)[0]
            frames_list.append(selected_frame) 

        mu_metrics.append(mu)
        annotation_times.append(cost)
        annotations_actions.append(ann_action)
    
    return mu_metrics, annotation_times, annotations_actions


def rl_agent_annotate(annotator, rl_agent, mivos_mask, gt_mask, im, frame_annots):
    mask_to_224 = transforms.Compose([
        transforms.Resize((224, )*2, interpolation=transforms.InterpolationMode.NEAREST),
    ])
    if frame_annots['metric'] == 20: # no object
        return gt_mask.unsqueeze(0), 3, 'no_object', None, None, None, None, 0

    annotator.set_image_to_sam(im)
    img_embedding = annotator.sam_controller.predictor.get_image_embedding()

    ma_224 = mask_to_224(mivos_mask.float())
    ma_224 = repeat(ma_224, 'b h w -> b 3 h w')
    avail_actions = ['3clicks', 'mask']
    with torch.no_grad():
        action, value = rl_agent.act(img_embedding, ma_224)
    ann_type = avail_actions[action]
    sam_mask, ann_cost, curr_iou, sam_logits, prompt_clicks, prompt_labels, bbox = annotate(annotator, ann_type, gt_mask, im, mivos_mask.unsqueeze(0).to(torch.bool), frame_annots=frame_annots)

    return sam_mask, ann_cost, ann_type, sam_logits, prompt_clicks, prompt_labels, bbox, value.squeeze().item()


def eva_vos(qnet, rl_agent, rounds, prop_model, fusion_model, data, annotator, annotation_types=['3clicks', 'mask'], eval_metric='j'):
    """EVA-VOS frame and annotation selection"""
    assert len(annotation_types) > 1, 'More than one annotation types are required'

    images, masks, num_frames, info, processor, frame_iteraction_type, metric, \
        frames_list, mu_metrics, _, pf_annots = initialize(prop_model, fusion_model, data, DEVICE)
    annotation_times = []
    masks_from_sam = {}
    fully_annotated = False
    rl_values = [-2]
    annotations_actions = []
    round_metrics = []
    for r in range(1, rounds+1):
        
        if (r >= num_frames and np.min(metric) == 1) or fully_annotated:
            # all frames are annotated with segmentation masks
            continue

        if metric is not None and not_avail_frames(ious=metric, interacted_frames=frames_list, num_frames=num_frames):          
            # all frames 
            continue
        
        frame = frames_list[-1]
        if r > 1:
            mivos_mask = gen_masks.squeeze()[frame]
           
            sam_mask, cost, ann_action, sam_logits, prompt_clicks, prompt_labels, bbox, rl_value  = rl_agent_annotate(annotator=annotator, rl_agent=rl_agent, gt_mask=masks[:,frame].squeeze(), \
                im=images[frame].squeeze(), mivos_mask=mivos_mask.unsqueeze(0).to(torch.bool), frame_annots=pf_annots[frame]
            )
            rl_values.append(rl_value)
            frame_iteraction_type, pf_annots, mask_for_iteraction = store_action_data(frame, ann_action, frame_iteraction_type, masks, sam_mask, masks_from_sam, pf_annots, prompt_labels, prompt_clicks, bbox, sam_logits)

        else :
            mask_for_iteraction = masks[:,frame]
            cost = 80
            ann_action = 'mask'
        
        if mask_for_iteraction.ndim == 3:
            mask_for_iteraction = mask_for_iteraction.unsqueeze(0)

        pf_annots[frame]['annotations'].append(ann_action)
        processor.interact(mask_for_iteraction,frame)

        mu,gen_masks, _, metric = eval_processor_metric(processor,data,frames_list, frame_iteraction_type, masks_from_sam, metric=eval_metric)

        gen_masks = torch.from_numpy(gen_masks).unsqueeze(0).unsqueeze(2).cuda().to(torch.float32)
        for ii, m in enumerate(metric):
            pf_annots[ii]['metric'] = m
        
        not_annotated_with_mask_frames = np.where((frame_iteraction_type != 1))[0]
        
        # select qnet frame
        gen_masks = gen_masks.unsqueeze(0).unsqueeze(2).cuda().to(torch.float32) 

        if r >= num_frames:
            not_annotated_with_mask_frames = np.where((frame_iteraction_type != 1))[0]
            if len(not_annotated_with_mask_frames) == 0:
                fully_annotated=True
                selected_frame=-1
            else :
                selected_frame = qnet_frame_selection(qnet, images, gen_masks, not_annotated_with_mask_frames)
        else :
            selected_frame = qnet_frame_selection(qnet, images, gen_masks, frames_list)

        frames_list.append(selected_frame) 

        mu_metrics.append(mu)
        annotation_times.append(cost)
        annotations_actions.append(ann_action)
        round_metrics.append(metric)
        
    return mu_metrics, annotation_times,rl_values, annotations_actions, round_metrics, frames_list[:-1]