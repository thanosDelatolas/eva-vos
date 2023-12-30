import torch
import numpy as np

from .eval import eval_processor_metric, not_avail_frames, initialize
from .policies import qnet_frame_selection, rand_frame_selection, get_frame_l2, get_frame_upper_bound

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""Mask interactions"""
def qnet_mask(qnet, rounds, prop_model, fusion_model, data, eval_metric='j'):

    images, masks, num_frames, info, processor, frame_iteraction_type, metric, \
        frames_list, mu_metrics,annotation_times, _ = initialize(prop_model, fusion_model, data, DEVICE)
    for r in range(1, rounds+1):

        if r >= num_frames:
            # all frames are annotated
            continue

        if metric is not None and not_avail_frames(ious=metric, interacted_frames=frames_list, num_frames=num_frames):          
            # all frames 
            continue

        # interact with a new frame each time
        frame = frames_list[r-1]
        processor.interact(masks[:,frame],frame)
        frame_iteraction_type[frame] = 1 # mask
        # evaluate
        mu,generated_masks,_,metric = eval_processor_metric(processor,data,frames_list, metric=eval_metric, frame_iteraction_type=frame_iteraction_type)
        mu_metrics.append(mu)

        # select qnet frame
        generated_masks = torch.from_numpy(generated_masks).unsqueeze(0).unsqueeze(2).cuda().to(torch.float32)
        selected_frame = qnet_frame_selection(qnet, images, generated_masks, frames_list)
        if metric[selected_frame] == 20: # no object
            annotation_times.append(3)
        else:
            annotation_times.append(80)

        frames_list.append(selected_frame)
    
    return mu_metrics, annotation_times[:-1]


def rand_mask(rounds, prop_model, fusion_model, data, eval_metric='j'):

    images, masks, num_frames, info, processor, frame_iteraction_type, metric, \
        frames_list, mu_metrics,annotation_times, _ = initialize(prop_model, fusion_model, data, DEVICE)
    for r in range(1, rounds+1):

        if r >= num_frames:
            # all frames are annotated
            continue

        if metric is not None and not_avail_frames(ious=metric, interacted_frames=frames_list, num_frames=num_frames):          
            # all frames 
            continue

        # interact with a new frame each time
        frame = frames_list[r-1]
        processor.interact(masks[:,frame],frame)
        frame_iteraction_type[frame] = 1 # mask
        # evaluate
        mu,generated_masks,_,metric = eval_processor_metric(processor,data,frames_list, metric=eval_metric, frame_iteraction_type=frame_iteraction_type)
        mu_metrics.append(mu)

        generated_masks = torch.from_numpy(generated_masks).unsqueeze(0).unsqueeze(2).cuda().to(torch.float32)
        selected_frame = rand_frame_selection(num_frames, frames_list)
        if metric[selected_frame] == 20: # no object
            annotation_times.append(3)
        else:
            annotation_times.append(80)

        frames_list.append(selected_frame)
    
    return mu_metrics, annotation_times[:-1]


def oracle_mask(rounds, prop_model, fusion_model, data, eval_metric='j'):

    images, masks, num_frames, info, processor, frame_iteraction_type, metric, \
        frames_list, mu_metrics,annotation_times, _ = initialize(prop_model, fusion_model, data, DEVICE)
    for r in range(1, rounds+1):

        if r >= num_frames:
            # all frames are annotated
            continue

        if metric is not None and not_avail_frames(ious=metric, interacted_frames=frames_list, num_frames=num_frames):          
            # all frames 
            continue

        # interact with a new frame each time
        frame = frames_list[r-1]
        processor.interact(masks[:,frame],frame)
        frame_iteraction_type[frame] = 1 # mask
        # evaluate
        mu,generated_masks,_,metric = eval_processor_metric(processor,data,frames_list, metric=eval_metric, frame_iteraction_type=frame_iteraction_type)
        mu_metrics.append(mu)

        generated_masks = torch.from_numpy(generated_masks).unsqueeze(0).unsqueeze(2).cuda().to(torch.float32)
        selected_frame = np.argmin(metric)
        if metric[selected_frame] == 20: # no object
            annotation_times.append(3)
        else:
            annotation_times.append(80)

        frames_list.append(selected_frame)
    
    return mu_metrics, annotation_times[:-1]


def oracle_mask_dataset(rounds, processor, data, eval_metric='j'):
    
    info = data['info']    
    k = len(info['labels'][0])
    num_frames = info['num_frames'].item()
    masks = data['gt'][0].to(DEVICE)

    frame_iteraction_type = np.zeros((num_frames,))
    metric = None
    generated_masks_per_round = []
    frames_list = [0]
    metric_list = []
    annotation_times = []
    for r in range(1, rounds+1):

        if r >= num_frames:
            # all frames are annotated
            continue

        if metric is not None and not_avail_frames(ious=metric, interacted_frames=frames_list, num_frames=num_frames):          
            # all frames are annotated
            continue

        # interact with a new frame each time
        frame = frames_list[r-1]
        processor.interact(masks[:,frame],frame)
        frame_iteraction_type[frame] = 1 # mask
        # evaluate
        mu,generated_masks,_,metric = eval_processor_metric(processor,data,frames_list, metric=eval_metric, frame_iteraction_type=frame_iteraction_type)

        generated_masks = torch.from_numpy(generated_masks).unsqueeze(0).unsqueeze(2).cuda().to(torch.float32)

        worst_frame = np.argmin(metric)
        frames_list.append(worst_frame)
        generated_masks_per_round.append(generated_masks)
        metric_list.append(metric)

        if metric[worst_frame] == 20:# 20 is a token for zero gt-masks!
            annotation_time = 3
        else :
            annotation_time = 80
        annotation_times.append(annotation_time)

    return generated_masks_per_round, frames_list[1:], metric_list, annotation_times


def l2_mask(encoder, rounds, prop_model, fusion_model, data, eval_metric='j'):
    """Used to select a frame with pretrained image encoders"""
    images, masks, num_frames, info, processor, frame_iteraction_type, metric, \
        frames_list, mu_metrics,annotation_times, _ = initialize(prop_model, fusion_model, data, DEVICE)
    
    encoder_images = data['images_for_encoder'][0].cuda()
    for r in range(1, rounds+1):

        if r >= num_frames:
            # all frames are annotated
            continue

        if metric is not None and not_avail_frames(ious=metric, interacted_frames=frames_list, num_frames=num_frames):          
            # all frames 
            continue

        # interact with a new frame each time
        frame = frames_list[r-1]
        processor.interact(masks[:,frame],frame)
        frame_iteraction_type[frame] = 1 # mask
        # evaluate
        mu,generated_masks,_,metric = eval_processor_metric(processor,data,frames_list, metric=eval_metric, frame_iteraction_type=frame_iteraction_type)
        mu_metrics.append(mu)

        # select frame
        generated_masks = torch.from_numpy(generated_masks).unsqueeze(0).unsqueeze(2).cuda().to(torch.float32)
        selected_frame = get_frame_l2(encoder_images, frames_list, encoder)
        if metric[selected_frame] == 20: # no object
            annotation_times.append(3)
        else:
            annotation_times.append(80)

        frames_list.append(selected_frame)
    
    return mu_metrics, annotation_times[:-1]


def upper_bound_mask(rounds, prop_model, fusion_model, data, eval_metric='j'):

    images, masks, num_frames, info, processor, frame_iteraction_type, metric, \
        frames_list, mu_metrics,annotation_times, _ = initialize(prop_model, fusion_model, data, DEVICE)
    
    for r in range(1, rounds+1):

        if r >= num_frames:
            # all frames are annotated
            continue

        if metric is not None and not_avail_frames(ious=metric, interacted_frames=frames_list, num_frames=num_frames):          
            # all frames 
            continue

        # interact with a new frame each time
        frame = frames_list[r-1]
        processor.interact(masks[:,frame],frame)
        frame_iteraction_type[frame] = 1 # mask
        # evaluate
        mu,generated_masks,_,metric = eval_processor_metric(processor,data,frames_list, metric=eval_metric, frame_iteraction_type=frame_iteraction_type)
        mu_metrics.append(mu)

        generated_masks = torch.from_numpy(generated_masks).unsqueeze(0).unsqueeze(2).cuda().to(torch.float32)
        selected_frame = get_frame_upper_bound(processor, data, frames_list, frame_iteraction_type, eval_metric)
        if metric[selected_frame] == 20: # no object
            annotation_times.append(3)
        else:
            annotation_times.append(80)

        frames_list.append(selected_frame)
    
    return mu_metrics, annotation_times[:-1]