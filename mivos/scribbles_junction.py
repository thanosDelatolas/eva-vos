import torch
import numpy as np
import cv2
from copy import deepcopy
from scribble_robot.utils import scribbles2mask
from .inference_core import InferenceCore

from mivos.model.s2m.s2m_network import deeplabv3plus_resnet50 as S2M
from .tensor_util import pad_divide_by
from mivos.model.aggregate import aggregate_sbg, aggregate_wbg

class MiVOSInteractive:
    """
    Acts as the junction between DAVIS interactive track and our inference_core
    """
    def __init__(self, prop_net, fuse_net, s2m_net, images, num_objects, device='cuda:0', dataset_name='DAVIS_17'):
        self.s2m_net = s2m_net.to(device, non_blocking=True)
        self.processor = InferenceCore(prop_net, fuse_net, images, num_objects, mem_profile=0, device=device)

        # True dimensions
        t = images.shape[1]
        h, w = images.shape[-2:]
        self.t, self.h, self.w = t, h, w

        images, self.pad = pad_divide_by(images, 16, images.shape[-2:])
        self.device = device

        # Padded dimensions
        nh, nw = images.shape[-2:]
        self.nh, self.nw = nh, nw

        self.k = num_objects
        
        self.interacted_count = 0
        self.interacted_frames = []
        self.scribble_masks = {}

        

    def to_mask(self, scribble, frame_idx):
        # # First we select the only frame with scribble
        # all_scr = scribble['scribbles']
        # # all_scr is a list. len(all_scr) == total number of frames
        # for idx, s in enumerate(all_scr):
        #     # The only non-empty element in all_scr is the frame that has been interacted with
        #     if len(s) != 0:
        #         scribble['scribbles'] = [s]
        #         # since we break here, idx will remain at the interacted frame and can be used below
        #         break
            
        # Pass to DAVIS to change the path to an array
        scr_mask = scribbles2mask(scribble, (self.h, self.w))[0]
        user_scribble = deepcopy(scr_mask)

        positive_scribbles = np.zeros((self.k, self.h, self.w), dtype=np.uint8)
        negative_scribbles = np.zeros((self.k, self.h, self.w), dtype=np.uint8)

        # Run our S2M
        kernel = np.ones((3,3), np.uint8)
        mask = torch.zeros((self.k, 1, self.nh, self.nw), dtype=torch.float32, device=self.device)
        for ki in range(1, self.k+1):
            p_srb = (scr_mask==ki).astype(np.uint8)
            p_srb = cv2.dilate(p_srb, kernel).astype(np.bool_)
            positive_scribbles[ki-1] = p_srb

            n_srb = ((scr_mask!=ki) * (scr_mask!=-1)).astype(np.uint8)
            n_srb = cv2.dilate(n_srb, kernel).astype(np.bool_)
            negative_scribbles[ki-1] = n_srb

            Rs = torch.from_numpy(np.stack([p_srb, n_srb], 0)).unsqueeze(0).float().to(self.device)
            Rs, _ = pad_divide_by(Rs, 16, Rs.shape[-2:])

            # Use hard mask because we train S2M with such
            inputs = torch.cat([self.processor.get_image_buffered(frame_idx), 
                        (self.processor.masks[frame_idx]==ki).to(self.device).float().unsqueeze(0), Rs], 1)
            mask[ki-1] = torch.sigmoid(self.s2m_net(inputs))
        mask = aggregate_wbg(mask, keep_bg=True, hard=True)

        return mask, positive_scribbles, negative_scribbles, user_scribble


    def interact(self, interaction_input, frame_idx, scribble_interaction=True):

        if not scribble_interaction:
            mask_to_interact = interaction_input
            user_scribble = positive_scribbles = negative_scribbles = None
        else :
            scribble_mask, positive_scribbles, negative_scribbles, user_scribble = self.to_mask(interaction_input, frame_idx)  
            mask_to_interact = scribble_mask

        out_masks = self.processor.interact(mask_to_interact, frame_idx, scribble=scribble_interaction)
        
        
        self.interacted_count += 1
        self.interacted_frames.append(frame_idx)

        # if scribble_interaction:
        #     # Trim paddings
        #     if self.pad[2]+self.pad[3] > 0:
        #         out_masks = out_masks[:,self.pad[2]:-self.pad[3],:]
                
        #     if self.pad[0]+self.pad[1] > 0:
        #         out_masks = out_masks[:,:,self.pad[0]:-self.pad[1]]

        return out_masks, positive_scribbles, negative_scribbles, user_scribble
