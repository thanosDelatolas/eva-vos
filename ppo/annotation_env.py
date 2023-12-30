import os
from copy import deepcopy
from einops import repeat

import torch
from torchvision import transforms
import numpy as np

# custom imports
from sam import SAMController
from robots.click_robot import ClickRobot
from robots.bbox_robot import BboxRobot

from util.mypath import Path
from util.helpers import ANNOTATION_COSTS
from interactions.eval import compute_iou
from annotator.util import *

class AnnotationEnv:
    """Environment used to train the RL Agent"""
    def __init__(self, image, gt_mask, init_mask, max_steps, use_cost=False, device='cuda'):
        super(AnnotationEnv, self).__init__()
        self.device = device
        self.sam_controller = SAMController('./model_weights/sam/sam.pth', self.device, verbose=False)
        self.click_robot = ClickRobot()
        self.bbox_robot = BboxRobot()
        self.image = image.squeeze()
        self.gt_mask = gt_mask.squeeze().to(torch.bool)
        self.init_mask = init_mask
        self.use_cost = use_cost
        self.max_steps = max_steps

        self.mask_to_224 = transforms.Compose([
            transforms.Resize((224, )*2, interpolation=transforms.InterpolationMode.NEAREST),
        ])

        self.reset()
        self.avail_actions = ['click', 'mask', 'stop'] 


    @torch.no_grad()
    def reset(self):
        self.set_image_to_sam(self.image)
        sam_logits, sam_mask, prompt_clicks, prompt_labels = self.create_similar_samlogits(self.init_mask.to(torch.bool))
        if sam_mask is None:
            sam_mask = torch.zeros_like(self.init_mask, dtype=torch.bool)

        self.sam_logits = sam_logits
        self.sam_mask = sam_mask
        self.prompt_clicks = prompt_clicks
        self.prompt_labels = prompt_labels
        self.iou = 0 if sam_mask is None else compute_iou(sam_mask, self.gt_mask.unsqueeze(0))
        self.init_iou = self.iou
        self.annotation_cost = 0
        self.bbox = None

        self.get_state()
        self.annotation_actions = []
        self.done = False
        


    def get_state(self):
        # self.state = torch.cat((self.image, self.sam_mask), dim=-0)
        # if self.use_cost:
        #     cost_mask = torch.ones_like(self.sam_mask) * self.annotation_cost
        #     self.state = torch.cat((self.image, cost_mask), dim=0)
        
        sam_mask = self.mask_to_224(self.sam_mask)
        self.state=[self.img_embedding.to(self.device), repeat(sam_mask, '1 h w -> 1 3 h w').to(self.device).float()]
        if self.use_cost:
            self.state.append(torch.tensor([[self.annotation_cost]], device=self.device).float())


    def set_image_to_sam(self, im):
        im = inv_im_trans(im)
        im = im.cpu().squeeze().numpy().transpose((1, 2, 0))
        im = (im * 255).astype(np.uint8)
        self.sam_controller.reset_image()
        self.sam_controller.set_image(im)
        self.img_embedding = deepcopy(self.sam_controller.predictor.get_image_embedding())
    

    def best_sam_mask(self, sam_masks, target_mask):
        mask_idx = -1
        max_iou = 0

        if target_mask.ndim == 4:
            target = target_mask.squeeze(0)
        elif target_mask.ndim == 2:
            target = target_mask.unsqueeze(0)
        else :
            target = target_mask

        for ii, gen_mask in enumerate(sam_masks):
            assert gen_mask.ndim == 3
            
            iou = compute_iou(gen_mask, target)
            if iou > max_iou:
                mask_idx = ii
                max_iou = iou
        
        return max_iou, mask_idx


    def create_similar_samlogits(self, pred_mask):
        """We cannot prompt SAM with MiVOS mask, we have to create a mask from SAM that is similar to MiVOS mask"""
        similar_iou_threshold = 0.8
        
        if torch.count_nonzero(pred_mask).item() == 0:
            return None, None, None, None
        
        click_coords_middle, click_labels_middle = self.click_robot.middle_click(pred_mask)
        sam_masks, _, logits = self.sam_controller.predict(click_coords=click_coords_middle, click_labels=click_labels_middle)        
        max_iou, mask_idx = self.best_sam_mask(sam_masks, pred_mask)        

        if max_iou>similar_iou_threshold:
            return logits[mask_idx][None, :, :], sam_masks[mask_idx], click_coords_middle, click_labels_middle
            
        # SAM did not provide similar mask to MiVOS
        best_sam_mask = sam_masks[mask_idx]
        best_sam_logits = logits[mask_idx]

        prev_clicks = click_coords_middle 
        prev_labels = click_labels_middle

        # 20 tries with sam to create a similar mask to MiVOS
        for _ in range(20): 
            click_coords, click_labels = self.click_robot.interact(best_sam_mask, pred_mask)

            if prev_clicks is None:
                prompt_clicks = click_coords
                prompt_labels = click_labels
            else :
                prompt_clicks = np.concatenate((prev_clicks, click_coords), axis=0)
                prompt_labels = np.concatenate((prev_labels, click_labels), axis=0)

            sam_masks, _, logits = self.sam_controller.predict(mask_input=best_sam_logits[None, :, :], 
                click_coords=prompt_clicks, click_labels=prompt_labels, multimask_output=True
            )
            max_iou, mask_idx = self.best_sam_mask(sam_masks, pred_mask)        

            best_sam_mask = sam_masks[mask_idx]
            best_sam_logits = logits[mask_idx]

            prev_clicks = prompt_clicks
            prev_labels = prompt_labels

            if max_iou>similar_iou_threshold:
                return best_sam_logits[None, :, :], best_sam_mask, prompt_clicks, prompt_labels
            
        return None, None, None, None
    

    def calculate_cost(self, action):
        self.annotation_cost += ANNOTATION_COSTS[action]
        self.curr_cost = ANNOTATION_COSTS[action]


    def click_mask(self):
        if self.sam_logits is None:
            self.prompt_clicks, self.prompt_labels = self.click_robot.middle_click(self.gt_mask)
        else:
            click_coords, click_labels = self.click_robot.interact(self.sam_mask, self.gt_mask)
            self.prompt_clicks = np.concatenate((self.prompt_clicks, click_coords), axis=0)
            self.prompt_labels = np.concatenate((self.prompt_labels, click_labels), axis=0)

        masks_from_sam, _, logits = self.sam_controller.predict(click_coords=self.prompt_clicks, 
            click_labels=self.prompt_labels, mask_input=self.sam_logits, multimask_output=True
        ) 

        curr_iou, mask_idx = self.best_sam_mask(masks_from_sam, self.gt_mask) 
        self.sam_mask = masks_from_sam[mask_idx]
        self.sam_logits = logits[mask_idx][None, :, :]
        self.iou = curr_iou


    def draw_mask(self):
        self.sam_mask = deepcopy(self.gt_mask).unsqueeze(0)
        self.iou = 1      
    
    def calculate_reward(self, prev_iou, action):
        if self.done:
            if 'click' in self.annotation_actions:
                self.annotation_cost += ANNOTATION_COSTS['click_overhead']

            # if len(self.annotation_actions) == 1 and action=='stop': # stop first action
            #     reward = -3

            # # elif 'click' in self.annotation_actions and 'mask' in self.annotation_actions:
            # #     reward = -3
            # else :
            reward = (self.iou - self.init_iou)/self.annotation_cost
        else :
            reward = (self.iou - prev_iou)/self.curr_cost
            
            
        return reward


    @torch.no_grad()
    def step(self, action_in):
        action = self.avail_actions[action_in]
        assert action in {'click', 'mask', 'stop'} and not self.done 

        self.annotation_actions.append(action)
        self.calculate_cost(action)
        done = False
        prev_iou = self.iou
        if action == 'click':
            self.click_mask()
            done = len(self.annotation_actions) == self.max_steps
        elif action == 'mask':
            self.draw_mask()
            done = True
        else : # stop
            done = True

        self.done =  done
        reward = self.calculate_reward(prev_iou, action)
        self.get_state()
        return reward, self.state, done


    def _set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        

        