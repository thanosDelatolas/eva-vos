import os
import matplotlib.pyplot as plt

import torch
import numpy as np

from sam import SAMController
from robots.click_robot import ClickRobot
from robots.bbox_robot import BboxRobot


from util.mypath import Path
from util.helpers import ANNOTATION_COSTS
from interactions.eval import compute_iou

from annotator.util import *


class Annotator:
    """Simulates an annotator"""
    def __init__(self, prompt_type='c', device='cuda', verbose=True):
        self.device = device
        self.sam_controller = SAMController('./model_weights/sam/sam.pth', self.device, verbose)
        self.click_robot = ClickRobot()
        self.bbox_robot = BboxRobot()
        assert prompt_type in {'a','b','c'}
        self.prompt_type=prompt_type
    

    def set_image_to_sam(self, im):
        im = inv_im_trans(im)
        im = im.cpu().squeeze().numpy().transpose((1, 2, 0))
        im = (im * 255).astype(np.uint8)
        self.sam_controller.reset_image()
        self.sam_controller.set_image(im)


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
    

    @torch.no_grad()
    def get_mask(self, annotation_type, gt_mask, im=None, num_prompts=1, mivos_mask=None, save_annotations_path=None, prev_iter_data=None):
        """Generates a mask at the input frame im.
        Parameters
        ----------
            annotation type: mask, click, bbox
            gt_mask: ground truth mask
            num_prompts: number of prompts
            prompt_type:
                a -> don't prompt sam with prev prompts or prev logits
                b -> prompt SAM with logits + new prompts
                c -> prompt SAM with prev prompts + new prompts 
            
            prev_iter_data : {
                'sam_logits' : <>,
                'click_coords': <>,
                'click_labels': <>,
                'bbox': <>,
            }
        """
        assert annotation_type in {'mask','click', 'bbox'}
        
        if torch.count_nonzero(gt_mask).item() == 0:
            return gt_mask, ANNOTATION_COSTS['no_object'], 20, None, None, None, None
        
        if annotation_type == 'mask':
            return gt_mask, ANNOTATION_COSTS[annotation_type], 1, None, None, None, None
        
        else: # bbox or click
            self.im = im
            self.set_image_to_sam(im)

            if annotation_type == 'click':
                return self.get_click_mask_n_prompts(gt_mask.to(torch.bool), num_prompts, mivos_mask=mivos_mask, prompt_type=self.prompt_type, save_annotations_path=save_annotations_path, prev_iter_data=prev_iter_data)
            else : 
                return self.get_bbox_mask_n_prompts(gt_mask.to(torch.bool), num_prompts, mivos_mask=mivos_mask, prompt_type=self.prompt_type, save_annotations_path=save_annotations_path, prev_iter_data=prev_iter_data)
    

    def get_initial_prompts(self, mivos_mask, prompt_type):
        sam_logits, sam_mask, mivos_clicks, mivos_labels = None, None, None, None
        if (prompt_type == 'b' or prompt_type == 'c') and mivos_mask is not None:
            sam_logits, sam_mask, mivos_clicks, mivos_labels = self.create_similar_samlogits(mivos_mask)

        return sam_logits, sam_mask, mivos_clicks, mivos_labels


    def save_annotation(self, save_path, click_coords, click_labels, mask, bbox=None):
        im = self.im
        plt.figure()
        plt.imshow(im_normalize(tens2image(im.cpu())))
        show_mask(mask.squeeze().cpu().numpy(), plt.gca())
        show_points(click_coords, click_labels, plt.gca())
        if bbox is not None:
            show_box(bbox, plt.gca())
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    def save_mask(self, save_path, mask):
        im = self.im
        plt.figure()
        plt.imshow(im_normalize(tens2image(im.cpu())))
        show_mask(mask.squeeze().cpu().numpy(), plt.gca())
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


    def get_prompts(self, mivos_mask, prompt_type, prev_iter_data):
        if prev_iter_data is None or prev_iter_data['sam_logits'] is None:
            sam_logits, sam_mask, prev_clicks, prev_labels = self.get_initial_prompts(mivos_mask, prompt_type)
            bbox = None
        else :
            sam_mask = mivos_mask 
            prev_clicks = prev_iter_data['click_coords']
            prev_labels = prev_iter_data['click_labels']
            sam_logits = prev_iter_data['sam_logits']
            bbox = prev_iter_data['bbox']

        
        if prompt_type == 'b': # prompt only with SAM logits
            prev_clicks, prev_labels, bbox = None, None, None

        return sam_logits, sam_mask, prev_clicks, prev_labels, bbox
        


    def get_click_mask_n_prompts(self, gt_mask, num_clicks, mivos_mask=None, prompt_type='a', save_annotations_path=None, prev_iter_data=None):
        curr_iou = 0
        cost = 0
        ii = 0

        sam_logits, sam_mask, prev_clicks, prev_labels, bbox = self.get_prompts(mivos_mask, prompt_type, prev_iter_data)
        sam_ious = []
        click_coords = None
        while ii < num_clicks:

            if prev_clicks is None: 
                if sam_mask is None:
                    prompt_clicks, prompt_labels = self.click_robot.middle_click(gt_mask)
                else :
                    prompt_clicks, prompt_labels = self.click_robot.interact(sam_mask, gt_mask)
                
                cost +=  ANNOTATION_COSTS['click']
            else:
                click_coords, click_labels = self.click_robot.interact(sam_mask, gt_mask)
                cost +=  click_labels.shape[0] * ANNOTATION_COSTS['click']
                prompt_clicks = np.concatenate((prev_clicks, click_coords), axis=0)
                prompt_labels = np.concatenate((prev_labels, click_labels), axis=0)
                 

            if sam_mask is not None and save_annotations_path is not None:
                save_path = os.path.join(save_annotations_path, f'annot_{ii+1}.png')
                self.save_annotation(save_path=save_path, click_coords=prompt_clicks, click_labels=prompt_labels, mask=sam_mask)

            masks_from_sam, _, logits = self.sam_controller.predict(click_coords=prompt_clicks, 
                click_labels=prompt_labels, mask_input=sam_logits, bbox=bbox, multimask_output=True
            )     
            curr_iou, mask_idx = self.best_sam_mask(masks_from_sam, gt_mask) 
            sam_mask = masks_from_sam[mask_idx]
            if save_annotations_path is not None:
                save_path = os.path.join(save_annotations_path, f'sam_out_{ii+1}.png')
                self.save_mask(save_path=save_path, mask=sam_mask)

            sam_logits = logits[mask_idx][None, :, :]

            prev_clicks = prompt_clicks
            prev_labels = prompt_labels

            sam_ious.append(curr_iou)
            ii+= 1

        # add click overhead
        cost += ANNOTATION_COSTS['click_overhead']
        return sam_mask, cost, curr_iou, sam_logits, prompt_clicks, prompt_labels, bbox
         
    def get_bbox_mask_n_prompts(self, gt_mask, prompts, mivos_mask=None, prompt_type='a', save_annotations_path=None, prev_iter_data=None):
        ii = 0
        curr_iou = 0
        cost = 0

        sam_logits, sam_mask, prev_clicks, prev_labels, prev_box = self.get_prompts(mivos_mask, prompt_type, prev_iter_data)
        assert prev_box is None
        new_clicks = False
        sam_ious = []
        while ii < prompts:

            if ii == 0: # initial bounding box
                bbox = self.bbox_robot.interact(gt_mask)               
                cost +=  ANNOTATION_COSTS['bbox']
                prompt_clicks = prev_clicks
                prompt_labels = prev_labels

            else:
                new_clicks = True
                click_coords, click_labels = self.click_robot.interact(sam_mask, gt_mask) # returns up two clicks (positive and negative)
                cost +=  click_labels.shape[0] * ANNOTATION_COSTS['click']

                if prev_labels is None:
                    prompt_clicks = click_coords
                    prompt_labels = click_labels
                else :
                    prompt_clicks = np.concatenate((prev_clicks, click_coords), axis=0)
                    prompt_labels = np.concatenate((prev_labels, click_labels), axis=0)  
                
            masks_from_sam, _, logits = self.sam_controller.predict(click_coords=prompt_clicks, 
                click_labels=prompt_labels, mask_input=sam_logits, bbox=bbox, multimask_output=True
            )
            curr_iou, mask_idx = self.best_sam_mask(masks_from_sam, gt_mask) 
            sam_mask = masks_from_sam[mask_idx]
            sam_logits = logits[mask_idx][None, :, :]

            ii+=1
            prev_clicks = prompt_clicks
            prev_labels = prompt_labels
            sam_ious.append(curr_iou)

        if new_clicks:
            cost += ANNOTATION_COSTS['click_overhead']
        return sam_mask, cost, curr_iou, sam_logits, prompt_clicks, prompt_labels, bbox
    
