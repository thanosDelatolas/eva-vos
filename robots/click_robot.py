import torch

import numpy as np
from skimage.measure import label

class ClickRobot(object):
    """ Simulates a user click
    """

    def __init__(self):
        """ Robot constructor
        """

    def interact(self,pred_mask, gt_mask, iou=None):
        pred = pred_mask.squeeze().cpu().numpy()
        gt = gt_mask.squeeze().cpu().numpy()
        clicks = []
        click_labels = []
        components_len = []

        # negative click
        false_positive_mask = np.logical_and(pred, np.logical_not(gt))
        false_positive_components, num_false_positive = label(false_positive_mask, connectivity=2, return_num=True)        

        
        if num_false_positive > 0:
            components = np.bincount(false_positive_components.flat)[1:]
            max_false_positive_component = np.argmax(components) + 1
            components_len.append(np.max(components))

            max_false_positive_indices = np.where(false_positive_components == max_false_positive_component)
            max_false_positive_center = (np.mean(max_false_positive_indices[0]), np.mean(max_false_positive_indices[1]))
            false_positive_click = [int(max_false_positive_center[1]), int(max_false_positive_center[0])]

            clicks.append(false_positive_click)
            click_labels.append(0)
        
        #positive click
        false_negative_mask = np.logical_and(np.logical_not(pred), gt)
        false_negative_components, num_false_negative = label(false_negative_mask, connectivity=2, return_num=True)

        if num_false_negative > 0:
            components = np.bincount(false_negative_components.flat)[1:]
            max_false_negative_component = np.argmax(components) + 1
            components_len.append(np.max(components))

            max_false_negative_indices = np.where(false_negative_components == max_false_negative_component)
            max_false_negative_center = (np.mean(max_false_negative_indices[0]), np.mean(max_false_negative_indices[1]))
            false_negative_click = (int(max_false_negative_center[1]), int(max_false_negative_center[0]))

            if not gt[false_negative_click[1], false_negative_click[0]]:
                y,x = np.where(gt == 1)
                distances = np.sqrt((x - false_negative_click[0]) ** 2 + (y - false_negative_click[1]) ** 2)
                nearest_index = np.argmin(distances)
                false_negative_click = (x[nearest_index], y[nearest_index])

            clicks.append(false_negative_click)
            click_labels.append(1)
        else: 
            false_negative_click = None
        
        try :
            max_component = np.argmax(components_len)
        except ValueError:
            return self.middle_click(gt_mask)
        
        clicks = clicks[max_component]
        click_labels = click_labels[max_component]
        click_labels = [click_labels]
        clicks = [clicks]
        if iou is not None and iou < 0.1 and click_labels[0] == 0 and false_negative_click is not None:
            # in case we have a mask in another object
            clicks.append([false_negative_click[0], false_negative_click[1]])
            click_labels = [0,1]
        return np.array(clicks), np.array(click_labels)
    

    def middle_click(self, gt_mask):
        clicks = []
        click_labels = []
        gt = gt_mask.squeeze().cpu().numpy()
        
        y,x = np.where(gt == 1)
        object_indices = np.column_stack((y, x))
        middle_y, middle_x = np.median(object_indices, axis=0)
        middle_y = int(middle_y)
        middle_x = int(middle_x)
        if gt[middle_y, middle_x] != 1:
            # Find the nearest point within the object region
            distances = np.sqrt((x - middle_x) ** 2 + (y - middle_y) ** 2)
            nearest_index = np.argmin(distances)
            middle_x = x[nearest_index]
            middle_y = y[nearest_index]

        middle_click = [int(middle_x), int(middle_y)]
        clicks.append(middle_click)
        click_labels.append(1)

        return np.array(clicks), np.array(click_labels)
    

    def three_pos_clicks(self, gt_mask):
        """Returns three positive clicks"""
        non_zero = torch.nonzero(gt_mask.squeeze())
        idxs = torch.tensor([0, non_zero.shape[0]//2, non_zero.shape[0]-1])
        click_coords = non_zero[idxs]

        click_coords[:, [0, 1]] = click_coords[:, [1, 0]]
        return click_coords.cpu().numpy(), np.ones((3,))
    

    def three_refinement_clicks(self,pred_mask, gt_mask):
        pred = pred_mask.squeeze().cpu().numpy()
        gt = gt_mask.squeeze().cpu().numpy()
        clicks = []
        click_labels = []
        components_len = []

        # negative click
        false_positive_mask = np.logical_and(pred, np.logical_not(gt))
        false_positive_components, num_false_positive = label(false_positive_mask, connectivity=2, return_num=True)        

        
        if num_false_positive > 0:
            components = np.bincount(false_positive_components.flat)[1:]
            sorted_components_indices = np.argsort(-components)  # Sorting in descending order

            for component_idx in sorted_components_indices:
                component_mask = (false_positive_components == component_idx + 1)
                component_len = np.sum(component_mask)
                components_len.append(component_len)
                
                component_indices = np.where(component_mask)
                component_center = (np.mean(component_indices[0]), np.mean(component_indices[1]))
                click = (int(component_center[1]), int(component_center[0]))
                clicks.append(click)
                click_labels.append(0)
        
        #positive click
        false_negative_mask = np.logical_and(np.logical_not(pred), gt)
        false_negative_components, num_false_negative = label(false_negative_mask, connectivity=2, return_num=True)

        if num_false_negative > 0:
            components = np.bincount(false_negative_components.flat)[1:]
            sorted_components_indices = np.argsort(-components)  # Sorting in descending order

            for component_idx in sorted_components_indices:
                component_mask = (false_negative_components == component_idx + 1)
                component_len = np.sum(component_mask)
                components_len.append(component_len)
                
                component_indices = np.where(component_mask)
                component_center = (np.mean(component_indices[0]), np.mean(component_indices[1]))
                click = (int(component_center[1]), int(component_center[0]))
                clicks.append(click)
                click_labels.append(1)
        
        components_len = np.array(components_len)
        sorted_idxs = (-components_len).argsort()[:len(components_len)][:3] # get the first three
        clicks = np.array(clicks)
        click_labels = np.array(click_labels)

        clicks = clicks[sorted_idxs]
        click_labels = click_labels[sorted_idxs]
        
        return clicks, click_labels