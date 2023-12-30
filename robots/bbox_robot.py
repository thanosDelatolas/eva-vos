from torchvision.ops import masks_to_boxes

class BboxRobot(object):
    """ Simulates a user bounding box
    """

    def __init__(self):
        """ Robot constructor
        """
    
    def interact(self,gt_mask):
        gt = gt_mask.squeeze(1)
        if gt.ndim != 3:
            gt = gt.unsqueeze(0)
        bbox = masks_to_boxes(gt).cpu().numpy()
        return bbox
        