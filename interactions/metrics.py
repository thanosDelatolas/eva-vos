import torch
import cv2
import math
import numpy as np
from skimage.morphology import disk
from torchmetrics import JaccardIndex

SMOOTH = 1e-6
def compute_iou(outputs: torch.Tensor, labels: torch.Tensor):
    assert outputs.ndim == labels.ndim == 3
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    #thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return iou.mean().item()


iou_comp = JaccardIndex(task="binary", num_classes=2)

def get_j_and_f(gt_mask, pred_mask):
    
    assert gt_mask.ndim == pred_mask.ndim == 3
    
    gt_mask = gt_mask.to(torch.bool)
    pred_mask = pred_mask.to(torch.bool)

    iou = iou_comp(gt_mask, pred_mask).item()
    f = f_measure(gt_mask.squeeze(), pred_mask.squeeze())

    return iou*0.5 + f*0.5


"""The code below is from davisinteractive"""
def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries. The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.

    # Arguments
        seg: Segments labeled from 1..k.
        width:	Width of desired bmap  <= seg.shape[1]
        height:	Height of desired bmap <= seg.shape[0]

    # Returns
        bmap (ndarray):	Binary boundary map.

    David Martin <dmartin@eecs.berkeley.edu>
    January 2003
    """

    seg = seg.astype(np.bool_)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width > w | height > h | abs(ar1 - ar2) >
                0.01), "Can't convert %dx%d seg to %dx%d bmap." % (w, h, width,
                                                                   height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


def f_measure(true_mask, pred_mask, bound_th=0.008):
    """F-measure for two 2D masks.

    # Arguments
        true_mask: Numpy Array, Binary array of shape (H x W) representing the
            ground truth mask.
        pred_mask: Numpy Array. Binary array of shape (H x W) representing the
            predicted mask.
        bound_th: Float. Optional parameter to compute the F-measure. Default is
            0.008.

    # Returns
        float: F-measure.
    """
    true_mask = np.asarray(true_mask, dtype=np.bool_)
    pred_mask = np.asarray(pred_mask, dtype=np.bool_)

    assert true_mask.shape == pred_mask.shape

    bound_pix = bound_th if bound_th >= 1 else (np.ceil(
        bound_th * np.linalg.norm(true_mask.shape)))

    fg_boundary = _seg2bmap(pred_mask)
    gt_boundary = _seg2bmap(true_mask)

    fg_dil = cv2.dilate(
        fg_boundary.astype(np.uint8),
        disk(bound_pix).astype(np.uint8))
    gt_dil = cv2.dilate(
        gt_boundary.astype(np.uint8),
        disk(bound_pix).astype(np.uint8))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F


