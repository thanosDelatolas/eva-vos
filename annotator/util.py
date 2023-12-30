from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

inv_im_trans = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)


inv_im_trans = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225])


def show_mask(mask, ax, color=None):
    if color is None:
        color = np.array([1,0,0, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_mask_thickness(mask, ax, color=None, thickness=1):
    if color is None:
        color = np.array([1, 0, 0, 0.6])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    if thickness > 1:
        from scipy.ndimage import binary_dilation
        struct = np.ones((thickness, thickness))
        mask_boundaries = np.logical_xor(mask, binary_dilation(mask, structure=struct))
        mask_boundaries = mask_boundaries.reshape(h, w, 1) * np.array([1, 1, 1, 1])
        mask_image = np.maximum(mask_image, mask_boundaries)

    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    extra_points = coords[labels==2]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(extra_points[:, 0], extra_points[:, 1], color='blue', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax, lw=2):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=lw))    


def tens2image(im):
    tmp = np.squeeze(im.numpy())
    if tmp.ndim == 2:
        return tmp
    else:
        return tmp.transpose((1, 2, 0))
    
def im_normalize(im):
    """
    Normalize image
    """
    imn = (im - im.min()) / max((im.max() - im.min()), 1e-8)
    return imn