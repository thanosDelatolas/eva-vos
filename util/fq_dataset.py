import os
import numpy as np
from PIL import Image

from torchvision import transforms
from .helpers import im_normalize, tens2image

mask_to_224 = transforms.Compose([
    transforms.Resize((224, )*2, interpolation=transforms.InterpolationMode.NEAREST),
])

im_to_224 = transforms.Compose([
    transforms.Resize((224, )*2, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
])


im_to_480p = transforms.Compose([
    transforms.Resize((480, 854), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
])

mask_to_480p = transforms.Compose([
    transforms.Resize((480, 854), interpolation=transforms.InterpolationMode.NEAREST, antialias=True),
])


def save_frames(images,video_name, db_dir, full_res=False):
    if full_res:
        res = '480p'
    else :
        res = '224'

    im_dir = os.path.join(db_dir, 'RGBFrames',res, video_name)
    os.makedirs(im_dir ,exist_ok=True)
    images = images.squeeze()

    for ii,rgb_frame in enumerate(images):
        rgb_frame = rgb_frame.cpu()
        # save the rgb frame
        if not full_res:
            rgb_frame = im_to_224(rgb_frame)   
        else :
            rgb_frame = im_to_480p(rgb_frame)
            
        rgb_img = im_normalize(tens2image(rgb_frame))
        canvas = (rgb_img * 255).astype(np.uint8)
        img_E = Image.fromarray(canvas)
        img_E.save(os.path.join(im_dir,'{:05d}.png'.format(ii)))


def saver(gen_mask_list, frame_choice_list, ious_list, video_name, id, db_dir, results_dict, full_res=False, dont_save=[]):
    """States come from gen_rand_oracle_states. First interaction is with any frame"""
    assert  len(gen_mask_list) == len(frame_choice_list) == len(ious_list), 'Invalid input!'

    n = len(gen_mask_list)
    for ii in range(n):
        if ii in dont_save or ious_list[ii] == 20: # 20 is for no object frames!
            continue
        
        if full_res:
            res = '480p'
        else :
            res = '224'
            
        masks_dir = os.path.join(db_dir, 'Annotations', f'{res}',f'{video_name}_round_{id}')
        os.makedirs(masks_dir, exist_ok=True)
       

        masks = gen_mask_list[ii].squeeze().cpu()
        n_frames = masks.shape[0]

        # save masks
        for f_t in range(n_frames):
            ma = masks[f_t]
            
            if not full_res:
                ma = mask_to_224(ma.unsqueeze(0))

            else :
                ma = mask_to_480p(ma.unsqueeze(0))

            ma_img = tens2image(ma)
            canvas = (ma_img * 255).astype(np.uint8)
            img_E = Image.fromarray(canvas)
            img_E.save(os.path.join(masks_dir, '{:05d}.png'.format(f_t)))

        results_dict['state_name'].append(f'{video_name}_round_{id}')
        results_dict['selected_frame'].append(frame_choice_list[ii])
        results_dict['ious'].append(ious_list[ii])

        id += 1
    return id, results_dict