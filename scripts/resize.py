import sys
import os
from os import path

from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm

new_min_size = 480

def resize_vid_jpeg(inputs):
    vid_name, folder_path, out_path = inputs

    vid_path = path.join(folder_path, vid_name)
    vid_out_path = path.join(out_path, 'JPEGImages', f'{new_min_size}p',vid_name)
    os.makedirs(vid_out_path, exist_ok=True)

    for im_name in os.listdir(vid_path):
        hr_im = Image.open(path.join(vid_path, im_name))
        w, h = hr_im.size

        ratio = new_min_size / min(w, h)

        lr_im = hr_im.resize((int(w*ratio), int(h*ratio)), Image.BICUBIC)
        lr_im.save(path.join(vid_out_path, im_name))

def resize_vid_anno(inputs):
    vid_name, folder_path, out_path = inputs

    vid_path = path.join(folder_path, vid_name)
    vid_out_path = path.join(out_path, 'Annotations', f'{new_min_size}p',vid_name)
    os.makedirs(vid_out_path, exist_ok=True)

    for im_name in os.listdir(vid_path):
        hr_im = Image.open(path.join(vid_path, im_name)).convert('P')
        w, h = hr_im.size

        ratio = new_min_size / min(w, h)

        lr_im = hr_im.resize((int(w*ratio), int(h*ratio)), Image.NEAREST)
        lr_im.save(path.join(vid_out_path, im_name))


def resize_all(in_path, out_path):
    for folder in os.listdir(in_path):

        if folder not in ['JPEGImages', 'Annotations']:
            continue
        folder_path = path.join(in_path, folder)
        videos = os.listdir(folder_path)

        videos = [(v, folder_path, out_path) for v in videos]

        if folder == 'JPEGImages':
            print('Processing images')
            os.makedirs(path.join(out_path, 'JPEGImages'), exist_ok=True)

            pool = Pool(processes=8)
            for _ in tqdm(pool.imap_unordered(resize_vid_jpeg, videos), total=len(videos)):
                pass
        else:
            print('Processing annotations')
            os.makedirs(path.join(out_path, 'Annotations'), exist_ok=True)

            pool = Pool(processes=8)
            for _ in tqdm(pool.imap_unordered(resize_vid_anno, videos), total=len(videos)):
                pass
