import os
import gdown
import tarfile
import zipfile
import shutil
import random

from scripts.resize import resize_all
from util.mypath import Path

random.seed(292910)

# MOSE
db_root_path = Path.db_root_path('MOSE')
os.makedirs(db_root_path, exist_ok=True)

if not os.path.exists(os.path.join(db_root_path, 'train.tar.gz')):
    print('Downloading MOSE...')
    gdown.download('https://drive.google.com/uc?id=10HYO-CJTaITalhzl_Zbz_Qpesh8F3gZR', output=db_root_path, quiet=False)

assert os.path.exists(os.path.join(db_root_path, 'train.tar.gz')), 'MOSE data are not downloaded!'

print('Extracting MOSE dataset...')
with tarfile.open(os.path.join(db_root_path, 'train.tar.gz'), 'r') as tfile:
    tfile.extractall(db_root_path)

print('Resizing MOSE to 480p...')

mose_480p_path = Path.db_root_path('MOSE_480p')
os.makedirs(mose_480p_path)
resize_all(os.path.join(db_root_path, 'train'), mose_480p_path)

print('Cleaning up...')
shutil.rmtree(db_root_path)
os.rename(mose_480p_path, db_root_path)

print('Generate train val test subsets ...')
all_mose_videos = list(os.listdir(os.path.join(db_root_path, 'JPEGImages/480p')))
videos = []
for vid in all_mose_videos:
    n_frames = len(os.listdir(os.path.join(os.path.join(db_root_path, 'JPEGImages/480p'), vid)))
    if n_frames > 15 and n_frames <= 104:
        videos.append(vid)

random.shuffle(videos)
train_videos = videos[:800]
val_videos = videos[800:950]
test_videos = videos[950:]
print(f'Train videos: {len(train_videos)}, Val videos: {len(val_videos)}, Test videos: {len(test_videos)}')
os.makedirs(os.path.join(db_root_path, 'ImageSets'), exist_ok=True)

with open(os.path.join(db_root_path, 'ImageSets/subset_train_4.txt'), 'w+') as fp:
    fp.write('\n'.join(train_videos))

with open(os.path.join(db_root_path,'ImageSets/val.txt'), 'w+') as fp:
    fp.write('\n'.join(val_videos))

with open(os.path.join(db_root_path, 'ImageSets/test.txt'), 'w+') as fp:
    fp.write('\n'.join(test_videos))

# DAVIS 17
data_path = Path.data_path()
os.makedirs(data_path, exist_ok=True)
gdown.download('https://drive.google.com/uc?id=1kiaxrX_4GuW6NmiVuKGSGVoKGWjOdp6d', output=os.path.join(data_path, 'DAVIS-2017-trainval-480p.zip'), quiet=False)

with zipfile.ZipFile(os.path.join(data_path, 'DAVIS-2017-trainval-480p.zip'), 'r') as zip_file:
    zip_file.extractall(data_path)

os.remove(os.path.join(data_path, 'DAVIS-2017-trainval-480p.zip'))
os.rename(os.path.join(data_path, 'DAVIS'), os.path.join(data_path, 'DAVIS_17'))