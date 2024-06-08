import os
import gdown
import zipfile
import wget

os.makedirs('./model_weights', exist_ok=True)
gdown.download('https://drive.google.com/uc?id=1pnOBaNZKSPddBuh9AqM48FFwmw_ln5lT', output='./model_weights.zip', quiet=False)

with zipfile.ZipFile('./model_weights.zip', 'r') as zip_file:
    zip_file.extractall('./')

os.remove('./model_weights.zip')

os.makedirs('./model_weights/sam/', exist_ok=True)
print('Downloading SAM ...')
wget.download('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth', './model_weights/sam/sam.pth')
print()