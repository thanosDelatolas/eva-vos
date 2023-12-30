import math

import torch
from torchvision import transforms

class DINOFeatureExtractor:
    def __init__(self, arch='large'):
        assert arch in {'small', 'base', 'large', 'giant'}

        if arch == 'small':
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
        elif arch == 'base':
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
        elif arch == 'large':
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)
        elif arch == 'giant':
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14', pretrained=True)
            
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()

        # from the official repo: https://github.com/facebookresearch/dinov2/blob/HEAD/dinov2/data/transforms.py
        self.transforms = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])


    def extract_features(self, X):
        # forward pass -- getting the outputs
        with torch.no_grad():
            y = self.model(X)
        
        return y
    

# class DINOPatchFeatureExtractor :
#     def __init__(self, arch='small'):
#         assert arch in {'small', 'base', 'large', 'giant'}

#         if arch == 'small':
#             self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
#         elif arch == 'base':
#             self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
#         elif arch == 'large':
#             self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)
#         elif arch == 'giant':
#             self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14', pretrained=True)
            
#         self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#         self.model.to(self.device)
#         self.model.eval()

#         # register forward hook
#         self.forward_hook = self.model.norm.register_forward_hook(self.getActivation())
    



#     def getActivation(self,):        
#         # the hook signature
#         def hook(model, input, output):
#             patch_features = output.detach()[:,1:] # ignore the cls token
#             num_patches = patch_features.shape[1]
#             patch_size = int(math.sqrt(num_patches))

#             patch_features = patch_features.view(-1, patch_features.shape[-1], patch_size, patch_size)
#             #patch_features = F.interpolate(patch_features, (224, 224), mode='bilinear')
#             self.patch_features = patch_features
#         return hook


#     def get_features(self, X):
#         # forward pass -- getting the outputs
#         with torch.no_grad():
#             y = self.model(X)
        
#         return self.patch_features
    

   
