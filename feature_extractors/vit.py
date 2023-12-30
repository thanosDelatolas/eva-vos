import math

import torch
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights, vit_l_16, ViT_L_16_Weights

class ViTFeatureExtractor:
    def __init__(self, arch):

        if arch == 'base':
            self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            self.transforms = ViT_B_16_Weights.DEFAULT.transforms()
        elif arch == 'large':
            self.model = vit_l_16(weights=ViT_L_16_Weights.DEFAULT)
            self.transforms = ViT_L_16_Weights.DEFAULT.transforms()

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()

        # register forward hook
        self.forward_hook = self.model.encoder.register_forward_hook(self.getActivation())
    
    def getActivation(self,):        
        # the hook signature
        def hook(model, input, output):
            features = output.detach()[:,0] # cls token
            self.features = features
        return hook

    
    def extract_features(self, X):
        # forward pass -- getting the outputs
        with torch.no_grad():
            y = self.model(X)
        
        return self.features



# class ViTPatchFeatureExtractor:
#     def __init__(self):
#         self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
#         self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#         self.model.to(self.device)
#         self.model.eval()

#         # register forward hook
#         self.forward_hook = self.model.encoder.register_forward_hook(self.getActivation())
    
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

