import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, resnet101, ResNet101_Weights, \
    vit_b_16, ViT_B_16_Weights, vit_b_32, ViT_B_32_Weights, vit_l_32, ViT_L_32_Weights


import numpy as np

from copy import deepcopy

class CNNBranch(nn.Module):
    """Three channel CNNBranch"""
    def __init__(self, arch='resnet18'):
        super(CNNBranch, self).__init__()
        self.arch = arch

        if self.arch == 'small':
            resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.last_layer_num = 1024

        elif self.arch == 'resnet18':
            resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.last_layer_num = 512  

        elif self.arch == 'resnet50':
            resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.last_layer_num = 2048
        
        elif self.arch == 'resnet101':
            resnet = resnet101(weights=ResNet101_Weights.DEFAULT)
            self.last_layer_num = 2048
       
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1 # 1/4, 256
        self.layer2 = resnet.layer2 # 1/8, 512
        self.layer3 = resnet.layer3 # 1/16, 1024

        if arch != 'small':
            self.layer4 = resnet.layer4 # 1/32, 2048
            self.avg_pool_layer = torch.nn.AvgPool2d(kernel_size=7)  
        else :
            self.avg_pool_layer = torch.nn.AvgPool2d(kernel_size=14)
    


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.layer1(x)   # 1/4, 256
        x = self.layer2(x) # 1/8, 512
        x = self.layer3(x) # 1/16, 1024

        if self.arch != 'small':
            x = self.layer4(x) # 1/32, 2048

        x  = self.avg_pool_layer(x)
            
        return x



class CNNBranch4D(nn.Module):
    """A branch that accepts 4d input"""
    def __init__(self, arch='small'):
        super(CNNBranch4D, self).__init__()
        self.arch = arch

        if self.arch == 'small':
            resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.last_layer_num = 1024

        elif self.arch == 'resnet18':
            resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.last_layer_num = 512  

        elif self.arch == 'resnet50':
            resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.last_layer_num = 2048
        
        elif self.arch == 'resnet101':
            resnet = resnet101(weights=ResNet101_Weights.DEFAULT)
            self.last_layer_num = 2048


        ### conv with 3 channels to conv with 4
        conv1_weight = resnet.conv1.weight

        I, J, K, L = conv1_weight.shape
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()

        conv1_3_chan_input = deepcopy(conv1_weight)

        # create last channel conv
        conv1_weight = conv1_weight.sum(dim=1, keepdim=True)

        conv1_weight = conv1_weight.to(conv1_type)
        conv1_weights_4d = torch.cat((conv1_3_chan_input, conv1_weight), dim=1)

        resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,bias=False)
        resnet.conv1.weight = nn.Parameter(conv1_weights_4d)


        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1 # 1/4, 256
        self.layer2 = resnet.layer2 # 1/8, 512
        self.layer3 = resnet.layer3 # 1/16, 1024

        if arch != 'small':
            self.layer4 = resnet.layer4 # 1/32, 2048  
            self.avg_pool_layer = torch.nn.AvgPool2d(kernel_size=7)
        else :
            self.avg_pool_layer = torch.nn.AvgPool2d(kernel_size=14)
            

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.layer1(x)   # 1/4, 256
        x = self.layer2(x) # 1/8, 512
        x = self.layer3(x) # 1/16, 1024

        if self.arch != 'small':
            x = self.layer4(x) # 1/32, 2048

        x = self.avg_pool_layer(x)
        
        return x

class Attention(nn.Module):
    def __init__(self, embed_dim):
        super(Attention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)

    def forward(self, query, key, value):
        attn_output, _ = self.attn(query, key, value)
        return attn_output


class VITBranch(nn.Module):
    def __init__(self, arch='vit_b_16'):
        super(VITBranch, self).__init__()
        
        if arch == 'vit_b_16':
            weights = ViT_B_16_Weights.DEFAULT
            vit = vit_b_16(weights=weights) 
            self.last_layer_num = 768       
        elif arch == 'vit_b_32' :
            weights = ViT_B_32_Weights.DEFAULT
            vit = vit_b_32(weights=weights)
            self.last_layer_num = 768
        elif arch == 'vit_l_32':
            weights = ViT_L_32_Weights.DEFAULT
            vit = vit_l_32(weights=weights)
            self.last_layer_num = 1024
        else :
            raise AttributeError('Invalid vit')
        
        # remove the last layer
        vit.heads = nn.Identity()

        self.vit = vit
        
    
    def forward(self, x):
        x = self.vit(x)
        return x