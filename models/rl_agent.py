import torch
import torch.nn as nn

from .modules import CNNBranch, VITBranch

class ActorCritic(nn.Module):
    """Actor critic network
    It takes as input the image embedding from SAM and a segmentation mask to output the most suitable annotation type
    """
    def __init__(self, out_dim, arch='resnet18', dropout=0.5, use_cost=False):
        super(ActorCritic, self).__init__()
        if arch.__contains__('vit'):
            self.mask_branch = VITBranch(arch=arch)
        else:
            self.mask_branch = CNNBranch(arch=arch)

        self.flattened_neurons = self.mask_branch.last_layer_num
        self.embed_branch = nn.Sequential( # sam_embedding branch
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(256, self.flattened_neurons),
        )
        if use_cost:
            self.cost_branch = nn.Sequential(
                nn.Linear(1, self.flattened_neurons),
                nn.ReLU(),
            ) 
            num_branches = 3
        else :
            num_branches = 2 
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        
        self.policy = nn.Linear(self.flattened_neurons*num_branches, out_dim)
        self.value = nn.Linear(self.flattened_neurons*num_branches, 1)

    
    def forward(self, x_img, x_mask, x_cost=None):
        """
        -> x_img: sam image embedding
        -> x_mask: segmentation mask
        """
        embed_out = self.embed_branch(x_img)
        embed_out = self.flatten(embed_out)
        mask_out = self.mask_branch(x_mask)
        mask_out = self.flatten(mask_out)
        if x_cost is not None:
            cost_out = self.cost_branch(x_cost)
            x = torch.cat((embed_out, mask_out, cost_out), dim=1)
        else :
            x = torch.cat((embed_out, mask_out), dim=1)
        x = self.dropout(x)
        p = self.policy(x)
        v = self.value(x)
        return p,v 