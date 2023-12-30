import torch
import torch.nn as nn
from .modules import CNNBranch
from .modules import Attention

class QualityNet(nn.Module):
    def __init__(self, merge_strategy='cat', arch='resnet18', n_labels=20):
        super(QualityNet, self).__init__()

        assert merge_strategy in {'add', 'cat', 'attn'}
        assert arch in {'small', 'resnet18', 'resnet50', 'resnet101'}

        self.neuron_out = 1 if n_labels == 2 else n_labels
        self.merge_strategy = merge_strategy

        self.arch = arch
        self.rgb_branch = CNNBranch(arch=arch)
        self.mask_branch = CNNBranch(arch=arch)

        self.flattened_neurons = self.rgb_branch.last_layer_num

        if self.merge_strategy == 'cat':
            self.flattened_neurons *= 2
        
        elif self.merge_strategy == 'attn':
            self.query_proj = nn.Linear(self.flattened_neurons, self.flattened_neurons)
            self.key_proj = nn.Linear(self.flattened_neurons, self.flattened_neurons)
            self.value_proj = nn.Linear(self.flattened_neurons, self.flattened_neurons)

            self.attn_mod = Attention(embed_dim=self.flattened_neurons)    

        self.flatten = nn.Flatten()

        if n_labels != -1 :
            self.dropout = nn.Dropout(0.5)
            self.out_layer = nn.Linear(self.flattened_neurons, self.neuron_out)

    
    def merge(self, rgb_out, mask_out):
        if self.merge_strategy == 'add':
            return rgb_out + mask_out
        elif self.merge_strategy == 'cat':
            return torch.cat((rgb_out, mask_out), dim=1)

        elif self.merge_strategy == 'attn':
            # B x flattened_neurons
            rgb_out = self.flatten(rgb_out)
            mask_out = self.flatten(mask_out)

            query = self.query_proj(mask_out)
            key = self.key_proj(rgb_out)
            value = self.value_proj(rgb_out)

            return self.attn_mod(query, key, value)



    def forward(self, x_rgb, x_mask):
        rgb_out = self.rgb_branch(x_rgb)
        mask_out = self.mask_branch(x_mask)
        
        x = self.merge(rgb_out, mask_out)

        if self.merge_strategy != 'attn':
            x = self.flatten(x)

        if self.neuron_out != -1:
            x = self.dropout(x)
            x = self.out_layer(x)
        return x
    
    def extract_features(self, x_rgb, x_mask):
        rgb_out = self.rgb_branch(x_rgb)
        mask_out = self.mask_branch(x_mask)
        
        x = self.merge(rgb_out, mask_out)
        
        x = self.flatten(x)
        return x
    
    def forward_and_features(self, x_rgb, x_mask):
        rgb_out = self.rgb_branch(x_rgb)
        mask_out = self.mask_branch(x_mask)
        
        x = self.merge(rgb_out, mask_out)
        
        features = self.flatten(x)
        x = self.out_layer(features)

        x = nn.Softmax(dim=1)(x)
        pred_quality = torch.argmax(x, dim=1)
        return pred_quality, features
