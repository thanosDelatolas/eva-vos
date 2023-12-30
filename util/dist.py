"""
Distributed training util
"""
import os
import numpy as np
import random

# torch imports
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

# custom imports
from datasets import MaskQualityDB
from util.mypath import Path

def setup(rank, world_size, port='2222'):    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port   
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def create_data_loaders(local_rank, world_size, batch_size=32, num_workers=4):
    db_root = Path.db_root_path(f'FQ_DB')
    imset_train = os.path.join(db_root, 'res_subset_train_4.csv')
    imset_val = os.path.join(db_root, 'res_val.csv')

    train_dataset = MaskQualityDB(db_root, imset_train)
    train_sampler = DistributedSampler(train_dataset, rank=local_rank, shuffle=True, num_replicas=world_size)
    train_loader = DataLoader(train_dataset, batch_size, sampler=train_sampler, 
        num_workers=num_workers, drop_last=True, pin_memory=True
    )

    val_dataset = MaskQualityDB(db_root, imset_val)
    val_sampler = DistributedSampler(val_dataset, rank=local_rank, shuffle=False, num_replicas=world_size)
    val_loader = DataLoader(val_dataset, batch_size=32, sampler=val_sampler,
        drop_last=False, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader


def move_to_cuda(data):
    for k, v in data.items():
        if type(v) != list and type(v) != dict and type(v) != int:
            data[k] = v.cuda(non_blocking=True)
    return data


def seed_everything(seed=29102910):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)