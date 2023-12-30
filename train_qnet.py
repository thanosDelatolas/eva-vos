import os
from tqdm import tqdm
from einops import repeat
import wandb

# torch imports
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# custom imports
from util.hyper_para import HyperParametersQNet
from util.dist import setup, cleanup, seed_everything, create_data_loaders, move_to_cuda
from models.qnet import QualityNet


def train(local_rank, world_size, args):
    """Dataloaders"""
    train_loader, val_loader = create_data_loaders(local_rank, world_size, args['batch_size'], args['num_workers'])
    
    """QNet"""
    model = QualityNet(arch=args['arch'], n_labels=20, merge_strategy='cat').to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    """Optimizer and Loss"""
    if args['optim'] == 'Adam':    
        optimizer = AdamW(model.parameters(), lr=args['lr'])
    elif args['optim'] == 'SGD' :
        optimizer = SGD(model.parameters(), lr=args['lr'], momentum=0.9)
    else :
        raise NotImplementedError('No implementation for this optimizer')
    
    loss_fn = nn.CrossEntropyLoss()
    activation_fn = nn.Softmax(dim=-1)

    if local_rank == 0:
        os.makedirs('model_weights/qnet', exist_ok=True)

        logger  = wandb.init(
            project="qnet",
            config={
                'Optim': args['optim'],
                'lr': args['lr'],
                'batch_size': args['batch_size'],
                'arch': args['arch'],
                'merge': 'cat'
            }
        )
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] Architecture: {args['arch']}")
        print(f'[INFO] Merge strategy: cat')
        print(f'[INFO] Trainable parameters: {total_params/1_000_000:.2f}M')
        print(f'[INFO] Available GPUs: {world_size}')
    
    for e in range(args['epochs']):
        # crucial for randomness!
        train_loader.sampler.set_epoch(e) 
        model.train()

        train_acc = 0.0
        train_loss = 0.0
        for data in tqdm(train_loader, total=len(train_loader), desc=f"Epoch: {e+1}/{args['epochs']}"):
            data = move_to_cuda(data)
            imgs = data['img']
            masks = data['mask']
            masks = repeat(masks, 'b h w -> b c h w', c=3)

            optimizer.zero_grad(set_to_none=True)
            y = model(imgs, masks)
        
            batch_loss = loss_fn(y, data['label'])
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss

            y = activation_fn(y)
            predicted = y.argmax(dim=-1)
            train_acc += (predicted == data['label']).sum().item()/data['label'].shape[0]
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        val_acc = 0
        for data in tqdm(val_loader, desc='Validation'):
            data = move_to_cuda(data)
            imgs = data['img']
            masks = data['mask']
            masks = repeat(masks, 'b h w -> b c h w', c=3)

            with torch.no_grad():
                y = model(imgs, masks)

            batch_loss = loss_fn(y, data['label'])

            y = activation_fn(y)
            predicted = y.argmax(dim=-1)
            val_acc += (predicted == data['label']).sum().item()/data['label'].shape[0]

        val_acc /= len(val_loader)

        if local_rank == 0:
            logger.log({
                'Train loss': train_loss,
                'Train acc': train_acc,
                'Val acc': val_acc
            })
    
    if local_rank == 0 :
        torch.save(model.module.state_dict(),f'model_weights/qnet/qnet.pth')


def main(local_rank, world_size):
    args = HyperParametersQNet()
    args.parse()
    
    setup(local_rank, world_size, args['port'])
    print(f'I am rank {local_rank} in the world of {world_size}!')
    torch.cuda.set_device(local_rank)
    seed_everything()

    try: 
        train(local_rank, world_size, args)
    finally:
        cleanup()


if __name__ == '__main__':

    world_size = torch.cuda.device_count() 
    # rank is automatically passed for each procces      
    mp.spawn(
        main,
        args=[world_size],
        nprocs=world_size
    )
