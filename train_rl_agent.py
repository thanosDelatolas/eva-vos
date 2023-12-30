import os
import wandb
from tqdm import tqdm

import hydra
from omegaconf import DictConfig

# torch imports
import torch
from torch.utils.data import DataLoader

# ppo imports
from ppo.ppo_trainer import PPOTrainer
from ppo.rollout import rollout
from ppo.storage import RolloutStorage

# custom imports
from datasets import AnnotTypeDB
from util.mypath import Path
from util.dist import seed_everything

@hydra.main(version_base=None, config_path="config", config_name="rl_agent")
def train(cfg : DictConfig):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    assert cfg.action_space in {2,3}
    assert cfg.ppo.advantages in {'gae', 'diff'}
    seed_everything()

    root = Path.db_root_path('AnnotDB')
    path_to_model = './model_weights/rl_agent/'
    os.makedirs(path_to_model, exist_ok=True)

    sample_size = 5 if cfg.sample else None
    train_db = AnnotTypeDB(root=root, imset=cfg.imset, sample_size=sample_size)
    train_loader = DataLoader(train_db, batch_size=cfg.num_envs, shuffle=True, drop_last=True) 

    wandb.init(project="eva-vos-rl-agent")
    print(f'[DB INFO] Number of images: {len(train_db)}')
    print(f'[DB INFO] Sample: {cfg.sample}')
    print(f'[OPTIM INFO] Envs: {cfg.num_envs}')
    print(f'[OPTIM INFO] Minibatch: {cfg.mini_batch}')
    print(f'[OPTIM INFO] Optim: {cfg.optim.optim_str}')
    print(f'[OPTIM INFO] LR: {cfg.optim.lr}')
    print(f'[MODEL INFO] Arch: {cfg.policy.arch}')
    print(f'[MODEL INFO] Actions: {cfg.action_space}')
    print(f'[PPO INFO] PPO epochs: {cfg.ppo.ppo_epochs}')
    print(f'[PPO INFO] Entropy coef: {cfg.ppo.entropy_coef}')
    print(f'[PPO INFO] target kl: {cfg.ppo.target_kl_div}')
    print(f'[PPO INFO] Advantages: {cfg.ppo.advantages}')
    print(f'[PPO INFO] clip param: {cfg.ppo.clip_param}')
    print(f'[PPO INFO] gamma: {cfg.ppo.gamma}')
    print(f'[PPO INFO] Steps: {cfg.num_steps}')

    """PPO trainer"""
    ppo_trainer = PPOTrainer(
        action_space=cfg.action_space, # click and mask
        ppo_epochs=cfg.ppo.ppo_epochs,
        clip_param=cfg.ppo.clip_param,
        value_loss_coef=cfg.ppo.value_loss_coef,
        entropy_coef=cfg.ppo.entropy_coef,
        target_kl_div=cfg.ppo.target_kl_div,
        
        lr=cfg.optim.lr,
        optim_str=cfg.optim.optim_str,

        arch=cfg.policy.arch,
        dropout=cfg.policy.dropout
    )
    
    rollouts = RolloutStorage(num_envs=cfg.num_envs, num_steps=cfg.num_steps, obs_shape=(3,224,224), num_mini_batch=cfg.mini_batch)
    total_iters = cfg.ppo_rollouts

    if cfg.resume:
        checkpoint_path = os.path.join(path_to_model, '_checkpoint.pth')
        checkpoint = torch.load(checkpoint_path)
        max_reward = checkpoint['max_reward']
        iters = checkpoint['iters']
        ppo_trainer.ac_net.load_state_dict(checkpoint['network'])
        print(f'Model is loaded! {iters} with max reward: {max_reward:.0e}')
    else :
        max_reward = -1e10
        iters = 0
    
    pbar = tqdm(total=int(total_iters), initial=iters, desc=f'Training')

    while iters < total_iters:
        avg_train_loss = 0
        avg_train_reward = 0
        for data in train_loader:
            for ii in range(cfg.num_envs):
                image = data['img'][ii].to(device)
                gt_mask = data['gt_mask'][ii].to(device).unsqueeze(0)
                init_mask = data['mask'][ii].to(device).unsqueeze(0)
                rollout(ppo_trainer, image, gt_mask, init_mask, ii, rollouts, cfg.num_steps, cfg.ppo.gamma, cfg.ppo.advantages, device)

                iters += 1
                pbar.update(1)

            valid_idxs = torch.where(rollouts.paddings == 0)
            mu_reward = torch.mean(rollouts.rewards[valid_idxs]).item()

            avg_train_loss += ppo_trainer.optimize(rollouts)
            avg_train_reward += mu_reward

        avg_train_loss /= len(train_loader)
        avg_train_reward /= len(train_loader)
        
        wandb.log({
            'Reward': avg_train_reward,
            'Loss': avg_train_loss,
            'Db step': iters//len(train_db) + 1
        })
        

        if avg_train_reward >= max_reward:
            torch.save(ppo_trainer.ac_net.state_dict(), os.path.join(path_to_model, 'model.pth'))
            max_reward = avg_train_reward
    
        checkpoint = {
            'network': ppo_trainer.ac_net.state_dict(),
            'iters': iters,
            'max_reward': max_reward,
        }
        torch.save(checkpoint, os.path.join(path_to_model, '_checkpoint.pth'))
        # sample new states!
        train_db.sample_df()

train()

