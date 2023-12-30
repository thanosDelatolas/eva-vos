import torch
import torch.nn.functional as F

from copy import deepcopy
from ppo.ppo_trainer import PPOTrainer
from .annotation_env import AnnotationEnv
from ppo.storage import RolloutStorage

def compute_returns(ep_rewards, gamma, paddings):
    ep_returns = []
    discounted_reward = 0
    for rew, pad in zip(reversed(ep_rewards), reversed(paddings)):
        if not pad:
            discounted_reward = rew + discounted_reward * gamma
            ep_returns.insert(0, discounted_reward)

    # Convert the retruns into a tensor
    ep_returns = torch.tensor(ep_returns, dtype=torch.float)
    return ep_returns


def calculate_gaes(rewards, values, gamma=0.99, decay=0.97):
    """
    Return the General Advantage Estimates from the given rewards and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf

    Args:
    rewards (torch.Tensor): 1D tensor of rewards
    values (torch.Tensor): 1D tensor of state values
    gamma (float): discount factor
    decay (float): decay factor for GAE

    Returns:
    torch.Tensor: 1D tensor of General Advantage Estimates (GAEs)
    """
    next_values = torch.cat([values[1:], torch.tensor([0.])])
    deltas = rewards + gamma * next_values - values

    gaes = [deltas[-1].item()]
    for i in reversed(range(len(deltas) - 1)):
        gaes.append(deltas[i].item() + decay * gamma * gaes[-1])

    return torch.tensor(gaes[::-1])



def rollout(ppo_trainer:PPOTrainer, image, gt_mask, init_mask, env_num, rollout_storage:RolloutStorage, num_steps, gamma, advantages, device):

    ep_masks = []       
    ep_actions = []    
    ep_rewards = []          
    ep_log_probs = []   
    ep_values = []    
    paddings = []
    ann_env = AnnotationEnv(image, gt_mask, init_mask, num_steps, device=device)
    
    for _ in range(num_steps):
        state = ann_env.state
        action, log_prob, value = ppo_trainer.act(*state, prev_actions=ann_env.annotation_actions)
        
        reward, _, done = ann_env.step(action.item())
        ep_masks.append(state[1])
        ep_actions.append(action.item())
        ep_rewards.append(reward)
        ep_log_probs.append(log_prob.item())     
        ep_values.append(value.item())
        paddings.append(False)
        if done: 
            break
    
    # if len(ep_masks) == 1:
    #     pass
    ep_returns = compute_returns(ep_rewards, gamma, paddings).cpu()
    ep_masks = torch.cat(ep_masks).cpu()
    ep_actions = torch.tensor(ep_actions).cpu()
    ep_rewards = torch.tensor(ep_rewards).cpu()
    ep_log_probs = torch.tensor(ep_log_probs).cpu()
    ep_values = torch.tensor(ep_values).cpu()

    if advantages == 'diff':
        ep_adv = ep_returns - ep_values
        #ep_adv = (ep_adv - ep_adv.mean()) / (ep_adv.std() + 1e-5)
    elif advantages == 'gae':
        ep_adv = calculate_gaes(ep_rewards, ep_values)
    else :
        raise AttributeError('Invalid adv type')
        
    img_embedding = deepcopy(state[0].squeeze()).cpu()
    
    steps = len(ep_masks)
    # torch.Size([3, 3, 224, 224])
    diff = num_steps-steps
    if diff != 0:
        paddings.extend([True] * diff)
        pad_shape = (0,0, 0,0, 0,0, 0,diff)
        ep_masks = F.pad(ep_masks, pad_shape, "constant", -1)
        ep_actions = F.pad(ep_actions, (0,diff), "constant", -1)
        ep_rewards = F.pad(ep_rewards, (0,diff), "constant", -1)
        ep_log_probs = F.pad(ep_log_probs, (0,diff), "constant", -1)
        ep_values = F.pad(ep_values, (0,diff), "constant", -1)
        ep_returns = F.pad(ep_returns, (0,diff), "constant", -1)
        ep_adv = F.pad(ep_adv, (0,diff), "constant", -1)
    
    paddings = torch.tensor(paddings, dtype=torch.bool, device='cpu')
    rollout_storage.insert(env_num, ep_masks, img_embedding, ep_actions, ep_log_probs, ep_values, ep_rewards, ep_returns, paddings, ep_adv)
    

