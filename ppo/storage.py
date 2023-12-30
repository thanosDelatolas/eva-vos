import torch
from einops import repeat, rearrange

class RolloutStorage(object):
    """
    Creates a minibatch with all the simulations from different environments.
    """
    def __init__(self, num_envs, num_steps, obs_shape, num_mini_batch):

        assert num_envs >= num_mini_batch

        self.num_envs = num_envs
        self.num_mini_batch = num_mini_batch
        self.num_steps = num_steps

        self.masks = torch.zeros(num_envs,num_steps, *obs_shape, device='cpu')
        self.img_embeddings = torch.zeros(num_envs, 256,64,64, device='cpu')

        self.rewards = torch.zeros(num_envs,num_steps, device='cpu')
        self.value_preds = torch.zeros(num_envs,num_steps, device='cpu')
        self.returns = torch.zeros(num_envs,num_steps, device='cpu')
        self.action_log_probs = torch.zeros(num_envs,num_steps, device='cpu')
        self.actions = torch.zeros(num_envs, num_steps, device='cpu')
        self.paddings = torch.zeros(num_envs, num_steps, device='cpu')
        self.advantages = torch.zeros(num_envs, num_steps, device='cpu')
        


    def insert(self, env_num, masks, img_embedding, actions, action_log_probs, value_preds, rewards, returns, paddings, adv):
        self.masks[env_num].copy_(masks)
        self.img_embeddings[env_num].copy_(img_embedding) 

        self.actions[env_num].copy_(actions)
        self.action_log_probs[env_num].copy_(action_log_probs)
        self.value_preds[env_num].copy_(value_preds)
        self.rewards[env_num].copy_(rewards)
        self.returns[env_num].copy_(returns)
        self.paddings[env_num].copy_(paddings)
        if adv is not None:
            self.advantages[env_num].copy_(adv)



    def data_generator(self, device):
        """
        Creates a minibatch with simulation for all environments
        """
        num_envs_per_batch = self.num_envs // self.num_mini_batch
        perm = torch.randperm(self.num_envs)

        for start_ind in range(0, self.num_envs, num_envs_per_batch):
            masks_batch = []
            img_embeddings_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                valid_idxs = torch.where(self.paddings[ind] == 0)[0]
                masks_batch.append(self.masks[ind][valid_idxs])

                img_embed = self.img_embeddings[ind]
                img_embed = repeat(img_embed, 'f h w -> v f h w', v=valid_idxs.shape[0])
                img_embeddings_batch.append(img_embed)
                actions_batch.append(self.actions[ind][valid_idxs])
                value_preds_batch.append(self.value_preds[ind][valid_idxs])
                return_batch.append(self.returns[ind][valid_idxs])
                old_action_log_probs_batch.append(self.action_log_probs[ind][valid_idxs])
                adv_targ.append(self.advantages[ind][valid_idxs])

            masks_batch = torch.cat(masks_batch).to(device)
            img_embeddings_batch = torch.cat(img_embeddings_batch).to(device)
            actions_batch = torch.cat(actions_batch).to(device)
            value_preds_batch = torch.cat(value_preds_batch).to(device)
            return_batch = torch.cat(return_batch).to(device)
            old_action_log_probs_batch = torch.cat(old_action_log_probs_batch).to(device)
            adv_targ = torch.cat(adv_targ).to(device)
            yield masks_batch, img_embeddings_batch, actions_batch, value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ
                