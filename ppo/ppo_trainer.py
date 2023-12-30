import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.distributions.categorical import Categorical

from models.rl_agent import ActorCritic

class PPOTrainer:
    def __init__(self,
        action_space,
        # ppo params
        ppo_epochs,
        clip_param,
        value_loss_coef,
        entropy_coef,
        target_kl_div,
        
        #optim parms
        lr,
        optim_str,

        # model params
        arch,
        dropout,
        ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert optim_str in {'Adam' , 'SGD'}
        self.action_space = action_space

        self.ppo_epochs = ppo_epochs
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.target_kl_div = target_kl_div

        self.ac_net = ActorCritic(out_dim=action_space, arch=arch, dropout=dropout)
        self.ac_net.to(self.device)
        self.ac_net.train()

        total_params = sum(p.numel() for p in self.ac_net.parameters() if p.requires_grad)
        print(f'Trainable parameters: {total_params/1_000_000:.2f}M')

        if optim_str == 'Adam':
            self.optimizer = AdamW(self.ac_net.parameters(), lr=lr)
        else:
            self.optimizer = SGD(self.ac_net.parameters(), lr=lr, momentum=0.9)
    
    def act(self, x_img, x_mask, x_cost=None, prev_actions=None):
        with torch.no_grad():
            logits, value = self.ac_net(x_img, x_mask, x_cost)
        
        # if self.action_space == 3 and len(prev_actions) == 0:
        #     logits = logits[:,:2] # invalid action

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value
    

    def evaluate(self, ep_actions, x_imgs, x_mask, x_cost=None):        
        logits, pred_values = self.ac_net(x_imgs, x_mask, x_cost)        
        dist = Categorical(logits=logits)
        action_logprobs = dist.log_prob(ep_actions)
        dist_entropy = dist.entropy()
        return action_logprobs, pred_values, dist_entropy


    def optimize(self, rollouts):
        total_loss = 0.0
        
        steps = 0
        self.ac_net.train()
        for _ in range(self.ppo_epochs):
            data_generator = rollouts.data_generator(self.device)
            for sample in data_generator:
                masks_batch, img_embeddings_batch, actions_batch, value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ = sample
                self.optimizer.zero_grad()

                curr_log_probs, values, dist_entropy = self.evaluate(actions_batch, img_embeddings_batch, masks_batch)

                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                # NOTE: we just subtract the logs, which is the same as
                # dividing the values and then canceling the log with e^log
                # reason here: https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms

                ratios = torch.exp(curr_log_probs - old_action_log_probs_batch)

                surr1 = ratios * adv_targ
                surr2 = torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param) * adv_targ
                critic_loss = nn.MSELoss()(values, return_batch.unsqueeze(1))

                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) + self.value_loss_coef * critic_loss - self.entropy_coef * dist_entropy
                loss.mean().backward()
                self.optimizer.step()
                steps+=1

                total_loss += loss.mean().item()

                kl_div = (old_action_log_probs_batch - curr_log_probs).mean()
                if self.target_kl_div is not None and kl_div >= self.target_kl_div:
                    break


        self.ac_net.eval()
        return total_loss/steps
