
import torch
from torch.distributions.categorical import Categorical
from models.rl_agent import ActorCritic

class PPOAgent:
    def __init__(self, 
        action_space,
        # model params
        arch,
        model_path,
        return_logits=False
        ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.action_space = action_space

        self.ac_net = ActorCritic(out_dim=action_space, arch=arch, dropout=0)
        self.ac_net.eval()
        self.ac_net.load_state_dict(torch.load(model_path))
        self.ac_net.to(self.device)
        self.return_logits = return_logits
    
    def act(self, x_img, x_mask, x_cost=None):
        with torch.no_grad():
            logits, value = self.ac_net(x_img, x_mask, x_cost)
        
        if not self.return_logits:
            dist = Categorical(logits=logits)
            #action = logits.argmax()
            action = dist.sample()
            return action.item(), value
        else :
            return logits, value
        
