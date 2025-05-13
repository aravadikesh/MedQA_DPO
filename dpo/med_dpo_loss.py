import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MedDPOLoss(nn.Module):
    def __init__(self, learned = False, weights =[.4,.3,.3] ):
        super().__init__()
        self.weights = nn.Parameter(data = torch.Tensor((1,1,1)), requires_grad = True) if learned else weights
        self.learned = learned

    def forward(self, 
                chosen_logits: torch.Tensor,
                rejected_logits: torch.Tensor,
                chosen_rewards,
                rejected_rewards):
        '''
        chosen_logits: output from model when prompted with preffered output.\n
        rejected_logits: output from model when prompted with rejected output.\n
        chosen_rewards: rewards given to chosen output by 4o (R_acc, R_saftey, R_expl).\n
        rejected_rewards: rewards given to rejected output by 4o (R_acc, R_saftey, R_expl)

        '''
        
        weights = torch.tensor(self.weights, device=chosen_rewards.device, dtype=chosen_rewards.dtype, requires_grad=False)
        
        chosen_reward = (chosen_rewards * weights).sum(dim=1).to("cuda")    
        rejected_reward = (rejected_rewards * weights).sum(dim=1).to("cuda")

        chosen_log_probs = torch.log_softmax(chosen_logits, dim = -1)
        rejected_log_probs = torch.log_softmax(rejected_logits, dim = -1)


        chosen_conf = chosen_log_probs.logsumexp(dim=-1).mean()
        rejected_conf = rejected_log_probs.logsumexp(dim=-1).mean()

        conf = chosen_conf - rejected_conf
      

        margins = torch.log(chosen_reward) - torch.log(rejected_reward)
        margins = margins / 1 + F.sigmoid(conf) #margins.clamp(-5.0, 5.0) allows qwen to train but due to overfitting, further training just cause the model to output nonsense
        loss = - F.logsigmoid( margins * (chosen_conf - rejected_conf))

        return loss