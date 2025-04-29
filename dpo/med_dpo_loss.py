import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MedDPOLoss(nn.Module):
    def __init__(self, learned = False, weights = [.4,.3,.3] ):
        super().__init__()
        # are beta, gamma, alpha learned? Hyper?
        # are their a different set of parameters for rejected and accepted?
        self.weights = nn.Parameter(data = torch.Tensor((1,1,1)), requires_grad = True) if learned else weights
        self.weights.data.uniform_(-1, 1)


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
        
        chosen_rewards = np.dot(self.weights, chosen_rewards)    
        rejected_rewards = np.dot(self.weights, rejected_rewards)

        log_pref_logits = torch.log(chosen_logits)
        log_reject_logits = torch.log(rejected_logits)

        conf = log_pref_logits - log_reject_logits

        margins = torch.log(chosen_rewards) - torch.log(rejected_rewards)
        margins /= 1 + F.sigmoid(conf)

        loss = - F.logsigmoid( margins * (log_pref_logits - log_reject_logits))

        return loss