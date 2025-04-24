import torch
import torch.nn as nn
import torch.nn.functional as F

class MedDPOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # are beta, gamma, alpha learned? Hyper?
    
    def forward(self, 
                preffered_logits,
                rejected_logits,
                preffered_rewards,
                rejected_rewards):
        
        log_pref_logits = torch.log(preffered_logits)
        log_reject_logits = torch.log(rejected_logits)

        conf = log_pref_logits - log_reject_logits

        margins = torch.log(preffered_rewards) - torch.log(rejected_rewards)
        margins /= 1 + F.sigmoid(conf)

        loss = - F.logsigmoid( margins * (log_pref_logits - log_reject_logits))

        return loss