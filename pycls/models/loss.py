import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmooth(nn.Module):
    def __init__(self, num_classes, eta=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.eta = eta
    
    def forward(self, input, target):
        soft_target = F.one_hot(target, self.num_classes)
        soft_target = (1 - self.eta) * soft_target + self.eta / self.num_classes
        log_probs = F.log_softmax(input, dim=1)
        loss = (-soft_target * log_probs).mean(0).sum() 
        return loss
