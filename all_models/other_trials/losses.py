import torch
from torch import Tensor

class FocalBCE():
    def __init__(self, reduction=True):
        self.reduction = reduction

    def __call__(self, probs: Tensor, target: Tensor, weights: Tensor, gamma: float) -> Tensor:

        probs = probs.type(torch.float32)
        target = target.type(torch.float32)
        weights = weights.type(torch.float32)

        pt = torch.zeros_like(probs)

        #TODO: Fix this. below is a dirty one
        if len(target.size()) != len(probs.size()):
            target.unsqueeze_(1)
        pt[target==0] = 1-probs[target==0] + 1e-10
        pt[target==1] = probs[target==1] + 1e-10

        log_loss = - torch.log(pt + 1e-35)
        focal_weight = (1-pt)**gamma + 1e-10

        _loss = weights*log_loss*focal_weight

        if self.reduction:
            return _loss.mean()
        else:
            return _loss