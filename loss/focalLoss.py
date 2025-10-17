import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0, weight: torch.Tensor = None, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.weight = weight
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_onehot = F.one_hot(target, num_classes=input.size(1))
        target_onehot_labelsmoothing = torch.clamp(target_onehot.float(),
                                                   min=self.label_smoothing / (input.size(1) - 1),
                                                   max=1.0 - self.label_smoothing)
        input_softmax = F.softmax(input, dim=1) + 1e-7
        input_logsoftmax = torch.log(input_softmax)
        ce = -1 * input_logsoftmax * target_onehot_labelsmoothing
        fl = torch.pow((1 - input_softmax), self.gamma) * ce
        fl = fl.sum(1) * self.weight[target.long()]
        return fl.mean()