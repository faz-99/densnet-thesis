import torch
import torch.nn as nn
import torch.optim as optim


class LMFloss(nn.Module):
    def __init__(self, num_classes, num_features, alpha=1.0, beta=1.0):
        super(LMFloss, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.alpha = alpha
        self.beta = beta

        self.centers = nn.Parameter(torch.randn(num_classes, num_features))
        self.scale = nn.Parameter(torch.ones(1))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, features, labels):
        # 计算Fisher's Loss
        dist = (features.unsqueeze(1) - self.centers.unsqueeze(0)).pow(2).sum(dim=2)
        fisher_loss = dist[torch.arange(features.size(0)), labels].mean()

        # 辅助损失函数
        aux_loss = nn.CrossEntropyLoss()(self.classifier(features), labels)

        # 总损失
        total_loss = self.alpha * fisher_loss + self.beta * aux_loss

        return total_loss