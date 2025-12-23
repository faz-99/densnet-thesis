# Cell 6: Class Imbalance Handling for BreakHis
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, num_classes=8, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.size_average = size_average
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class ClassBalancer:
    def __init__(self, labels):
        self.labels = labels
        self.class_counts = Counter(labels)
        self.num_classes = len(self.class_counts)
        
    def get_class_weights(self):
        """Calculate class weights for loss function"""
        total_samples = len(self.labels)
        weights = []
        
        for i in range(self.num_classes):
            weight = total_samples / (self.num_classes * self.class_counts.get(i, 1))
            weights.append(weight)
            
        return torch.FloatTensor(weights)
    
    def get_sample_weights(self):
        """Calculate sample weights for WeightedRandomSampler"""
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[label] for label in self.labels]
        return torch.FloatTensor(sample_weights)
    
    def create_balanced_sampler(self):
        """Create WeightedRandomSampler for balanced training"""
        sample_weights = self.get_sample_weights()
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

class_names = ['Adenosis', 'Fibroadenoma', 'Phyllodes_tumor', 'Tubular_adenoma',
               'Ductal_carcinoma', 'Lobular_carcinoma', 'Mucinous_carcinoma', 'Papillary_carcinoma']

focal_loss = FocalLoss(alpha=1, gamma=2, num_classes=8)
print("Class imbalance handling initialized with Focal Loss")