# Cell 4: Advanced Augmentations (MixUp, CutMix, Random Erasing)
import torch
import numpy as np
import random

class MixUp:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def __call__(self, batch_x, batch_y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * batch_x + (1 - lam) * batch_x[index, :]
        y_a, y_b = batch_y, batch_y[index]
        
        return mixed_x, y_a, y_b, lam

class CutMix:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, batch_x, batch_y):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size)
        
        y_a, y_b = batch_y, batch_y[index]
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(batch_x.size(), lam)
        batch_x[:, :, bbx1:bbx2, bby1:bby2] = batch_x[index, :, bbx1:bbx2, bby1:bby2]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch_x.size()[-1] * batch_x.size()[-2]))
        
        return batch_x, y_a, y_b, lam
    
    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2

class RandomErasing:
    def __init__(self, probability=0.15, sl=0.02, sh=0.15, r1=0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        
    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
            
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)
            
            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                img[0, x1:x1+h, y1:y1+w] = random.random()
                img[1, x1:x1+h, y1:y1+w] = random.random()
                img[2, x1:x1+h, y1:y1+w] = random.random()
                return img
                
        return img

mixup = MixUp(alpha=0.2)
cutmix = CutMix(alpha=1.0)
random_erasing = RandomErasing(probability=0.15)
print("Advanced augmentation methods initialized")