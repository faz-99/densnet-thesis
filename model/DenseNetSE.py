import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SENetDenseNet(nn.Module):
    def __init__(self):
        super(SENetDenseNet, self).__init__()
        self.densenet = timm.create_model('densenet201', pretrained=False,num_classes=8)
        for name, module in self.densenet.named_modules():
            if isinstance(module, timm.models.densenet.DenseBlock):
                blocks = list(module.children())
                num_blocks = len(blocks)
                for i in range(num_blocks - 1):
                    blocks.insert((2*i)+1, SEBlock(module.num_features))
                setattr(self.densenet, name, nn.Sequential(*blocks))

    def forward(self, x):
        return self.densenet(x)



if __name__=='__main__':
    # 创建SENetDenseNet实例
    model = SENetDenseNet()
    print(model)