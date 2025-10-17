"""
加载网络模型
@date: 2022/05/01
@author: wuqiwei
"""
import torch
from torch import nn
from torch.nn import functional

import config
#from model import inception, resnet, se_resnet, cbam_resnet, swintransformer_new, efficientnet, convnext


class LoadModel(nn.Module):
    def __init__(self, class_num=2):
        super(LoadModel, self).__init__()
        if config.net_name == "Inception":
            # 特征向量是2048维的
            self.subnet = inception.inception_v3()
            self.fc = torch.nn.Linear(2048, class_num)
            # 加载预训练权重
            self.subnet.load_state_dict(torch.load(config.base_weight1), strict=False)
        elif config.net_name == "ResNet":
            # 特征向量是2048维的
            self.subnet = cbam_resnet.resnet50()
            self.fc = torch.nn.Linear(2048, class_num)
            # 加载预训练权重
            self.subnet.load_state_dict(torch.load(config.base_weight2), strict=False)
        elif config.net_name == "swintransformer":
            # 特征向量是2048维的
            self.subnet = swintransformer_new.swin_tiny_patch4_window7_224(num_classes=3)
            # self.fc = torch.nn.Linear(2048, class_num)
            # 加载预训练权重
            self.subnet.load_state_dict(torch.load(config.base_weight3), strict=False)
        elif config.net_name == "efficientnet":
            # 特征向量是2048维的
            self.subnet = efficientnet.efficientnet_b0(num_classes=8)
            # self.fc = torch.nn.Linear(2048, class_num)
            # 加载预训练权重
            self.subnet.load_state_dict(torch.load(config.base_weight4), strict=False)
        elif config.net_name == "convnext":
            # 特征向量是2048维的
            self.subnet = convnext.convnext_base(num_classes=8)
            # self.fc = torch.nn.Linear(2048, class_num)
            # 加载预训练权重
            self.subnet.load_state_dict(torch.load(config.base_weight5), strict=False)
        else:
            raise ValueError("load net is error")

    def forward(self, x):
        # 获取到x的形状
        b, c, h, w = x.size()
        # 获取提取出来的特征向量
        feature = self.subnet(x)
        # 进行全局平均池化
        # feature = functional.adaptive_avg_pool2d(feature, 1).view(b, -1)
        # dropout防止过拟合
        # feature = functional.dropout(feature, p=0.3, training=self.training)
        # 全连接层进行分类
        # feature = self.fc(feature)
        return feature
