"""
加载网络模型
@date: 2022/05/01
@author: wuqiwei
"""
import os

import torch
from torch import nn, device
from torch.nn import functional

import config
from model import inception, resnet, se_resnet, cbam_resnet, swintransformer_new, efficientnet, convnext, swinconv


class LoadModel(nn.Module):
    def __init__(self, class_num=8):
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
        #elif config.net_name == "efficientnet":
            ## 特征向量是2048维的
            #self.subnet = efficientnet.efficientnet_b5(num_classes=1000)
            ## self.fc = torch.nn.Linear(2048, class_num)
            ## 加载预训练权重
            #self.subnet.load_state_dict(torch.load(config.base_weight7), strict=False)
        elif config.net_name == "efficientnet":
            # 加载模型
            self.subnet = efficientnet.efficientnet_b6(num_classes=8)
            # 加载预训练权重
            model_weight_path = "weight/base/efficientnetb6.pth"  # 后期可以使用自己数据集的训练权重作为预训练模型，当为老师，实现半监督（知识蒸馏）
            # 判断预训练权重文件在不在
            assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)