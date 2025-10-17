"""
加载数据集
@date: 2022/05/01
@author: wuqiwei
"""
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
# from torch.utils.data.sampler import WeightedRandomSampler
# from torch.utils.data.sampler import *

# from imblearn.over_sampling import RandomOverSampler

from config import *
# from utils.sample import *

import numpy as np
import pandas as pd

img_size = img_s

def get_data_loader():
    """
    ImageFolder和DataLoader加载数据集
    :return: train_loader, valid_loader
    """
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(1.0, 1.0)),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # 高斯噪声
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])

    transform_valid = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])

    train_path = "/home/wu/Jia/steel_cls/data/data_40/train"
    valid_path = "/home/wu/Jia/steel_cls/data/data_40/val"

    train_dataset = ImageFolder(train_path, transform=transform_train)
    valid_dataset = ImageFolder(valid_path, transform=transform_valid)

    class_names = train_dataset.classes
    n_class = len(class_names)
    # 映射关系：类别 到 索引号
    train_dataset.class_to_idx
    # 映射关系：索引号 到 类别
    idx_to_labels = {y: x for x, y in train_dataset.class_to_idx.items()}
    # 保存为本地的 npy 文件
    np.save('idx_to_labels.npy', idx_to_labels)
    np.save('labels_to_idx.npy', train_dataset.class_to_idx)

    target = train_dataset.targets
    target1 = valid_dataset.targets
    class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    class_sample_count1 = np.array([len(np.where(target1 == t)[0]) for t in np.unique(target1)])
    # weight = 1. / class_sample_count
    # samples_weight = torch.from_numpy(weight)
    # samples_weight = np.array([samples_weight[t] for t in target])
    # samples_weight = torch.from_numpy(samples_weight)
    # samples_weight = samples_weight.double()

    print('训练集图像数量', len(train_dataset))
    print('类别个数', len(train_dataset.classes))
    print('各类别名称', train_dataset.classes)
    print("各类被数量", class_sample_count)
    # print("各类别权重", weight)

    print('测试集图像数量', len(valid_dataset))
    print('类别个数', len(valid_dataset.classes))
    print('各类别名称', valid_dataset.classes)
    print("各类被数量", class_sample_count1)

    # trainsampler = WeightedRandomSampler(samples_weight, len(train_dataset))
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              # sampler=trainsampler,
                              shuffle=True,
                              # pin_memory=True,
                              num_workers=num_workers)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              # pin_memory=True,
                              num_workers=num_workers,
                              )

    # 输出每个batch的图像标签
    # for i, (images, labels) in enumerate(train_loader):
    #     print('Batch', i + 1, 'labels:', labels)
    # num_training_steps_per_epoch = len(train_dataset) // batch_size

    return train_loader, valid_loader

if __name__ == "__main__":
    get_data_loader()