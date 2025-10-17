"""
计算数据集的均值和方差
@date: 2022/05/01
@author: wuqiwei
"""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from config import *

mean_std_transform = transforms.Compose([
    transforms.Resize([300, 300]),
    transforms.ToTensor()
])


def get_mean_std(dataset):
    """
    计算数据集的均值和方差
    :param dataset: 数据集
    :return: 均值和方差
    """
    loader = DataLoader(dataset, batch_size=16, num_workers=6)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for x, y in loader:
        for i in range(3):
            mean[i] += x[:, i, :, :].mean()
            std[i] += x[:, i, :, :].std()
    mean.div_(len(loader))
    std.div_(len(loader))
    return mean, std


if __name__ == "__main__":
    ds = ImageFolder(train, transform=mean_std_transform)
    m, s = get_mean_std(ds)
    print(m, s)
