#按照不同的类别按照一定比例进行采样

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import os
import random


class MySampler(Sampler):
    def __init__(self, labels, ratio):
        self.labels = labels
        self.ratio = ratio

    def __iter__(self):
        indices = []
        for label in set(self.labels):
            class_indices = [i for i, l in enumerate(self.labels) if l == label]
            sample_size = int(len(class_indices) * self.ratio[label])
            sampled_indices = random.sample(class_indices, sample_size)
            indices.extend(sampled_indices)
        return iter(indices)

    def __len__(self):
        return len(self.labels)


class MyDataset(Dataset):
    def __init__(self, root_dir, ratio):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        class_names = os.listdir(root_dir)
        class_labels = {class_name: i for i, class_name in enumerate(class_names)}
        for class_name in class_names:
            class_path = os.path.join(root_dir, class_name)
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                self.data.append(file_path)
                self.labels.append(class_labels[class_name])

        self.sampler = MySampler(self.labels, ratio)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        # load data and preprocess if necessary
        return data, label


# root_dir = '/path/to/data'
# # ratio = {0: 0.3, 1: 0.3, 2: 0.3, 3: 0.3, 4: 0.2, 5: 0.2, 6: 0.2, 7: 0.2}
ratio = {0: 0.2, 1: 0.8}
#
# dataset = MyDataset(root_dir, ratio)
# dataloader = DataLoader(dataset, batch_size=32, sampler=dataset.sampler)

# for batch_data, batch_label in dataloader:
# # do something with the batch data
