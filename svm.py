import sys

import torch
import torchvision.models as models
from sklearn import svm
from sklearn.metrics import accuracy_score
from torch import nn
from tqdm import tqdm

from model.model import getModel
from utils.load_dataset2 import get_data_loader

model=getModel()
weights_pth="./weight/save/densenet201_SE100.0005.pth"
model.load_state_dict(torch.load(weights_pth),strict=False)


# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 移除原始模型的分类层
model = nn.Sequential(*list(model.children())[:-1])

train_loader, valid_loader, weights = get_data_loader()

# 提取特征
features = []
labels = []
train_loader = tqdm(train_loader, file=sys.stdout)
for images, targets in train_loader:
    with torch.no_grad():
        batch_features = model(images).squeeze()
        batch_size = batch_features.size(0)
        features.append(batch_features)
    labels.extend(targets.numpy())

print(len(features))
print(features[0].shape)
print(len(labels))
features = torch.cat(features, dim=0).numpy()

# 训练SVM模型
svm_model = svm.SVC()
svm_model.fit(features, labels)

# 在测试集上进行预测并计算准确率

test_labels = []
pred_labels = []
valid_loader = tqdm(valid_loader, file=sys.stdout)
for images, targets in valid_loader:
    with torch.no_grad():
        features = model(images).squeeze().numpy()
        pred = svm_model.predict(features)
        pred_labels.extend(pred)
        test_labels.extend(targets.numpy())

accuracy = accuracy_score(test_labels, pred_labels)
print("模型准确率：{:.2f}%".format(accuracy * 100))

#print(model)