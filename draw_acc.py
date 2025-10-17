import matplotlib.pyplot as plt
import numpy as np

import config


def draw_acc(train_acc,val_acc,epoch):
    # 保存损失图像为文件
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1,epoch+1),train_acc, label='Training Accuracy')
    plt.plot(np.arange(1,epoch+1),val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(False)
    # TODO
    plt.savefig(f'csv/40/{config.net_name}ACC_5.png')