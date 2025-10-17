import matplotlib.pyplot as plt
import numpy as np

import config


def draw_loss(train_losses,val_losses,epoch):
    # 保存损失图像为文件
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1,epoch+1),train_losses, label='Training Loss')
    plt.plot(np.arange(1,epoch+1),val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(False)
    # TODO
    plt.savefig(f'csv/40/{config.net_name}LOSS_5.png')

