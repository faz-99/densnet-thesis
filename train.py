"""
@date: 2022/05/01
@author: wuqiwei
"""
import argparse
import platform
import pathlib

from loss.focalLoss import FocalLoss
from model.CBAM import CBAMLayer
from model.model import getModel

plt = platform.system()
if plt != 'Windows':
  pathlib.WindowsPath = pathlib.PosixPath

# 正常显示混淆矩阵
import numpy as np
#from timm import create_model
from torch.distributed.run import get_args_parser
from drawLoss import draw_loss
from draw_acc import draw_acc

# TODO
from utils.load_dataset2 import get_data_loader
from model.model import getModel,class_model

np.set_printoptions(suppress=True)

# 忽略烦人的红色提示
import warnings
warnings.filterwarnings("ignore")

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

import os
import sys
import time

import numpy
import timm
import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.optim import SGD

#from model import jiangzao
import config

#from model.SC3cat import convnext_base_in22k as create_model
# from model.convnext_iaff import convnext_base_in22k as create_model
# from model.SC import convnext_base_in22k as create_model
# from model.convnext_eca import convnext_base_in22k as create_model
# from model.convnext import convnext_base_in22k as create_model
# from model.SCcov import convnext_base_in22ft1k as create_model
# from model.CONV2NEXT import convnext_base as create_model
# from model.effi import efficientnet_b4 as create_model
#from model.swintransformer import swin_base_patch4_window7_224 as create_model
#from utils.load_dataset2 import *
# from utils import load_model
from utils.load_model import *
from utils.lion import *
from model.SENet import SELayer
from model.ECALayer import ECALayer

from utils.confusion_matrix import ConfusionMatrix

from torch.utils.tensorboard import SummaryWriter
#writer =SummaryWriter('./steel')

import matplotlib.pyplot as plt
from utils.lr import *
from tqdm import tqdm
from model.swin_transformer import swin_tiny_patch4_window7_224 as create_model
#from model.efficientnet_v2 import efficientnetv2_s as create_model
from model.efficientnet.model import EfficientNet

from acb import ACBlock

#引入评估指标
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#设置中文字体
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

# 指定显卡
if config.is_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

#from timm.models.vision_transformer import checkpoint_filter_fn, get_init_weights_vit

device = torch.device(config.device if torch.cuda.is_available() else "cpu")
#

# 计算参数量
def printparam(net):
    sum_ = 0
    for name, param in net.named_parameters():
        mul = 1
        for size_ in param.shape:
            mul *= size_  # 统计每层参数个数
        sum_ += mul  # 累加每层参数个数
        # print('%14s : %s' % (name, param.shape))  # 打印参数名和参数数量
    # print('%s' % param)                 # 这样可以打印出参数，由于过多，我就不打印了
    print('参数个数：', sum_)  # 打印参数量



def parmhead (model):
    for name, parameters in model.named_parameters():
        # print(name, ':', parameters.size())
        # if "iaff" in name:
        #     print(name, ':', parameters.size(), parameters)
        if "head" in name:
            print(name, ':', parameters.size(), parameters)

def parm (model):

    for name, parameters in model.named_parameters():
        # print(name, ':', parameters.size())
        if "iaff" in name:
            print(name, ':', parameters.size(), parameters)
        if "eca" in name:
            print(name, ':', parameters.size(), parameters)

def train():
    t_loss=[]
    val_loss=[]
    t_acc=[]
    val_acc=[]

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    # 最好的准确率
    best_acc = 0
    # 最好的混淆矩阵
    #best_matrix = numpy.zeros((config.class_num, config.class_num))
    # 加载数据集
    #args,weights
    # TODO
    #args=get_args_parser()
    train_loader, valid_loader,weights= get_data_loader()
    #train_loader, valid_loader= get_data_loader()
    # 加载模型


   #  input_size = getModel().num_features
   # # # input_size=896
   #  hidden_size = 32
   #  output_size=config.class_num
   #  model=ClassificationModel(input_size,hidden_size,output_size)
   #  model.to(device)
   #  print(model)


    #获取debseblock层的输出通道数
    # TODO
    model=class_model()
    model.to(device)
    print(model)


    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)



    # 优化器和学习率调度器
    optimizer = SGD(params=model.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.max_epoch, eta_min=1e-6)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)

    # 损失函数和混淆矩阵
    #print(weights)
    # TODO
    loss_function = nn.CrossEntropyLoss(weight=weights).float().to(device)
    #loss_function=FocalLoss(weight=weights).to(device)
    #loss_function = nn.CrossEntropyLoss()
    confusion_matrix = ConfusionMatrix(config.class_num)
    # 开始循环
    for i in range(1, config.max_epoch + 1):
        # 分配学习率
        # lr = scheduler.get_last_lr()
        lr = scheduler.get_last_lr()
        print("{} epoch: {}, lr: {}".format(time.strftime("%Y-%m-%d %H:%M:%S"), i, lr))
        # 开始训练
        model.train()
        train_loss = 0
        confusion_matrix.__init__(config.class_num)

        optimizer.zero_grad()

        train_loader = tqdm(train_loader, file=sys.stdout)
        for inputs, labels in train_loader:
            if config.is_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            # 清空梯度值，否则在每次进行反向传播时都会累加
            # optimizer.zero_grad()
            outputs = model(inputs)


            # 计算loss和acc
            step_loss = loss_function(outputs, labels)
            train_loss += step_loss.item()
            #print(outputs.argmax(1).cpu().numpy())
            #print(labels.cpu().numpy())
            confusion_matrix.update(outputs.argmax(1).cpu().numpy(), labels.cpu().numpy())
            # 反向传播
            step_loss.backward()

            train_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.8f}".format(
                i,
                train_loss / len(train_loader),
                confusion_matrix.acc(),
                optimizer.param_groups[0]["lr"]
            )

            # 梯度更新
            optimizer.step()
            optimizer.zero_grad()
            del (step_loss, inputs, outputs, labels)

            # optimizer.step()
            # optimizer.zero_grad()
            # # update lr
            # scheduler.step()
        t_loss.append(train_loss / len(train_loader))
        scheduler.step()
        # 训练情况
        train_loss = train_loss / len(train_loader)
        train_acc = confusion_matrix.acc()
        t_acc.append(train_acc)

        # 开始测试
        model.eval()
        valid_loss = 0
        confusion_matrix.__init__(config.class_num)
        valid_loader = tqdm(valid_loader, file=sys.stdout)
        for inputs, labels in valid_loader:
            if config.is_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)
            labels_t=labels.detach().cpu().numpy()
            outputs_t=outputs.detach().cpu().numpy()

            # 找到每一行的最大值索引
            max_indices = np.argmax(outputs_t, axis=1)

            # accuracy=accuracy_score(labels_t,max_indices)

            step_loss = loss_function(outputs, labels)
            valid_loss += step_loss.item()
            # torch.nn.functional.softmax(outputs, dim=-1)
            # acc = accuracy_score(outputs, labels)
            confusion_matrix.update(outputs.argmax(1).cpu().numpy(), labels.cpu().numpy())
            # precision = confusion_matrix.pre()
            # recall = confusion_matrix.recall()
            # f1 = confusion_matrix.f1_score()

            valid_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
                i,
                valid_loss / len(valid_loader),
                confusion_matrix.acc(),
                # precision,
                # recall,
                # f1
            )
            del (step_loss, inputs, outputs, labels)
        val_loss.append(valid_loss / len(valid_loader))
        print(confusion_matrix.report())
        # 验证情况
        valid_loss = valid_loss / len(valid_loader)
        valid_acc = confusion_matrix.acc()
        val_acc.append(valid_acc)
        # 保存最好模型、最好准确率、最好混淆矩阵
        if best_acc < valid_acc:
            best_acc = valid_acc
            best_matrix = confusion_matrix.matrix
            best_report=confusion_matrix.report()
            save_model = {
                'epoch': i,
                'best_acc': best_acc,
                'best_matrix': best_matrix,
                'best_report': best_report,
                'model': model,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': {
                    'class_num': config.class_num,
                    'img_s': config.img_s,
                    'dataset_mean': config.dataset_mean,
                    'dataset_std': config.dataset_std,
                    'net_name': config.net_name
                }
            }
            
            # Ensure save directory exists
            save_dir = "weight/save/40"
            os.makedirs(save_dir, exist_ok=True)
            
            model_save_path = os.path.join(save_dir, f"{config.net_name}_5.pth")
            torch.save(save_model, model_save_path)
            
            print("{} epoch: {}, train_loss: {:.6f}, train_acc: {:.4f}, valid_loss: {:.6f}, valid_acc: {:.4f}"
                  .format(time.strftime("%Y-%m-%d %H:%M:%S"), i, train_loss, train_acc, valid_loss, valid_acc))
            print("{} epoch: {}, matrix:\n{}".format(time.strftime("%Y-%m-%d %H:%M:%S"), i, confusion_matrix.matrix))
            print("Model saved to: {}".format(model_save_path))

        # print("{} epoch: {}, matrix:\n{}".format(time.strftime("%Y-%m-%d %H:%M:%S"), i, confusion_matrix.matrix))
        # 保存这一轮训练和验证结果
        with open("csv/40/{}_5.csv".format(config.net_name), "a", encoding="utf-8") as file:
            content = "{},{:.6f},{:.4f},{:.6f},{:.4f},{}\n" \
                .format(i, train_loss, train_acc, valid_loss, valid_acc, time.strftime("%Y-%m-%d %H:%M:%S"))
            file.write(content)

        #writer.add_scalars("loss_5", {'train_loss': train_loss, 'valid_loss': valid_loss}, i)
        #writer.add_scalars("acc_5", {'train_acc': train_acc, 'valid_acc': valid_acc}, i)
        #writer.close()

        # parm(model)

    draw_loss(t_loss,val_loss,config.max_epoch)
    draw_acc(t_acc,val_acc,config.max_epoch)
    print("best_acc:{}".format(best_acc))
    print("best_matrix:\n{}".format(best_matrix))
    print("best_report:\n{}".format(best_report))

if __name__ == "__main__":
    # parents=[get_args_parser()]
    parser = argparse.ArgumentParser('ConvNeXt trai parents=[get_args_parser()]ning and evaluation script')
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='./weight/swin_tiny_patch4_window7_224.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    args = parser.parse_args()
    train()











