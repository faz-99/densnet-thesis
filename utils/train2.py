"""
@date: 2022/05/01
@author: wuqiwei
"""
# 正常显示混淆矩阵
import numpy as np
np.set_printoptions(suppress=True)

# 忽略烦人的红色提示
import warnings
warnings.filterwarnings("ignore")

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

import os
import time

import numpy
import timm
import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.optim import SGD

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data.mixup import Mixup
from model import jiangzao
import config

# from model.canchagaibian import convnext_base as create_model
# from model.SC import convnext_base as create_model
# from model.convnext_bif import convnext_base as create_model
# from model.dcnv3 import convnext_base as create_model
from model.convnext import convnext_base as create_model
# from model.SPPCSPC_convnext import convnext_base as create_model
# from model.convnext_aspp import convnext_base as create_model
# from model.duochidu import convnext_base as create_model
# from model.SPPCSPC_convnext import convnext_base as create_model
# from model.PCBconv import convnext_base_in22ft1k as create_model
# from model.ASPPconv import convnext_base_in22ft1k as create_model
# from model.SCcov import convnext_base_in22ft1k as create_model
# from model.CONV2NEXT import convnext_base as create_model
# from model.effi import efficientnet_b4 as create_model
# from model.new2 import convnext_base as create_model
# from model.swintransformer_new import swin_base_patch4_window7_224
from utils.load_dataset import *
# from utils import load_model
from utils.load_model import *
from utils.lion import *

from utils.confusion_matrix import ConfusionMatrix

from torch.utils.tensorboard import SummaryWriter
writer =SummaryWriter('./logs/')

import matplotlib.pyplot as plt
from utils.lr import *
from tqdm import tqdm

#设置中文字体
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

# 指定显卡
if config.is_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

from timm.models.vision_transformer import checkpoint_filter_fn, get_init_weights_vit
# 修改预训练权重
def checkpoint_filter_fn1(state_dict, model):
    """ Remap FB checkpoints -> timm """
    if 'head.norm.weight' in state_dict or 'norm_pre.weight' in state_dict:
        return state_dict  # non-FB checkpoint
    if 'model' in state_dict:
        state_dict = state_dict['model']
    out_dict = {}
    import re
    for k, v in state_dict.items():
        k = k.replace('downsample_layers.0.', 'stem.')
        k = re.sub(r'stages.([0-9]+).([0-9]+)', r'stages.\1.blocks.\2', k)
        k = re.sub(r'downsample_layers.([0-9]+).([0-9]+)', r'stages.\1.downsample.\2', k)
        k = k.replace('dwconv', 'conv_dw')
        k = k.replace('pwconv', 'mlp.fc')
        k = k.replace('head.', 'head.fc.')
        if k.startswith('norm.'):
            k = k.replace('norm', 'head.norm')
        if v.ndim == 2 and 'head' not in k:
            model_shape = model.state_dict()[k].shape
            v = v.reshape(model_shape)
        out_dict[k] = v
    return out_dict

def checkpoint_filter_fn2(state_dict, model):
    """ Remap FB checkpoints -> timm """
    if 'head.norm.weight' in state_dict or 'norm_pre.weight' in state_dict:
        return state_dict  # non-FB checkpoint
    if 'model' in state_dict:
        state_dict = state_dict['model']
    out_dict = {}
    import re
    for k, v in state_dict.items():
        k = k.replace('downsample_layers.0.', 'stem.')
        k = re.sub(r'stages.([0-9]+).([0-9]+)', r'stages.\1.blocks.\2', k)
        k = re.sub(r'downsample_layers.([0-9]+).([0-9]+)', r'stages.\1.downsample.\2', k)
        k = k.replace('dwconv', 'conv_dw')
        k = k.replace('pwconv', 'mlp.fc')
        k = k.replace('head.', 'head.fc.')
        if k.startswith('norm.'):
            k = k.replace('norm', 'head.norm')

        if k == "stages.0.blocks.0.attn.qkv.qkv.weight":
            print(k)
            model_shape = model.state_dict()[k].shape
            v = v.reshape(model_shape[0], model_shape[1])  # Reshape correctly

        if v.ndim == 2 and 'head' not in k:
            print(k)
            model_shape = model.state_dict()[k].shape
            v = v.reshape(model_shape[0], model_shape[1])  # Reshape correctly
        out_dict[k] = v
    return out_dict

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

# 加载预训练模型
def load_one (model):
    checkpoint = torch.load(config.base_weight5, map_location=device)
    checkpoint_model = None
    for model_key in ["model"]:
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break

    if checkpoint_model is None:
        checkpoint_model = checkpoint

    # 改变预训练权重里键名
    state_dict = model.state_dict()
    # print(model)
    print(state_dict.keys())
    print(checkpoint_model.keys())
    # print(checkpoint_model1.keys())
    checkpoint_model = checkpoint_filter_fn1(checkpoint_model, model)
    print(checkpoint_model.keys())

    import torch.nn.init as init
    for k in state_dict.keys():
        if k in ['head.fc.weight', 'head.fc.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
                continue
        if k not in checkpoint_model or checkpoint_model[k].shape != state_dict[k].shape:
            if "norm" in k :
                if "weight" in k:
                    nn.init.constant_(state_dict[k], 1.0)
                if "bias" in k:
                    nn.init.constant_(state_dict[k], 0.)
                # init.constant_(state_dict[k], 1)
                # from timm.models.layers import trunc_normal_
                # trunc_normal_(state_dict[k], std=.02)
                print(k)
                continue
            if "weight" in k and "norm" not in k:
                # if len(state_dict[k].shape) != 1:
                init.kaiming_normal_(state_dict[k])  # init.kaiming_normal_ 方法使用 Kaiming 初始化方法对权重进行初始化，
                # 该方法根据权重的形状和深度来计算标准差，并以标准差为参数进行正态分布初始化。
                print(k)
                continue
            if "gamma" in k:
                init.constant_(state_dict[k], 0.)  # init.constant_ 方法将指定权重的值初始化为常数
                print(k)
                continue
            if "bias" in k and "norm" not in k:
                init.constant_(state_dict[k], 0.)  # init.constant_ 方法将指定权重的值初始化为常数
                print(k)

    print(model.load_state_dict(checkpoint_model, strict=False))
    return model

def load_two (model):
    checkpoint = torch.load(config.base_weight5, map_location=device)
    checkpoint1 = torch.load(config.base_weight6, map_location=device)
    checkpoint_model = checkpoint["model"]
    checkpoint_model1 = checkpoint1["model"]
    # for model_key in ["model"]:
    #     if model_key in checkpoint:
    #         checkpoint_model = checkpoint["model"]
    #         print("Load state_dict by model_key = %s" % model_key)
    #         break
    #     if model_key in checkpoint1:
    #         checkpoint_model1 = checkpoint1[model_key]
    #         print("Load state_dict by model_key = %s" % model_key)
    #     break

    if checkpoint_model is None:
        checkpoint_model = checkpoint
        checkpoint_model1 = checkpoint1

    # 改变预训练权重里键名
    state_dict = model.state_dict()
    # print(model)
    print(state_dict.keys())
    print(checkpoint_model.keys())
    print(checkpoint_model1.keys())

    checkpoint_model = checkpoint_filter_fn1(checkpoint_model, model)
    print(checkpoint_model.keys())

    checkpoint_model1 = checkpoint_filter_fn(checkpoint_model1, model)
    checkpoint_model1 = checkpoint_filter_fn1(checkpoint_model1, model)
    print(checkpoint_model1.keys())

    for k in checkpoint_model:
        if k in checkpoint_model1:
            print(k)
            del checkpoint_model1[k]

    merged_checkpoint = {}
    merged_checkpoint.update(checkpoint_model)
    merged_checkpoint.update(checkpoint_model1)
    print(merged_checkpoint.keys())

    import torch.nn.init as init
    for k in state_dict.keys():
        if k in ['head.fc.weight', 'head.fc.bias']:
            if k in merged_checkpoint and merged_checkpoint[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del merged_checkpoint[k]
                continue
        if k not in merged_checkpoint or merged_checkpoint[k].shape != state_dict[k].shape:
            if "norm" in k and "bias" not in k:
                # if "weight" in k:
                #     nn.init.constant_(state_dict[k], 1.0)
                # if "bias" in k:
                #     nn.init.constant_(state_dict[k], 0.)
                # init.constant_(state_dict[k], 1)
                from timm.models.layers import trunc_normal_
                trunc_normal_(state_dict[k], std=.02)
                print(k)
                continue
            if "weight" in k and "norm" not in k:
                # if len(state_dict[k].shape) != 1:
                init.kaiming_normal_(state_dict[k])  # init.kaiming_normal_ 方法使用 Kaiming 初始化方法对权重进行初始化，
                # 该方法根据权重的形状和深度来计算标准差，并以标准差为参数进行正态分布初始化。
                print(k)
                continue
            if "gamma" in k:
                init.constant_(state_dict[k], 0.)  # init.constant_ 方法将指定权重的值初始化为常数
                print(k)
                continue
            if "bias" in k:
                init.constant_(state_dict[k], 0.)  # init.constant_ 方法将指定权重的值初始化为常数
                print(k)

    print(model.load_state_dict(merged_checkpoint, strict=False))
    return model

def train():

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    # 最好的准确率
    best_acc = 0
    # 最好的混淆矩阵
    best_matrix = numpy.zeros((config.class_num, config.class_num))
    # 加载数据集
    train_loader, valid_loader, num_training_steps_per_epoch = get_data_loader()
    # 加载模型
    # model = timm.models.convnext.convnext_base(pretrained=False, num_classes=config.class_num).to(device)
    # model = timm.models.resnet.resnetrs50(pretrained=True, num_classes=config.class_num).to(device)
    # model = timm.models.efficientnet.efficientnet_b4(pretrained=False, num_classes=config.class_num).to(device)
    model = create_model(pretrained=False, num_classes=config.class_num).to(device)

    # checkpoint = torch.load(config.base_weight5, map_location=device)
    # checkpoint_model = None
    # for model_key in ["model"]:
    #     if model_key in checkpoint:
    #         checkpoint_model = checkpoint[model_key]
    #         print("Load state_dict by model_key = %s" % model_key)
    #         break
    #
    # if checkpoint_model is None:
    #     checkpoint_model = checkpoint
    #
    # # 改变预训练权重里键名
    # state_dict = model.state_dict()
    # # print(model)
    # print(state_dict.keys())
    # print(checkpoint_model.keys())
    # # print(checkpoint_model1.keys())
    # checkpoint_model = checkpoint_filter_fn1(checkpoint_model, model)
    # print(checkpoint_model.keys())
    #
    # import torch.nn.init as init
    # for k in state_dict.keys():
    #     if k in ['head.fc.weight', 'head.fc.bias']:
    #         if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
    #             print(f"Removing key {k} from pretrained checkpoint")
    #             del checkpoint_model[k]
    #             continue
    #     if k not in checkpoint_model or checkpoint_model[k].shape != state_dict[k].shape:
    #         if "norm" in k and "bias" not in k:
    #             init.constant_(state_dict[k], 1)
    #             print(k)
    #             continue
    #         if "weight" in k:
    #             if len(state_dict[k].shape) != 1:
    #                 init.kaiming_normal_(state_dict[k])  # init.kaiming_normal_ 方法使用 Kaiming 初始化方法对权重进行初始化，
    #                 # 该方法根据权重的形状和深度来计算标准差，并以标准差为参数进行正态分布初始化。
    #             print(k)
    #             continue
    #         if "bias" in k:
    #             init.constant_(state_dict[k], 0.1)  # init.constant_ 方法将指定权重的值初始化为常数
    #             print(k)
    #
    # model.load_state_dict(checkpoint_model, strict=False)
    # print(model.load_state_dict(checkpoint_model, strict=False))

    # model = load_two(model)
    model = load_one(model)

    # jiangzaomodel = jiangzao.jiangzaonet(channels=3)
    # model = jiangzao.zong(jiangzaomodel, model)
    # from model import PRID
    # jiangzaomodel = PRID.PRIDNet()
    # model = jiangzao.zong(jiangzaomodel, model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)
    # printparam(model)  # 参数量
    # model = load_model.LoadModel(config.class_num)
    # torchvision.models.resnet50(num_classes=config.class_num, pretrained=True) # 用官方模型代码，预训练权重也不需要额外下载加载
    # cuda计算和多cuda并行计算
    # if config.is_cuda:
    #     model = model.cuda()
    # if config.is_parallel:
    #     model = nn.DataParallel(model, device_ids=config.gpu_ids)
    #     cudnn.benchmark = True

    # 优化器和学习率调度器
    # optimizer = SGD(params=model.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-4)
    optimizer = Lion(params=model.parameters(), lr=0.0005, weight_decay=0.05)
    # optimizer = Lion(params=model.parameters(), lr=0.0005, weight_decay=1e-4)
    # pg = get_params_groups(model, weight_decay=config.wd)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    # optimizer = optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=0.05)
    # scheduler = create_lr_scheduler(optimizer, num_step=len(train_loader), epochs=config.max_epoch, warmup=True, warmup_epochs=5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.max_epoch, eta_min=1e-6)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)

    mixup_fn = None
    mixup_active = config.mixup > 0 or config.cutmix > 0. or config.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=config.mixup, cutmix_alpha=config.cutmix, cutmix_minmax=config.cutmix_minmax,
            prob=config.mixup_prob, switch_prob=config.mixup_switch_prob, mode=config.mixup_mode,
            label_smoothing=config.smoothing, num_classes=config.class_num)

    print("Use Cosine LR scheduler")
    lr_schedule_values = cosine_scheduler(
        config.lr, config.min_lr, config.max_epoch, num_training_steps_per_epoch,
        warmup_epochs=config.warmup_epochs, warmup_steps=config.warmup_steps,
    )
    weight_decay_end = config.weight_decay
    wd_schedule_values = cosine_scheduler(
        config.weight_decay, weight_decay_end, config.max_epoch, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # 损失函数和混淆矩阵
    loss_function = nn.CrossEntropyLoss()
    confusion_matrix = ConfusionMatrix(config.class_num)
    # loss_function = nn.CrossEntropyLoss(weight=weights).float()
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
            del (step_loss, inputs, outputs, labels)

            # optimizer.step()
            optimizer.zero_grad()
            # # update lr
            # scheduler.step()

        scheduler.step()
        # 训练情况
        train_loss = train_loss / len(train_loader)
        train_acc = confusion_matrix.acc()

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
            step_loss = loss_function(outputs, labels)
            valid_loss += step_loss.item()
            # torch.nn.functional.softmax(outputs, dim=-1)
            # acc = accuracy_score(outputs, labels)
            confusion_matrix.update(outputs.argmax(1).cpu().numpy(), labels.cpu().numpy())
            valid_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
                i,
                valid_loss / len(valid_loader),
                confusion_matrix.acc(),
            )
            del (step_loss, inputs, outputs, labels)

        # 验证情况
        valid_loss = valid_loss / len(valid_loader)
        valid_acc = confusion_matrix.acc()
        # 保存最好模型、最好准确率、最好混淆矩阵
        if best_acc < valid_acc:
            best_acc = valid_acc
            best_matrix = confusion_matrix.matrix
            save_model = {
                'epoch': i,
                'best_acc': best_acc,
                'best_matrix': best_matrix,
                'model': model,
                 'model_state_dict': model.state_dict()
            }
            torch.save(save_model, "weight/save/{}_lion.pth".format(config.net_name))
            print("{} epoch: {}, train_loss: {:.6f}, train_acc: {:.4f}, valid_loss: {:.6f}, valid_acc: {:.4f}"
                  .format(time.strftime("%Y-%m-%d %H:%M:%S"), i, train_loss, train_acc, valid_loss, valid_acc))
            print("{} epoch: {}, matrix:\n{}".format(time.strftime("%Y-%m-%d %H:%M:%S"), i, confusion_matrix.matrix))

        # print("{} epoch: {}, matrix:\n{}".format(time.strftime("%Y-%m-%d %H:%M:%S"), i, confusion_matrix.matrix))
        # 保存这一轮训练和验证结果
        with open("csv/{}lion.csv".format(config.net_name), "a", encoding="utf-8") as file:
            content = "{},{:.6f},{:.4f},{:.6f},{:.4f},{}\n" \
                .format(i, train_loss, train_acc, valid_loss, valid_acc, time.strftime("%Y-%m-%d %H:%M:%S"))
            file.write(content)

        writer.add_scalars("loss", {'train_loss': train_loss, 'valid_loss': valid_loss}, i)
        writer.add_scalars("acc", {'train_acc': train_acc, 'valid_acc': valid_acc}, i)
        writer.close()

    print("best_acc:{}".format(best_acc))
    print("best_matrix:\n{}".format(best_matrix))

if __name__ == "__main__":
    train()
