

import os
import sys
import json
import pickle
import random
import math

import pandas as pd
import numpy as np

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()

    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        train_loss=accu_loss.item() / (step + 1)
        train_acc=accu_num.item() / sample_num

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            train_loss,
            train_acc,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()

        log_train = {}
        log_train['epoch'] = epoch
        log_train['train_loss'] = train_loss
        log_train['train_accuracy'] = train_acc

    return log_train


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):

    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()

    # accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    # accu_loss = torch.zeros(1).to(device)  # 累计损失

    loss_list = []
    labels_list = []
    preds_list = []

    sample_num = 0
    # accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    # accu_loss = torch.zeros(1).to(device)  # 累计损失

    # sample_num = 0
    # data_loader = tqdm(data_loader, file=sys.stdout)
    data_loader = tqdm(data_loader, file=sys.stdout)

    # for step, data in enumerate(data_loader):
    #     images, labels = data
        # sample_num += images.shape[0]

    for images, labels in data_loader:  # 生成一个 batch 的数据和标注
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]  # 获得当前 batch 所有图像的预测类别
        pred_classes = pred_classes.cpu().numpy()
        # accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        # accu_loss += loss
        loss = loss.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        loss_list.append(loss)
        labels_list.extend(labels)
        preds_list.extend(pred_classes)

        val_loss = np.mean(loss)
        val_acc = accuracy_score(labels_list, preds_list)

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            val_loss,
            val_acc)


    # val_loss = np.mean(loss)
    # val_acc= accuracy_score(labels_list, preds_list)
    #
    # data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
    #     epoch,
    #     val_loss,
    #     val_acc)


    log_val = {}
    log_val['epoch'] = epoch
    # 计算分类评估指标
    log_val['val_loss'] = np.mean(loss)
    log_val['val_acc'] = accuracy_score(labels_list, preds_list)
    log_val['val_precision'] = precision_score(labels_list, preds_list, average='macro')
    log_val['val_recall'] = recall_score(labels_list, preds_list, average='macro')
    log_val['val_f1-score'] = f1_score(labels_list, preds_list, average='macro')
    # sample_num += images.shape[0]

    # pred = model(images.to(device))
    # pred_classes = torch.max(pred, dim=1)[1]
    # accu_num += torch.eq(pred_classes, labels.to(device)).sum()

    # loss = loss_function(pred, labels.to(device))
    # accu_loss += loss

    # data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
    #         epoch,
    #         log_val['val_loss'],
    #         log_val['val_acc'])

    return log_val


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num