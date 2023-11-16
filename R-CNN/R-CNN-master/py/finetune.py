# -*- coding: utf-8 -*-

"""
@date: 2020/3/1 上午9:54
@file: finetune.py
@author: zj
@description: 
"""

import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from utils.data.custom_finetune_dataset import CustomFinetuneDataset
from utils.data.custom_batch_sampler import CustomBatchSampler
from utils.util import check_dir


def load_data(data_root_dir):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),  # 随机水平翻转，做数据增强
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_loaders = {}
    data_sizes = {}
    for name in ['train', 'val']:
        data_dir = os.path.join(data_root_dir, name)  # ./data/finetune_car/train(val)
        data_set = CustomFinetuneDataset(data_dir, transform=transform) #微调数据集类的实例化对象
        data_sampler = CustomBatchSampler(data_set.get_positive_num(), data_set.get_negative_num(), 32, 96) #小批量采样器
        data_loader = DataLoader(data_set, batch_size=128, sampler=data_sampler, num_workers=8, drop_last=True)

        data_loaders[name] = data_loader #data_loaders = {'train':train的data_loader, 'val':val的data_loader}
        data_sizes[name] = data_sampler.__len__()   #data_sizes = {'train':train的data_size, 'val':val的data_size}

    return data_loaders, data_sizes


def train_model(data_loaders, model, criterion, optimizer, lr_scheduler, num_epochs=3, device=None):
    since = time.time()

    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print('111111')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            # num1 = 0
            # print(data_loaders['train'])
            for inputs, labels in data_loaders[phase]:
                # num1 += 1
                # print(num1)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) #inputs的维度(batch,C,H,W),outputs的维度是(batch,2),因为最后一层是Linear(4096,2)
                    _, preds = torch.max(outputs, 1)    # torch.max()
                    loss = criterion(outputs, labels)   # 计算损失

                    # backward + optimize only if in training phase
                    if phase == 'train':    # 如果是训练过程，就反向传播，更新参数
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                lr_scheduler.step()

            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects.double() / data_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_loaders, data_sizes = load_data('./data/finetune_car')

    model = models.alexnet(pretrained=True)
    # print(model)
    num_features = model.classifier[6].in_features #alexnet模型的classifier的第6层的in_features(第6层输入特征的大小)
    model.classifier[6] = nn.Linear(num_features, 2) #修改AlexNet的classifier的第6层
    # print(model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) #调整学习率，每7轮调整一次，即原来的学习率乘以0.1

    best_model = train_model(data_loaders, model, criterion, optimizer, lr_scheduler, device=device, num_epochs=3)
    # 保存最好的模型参数
    check_dir('./models')
    torch.save(best_model.state_dict(), 'models/alexnet_car.pth')
