import csv
import os
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
from torch import Tensor

from torch.utils.data import Dataset

import torch.nn.functional as F

import random

from torch.optim.lr_scheduler import StepLR

from albumentations.augmentations.transforms import Lambda, Normalize, RandomBrightnessContrast
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate, HorizontalFlip
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.crops.transforms import RandomResizedCrop
from albumentations import Compose, Resize

import warnings

import torchvision.transforms as transforms
from utils.func import LDL
rewrite_print = print


# 定义新的print函数。
def print(*arg):
    # 首先，调用原始的print函数将内容打印到控制台。
    rewrite_print(*arg)

    log_name = 'log.txt'
    filename = os.path.join(save_path, log_name)
    rewrite_print(*arg, file=open(filename, "a"))



warnings.filterwarnings("ignore")

seed = 1#seed必须是int，可以自行设置
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)#让显卡产生的随机数一致
torch.cuda.manual_seed_all(seed)#多卡模式下，让所有显卡生成的随机数一致？这个待验证
np.random.seed(seed)#numpy产生的随机数一致
random.seed(seed)

# CUDA中的一些运算，如对sparse的CUDA张量与dense的CUDA张量调用torch.bmm()，它通常使用不确定性算法。
# 为了避免这种情况，就要将这个flag设置为True，让它使用确定的实现。
torch.backends.cudnn.deterministic = True

# 设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
# 但是由于噪声和不同的硬件条件，即使是同一台机器，benchmark都可能会选择不同的算法。为了消除这个随机性，设置为 False
torch.backends.cudnn.benchmark = False


norm_mean = [0.143]  # 0.458971
norm_std = [0.144]  # 0.225609

RandomErasing = transforms.RandomErasing(scale=(0.02, 0.08), ratio=(0.5, 2), p=0.8)


def randomErase(image, **kwargs):
    return RandomErasing(image)


def sample_normalize(image, **kwargs):
    image = image / 255
    channel = image.shape[2]
    mean, std = image.reshape((-1, channel)).mean(axis=0), image.reshape((-1, channel)).std(axis=0)
    return (image - mean) / (std + 1e-3)

train_norm_mean = [0.14691083, 0.14691083, 0.14691083]  # 0.458971
train_norm_std = [0.27544162, 0.27544162, 0.27544162]  # 0.225609

valid_norm_mean = [0.14691083, 0.14691083, 0.14691083]
valid_norm_std = [0.27544162, 0.27544162, 0.27544162]


transform_train = Compose([
    # RandomBrightnessContrast(p = 0.8),
    # Resize(height=512, width=512),
    RandomResizedCrop(256, 256, (0.5, 1.0), p=0.5),
    ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, value=0.0,
                     p=0.8),
    # HorizontalFlip(p = 0.5),

    # ShiftScaleRotate(shift_limit = 0.2, scale_limit = 0.2, rotate_limit=20, p = 0.8),
    HorizontalFlip(p=0.5),
    RandomBrightnessContrast(p=0.8, contrast_limit=(-0.3, 0.2)),
    # Lambda(image=sample_normalize),
    Normalize(mean=train_norm_mean, std=train_norm_std),
    ToTensorV2(),
    Lambda(image=randomErase)

])

transform_val = Compose([
    # Lambda(image=sample_normalize),
    Normalize(mean=valid_norm_mean, std=valid_norm_std),
    ToTensorV2(),
])

transform_test = Compose([
    Lambda(image=sample_normalize),
    ToTensorV2(),
])



class BAATrainDataset(Dataset):
    def __init__(self, df, file_path):
        def preprocess_df(df):
            # nomalize boneage distribution
            # df['zscore'] = df['boneage'].map(lambda x: (x - boneage_mean) / boneage_div)
            # change the type of gender, change bool variable to float32
            df['male'] = df['male'].astype('float32')
            df['bonage'] = df['boneage'].astype('float32')
            return df

        self.df = preprocess_df(df)
        self.file_path = file_path

    def __getitem__(self, index):
        row = self.df.iloc[index]
        num = int(row['id'])
        return transform_train(image=cv2.imread(f"{self.file_path}/{num}.png", cv2.IMREAD_COLOR))['image'], row['boneage'], LDL(row[['boneage']])

    def __len__(self):
        return len(self.df)


class BAAValDataset(Dataset):
    def __init__(self, df, file_path):
        def preprocess_df(df):
            # change the type of gender, change bool variable to float32
            df['male'] = df['male'].astype('float32')
            df['bonage'] = df['boneage'].astype('float32')
            return df

        self.df = preprocess_df(df)
        self.file_path = file_path

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return transform_val(image=cv2.imread(f"{self.file_path}/{int(row['id'])}.png", cv2.IMREAD_COLOR))['image'], row['boneage']

    def __len__(self):
        return len(self.df)


def create_data_loader(train_df, val_df, train_root, val_root):
    return BAATrainDataset(train_df, train_root), BAAValDataset(val_df, val_root)


def L1_penalty(net, alpha):
    l1_penalty = torch.nn.L1Loss(size_average=False)
    loss = 0
    for param in net.MLP.parameters():
        loss += torch.sum(torch.abs(param))
    for param2 in net.classifier.parameters():
        loss += torch.sum(torch.abs(param2))

    return alpha * loss



def train_fn(net, train_loader, loss_fn, loss_KL, epoch, optimizer):
    '''
    checkpoint is a dict
    '''
    global total_size
    global training_loss
    global training_KL_loss
    ageTable = torch.arange(1, 229, requires_grad=False).type(torch.FloatTensor).cuda()
    net.train()
    for batch_idx, data in enumerate(train_loader):
        image = data[0]
        image = image.type(torch.FloatTensor).cuda()

        batch_size = len(data[1])
        # label = F.one_hot(data[1]-1, num_classes=230).float().cuda()
        label = data[1].cuda()
        # label_LDL = data[2].cuda()

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        y_pred = net(image)
        y_pred = F.softmax(y_pred, dim=1)
        age_pred = (y_pred*ageTable).sum(dim=-1).view(-1)
        label = label.view(-1)

        # KL
        # KLLoss = loss_KL(y_pred.log(), label_LDL)

        # print(y_pred)
        # print(y_pred, label)
        loss = loss_fn(age_pred, label)
        # backward,calculate gradients
        # total_loss = 0.5*KLLoss + 0.5*loss + L1_penalty(net, 1e-5)
        total_loss = loss + L1_penalty(net, 1e-5)
        total_loss.backward()
        # backward,update parameter
        optimizer.step()
        batch_loss = loss.item()
        # batch_KL_loss = KLLoss.item()

        training_loss += batch_loss
        # training_KL_loss += batch_KL_loss
        total_size += batch_size
    return training_loss / total_size


def evaluate_fn(net, val_loader):
    net.eval()

    global mae_loss
    global val_total_size
    ageTable = torch.arange(1, 229, requires_grad=False).type(torch.FloatTensor).cuda()
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            val_total_size += len(data[1])

            image = data[0]
            image = image.type(torch.FloatTensor).cuda()

            label = data[1].cuda()

            y_pred = net(image)
            y_pred = F.softmax(y_pred, dim=1)
            y_pred = (y_pred * ageTable).sum(dim=-1).view(-1)
            label = label.view(-1)

            batch_loss = F.l1_loss(y_pred, label, reduction='sum').item()
            # print(batch_loss/len(data[1]))
            mae_loss += batch_loss
    return mae_loss


import time
from model import baselineSingleGender, get_My_resnet50


def map_fn(flags):
    model_name = f'Res50_LDL_256_male'
    # Acquires the (unique) Cloud TPU core corresponding to this process's index
    # gpus = [0, 1]
    # torch.cuda.set_device('cuda:{}'.format(gpus[0]))

    mymodel = baselineSingleGender(*get_My_resnet50(pretrained=True)).cuda()
    #   mymodel.load_state_dict(torch.load('/content/drive/My Drive/BAA/resnet50_pr_2/best_resnet50_pr_2.bin'))
    # mymodel = nn.DataParallel(mymodel.cuda(), device_ids=gpus, output_device=gpus[0])

    train_set, val_set = create_data_loader(train_df, valid_df, train_path, valid_path)
    print(train_set.__len__())
    # Creates dataloaders, which load data in batches
    # Note: test loader is not shuffled or sampled
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=flags['batch_size'],
        shuffle=True,
        num_workers=flags['num_workers'],
        drop_last=True,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=flags['batch_size'],
        shuffle=False,
        num_workers=flags['num_workers'],
        pin_memory=True
    )

    ## Network, optimizer, and loss function creation

    global best_loss
    best_loss = float('inf')
    #   loss_fn =  nn.MSELoss(reduction = 'sum')
    loss_fn = nn.L1Loss(reduction='sum')
    loss_KL = nn.KLDivLoss(reduction='sum')
    # loss_fn = nn.BCELoss(reduction='sum')
    # loss_fn = nn.CrossEntropyLoss(reduction='sum')
    lr = flags['lr']

    wd = 0

    optimizer = torch.optim.Adam(mymodel.parameters(), lr=lr, weight_decay=wd)
    #   optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay = wd)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    ## Trains
    for epoch in range(flags['num_epochs']):
        global training_loss
        training_loss = torch.tensor([0], dtype=torch.float32)
        global training_KL_loss
        training_KL_loss = torch.tensor([0], dtype=torch.float32)
        global total_size
        total_size = torch.tensor([0], dtype=torch.float32)

        global mae_loss
        mae_loss = torch.tensor([0], dtype=torch.float32)
        global val_total_size
        val_total_size = torch.tensor([0], dtype=torch.float32)

        start_time = time.time()
        train_fn(mymodel, train_loader, loss_fn, loss_KL, epoch, optimizer)

        ## Evaluation
        # Sets net to eval and no grad context
        evaluate_fn(mymodel, val_loader)

        train_loss, train_KL_loss, val_mae = training_loss / total_size, training_KL_loss / total_size, mae_loss / val_total_size
        if val_mae < best_loss:
            best_loss = val_mae
            torch.save(mymodel.state_dict(), '/'.join([save_path, f'{model_name}.bin']))
        print(
            f'training loss is {train_loss}, training KL loss is {train_KL_loss}, val loss is {val_mae}, time : {time.time() - start_time}, lr:{optimizer.param_groups[0]["lr"]}')
        scheduler.step()

    print(f'best loss: {best_loss}')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    save_path = '../../autodl-tmp/256_res50_male_LDL'
    os.makedirs(save_path, exist_ok=True)

    flags = {}
    flags['lr'] = 5e-4
    flags['batch_size'] = 32
    flags['num_workers'] = 8
    flags['num_epochs'] = 100
    flags['seed'] = 1

    data_dir = '../../autodl-tmp/archiveMale/'
    # data_dir = r'E:/code/archive/masked_1K_fold/fold_1'

    train_csv = os.path.join(data_dir, "train.csv")
    train_df = pd.read_csv(train_csv)
    valid_csv = os.path.join(data_dir, "valid.csv")
    valid_df = pd.read_csv(valid_csv)
    train_path = os.path.join(data_dir, "train")
    valid_path = os.path.join(data_dir, "valid")

    boneage_mean = train_df['boneage'].mean()
    boneage_div = train_df['boneage'].std()

    # balanced_df = balance_data(data_dir, "train.csv", 10, 640)

    # train_ori_dir = '../../autodl-tmp/ori_4K_fold/'
    # train_ori_dir = '../archive/masked_1K_fold/'
    print(f'{save_path} start')
    map_fn(flags)