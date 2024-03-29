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
from utils.func import print

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


transform_train = Compose([
    # RandomBrightnessContrast(p = 0.8),
    # Resize(height=512, width=512),
    RandomResizedCrop(512, 512, (0.5, 1.0), p=0.5),
    ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, value=0.0,
                     p=0.8),
    # HorizontalFlip(p = 0.5),

    # ShiftScaleRotate(shift_limit = 0.2, scale_limit = 0.2, rotate_limit=20, p = 0.8),
    HorizontalFlip(p=0.5),
    RandomBrightnessContrast(p=0.8, contrast_limit=(-0.3, 0.2)),
    Lambda(image=sample_normalize),
    ToTensorV2(),
    Lambda(image=randomErase)

])

transform_val = Compose([
    Lambda(image=sample_normalize),
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
        # return (transform_train(image=read_image(f"{self.file_path}/{num}.png"))['image'],
        #         Tensor([row['male']])), row['zscore']
        return (transform_train(image=cv2.imread(f"{self.file_path}/{num}.png", cv2.IMREAD_COLOR))['image'],
                # Tensor([row['male']])), Tensor([row['boneage']]).to(torch.int64)
                Tensor([row['male']])), row['boneage']

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
        return (transform_val(image=cv2.imread(f"{self.file_path}/{int(row['id'])}.png", cv2.IMREAD_COLOR))['image'],
                Tensor([row['male']])), row['boneage']

    def __len__(self):
        return len(self.df)


def create_data_loader(train_df, val_df, train_root, val_root):
    return BAATrainDataset(train_df, train_root), BAAValDataset(val_df, val_root)


def L1_penalty(net, alpha):
    l1_penalty = torch.nn.L1Loss(size_average=False)
    loss = 0
    for param in net.MLP.parameters():
        loss += torch.sum(torch.abs(param))
    # for param2 in net.classifer.parameters():
    #     loss += torch.sum(torch.abs(param2))

    return alpha * loss


def L1_penalty_multi(net, alpha):
    l1_penalty = torch.nn.L1Loss(size_average=False)
    loss = 0
    for param in net.module.fc.parameters():
        loss += torch.sum(torch.abs(param))

    return alpha * loss


def train_fn(net, train_loader, target_loader, loss_fn, optimizer):
    '''
    checkpoint is a dict
    '''
    global total_size
    global training_loss

    net.train()
    data_length = len(train_loader) * 32
    record = torch.zeros((data_length, data_length), requires_grad=True).type(torch.FloatTensor).cuda()

    total_labels = torch.zeros((1), requires_grad=False).type(torch.LongTensor).cuda()
    total_genders = torch.zeros((1), requires_grad=False).type(torch.LongTensor).cuda()
    total_labels_con = torch.zeros((1), requires_grad=False).type(torch.LongTensor).cuda()
    total_genders_con = torch.zeros((1), requires_grad=False).type(torch.LongTensor).cuda()
    for batch_idx, data in enumerate(train_loader):
        image, gender = data[0]
        label = (data[1] - 1).type(torch.LongTensor).cuda()
        image, gender = image.type(torch.FloatTensor).cuda(), gender.type(torch.FloatTensor).cuda()

        batch_size = len(data[1])
        logits = net(image, gender)

        label = label.data.view(-1)
        total_labels = torch.cat((total_labels, label), dim=0)
        gender = gender.data.view(-1)
        total_genders = torch.cat((total_genders, gender), dim=0)
        row_ptr = batch_idx * batch_size
        total_size += batch_size

        for batch_idx_con, data_con in enumerate(target_loader):
            col_ptr = batch_idx_con * batch_size
            image_con, gender_con = data_con[0]
            label_con = (data_con[1] - 1).type(torch.LongTensor).cuda()
            image_con, gender_con = image_con.type(torch.FloatTensor).cuda(), gender_con.type(torch.FloatTensor).cuda()
            print(batch_idx_con)
            logits_con = net(image_con, gender_con)

            if batch_idx == 0:
                label_con = label_con.data.view(-1)
                total_labels_con = torch.cat((total_labels_con, label_con), dim=0)
                gender_con = gender_con.data.view(-1)
                total_genders_con = torch.cat((total_genders_con, gender_con), dim=0)

            batch_similarity = cos_similarity(logits, logits_con) / 2 # BxB
            # print(batch_similarity.shape)
            record[row_ptr:row_ptr + batch_size, col_ptr:col_ptr + batch_size] = batch_similarity.detach()
            del image_con
            del gender_con
            del label_con
            del logits_con
            del batch_similarity

    total_labels = total_labels[1:]
    total_genders = total_genders[1:]
    total_labels_con = total_labels_con[1:]
    total_genders_con = total_genders_con[1:]

    target = get_align_target(total_labels, total_genders, total_labels_con, total_genders_con)
    loss = loss_fn(record, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    training_loss = loss.item()

    return training_loss / total_size


def evaluate_fn(net, val_loader, loss_fn):
    net.eval()
    # net.train()

    feature = torch.zeros((1, 1024), requires_grad=False).type(torch.FloatTensor).cuda()
    total_labels = torch.zeros((1), requires_grad=False).type(torch.LongTensor).cuda()
    total_genders = torch.zeros((1), requires_grad=False).type(torch.LongTensor).cuda()
    global mae_loss
    global val_total_size
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            val_total_size += len(data[1])

            image, gender = data[0]
            label = (data[1] - 1).type(torch.LongTensor).cuda()
            image, gender = image.type(torch.FloatTensor).cuda(), gender.type(torch.FloatTensor).cuda()

            logits = net(image, gender)
            # label = label.squeeze()
            feature = torch.cat((feature, logits), dim=0)
            label = label.clone().view(-1)
            total_labels = torch.cat((total_labels, label), dim=0)
            gender = gender.clone().view(-1)
            total_genders = torch.cat((total_genders, gender), dim=0)

        feature = feature[1:]
        total_labels = total_labels[1:]
        total_genders = total_genders[1:]
        align_target = get_align_target(total_labels, total_genders, total_labels, total_genders)
        similarity = cos_similarity(feature, feature)
        loss_similarity = loss_fn(similarity, align_target) / 2
        mae_loss = loss_similarity.item()
    return mae_loss


import time
from model import Res50Align, get_My_resnet50


def map_fn(flags):
    model_name = f'Pretrained_align_modify'
    # Acquires the (unique) Cloud TPU core corresponding to this process's index
    # gpus = [0, 1]
    # torch.cuda.set_device('cuda:{}'.format(gpus[0]))

    mymodel = Res50Align(32, *get_My_resnet50(pretrained=True)).cuda()
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

    target_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=flags['batch_size'],
        shuffle=True,
        num_workers=flags['num_workers'],
        drop_last=False,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=flags['batch_size'],
        shuffle=False,
        num_workers=flags['num_workers'],
        drop_last=True,
        pin_memory=True
    )


    ## Network, optimizer, and loss function creation

    global best_loss
    best_loss = float('inf')
    # loss_fn = nn.CrossEntropyLoss(reduction='sum')
    loss_fn = nn.MSELoss(reduction='sum')
    # loss_fn_2 = nn.L1Loss(reduction='sum')
    lr = flags['lr']

    wd = 0

    optimizer = torch.optim.Adam(mymodel.parameters(), lr=lr, weight_decay=wd)
    #   optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay = wd)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    ## Trains
    for epoch in range(flags['num_epochs']):
        global training_loss
        training_loss = torch.tensor([0], dtype=torch.float32)
        global total_size
        total_size = torch.tensor([0], dtype=torch.float32)

        global mae_loss
        mae_loss = torch.tensor([0], dtype=torch.float32)
        global val_total_size
        val_total_size = torch.tensor([0], dtype=torch.float32)


        start_time = time.time()
        train_fn(mymodel, train_loader, target_loader, loss_fn, optimizer)

        ## Evaluation
        # Sets net to eval and no grad context
        evaluate_fn(mymodel, val_loader, loss_fn)

        train_loss, val_loss = training_loss / total_size, mae_loss / val_total_size
        if val_loss < best_loss:
            best_loss = val_loss
        print(
            f'training loss is {train_loss}, val loss is {val_loss}, time : {time.time() - start_time}, lr:{optimizer.param_groups[0]["lr"]}')
        scheduler.step()

    print(f'best loss: {best_loss}')
    torch.save(mymodel.state_dict(), '/'.join([save_path, f'{model_name}.bin']))
    # if use multi-gpu
    # torch.save(mymodel.module.state_dict(), '/'.join([save_path, f'{model_name}.bin']))


def delete_diag(batch):
    mat_ones = torch.ones((batch, batch), requires_grad=False)
    delete_diag_mat = mat_ones - torch.diag_embed(torch.diag(mat_ones))
    return delete_diag_mat


def get_align_target(labels, gender, target_labels, target_gender):
    row_idx = labels
    col_idx = target_labels
    labels_mat = torch.index_select(torch.index_select(dis, 0, row_idx), 1, col_idx)
    gender_row = F.one_hot(gender.type(torch.LongTensor), num_classes=2).squeeze().float().cuda()
    gender_col = F.one_hot(target_gender.type(torch.LongTensor), num_classes=2).squeeze().float().cuda()
    # labels_mat = torch.matmul(one_hot, one_hot.t())
    gender_mat = torch.matmul(gender_row, gender_col.t())

    return (labels_mat * gender_mat).float().detach()


def cos_similarity(logits, target):
    logit_nrom = F.normalize(logits)
    target_norm = F.normalize(target)
    similarity = torch.mm(logit_nrom, target_norm.t())
    return similarity


def relative_pos_dis():
    dis = torch.zeros((1, 230))
    for i in range(230):
        age_vector = torch.zeros((1, 230))
        if i < 2:
            j = i
            age_vector[0][i + 1] = 1
            age_vector[0][i + 2] = 1
            while j >= 0:
                age_vector[0][j] = 1
                j -= 1
            dis = torch.cat((dis, age_vector), dim=0)
            continue
        if i > 227:
            j = i
            age_vector[0][i - 1] = 1
            age_vector[0][i - 2] = 1
            while j < 230:
                age_vector[0][j] = 1
                j += 1
            dis = torch.cat((dis, age_vector), dim=0)
            continue
        age_vector[0][i-2] = 1
        age_vector[0][i-1] = 1
        age_vector[0][i] = 1
        age_vector[0][i+1] = 1
        age_vector[0][i+2] = 1
        dis = torch.cat((dis, age_vector), dim=0)
    return dis[1:]



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    save_path = '../../autodl-tmp/Pretrained_1_50epoch_alignAll'
    # os.makedirs(save_path, exist_ok=True)

    flags = {}
    flags['lr'] = 5e-4
    flags['batch_size'] = 2
    flags['num_workers'] = 4
    flags['num_epochs'] = 50
    flags['seed'] = 1

    # data_dir = '../../autodl-tmp/archive'
    data_dir = r'E:/code/archive/masked_1K_fold/fold_1'

    train_csv = os.path.join(data_dir, "train.csv")
    train_df = pd.read_csv(train_csv)
    valid_csv = os.path.join(data_dir, "valid.csv")
    valid_df = pd.read_csv(valid_csv)
    train_path = os.path.join(data_dir, "train")
    valid_path = os.path.join(data_dir, "valid")

    # train_ori_dir = '../../autodl-tmp/ori_4K_fold/'
    # train_ori_dir = '../archive/masked_1K_fold/'

    # delete_diag_mat = delete_diag(flags['batch_size']).cuda()
    dis = relative_pos_dis().detach().cuda()

    print(flags)
    print(f'{save_path} start')
    map_fn(flags)