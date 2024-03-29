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
    for param2 in net.classifer.parameters():
        loss += torch.sum(torch.abs(param2))

    return alpha * loss


def L1_penalty_multi(net, alpha):
    l1_penalty = torch.nn.L1Loss(size_average=False)
    loss = 0
    for param in net.module.fc.parameters():
        loss += torch.sum(torch.abs(param))

    return alpha * loss


def train_fn(net, train_loader, loss_fn, loss_fn_2, optimizer, queue, queue_dataLen):
    '''
    checkpoint is a dict
    '''
    global total_size
    global training_loss

    net.train()
    use_the_queue = False
    for batch_idx, data in enumerate(train_loader):
        image, gender = data[0]
        label = (data[1] - 1).type(torch.LongTensor).cuda()
        image, gender = image.type(torch.FloatTensor).cuda(), gender.type(torch.FloatTensor).cuda()

        batch_size = len(data[1])
        # label = label.type(torch.LongTensor).cuda()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        y_pred, logits = net(image, gender)
        y_pred = y_pred.squeeze()
        label = label.squeeze()

        # time to use queue
        if queue is not None:
            if use_the_queue or not torch.all(queue[0][-1, :] == 0):
                use_the_queue = True
            # fill the queue
            queue[0][batch_size:] = queue[0][:-batch_size].clone()
            temp = logits.clone().detach()
            queue[0][:batch_size] = temp / torch.norm(temp, dim=1, keepdim=True)
            queue[1][batch_size:] = queue[1][:-batch_size].clone()
            queue[1][:batch_size] = gender.squeeze().clone().detach()
            queue[2][batch_size:] = queue[2][:-batch_size].clone()
            queue[2][:batch_size] = label.clone().detach()
            queue_dataLen += batch_size

        loss = loss_fn(y_pred, label)

        align_target = get_align_target(label, gender, queue, queue_dataLen)
        similarity = cos_similarity(logits, queue, queue_dataLen)
        loss_similarity = loss_fn_2(similarity, align_target)

        # backward,calculate gradients
        total_loss = loss + L1_penalty(net, 1e-5) + 0.5 * loss_similarity
        total_loss.backward()
        # backward,update parameter
        optimizer.step()
        batch_loss = loss.item()

        training_loss += batch_loss
        total_size += batch_size
    return queue, queue_dataLen


def evaluate_fn(net, val_loader):
    net.eval()

    global mae_loss
    global val_total_size
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            val_total_size += len(data[1])

            image, gender = data[0]
            image, gender = image.type(torch.FloatTensor).cuda(), gender.type(torch.FloatTensor).cuda()

            label = data[1].cuda()

            y_pred, _ = net(image, gender)
            # y_pred = net(image, gender)
            y_pred = torch.argmax(y_pred, dim=1)+1

            y_pred = y_pred.squeeze()
            label = label.squeeze()

            batch_loss = F.l1_loss(y_pred, label, reduction='sum').item()
            # print(batch_loss/len(data[1]))
            mae_loss += batch_loss
    return mae_loss


import time
from model import Res50Align, get_My_resnet50


def map_fn(flags):
    model_name = f'Res50_All_align'
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

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=flags['batch_size'],
        shuffle=False,
        num_workers=flags['num_workers'],
        pin_memory=True
    )

    # build the queue
    queue = None
    queue_dataLen = None
    # the queue needs to be divisible by the batch size
    flags['queue_length'] -= flags['queue_length'] % flags['batch_size']

    ## Network, optimizer, and loss function creation

    global best_loss
    best_loss = float('inf')
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    loss_fn_2 = nn.MSELoss(reduction='sum')
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

        # optionally starts a queue
        if flags['queue_length'] > 0 and epoch+1 >= args.epoch_queue_starts and queue is None:
            queue_logits = torch.zeros(
                flags['queue_length'],
                512,
            ).cuda()
            queue_labels = torch.zeros(flags['queue_length']).cuda()
            queue_gender = torch.zeros(flags['queue_length']).cuda()
            # 0->logits, 1->gender, 2->labels
            queue = (queue_logits, queue_gender, queue_labels)
            queue_dataLen = 0


        start_time = time.time()
        queue, queue_dataLen = train_fn(mymodel, train_loader, loss_fn, loss_fn_2, optimizer, queue, queue_dataLen)

        ## Evaluation
        # Sets net to eval and no grad context
        evaluate_fn(mymodel, val_loader)

        train_loss, val_mae = training_loss / total_size, mae_loss / val_total_size
        if val_mae < best_loss:
            best_loss = val_mae
        print(
            f'training loss is {train_loss}, val loss is {val_mae}, time : {time.time() - start_time}, lr:{optimizer.param_groups[0]["lr"]}')
        scheduler.step()

    print(f'best loss: {best_loss}')
    torch.save(mymodel.state_dict(), '/'.join([save_path, f'{model_name}.bin']))
    # if use multi-gpu
    # torch.save(mymodel.module.state_dict(), '/'.join([save_path, f'{model_name}.bin']))

    train_test_dataset = BAAValDataset(train_df, train_path)
    train_test_dataloader = torch.utils.data.DataLoader(
        train_test_dataset,
        batch_size=flags['batch_size'],
        shuffle=False,
        num_workers=flags['num_workers'],
        pin_memory=True
    )


    # save log
    with torch.no_grad():
        train_record = [['label', 'pred']]
        train_record_path = os.path.join(save_path, f"train_result.csv")
        train_length = 0.
        total_loss = 0.
        mymodel.eval()
        for idx, data in enumerate(train_test_dataloader):
            image, gender = data[0]
            image, gender = image.type(torch.FloatTensor).cuda(), gender.type(torch.FloatTensor).cuda()

            batch_size = len(data[1])
            label = data[1].cuda()

            y_pred, _ = mymodel(image, gender)

            output = torch.argmax(y_pred, dim=1)+1

            output = torch.squeeze(output)
            label = torch.squeeze(label)
            for i in range(output.shape[0]):
                train_record.append([label[i].item(), round(output[i].item(), 2)])
            assert output.shape == label.shape, "pred and output isn't the same shape"

            total_loss += F.l1_loss(output, label, reduction='sum').item()
            train_length += batch_size
        print(f"training dataset length :{train_length}")
        print(f'final training loss: {round(total_loss / train_length, 3)}')
        with open(train_record_path, 'w', newline='') as csvfile:
            writer_train = csv.writer(csvfile)
            for row in train_record:
                writer_train.writerow(row)

    with torch.no_grad():
        val_record = [['label', 'pred']]
        val_record_path = os.path.join(save_path, f"val_result.csv")
        val_length = 0.
        val_loss = 0.
        mymodel.eval()
        for idx, data in enumerate(val_loader):
            image, gender = data[0]
            image, gender = image.type(torch.FloatTensor).cuda(), gender.type(torch.FloatTensor).cuda()

            batch_size = len(data[1])
            label = data[1].cuda()

            y_pred, _ = mymodel(image, gender)

            output = torch.argmax(y_pred, dim=1)+1
            if output.shape[0] != 1:
                output = torch.squeeze(output)
                label = torch.squeeze(label)
            for i in range(output.shape[0]):
                val_record.append([label[i].item(), round(output[i].item(), 2)])
            # assert output.shape == label.shape, "pred and output isn't the same shape"

            val_loss += F.l1_loss(output, label, reduction='sum').item()
            val_length += batch_size
        print(f"valid dataset length :{val_length}")
        print(f'final val loss: {round(val_loss / val_length, 3)}')
        with open(val_record_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in val_record:
                writer.writerow(row)


def delete_diag(batch):
    mat_ones = torch.ones((batch, batch), requires_grad=False)
    delete_diag_mat = mat_ones - torch.diag_embed(torch.diag(mat_ones))
    return delete_diag_mat


def get_align_target(labels, gender, queue, queue_dataLen):
    idx = labels
    gender = gender.squeeze()
    if queue is not None:
        labels_mat = torch.index_select(torch.index_select(dis, 0, idx), 1, queue[2][:queue_dataLen].type(idx.dtype))
        one_hot_gender = F.one_hot(gender.type(torch.LongTensor), num_classes=2).squeeze().float().cuda()
        one_hot_gender_q = F.one_hot(queue[1][:queue_dataLen].type(torch.LongTensor), num_classes=2).squeeze().float().cuda()
        # labels_mat = torch.matmul(one_hot, one_hot.t())
        gender_mat = torch.matmul(one_hot_gender, one_hot_gender_q.t())
        return labels_mat * gender_mat

    labels_mat = torch.index_select(torch.index_select(dis, 0, idx), 1, idx)
    one_hot_gender = F.one_hot(gender.type(torch.LongTensor), num_classes=2).squeeze().float().cuda()
    # labels_mat = torch.matmul(one_hot, one_hot.t())
    gender_mat = torch.matmul(one_hot_gender, one_hot_gender.t())

    return labels_mat * gender_mat * delete_diag_mat


def cos_similarity(logits, queue, queue_dataLen):
    if queue is not None:
        logit_nrom = logits / torch.norm(logits, dim=1, keepdim=True)
        # similarity = torch.matmul(logit_nrom, logit_nrom.t())
        similarity = torch.matmul(logit_nrom, queue[0][:queue_dataLen].t())
        return similarity
    logit_nrom = logits / torch.norm(logits, dim=1, keepdim=True)
    similarity = torch.matmul(logit_nrom, logit_nrom.t())
    # similarity = torch.matmul(logit_nrom, queue[0].t())
    return similarity * delete_diag_mat


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
    parser.add_argument("--epoch_queue_starts", type=int, default=0,
                        help="from this epoch, we start using a queue")
    args = parser.parse_args()
    save_path = '../../autodl-tmp/Res50_AllPre_1_100epoch_align2_addQueue'
    os.makedirs(save_path, exist_ok=True)

    flags = {}
    flags['lr'] = 5e-4
    flags['batch_size'] = 32
    flags['num_workers'] = 8
    flags['num_epochs'] = 100
    flags['seed'] = 1
    flags['queue_length'] = 480

    data_dir = '../../autodl-tmp/archive'
    # data_dir = r'E:/code/archive/masked_1K_fold/fold_1'

    train_csv = os.path.join(data_dir, "train.csv")
    train_df = pd.read_csv(train_csv)
    valid_csv = os.path.join(data_dir, "valid.csv")
    valid_df = pd.read_csv(valid_csv)
    train_path = os.path.join(data_dir, "train")
    valid_path = os.path.join(data_dir, "valid")

    # train_ori_dir = '../../autodl-tmp/ori_4K_fold/'
    # train_ori_dir = '../archive/masked_1K_fold/'

    delete_diag_mat = delete_diag(flags['batch_size']).cuda()
    dis = relative_pos_dis().cuda()

    print(flags)
    print(f'{save_path} start')
    map_fn(flags)