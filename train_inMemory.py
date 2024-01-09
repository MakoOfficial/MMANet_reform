import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
import random
import warnings
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import csv

warnings.filterwarnings("ignore")

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # 让显卡产生的随机数一致
torch.cuda.manual_seed_all(seed)  # 多卡模式下，让所有显卡生成的随机数一致？这个待验证
np.random.seed(seed)  # numpy产生的随机数一致
random.seed(seed)

# CUDA中的一些运算，如对sparse的CUDA张量与dense的CUDA张量调用torch.bmm()，它通常使用不确定性算法。
# 为了避免这种情况，就要将这个flag设置为True，让它使用确定的实现。
torch.backends.cudnn.deterministic = True

# 设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
# 但是由于噪声和不同的硬件条件，即使是同一台机器，benchmark都可能会选择不同的算法。为了消除这个随机性，设置为 False
torch.backends.cudnn.benchmark = False

norm_mean = [0.143]  # 0.458971
norm_std = [0.144]  # 0.225609

# RandomErasing = transforms.RandomErasing(scale=(0.02, 0.08), ratio=(0.5, 2), p=0.8)
#
#
# def randomErase(image, **kwargs):
#     return RandomErasing(image)


# def sample_normalize(image, **kwargs):
#     image = image / 255
#     channel = image.shape[2]
#     mean, std = image.reshape((-1, channel)).mean(axis=0), image.reshape((-1, channel)).std(axis=0)
#     return (image - mean) / (std + 1e-3)


transform_train = transforms.Compose([
    transforms.RandomResizedCrop(512, (0.5, 1.0)),
    transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2),
                            interpolation=transforms.InterpolationMode.NEAREST),
    transforms.RandomHorizontalFlip(p=0.5),
])

transform_init = transforms.ToTensor()


def read_a_img(img_path):
    # input channels is 3
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    return transform_init(img)


def get_label(df, filename):
    # print(f"image_id: {image_index}")
    image_id = filename.split('.')[0]
    row = df[df['id'] == int(image_id)]
    boneage = np.array(row['boneage'])
    male = np.array(row['male'].astype('float32'))
    return torch.Tensor(boneage), torch.Tensor(male)


def L1_penalty(net, alpha):
    l1_penalty = torch.nn.L1Loss(size_average=False)
    loss = 0
    for param in net.MLP.parameters():
        loss += torch.sum(torch.abs(param))

    return alpha * loss


def train_fn(net, train_loader, loss_fn, optimizer):
    '''
        checkpoint is a dict
        '''
    global total_size
    global training_loss

    net.train()
    for batch_idx, data in enumerate(train_loader):
        image, gender, label = data
        image = transform_train(image)
        image, gender = image.type(torch.FloatTensor).cuda(), gender.type(torch.FloatTensor).cuda()
        label = (label - 1).type(torch.LongTensor).cuda()
        batch_size = len(label.shape[0])

        optimizer.zero_grad()
        # forward
        y_pred = net(image, gender)
        y_pred = y_pred.squeeze()
        label = label.squeeze()
        # print(y_pred, label)
        loss = loss_fn(y_pred, label)
        # backward,calculate gradients
        total_loss = loss + L1_penalty(net, 1e-5)
        total_loss.backward()
        # backward,update parameter
        optimizer.step()
        batch_loss = loss.item()

        training_loss += batch_loss
        total_size += batch_size
    return training_loss / total_size


def evaluate_fn(net, val_loader):
    net.eval()

    global mae_loss
    global val_total_size

    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            image, gender, label = data
            image, gender = image.type(torch.FloatTensor).cuda(), gender.type(torch.FloatTensor).cuda()

            label = label.cuda()

            y_pred = net(image, gender)
            # y_pred = net(image, gender)
            y_pred = torch.argmax(y_pred, dim=1) + 1

            y_pred = y_pred.squeeze()
            label = label.squeeze()

            batch_loss = F.l1_loss(y_pred, label, reduction='sum').item()
            # print(batch_loss/len(data[1]))
            mae_loss += batch_loss
    return mae_loss


import time
from model import get_My_resnet50, ResNet

def map_fn(flags):
    mymodel = ResNet(32, *get_My_resnet50()).cuda()

    print(trainDataset.__len__())
    train_loader = torch.utils.data.DataLoader(
        trainDataset,
        batch_size=flags['batch_size'],
        shuffle=True,
        num_workers=flags['num_workers'],
        drop_last=True,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        validDataset,
        batch_size=flags['batch_size'],
        shuffle=False,
        num_workers=flags['num_workers'],
        pin_memory=True
    )

    ## Network, optimizer, and loss function creation
    global best_loss
    best_loss = float('inf')
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    lr = flags['lr']

    wd = 0

    optimizer = torch.optim.Adam(mymodel.parameters(), lr=lr, weight_decay=wd)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    ## Trains
    start_time = time.time()
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
        train_fn(mymodel, train_loader, loss_fn, optimizer)

        evaluate_fn(mymodel, val_loader)
        train_loss, val_mae = training_loss / total_size, mae_loss / val_total_size
        if val_mae < best_loss:
            best_loss = val_mae
        print(
            f'training loss is {train_loss}, val loss is {val_mae}, time : {time.time() - start_time}, lr:{optimizer.param_groups[0]["lr"]}')
        scheduler.step()

    print(f'best loss: {best_loss}')

    model_name = f'Res50_All'
    torch.save(mymodel.state_dict(), '/'.join([save_path, f'{model_name}.bin']))

    # save log
    with torch.no_grad():
        train_record = [['label', 'pred']]
        train_record_path = os.path.join(save_path, f"train.csv")
        train_length = 0.
        total_loss = 0.
        mymodel.eval()
        for idx, data in enumerate(train_loader):
            image, gender = data[0]
            image, gender = image.type(torch.FloatTensor).cuda(), gender.type(torch.FloatTensor).cuda()

            batch_size = len(data[1])
            label = data[1].cuda()

            y_pred = mymodel(image, gender)

            output = torch.argmax(y_pred, dim=1) + 1

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
        val_record_path = os.path.join(save_path, f"val.csv")
        val_length = 0.
        val_loss = 0.
        mymodel.eval()
        for idx, data in enumerate(val_loader):
            image, gender = data[0]
            image, gender = image.type(torch.FloatTensor).cuda(), gender.type(torch.FloatTensor).cuda()

            batch_size = len(data[1])
            label = data[1].cuda()

            y_pred = mymodel(image, gender)

            output = torch.argmax(y_pred, dim=1) + 1

            output = torch.squeeze(output)
            label = torch.squeeze(label)
            for i in range(output.shape[0]):
                val_record.append([label[i].item(), round(output[i].item(), 2)])
            assert output.shape == label.shape, "pred and output isn't the same shape"

            val_loss += F.l1_loss(output, label, reduction='sum').item()
            val_length += batch_size
        print(f"valid dataset length :{val_length}")
        print(f'final val loss: {round(val_loss / val_length, 3)}')
        with open(val_record_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in val_record:
                writer.writerow(row)


def getDataset(data_path, df_path):
    imgList = os.listdir(data_path)
    df = pd.read_csv(df_path)

    List = []
    ageList = []
    maleList = []
    for filename in imgList:
        List.append(read_a_img(img_path=os.path.join(data_path, filename)))
        age, male = get_label(df, filename)
        ageList.append(age)
        maleList.append(male)

    TensorSet = torch.stack(List)
    TensorAge = torch.stack(ageList)
    TensorMale = torch.stack(maleList)

    return TensorDataset(TensorSet, TensorMale, TensorAge)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('lr', type=float)
    parser.add_argument('batch_size', type=int)
    parser.add_argument('num_epochs', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    save_path = '../../autodl-tmp/Res50_All'
    os.makedirs(save_path, exist_ok=True)

    flags = {}
    flags['lr'] = args.lr
    flags['batch_size'] = args.batch_size
    flags['num_workers'] = 8
    flags['num_epochs'] = args.num_epochs
    flags['seed'] = 1

    data_dir = '../../archive/'

    train_csv = os.path.join(data_dir, "train.csv")
    valid_csv = os.path.join(data_dir, "valid.csv")
    train_path = os.path.join(data_dir, "train")
    valid_path = os.path.join(data_dir, "valid")
    time1 = time.time()
    trainDataset = getDataset(train_path, train_csv)
    validDataset = getDataset(valid_path, valid_csv)

    print(f'dataset is prepared!, cost time: {time.time() - time1}')
    print("training start!")
    map_fn(flags)
