import model
import os
import torch
from torch import nn
import random
import numpy as np

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

# 保存原始的print函数，以便稍后调用它。
rewrite_print = print


# 定义新的print函数。
def print(*arg):
    # 首先，调用原始的print函数将内容打印到控制台。
    rewrite_print(*arg)

    # 如果日志文件所在的目录不存在，则创建一个目录。
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开（或创建）日志文件并将内容写入其中。
    log_name = 'log.txt'
    filename = os.path.join(output_dir, log_name)
    rewrite_print(*arg, file=open(filename, "a"))


def eval_func(net, val_loader):
    # valid process
    net.eval()
    val_loss = 0.
    val_length = 0.
    loss_func = nn.L1Loss(reduction="sum")
    with torch.no_grad():
        for idx, patch in enumerate(val_loader):
            patch_len = patch[0].shape[0]
            images = patch[0].cuda()
            cannys = patch[1].cuda()
            boneage = patch[2].cuda()
            male = patch[3].cuda()
            output = net(images, cannys, male)

            # output = (output.cpu() * div) + mean
            # boneage = (boneage.cpu() * div) + mean

            output = torch.squeeze(output)
            boneage = torch.squeeze(boneage)

            assert output.shape == boneage.shape, "pred and output isn't the same shape"

            loss = loss_func(output, boneage)
            val_loss += loss.item()
            val_length += patch_len

    # print(f'valid sum loss is {val_loss}\nval_length: {val_length}')
    return val_loss / val_length

def eval_func_MMANet(net, val_loader):
    # valid process
    net.eval()
    val_loss = 0.
    val_length = 0.
    loss_func = nn.L1Loss(reduction="sum")
    with torch.no_grad():
        for idx, patch in enumerate(val_loader):
            patch_len = patch[0].shape[0]
            images = patch[0].type(torch.FloatTensor).cuda()
            boneage = patch[1].type(torch.FloatTensor).cuda()
            male = patch[2].type(torch.FloatTensor).cuda()
            _, _, _, output = net(images, male)

            # output = (output.cpu() * div) + mean
            # boneage = (boneage.cpu() * div) + mean

            output = torch.squeeze(output)
            boneage = torch.squeeze(boneage)

            assert output.shape == boneage.shape, "pred and output isn't the same shape"

            loss = loss_func(output, boneage)
            val_loss += loss.item()
            val_length += patch_len

    # print(f'valid sum loss is {val_loss}\nval_length: {val_length}')
    return val_loss / val_length

def eval_func_dist(net, val_loader, mean, div):
    # valid process
    net.eval()
    val_loss = 0.
    val_length = 0.
    loss_func = nn.L1Loss(reduction="sum")
    with torch.no_grad():
        for idx, patch in enumerate(val_loader):
            patch_len = patch[0].shape[0]
            images = patch[0].cuda()
            cannys = patch[1].cuda()
            boneage = patch[2].cuda()
            male = patch[3].cuda()
            output = net(images, male)

            # output = (output.cpu() * div) + mean
            # boneage = (boneage.cpu() * div) + mean

            output = torch.squeeze(output)
            boneage = torch.squeeze(boneage)

            assert output.shape == boneage.shape, "pred and output isn't the same shape"

            loss = loss_func(output, boneage)
            val_loss += loss.item()
            val_length += patch_len

    # print(f'valid sum loss is {val_loss}\nval_length: {val_length}')
    return val_loss / val_length


def normalize_age(df):
    boneage_mean = df['boneage'].mean()
    boneage_div = df['boneage'].std()
    df['zscore'] = df['boneage'].map(lambda x: (x - boneage_mean) / boneage_div)
    return df, boneage_mean, boneage_div


def L1_regular(net, alpha):
    loss = 0.
    for param in net.parameters():
        if param.requires_grad:
            loss += torch.sum(torch.abs(param))

    return alpha * loss


# def mat_age(train_set, )