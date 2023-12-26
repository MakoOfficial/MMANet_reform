import numpy as np 
import pandas as pd 
import os, sys, random
import numpy as np
import pandas as pd
import cv2
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch import Tensor

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import _utils

from random import choice

from skimage import io
from PIL import Image, ImageOps

import glob

#from torchsummary import summary
import logging

import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.models as models
# from tqdm.notebook import tqdm
from tqdm import tqdm
from sklearn.utils import shuffle
# from apex import amp

import random

import time

from torch.optim.lr_scheduler import StepLR
from torch.nn.parameter import Parameter

from albumentations.augmentations.transforms import Lambda, Normalize, RandomBrightnessContrast
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate, HorizontalFlip
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.crops.transforms import RandomResizedCrop
from albumentations import Compose, OneOrOther

import albumentations

import warnings


import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import time
from utils.func import print

warnings.filterwarnings("ignore")


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


norm_mean = [0.143] #0.458971
norm_std = [0.144] #0.225609

RandomErasing = transforms.RandomErasing(scale=(0.02, 0.08), ratio = (0.5, 2), p = 0.8)

def randomErase(image, **kwargs):
    return RandomErasing(image)

def sample_normalize(image, **kwargs):
    image = image/255
    channel = image.shape[2]
    mean, std = image.reshape((-1, channel)).mean(axis = 0), image.reshape((-1, channel)).std(axis = 0)
    return (image-mean)/(std + 1e-3)

transform_train = Compose([
    # RandomBrightnessContrast(p = 0.8),
    RandomResizedCrop(512, 512, (0.5, 1.0), p = 0.5),
    ShiftScaleRotate(shift_limit = 0.2, scale_limit = 0.2, rotate_limit=20, border_mode = cv2.BORDER_CONSTANT, value = 0.0, p = 0.8),
    # HorizontalFlip(p = 0.5),
    
    # ShiftScaleRotate(shift_limit = 0.2, scale_limit = 0.2, rotate_limit=20, p = 0.8),
    HorizontalFlip(p = 0.5),
    RandomBrightnessContrast(p = 0.8, contrast_limit=(-0.3, 0.2)),                             
    Lambda(image = sample_normalize),
    ToTensorV2(),
    Lambda(image = randomErase) 
    
])

transform_val = Compose([                                   
    Lambda(image = sample_normalize),
    ToTensorV2(),
])

transform_test = Compose([                                   
    Lambda(image = sample_normalize),
    ToTensorV2(),
])


def L1_penalty(net, alpha):
    l1_penalty = torch.nn.L1Loss(size_average = False)
    loss = 0
    for param in net.module.fc.parameters():
        loss += torch.sum(torch.abs(param))

    return alpha*loss


def train_fn(net, train_loader, loss_fn, epoch, optimizer):
    '''
    checkpoint is a dict
    '''
    global total_size 
    global training_loss 

    net.train()
    for batch_idx, data in enumerate(train_loader):
        image, gender = data[0]
        image, gender= image.cuda(), gender.cuda()

        batch_size = len(data[1])
        label = data[1].cuda()
        
        # zero the parameter gradients
        optimizer.zero_grad()
        #forward
        _, _, _, y_pred = net(image, gender)
        y_pred = y_pred.squeeze()
        label = label.squeeze()

        # print(y_pred, label)
        loss = loss_fn(y_pred, label)
        #backward,calculate gradients
        total_loss = loss + L1_penalty(net, 1e-5)
        total_loss.backward()
        #backward,update parameter
        optimizer.step()
        batch_loss = loss.item()

        training_loss += batch_loss
        total_size += batch_size
    return training_loss/total_size 

def evaluate_fn(net, val_loader):
    net.eval()
    
    global mae_loss 
    global val_total_size 
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            val_total_size += len(data[1])

            image, gender = data[0]
            image, gender= image.cuda(), gender.cuda()

            label = data[1].cuda()

            _, _, _, y_pred = net(image, gender)
            # y_pred = net(image, gender)
            y_pred = (y_pred.cpu() * boneage_div) + boneage_mean
            label = (label.cpu() * boneage_div) + boneage_mean

            y_pred = y_pred.squeeze()
            label = label.squeeze()

            batch_loss = F.l1_loss(y_pred, label, reduction='sum').item()
            # print(batch_loss/len(data[1]))
            mae_loss += batch_loss
    return mae_loss


def reduce_fn(vals):
    return sum(vals)
import time



def map_fn(flags, train_set_k, val_set_k, k):
  root = './output'
  model_name = 'rsa50'
  path = f'{root}/{model_name}_fold{k}'
  # Sets a common random seed - both for initialization and ensuring graph is the same
  seed_everything(seed=flags['seed'])

  # Acquires the (unique) Cloud TPU core corresponding to this process's index
  gpus = [0, 1]
  torch.cuda.set_device('cuda:{}'.format(gpus[0]))


#   mymodel = BAA_base(32)
  mymodel = BAA_New(32, *get_My_resnet50())
#   mymodel.load_state_dict(torch.load('/content/drive/My Drive/BAA/resnet50_pr_2/best_resnet50_pr_2.bin'))
  mymodel = nn.DataParallel(mymodel.cuda(), device_ids=gpus, output_device=gpus[0])

  
  # Creates dataloaders, which load data in batches
  # Note: test loader is not shuffled or sampled
  train_loader = torch.utils.data.DataLoader(
      train_set_k,
      batch_size=flags['batch_size'],
      shuffle=True,
      num_workers=flags['num_workers'],
      drop_last=True)

  val_loader = torch.utils.data.DataLoader(
      val_set_k,
      batch_size=flags['batch_size'],
      shuffle=False,
      num_workers=flags['num_workers'])


  ## Network, optimizer, and loss function creation

  # Creates AlexNet for 10 classes
  # Note: each process has its own identical copy of the model
  #  Even though each model is created independently, they're also
  #  created in the same way.

  global best_loss 
  best_loss = float('inf')
#   loss_fn =  nn.MSELoss(reduction = 'sum')
  loss_fn = nn.L1Loss(reduction = 'sum')
  lr = flags['lr']

  wd = 0
    
  optimizer = torch.optim.Adam(mymodel.parameters(), lr=lr, weight_decay=wd)
#   optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay = wd)
  scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

  ## Trains
  train_start = time.time()
  for epoch in range(flags['num_epochs']):
    global training_loss 
    training_loss = torch.tensor([0], dtype = torch.float32)
    global total_size 
    total_size = torch.tensor([0], dtype = torch.float32)

    global mae_loss 
    mae_loss = torch.tensor([0], dtype = torch.float32)
    global val_total_size 
    val_total_size = torch.tensor([0], dtype = torch.float32)

    global test_mae_loss 
    test_mae_loss = torch.tensor([0], dtype = torch.float32)
    global test_total_size 
    test_total_size = torch.tensor([0], dtype = torch.float32)

    start_time = time.time()
    train_fn(mymodel, train_loader, loss_fn, epoch, optimizer)
    
    ## Evaluation
    # Sets net to eval and no grad context
    evaluate_fn(mymodel, val_loader)

    scheduler.step()

    print(test_total_size)
    train_loss, val_mae, test_mae = training_loss/total_size, mae_loss/val_total_size, test_mae_loss/test_total_size
    print(f'training loss is {train_loss}, val loss is {val_mae}, test loss is {test_mae}, time : {time.time() - start_time}, lr:{optimizer.param_groups[0]["lr"]}')
  torch.save(mymodel.module.state_dict(), '/'.join([path, f'{model_name}.bin']))



if __name__ == "__main__":
    from model import BAA_New, get_My_resnet50
    from utils import datasets, func
    from sklearn.model_selection import KFold
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type')
    parser.add_argument('lr', type=float)
    parser.add_argument('batch_size', type = int)
    parser.add_argument('num_epochs', type = int)
    parser.add_argument('seed', type = int)
    args = parser.parse_args()

    model = BAA_New(32, *get_My_resnet50())

    flags = {}
    flags['lr'] = args.lr
    flags['batch_size'] = args.batch_size
    flags['num_workers'] = 2
    flags['num_epochs'] = args.num_epochs
    flags['seed'] = args.seed

    train_df = pd.read_csv(f'../archive/boneage-training-dataset.csv')
    train_df, boneage_mean, boneage_div = func.normalize_age(train_df)
    train_ori_dir = '../../autodl-tmp/ori'
    train_dataset = datasets.MMANetDataset(df=train_df, data_dir=train_ori_dir)
    print(f'Training dataset info:\n{train_dataset}')
    data_len = train_dataset.__len__()
    X = torch.randn(data_len, 2)
    kf = KFold(n_splits=5, shuffle=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=X)):
        print(f"Fold {fold+1}/5")
        ids, zscore, male = train_dataset[train_idx]
        train_set = datasets.Kfold_MMANet_Dataset(ids, zscore, male, train_ori_dir, transforms=transform_train)
        ids1, zscore1, male1 = train_dataset[val_idx]
        val_set = datasets.Kfold_MMANet_Dataset(ids1, zscore1, male1, train_ori_dir, transforms=transform_val)
        torch.set_default_tensor_type('torch.FloatTensor')
        map_fn(flags, train_set_k=train_set, val_set_k=val_set, k=fold+1)

