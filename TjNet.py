import torch
import torch.nn as nn
import numpy as np
import random
from model import BAA_New, get_My_resnet50
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
class TjNet(nn.Module):
    def __init__(self):
        super(TjNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=80, kernel_size=(1, 1)),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.Conv2d(in_channels=80, out_channels=192, kernel_size=(3, 3)),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=512, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, padding='same'),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=(3, 3), stride=(2, 2), padding=2),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=(5, 5), stride=1, padding='same'),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(8, 8))
        )

        self.genderFc = nn.Sequential(
            nn.Linear(in_features=1, out_features=64),
            nn.Dropout(0.5)
        )

        self.featureCombine = nn.Sequential(
            nn.Linear(in_features=2048 + 64, out_features=1024),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=512),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=230),
            # nn.Softmax()
        )

    def forward(self, x, gender):
        graphFeature = self.block5(self.block4(self.block3(self.block2(self.block1(x))))).squeeze()
        genderFeature = self.genderFc(gender)

        return self.featureCombine(torch.cat([graphFeature, genderFeature], dim=1))
        # return self.block5(self.block4(self.block3(self.block2(self.block1(x)))))

# data = torch.ones((10, 500, 500)).unsqueeze(1)
# print(data.shape)
# #
# model = TjNet()
# print(model(data, torch.ones(10, 1)).shape)
# # cnt = 0
#
# MMANet = BAA_New(32, *get_My_resnet50())
# #
# print(sum(p.numel() for p in MMANet.parameters() if p.requires_grad))
# print(sum(p.numel() for p in model.parameters() if p.requires_grad))
