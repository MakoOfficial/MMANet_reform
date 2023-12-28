import torch
import torch.nn as nn
from model import BAA_New, get_My_resnet50

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
            nn.Softmax()
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
