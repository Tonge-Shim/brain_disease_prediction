import torch
import torch.nn as nn
from model.units import Conv2d, Reduction_A, Conv3d


class Stem(nn.Module):
    def __init__(self, in_channels):
        super(Stem, self).__init__()
        self.features = nn.Sequential(
            Conv3d(in_channels, 32, 3, stride=2, padding=0, bias=False), # 149 x 149 x 32
            Conv3d(32, 32, 3, stride=1, padding=0, bias=False), # 147 x 147 x 32
            Conv3d(32, 64, 3, stride=1, padding=1, bias=False), # 147 x 147 x 64
            nn.MaxPool3d(3, stride=2, padding=0), # 73 x 73 x 64
            Conv3d(64, 80, 1, stride=1, padding=0, bias=False), # 73 x 73 x 80
            Conv3d(80, 192, 3, stride=1, padding=0, bias=False), # 71 x 71 x 192
            nn.MaxPool3d(3, stride=2, padding=0), # 35 x 35 x 192
        )
        self.branch_0 = Conv3d(192, 96, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv3d(192, 48, 1, stride=1, padding=0, bias=False),
            Conv3d(48, 64, 5, stride=1, padding=2, bias=False),
        )
        self.branch_2 = nn.Sequential(
            Conv3d(192, 64, 1, stride=1, padding=0, bias=False),
            Conv3d(64, 96, 3, stride=1, padding=1, bias=False),
            Conv3d(96, 96, 3, stride=1, padding=1, bias=False),
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool3d(3, stride=1, padding=1, count_include_pad=False),
            Conv3d(192, 64, 1, stride=1, padding=0, bias=False)
        )
    def forward(self, x):
        x = self.features(x)
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class Inception_ResNet_A(nn.Module):
    def __init__(self, in_channels, scale=1.0):
        super(Inception_ResNet_A, self).__init__()
        self.scale = scale
        self.branch_0 = Conv3d(in_channels, 32, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv3d(in_channels, 32, 1, stride=1, padding=0, bias=False),
            Conv3d(32, 32, 3, stride=1, padding=1, bias=False)
        )
        self.branch_2 = nn.Sequential(
            Conv3d(in_channels, 32, 1, stride=1, padding=0, bias=False),
            Conv3d(32, 48, 3, stride=1, padding=1, bias=False),
            Conv3d(48, 64, 3, stride=1, padding=1, bias=False)
        )
        self.conv = nn.Conv3d(128, 320, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x_res = torch.cat((x0, x1, x2), dim=1)
        x_res = self.conv(x_res)
        return self.relu(x + self.scale * x_res)

###changed###
class Inception_ResNet_B(nn.Module):
    def __init__(self, in_channels, scale=1.0):
        super(Inception_ResNet_B, self).__init__()
        self.scale = scale
        self.branch_0 = Conv3d(in_channels, 192, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv3d(in_channels, 128, 1, stride=1, padding=0, bias=False),
            Conv3d(128, 149, (1, 1, 7), stride=1, padding=(0, 0, 3), bias=False),#21
            Conv3d(149, 170, (1, 7, 1), stride=1, padding=(0, 3, 0), bias=False),#21
            Conv3d(170, 192, (7, 1, 1), stride=1, padding=(3, 0, 0), bias=False) #22
        )
        self.conv = nn.Conv3d(384, 1088, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)
        return self.relu(x + self.scale * x_res)


class Reduciton_B(nn.Module):
    def __init__(self, in_channels):
        super(Reduciton_B, self).__init__()
        self.branch_0 = nn.Sequential(
            Conv3d(in_channels, 256, 1, stride=1, padding=0, bias=False),
            Conv3d(256, 384, 3, stride=2, padding=0, bias=False)
        )
        self.branch_1 = nn.Sequential(
            Conv3d(in_channels, 256, 1, stride=1, padding=0, bias=False),
            Conv3d(256, 288, 3, stride=2, padding=0, bias=False),
        )
        self.branch_2 = nn.Sequential(
            Conv3d(in_channels, 256, 1, stride=1, padding=0, bias=False),
            Conv3d(256, 288, 3, stride=1, padding=1, bias=False),
            Conv3d(288, 320, 3, stride=2, padding=0, bias=False)
        )
        self.branch_3 = nn.MaxPool3d(3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)

#changed#
class Inception_ResNet_C(nn.Module):
    def __init__(self, in_channels, scale=1.0, activation=True):
        super(Inception_ResNet_C, self).__init__()
        self.scale = scale
        self.activation = activation
        self.branch_0 = Conv3d(in_channels, 192, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv3d(in_channels, 192, 1, stride=1, padding=0, bias=False),
            Conv3d(192, 213, (1, 1, 3), stride=1, padding=(0, 0, 1), bias=False),
            Conv3d(213, 234, (1, 3, 1), stride=1, padding=(0, 1, 0), bias=False),
            Conv3d(234, 256, (3, 1, 1), stride=1, padding=(1, 0, 0), bias=False)
        )
        self.conv = nn.Conv3d(448, 2080, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)#
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)
        if self.activation:
            return self.relu(x + self.scale * x_res)
        return x + self.scale * x_res


class Inception_ResNetv2(nn.Module):
    def __init__(self, in_channels=1, classes=5, k=256, l=256, m=384, n=384):
        super(Inception_ResNetv2, self).__init__()
        blocks = []
        blocks.append(Stem(in_channels))
        for i in range(10):
            blocks.append(Inception_ResNet_A(320, 0.17))
        blocks.append(Reduction_A(320, k, l, m, n))
        for i in range(20):
            blocks.append(Inception_ResNet_B(1088, 0.10))
        blocks.append(Reduciton_B(1088))
        for i in range(9):
            blocks.append(Inception_ResNet_C(2080, 0.20))
        blocks.append(Inception_ResNet_C(2080, activation=False))
        self.features = nn.Sequential(*blocks)
        self.conv = Conv3d(2080, 1536, 1, stride=1, padding=0, bias=False)
        self.global_average_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))#changed#
        self.linear = nn.Sequential(nn.Linear(1536, 768), nn.Linear(768, 384), nn.Linear(384, classes))
        #nn.Linear(1536, classes)

    def forward(self, x):
        x = self.features(x)#
        x = self.conv(x)
        x = self.global_average_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
