import torch
import torch.nn as nn
from model.units import Conv3d, Reduction_A


class Stem(nn.Module):
    def __init__(self, in_channels):
        super(Stem, self).__init__()
        self.conv3d_1a_3x3x3 = Conv3d(in_channels, 32, 3, stride=2, padding=0, bias=False)

        self.conv3d_2a_3x3x3 = Conv3d(32, 32, 3, stride=1, padding=0, bias=False)
        self.conv3d_2b_3x3x3 = Conv3d(32, 64, 3, stride=1, padding=1, bias=False)

        self.mixed_3a_branch_0 = nn.MaxPool3d(3, stride=2, padding=0)
        self.mixed_3a_branch_1 = Conv3d(64, 96, 3, stride=2, padding=0, bias=False)

        self.mixed_4a_branch_0 = nn.Sequential(
            Conv3d(160, 64, 1, stride=1, padding=0, bias=False),
            Conv3d(64, 96, 3, stride=1, padding=0, bias=False),
        )
        self.mixed_4a_branch_1 = nn.Sequential(
            Conv3d(160, 64, 1, stride=1, padding=0, bias=False),
            Conv3d(64, 64, (1, 1, 7), stride=1, padding=(0, 0, 3), bias=False),
            Conv3d(64, 64, (1, 7, 1), stride=1, padding=(0, 3, 0), bias=False),
            Conv3d(64, 64, (7, 1, 1), stride=1, padding=(3, 0, 0), bias=False),
            Conv3d(64, 96, 3, stride=1, padding=0, bias=False)
        )

        self.mixed_5a_branch_0 = Conv3d(192, 192, 3, stride=2, padding=0, bias=False)
        self.mixed_5a_branch_1 = nn.MaxPool3d(3, stride=2, padding=0)

    def forward(self, x):
        x = self.conv3d_1a_3x3x3(x) # 149 x 149 x 32
        x = self.conv3d_2a_3x3x3(x) # 147 x 147 x 32
        x = self.conv3d_2b_3x3x3(x) # 147 x 147 x 64
        x0 = self.mixed_3a_branch_0(x)
        x1 = self.mixed_3a_branch_1(x)
        x = torch.cat((x0, x1), dim=1) # 73 x 73 x 160
        x0 = self.mixed_4a_branch_0(x)
        x1 = self.mixed_4a_branch_1(x)
        x = torch.cat((x0, x1), dim=1) # 71 x 71 x 192
        x0 = self.mixed_5a_branch_0(x)
        x1 = self.mixed_5a_branch_1(x)
        x = torch.cat((x0, x1), dim=1) # 35 x 35 x 384
        return x


class Inception_A(nn.Module):
    def __init__(self, in_channels):
        super(Inception_A, self).__init__()
        self.branch_0 = Conv3d(in_channels, 96, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv3d(in_channels, 64, 1, stride=1, padding=0, bias=False),
            Conv3d(64, 96, 3, stride=1, padding=1, bias=False),
        )
        self.branch_2 = nn.Sequential(
            Conv3d(in_channels, 64, 1, stride=1, padding=0, bias=False),
            Conv3d(64, 96, 3, stride=1, padding=1, bias=False),
            Conv3d(96, 96, 3, stride=1, padding=1, bias=False),
        )
        self.brance_3 = nn.Sequential(
            nn.AvgPool3d(3, 1, padding=1, count_include_pad=False),
            Conv3d(384, 96, 1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.brance_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class Inception_B(nn.Module):
    def __init__(self, in_channels):
        super(Inception_B, self).__init__()
        self.branch_0 = Conv3d(in_channels, 384, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv3d(in_channels, 192, 1, stride=1, padding=0, bias=False),
            Conv3d(192, 213, (1, 1, 7), stride=1, padding=(0, 0, 3), bias=False),
            Conv3d(213, 234, (1, 7, 1), stride=1, padding=(0, 3, 0), bias=False),
            Conv3d(234, 256, (7, 1, 1), stride=1, padding=(3, 0, 0), bias=False),
        )
        self.branch_2 = nn.Sequential(
            Conv3d(in_channels, 192, 1, stride=1, padding=0, bias=False),
            Conv3d(192, 202, (7, 1, 1), stride=1, padding=(3, 0, 0), bias=False),
            Conv3d(202, 212, (1, 7, 1), stride=1, padding=(0, 3, 0), bias=False),
            Conv3d(212, 224, (1, 1, 7), stride=1, padding=(0, 0, 3), bias=False),
            Conv3d(224, 234, (7, 1, 1), stride=1, padding=(3, 0, 0), bias=False),
            Conv3d(234, 244, (1, 7, 1), stride=1, padding=(0, 3, 0), bias=False),
            Conv3d(244, 256, (1, 1, 7), stride=1, padding=(0, 0, 3), bias=False)
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool3d(3, stride=1, padding=1, count_include_pad=False),
            Conv3d(in_channels, 128, 1, stride=1, padding=0, bias=False)
        )
    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class Reduction_B(nn.Module):
    # 17 -> 8
    def __init__(self, in_channels):
        super(Reduction_B, self).__init__()
        self.branch_0 = nn.Sequential(
            Conv3d(in_channels, 192, 1, stride=1, padding=0, bias=False),
            Conv3d(192, 192, 3, stride=2, padding=0, bias=False),
        )
        self.branch_1 = nn.Sequential(
            Conv3d(in_channels, 256, 1, stride=1, padding=0, bias=False),
            Conv3d(256, 277, (1, 1, 7), stride=1, padding=(0, 0, 3), bias=False),
            Conv3d(277, 298, (1, 7, 1), stride=1, padding=(0, 3, 0), bias=False),
            Conv3d(298, 320, (7, 1, 1), stride=1, padding=(3, 0, 0), bias=False),
            Conv3d(320, 320, 3, stride=2, padding=0, bias=False)
        )
        self.branch_2 = nn.MaxPool3d(3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        return torch.cat((x0, x1, x2), dim=1)  # 8 x 8 x 1536


class Inception_C(nn.Module):
    def __init__(self, in_channels):
        super(Inception_C, self).__init__()
        self.branch_0 = Conv3d(in_channels, 256, 1, stride=1, padding=0, bias=False)

        self.branch_1 = Conv3d(in_channels, 384, 1, stride=1, padding=0, bias=False)
        self.branch_1_1 = Conv3d(384, 256, (1, 1, 3), stride=1, padding=(0, 0, 1), bias=False)
        self.branch_1_2 = Conv3d(384, 256, (1, 3, 1), stride=1, padding=(0, 1, 0), bias=False)
        self.branch_1_3 = Conv3d(384, 256, (3, 1, 1), stride=1, padding=(1, 0, 0), bias=False)

        self.branch_2 = nn.Sequential(
            Conv3d(in_channels, 384, 1, stride=1, padding=0, bias=False),
            Conv3d(384, 426, (1, 1, 3), stride=1, padding=(0, 0, 1), bias=False),
            Conv3d(426, 468, (1, 3, 1), stride=1, padding=(0, 1, 0), bias=False),
            Conv3d(468, 512, (3, 1, 1), stride=1, padding=(1, 0, 0), bias=False),
        )
        self.branch_2_1 = Conv3d(512, 256, (1, 1, 3), stride=1, padding=(0, 0, 1), bias=False)
        self.branch_2_2 = Conv3d(512, 256, (1, 3, 1), stride=1, padding=(0, 1, 0), bias=False)
        self.branch_2_3 = Conv3d(512, 256, (3, 1, 1), stride=1, padding=(1, 0, 0), bias=False)

        self.branch_3 = nn.Sequential(
            nn.AvgPool3d(3, stride=1, padding=1, count_include_pad=False),
            Conv3d(in_channels, 256, 1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x1_1 = self.branch_1_1(x1)
        x1_2 = self.branch_1_2(x1)
        x1_3 = self.branch_1_3(x1)
        x1 = torch.cat((x1_1, x1_2, x1_3), 1)
        x2 = self.branch_2(x)
        x2_1 = self.branch_2_1(x2)
        x2_2 = self.branch_2_2(x2)
        x2_3 = self.branch_2_3(x2)
        x2 = torch.cat((x2_1, x2_2, x2_3), dim=1)
        x3 = self.branch_3(x)#####################################error prob
        return torch.cat((x0, x1, x2, x3), dim=1) # 8 x 8 x 1536=512*3/////2048


class Inceptionv4(nn.Module):
    def __init__(self, in_channels=1, classes=5, k=192, l=224, m=256, n=384):
        super(Inceptionv4, self).__init__()
        blocks = []
        blocks.append(Stem(in_channels))
        for i in range(4):
            blocks.append(Inception_A(384))
        blocks.append(Reduction_A(384, k, l, m, n))
        for i in range(7):
            blocks.append(Inception_B(1024))
        blocks.append(Reduction_B(1024))
        for i in range(3):
            blocks.append(Inception_C(1536))
        self.features = nn.Sequential(*blocks) 
        self.global_average_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.linear = nn.Linear(2048, classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_average_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

