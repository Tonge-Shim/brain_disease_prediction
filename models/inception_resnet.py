import torch
from torch import nn

'''
참고링크:
(1) Inception 논문: https://arxiv.org/abs/1602.07261
(2) https://deep-learning-study.tistory.com/537
'''


class BasicConv3d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm3d(out_planes, eps=0.001, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Mixed3a(nn.Module):

    def __init__(self):
        super(Mixed3a, self).__init__()
        self.maxpool = nn.MaxPool3d(3, stride=2)
        self.conv = BasicConv3d(64, 96, 3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed4a(nn.Module):

    def __init__(self):
        super(Mixed4a, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv3d(160, 64, 1),
            BasicConv3d(64, 96, 3))
        self.branch1 = nn.Sequential(
            BasicConv3d(160, 64, 1),
            BasicConv3d(64, 64, (1, 7, 1), padding=(0, 3, 0)),
            BasicConv3d(64, 64, (1, 1, 7), padding=(0, 0, 3)),
            BasicConv3d(64, 96, 3))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed5a(nn.Module):

    def __init__(self):
        super(Mixed5a, self).__init__()
        self.conv = BasicConv3d(192, 192, 3, stride=2)
        self.maxpool = nn.MaxPool3d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class Stem(nn.Module):

    def __init__(self, in_channels):
        super(Stem, self).__init__()
        self.conv = nn.Sequential(
            BasicConv3d(in_channels, 32, 3, stride=2),  # 32x___x149x149
            BasicConv3d(32, 32, 3),  # 32x___x147x147
            BasicConv3d(32, 64, 3, padding=1))  # 64x___x147x147
        self.mixed_3a = Mixed3a()  # 160x___x73x73
        self.mixed_4a = Mixed4a()  # 192x___x71x71
        self.mixed_5a = Mixed5a()  # 384x___x35x35
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.mixed_3a(x)
        x = self.mixed_4a(x)
        out = self.mixed_5a(x)
        return self.relu(out)


class InceptionResnetA(nn.Module):

    def __init__(self, scale=1.0):
        super(InceptionResnetA, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv3d(384, 32, 1)
        self.branch1 = nn.Sequential(
            BasicConv3d(384, 32, 1),
            BasicConv3d(32, 32, 3, padding=1))
        self.branch2 = nn.Sequential(
            BasicConv3d(384, 32, 1),
            BasicConv3d(32, 48, 3, padding=1),
            BasicConv3d(48, 64, 3, padding=1))
        self.conv = nn.Conv3d(128, 384, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv(out)
        out = self.scale * out + x
        return self.relu(out)


class ReductionA(nn.Module):

    def __init__(self):
        super(ReductionA, self).__init__()
        self.branch0 = nn.MaxPool3d(3, stride=2)
        self.branch1 = BasicConv3d(384, 384, 3, stride=2)
        self.branch2 = nn.Sequential(
            BasicConv3d(384, 256, 1),
            BasicConv3d(256, 256, 3, padding=1),
            BasicConv3d(256, 384, 3, stride=2))
        self.relu = nn.ReLU()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return self.relu(out)


class InceptionResnetB(nn.Module):

    def __init__(self, scale=1.0):
        super(InceptionResnetB, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv3d(1152, 192, 1)
        self.branch1 = nn.Sequential(
            BasicConv3d(1152, 128, 1),
            BasicConv3d(128, 160, (1, 1, 7), padding=(0, 0, 3)),
            BasicConv3d(160, 192, (1, 7, 1), padding=(0, 3, 0)))
        self.conv = nn.Conv3d(384, 1152, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv(out)
        out = self.scale * out + x
        return self.relu(out)


class ReductionB(nn.Module):

    def __init__(self):
        super(ReductionB, self).__init__()
        self.branch0 = nn.MaxPool3d(3, stride=2)
        self.branch1 = nn.Sequential(
            BasicConv3d(1152, 256, 1),
            BasicConv3d(256, 384, 3, stride=2))
        self.branch2 = nn.Sequential(
            BasicConv3d(1152, 256, 1),
            BasicConv3d(256, 288, 3, stride=2))
        self.branch3 = nn.Sequential(
            BasicConv3d(1152, 256, 1),
            BasicConv3d(256, 288, 3, padding=1),
            BasicConv3d(288, 320, 3, stride=2))
        self.relu = nn.ReLU()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return self.relu(out)


class InceptionResnetC(nn.Module):

    def __init__(self, scale=1.0):
        super(InceptionResnetC, self).__init__()
        self.scale = scale
        self.branch0 = nn.Sequential(
            BasicConv3d(2144, 192, 1))
        self.branch1 = nn.Sequential(
            BasicConv3d(2144, 192, 1),
            BasicConv3d(192, 224, (1, 1, 3), padding=(0, 0, 1)),
            BasicConv3d(224, 256, (1, 3, 1), padding=(0, 1, 0)))
        self.conv = nn.Conv3d(448, 2144, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv(out)
        out = self.scale * out + x
        return self.relu(out)


class InceptionResnet(nn.Module):

    def __init__(self, in_channels=1, num_classes=5):
        super(InceptionResnet, self).__init__()
        self.features = nn.Sequential(
            Stem(in_channels),  # 384x___x35x35
            InceptionResnetA(scale=0.17),
            InceptionResnetA(scale=0.17),
            InceptionResnetA(scale=0.17),
            InceptionResnetA(scale=0.17),
            InceptionResnetA(scale=0.17),  # 384x___x35x35
            ReductionA(),  # 1152x___x17x17
            InceptionResnetB(scale=0.10),
            InceptionResnetB(scale=0.10),
            InceptionResnetB(scale=0.10),
            InceptionResnetB(scale=0.10),
            InceptionResnetB(scale=0.10),
            InceptionResnetB(scale=0.10),
            InceptionResnetB(scale=0.10),
            InceptionResnetB(scale=0.10),
            InceptionResnetB(scale=0.10),
            InceptionResnetB(scale=0.10),  # 1152x___x17x17
            ReductionB(),  # 2144x___x8x8
            InceptionResnetC(scale=0.20),
            InceptionResnetC(scale=0.20),
            InceptionResnetC(scale=0.20),
            InceptionResnetC(scale=0.20),
            InceptionResnetC(scale=0.20))  # 2144x___x8x8
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout3d(0.2)
        self.fc = nn.Linear(2144, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


def inception_resnet(**kwargs):
    model = InceptionResnet(**kwargs)
    return model