""" Pytorch Inception-V4 implementation
Sourced from https://github.com/Cadene/tensorflow-model-zoo.torch (MIT License) which is
based upon Google's Tensorflow implementation and pretrained weights (Apache 2.0 License)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['InceptionV4', 'inception_v4']


class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_planes, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed3a(nn.Module):
    def __init__(self):
        super(Mixed3a, self).__init__()
        self.maxpool = nn.MaxPool3d(3, stride=2)
        self.conv = BasicConv3d(64, 96, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed4a(nn.Module):
    def __init__(self):
        super(Mixed4a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(160, 64, kernel_size=1, stride=1),
            BasicConv3d(64, 96, kernel_size=3, stride=1)
        )

        self.branch1 = nn.Sequential(
            BasicConv3d(160, 64, kernel_size=1, stride=1),
            BasicConv3d(64, 64, kernel_size=(1, 1, 7), stride=1, padding=(0, 0, 3)),
            BasicConv3d(64, 64, kernel_size=(1, 7, 1), stride=1, padding=(0, 3, 0)),
            BasicConv3d(64, 96, kernel_size=(3, 3, 3), stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed5a(nn.Module):
    def __init__(self):
        super(Mixed5a, self).__init__()
        self.conv = BasicConv3d(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool3d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class InceptionA(nn.Module):
    def __init__(self):
        super(InceptionA, self).__init__()
        self.branch0 = BasicConv3d(384, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv3d(384, 64, kernel_size=1, stride=1),
            BasicConv3d(64, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv3d(384, 64, kernel_size=1, stride=1),
            BasicConv3d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv3d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool3d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv3d(384, 96, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class ReductionA(nn.Module):
    def __init__(self):
        super(ReductionA, self).__init__()
        self.branch0 = BasicConv3d(384, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv3d(384, 192, kernel_size=1, stride=1),
            BasicConv3d(192, 224, kernel_size=3, stride=1, padding=1),
            BasicConv3d(224, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool3d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class InceptionB(nn.Module):
    def __init__(self):
        super(InceptionB, self).__init__()
        self.branch0 = BasicConv3d(1024, 384, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv3d(1024, 192, kernel_size=1, stride=1),
            BasicConv3d(192, 224, kernel_size=(1, 1, 7), stride=1, padding=(0, 0, 3)),
            BasicConv3d(224, 256, kernel_size=(1, 7, 1), stride=1, padding=(0, 3, 0))
        )

        self.branch2 = nn.Sequential(
            BasicConv3d(1024, 192, kernel_size=1, stride=1),
            BasicConv3d(192, 192, kernel_size=(1, 7, 1), stride=1, padding=(0, 3, 0)),
            BasicConv3d(192, 224, kernel_size=(1, 1, 7), stride=1, padding=(0, 0, 3)),
            BasicConv3d(224, 224, kernel_size=(1, 7, 1), stride=1, padding=(0, 3, 0)),
            BasicConv3d(224, 256, kernel_size=(1, 1, 7), stride=1, padding=(0, 0, 3))
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool3d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv3d(1024, 128, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class ReductionB(nn.Module):
    def __init__(self):
        super(ReductionB, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(1024, 192, kernel_size=1, stride=1),
            BasicConv3d(192, 192, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv3d(1024, 256, kernel_size=1, stride=1),
            BasicConv3d(256, 256, kernel_size=(1, 1, 7), stride=1, padding=(0, 0, 3)),
            BasicConv3d(256, 320, kernel_size=(1, 7, 1), stride=1, padding=(0, 3, 0)),
            BasicConv3d(320, 320, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool3d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class InceptionC(nn.Module):
    def __init__(self):
        super(InceptionC, self).__init__()

        self.branch0 = BasicConv3d(1536, 256, kernel_size=1, stride=1)

        self.branch1_0 = BasicConv3d(1536, 384, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv3d(384, 256, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
        self.branch1_1b = BasicConv3d(384, 256, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))

        self.branch2_0 = BasicConv3d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv3d(384, 448, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))
        self.branch2_2 = BasicConv3d(448, 512, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
        self.branch2_3a = BasicConv3d(512, 256, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
        self.branch2_3b = BasicConv3d(512, 256, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))

        self.branch3 = nn.Sequential(
            nn.AvgPool3d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv3d(1536, 256, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)

        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)

        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)

        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionV4(nn.Module):
    def __init__(self, num_classes=1000, in_channels=3, output_stride=32, drop_rate=0.):
        super(InceptionV4, self).__init__()
        assert output_stride == 32
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.num_features = 1536

        self.features = nn.Sequential(
            BasicConv3d(in_channels, 32, kernel_size=3, stride=2),
            BasicConv3d(32, 32, kernel_size=3, stride=1),
            BasicConv3d(32, 64, kernel_size=3, stride=1, padding=1),
            Mixed3a(),
            Mixed4a(),
            Mixed5a(),
            InceptionA(),
            InceptionA(),
            InceptionA(),
            InceptionA(),
            ReductionA(),  # Mixed6a
            InceptionB(),
            InceptionB(),
            InceptionB(),
            InceptionB(),
            InceptionB(),
            InceptionB(),
            InceptionB(),
            ReductionB(),  # Mixed7a
            InceptionC(),
            InceptionC(),
            InceptionC(),
        )
        self.feature_info = [
            dict(num_chs=64, reduction=2, module='features.2'),
            dict(num_chs=160, reduction=4, module='features.3'),
            dict(num_chs=384, reduction=8, module='features.9'),
            dict(num_chs=1024, reduction=16, module='features.17'),
            dict(num_chs=1536, reduction=32, module='features.21'),
        ]

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(1536, num_classes)

    def forward_features(self, x):
        return self.features(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.fc(x)
        return x


def inception_v4(**kwargs):
    return InceptionV4(**kwargs)