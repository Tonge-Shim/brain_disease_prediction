""" Pytorch Inception-Resnet-V2 3D version implementation
Sourced from https://github.com/Cadene/tensorflow-model-zoo.torch (MIT License) which is
based upon Google's Tensorflow implementation and pretrained weights (Apache 2.0 License)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['InceptionResnetV2', 'inception_resnet_v2']


class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_planes, eps=.001)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Module):
    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = BasicConv3d(192, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv3d(192, 48, kernel_size=1, stride=1),
            BasicConv3d(48, 64, kernel_size=5, stride=1, padding=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv3d(192, 64, kernel_size=1, stride=1),
            BasicConv3d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv3d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool3d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv3d(192, 64, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):
    def __init__(self, scale=1.0):
        super(Block35, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv3d(320, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv3d(320, 32, kernel_size=1, stride=1),
            BasicConv3d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv3d(320, 32, kernel_size=1, stride=1),
            BasicConv3d(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv3d(48, 64, kernel_size=3, stride=1, padding=1)
        )

        self.conv3d = nn.Conv3d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv3d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Module):
    def __init__(self):
        super(Mixed_6a, self).__init__()

        self.branch0 = BasicConv3d(320, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv3d(320, 256, kernel_size=1, stride=1),
            BasicConv3d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv3d(256, 384, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool3d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):
    def __init__(self, scale=1.0):
        super(Block17, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv3d(1088, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv3d(1088, 128, kernel_size=1, stride=1),
            BasicConv3d(128, 160, kernel_size=(1, 1, 7), stride=1, padding=(0, 0, 3)),
            BasicConv3d(160, 192, kernel_size=(1, 7, 1), stride=1, padding=(0, 3, 0))
        )

        self.conv3d = nn.Conv3d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv3d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Module):
    def __init__(self):
        super(Mixed_7a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(1088, 256, kernel_size=1, stride=1),
            BasicConv3d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv3d(1088, 256, kernel_size=1, stride=1),
            BasicConv3d(256, 288, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv3d(1088, 256, kernel_size=1, stride=1),
            BasicConv3d(256, 288, kernel_size=3, stride=1, padding=1),
            BasicConv3d(288, 320, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool3d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, no_relu=False):
        super(Block8, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv3d(2080, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv3d(2080, 192, kernel_size=1, stride=1),
            BasicConv3d(192, 224, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1)),
            BasicConv3d(224, 256, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))
        )

        self.conv3d = nn.Conv3d(448, 2080, kernel_size=1, stride=1)
        self.relu = None if no_relu else nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv3d(out)
        out = out * self.scale + x
        if self.relu is not None:
            out = self.relu(out)
        return out


class InceptionResnetV2(nn.Module):
    def __init__(self, num_classes=1000, in_channels=3, drop_rate=0., output_stride=32):
        super(InceptionResnetV2, self).__init__()
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.num_features = 1536
        assert output_stride == 32

        self.conv3d_1a = BasicConv3d(in_channels, 32, kernel_size=3, stride=2)
        self.conv3d_2a = BasicConv3d(32, 32, kernel_size=3, stride=1)
        self.conv3d_2b = BasicConv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.feature_info = [dict(num_chs=64, reduction=2, module='conv3d_2b')]

        self.maxpool_3a = nn.MaxPool3d(3, stride=2)
        self.conv3d_3b = BasicConv3d(64, 80, kernel_size=1, stride=1)
        self.conv3d_4a = BasicConv3d(80, 192, kernel_size=3, stride=1)
        self.feature_info += [dict(num_chs=192, reduction=4, module='conv3d_4a')]

        self.maxpool_5a = nn.MaxPool3d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.repeat = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17)
        )
        self.feature_info += [dict(num_chs=320, reduction=8, module='repeat')]

        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10)
        )
        self.feature_info += [dict(num_chs=1088, reduction=16, module='repeat_1')]

        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20)
        )
        self.block8 = Block8(no_relu=True)
        self.conv3d_7b = BasicConv3d(2080, self.num_features, kernel_size=1, stride=1)
        self.feature_info += [dict(num_chs=self.num_features, reduction=32, module='conv3d_7b')]

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(1536, num_classes)

    def forward_features(self, x):
        x = self.conv3d_1a(x)
        x = self.conv3d_2a(x)
        x = self.conv3d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv3d_3b(x)
        x = self.conv3d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv3d_7b(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.fc(x)
        return x


def inception_resnet_v2(**kwargs):
    return InceptionResnetV2(**kwargs)