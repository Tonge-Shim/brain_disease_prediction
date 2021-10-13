
import torch
import torch.nn as nn


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
#################################3dconv#################################################33
class Conv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding, stride = 1, bias = True):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size, stride = stride, padding = padding, bias = bias)
        self.bn = nn.BatchNorm3d(out_planes, eps = 0.001, momentum = 0.1)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)#
        x = self.bn(x)
        x = self.relu(x)
        # print(x.shape)
        return x
##################################################################################################
class Reduction_A(nn.Module):
    # 35 -> 17
    def __init__(self, in_channels, k, l, m, n):
        super(Reduction_A, self).__init__()
        self.branch_0 = Conv3d(in_channels, n, 3, stride=2, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv3d(in_channels, k, 1, stride=1, padding=0, bias=False),
            Conv3d(k, l, 3, stride=1, padding=1, bias=False),
            Conv3d(l, m, 3, stride=2, padding=0, bias=False),
        )
        self.branch_2 = nn.MaxPool3d(3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        return torch.cat((x0, x1, x2), dim=1) # 17 x 17 x 1024



