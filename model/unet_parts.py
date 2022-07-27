

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# torch.set_default_tensor_type(torch.DoubleTensor)  #不然网络参数格式不统一

class DilateDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

def mask_gen():
    # 中心为0，周围为1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    center_zero = torch.tensor(np.ones(shape=(3, 3)))  # 卷积核大小是3*3
    center_zero[1, 1] = 0
    # I型卷积核
    #     v1:[1,1,1] v2:[0,0,0]
    #        [1,0,0]    [0,0,1]
    #        [0,0,0]    [1,1,1]
    I_v1 = torch.tensor(np.zeros(shape=(3, 3)))
    I_v1[0, :] = 1
    I_v1[1, 0] = 1
    # print(self.I_v1)
    I_v2 = torch.tensor(np.zeros(shape=(3, 3)))
    I_v2[2, :] = 1
    I_v2[1, 2] = 1
    # print(self.I_v2)
    # II型卷积核
    #     v1:[1,1,1] v2:[0,0,0]
    #        [0,0,1]    [1,0,0]
    #        [0,0,0]    [1,1,1]
    II_v1 = torch.tensor(np.zeros(shape=(3, 3)))
    II_v1[0, :] = 1
    II_v1[1, 2] = 1
    # print(self.II_v1)
    II_v2 = torch.tensor(np.zeros(shape=(3, 3)))
    II_v2[2, :] = 1
    II_v2[1, 0] = 1
    # print(self.II_v2)
    # III型卷积核
    #     v1:[0,1,1] v2:[1,0,0]
    #        [0,0,1]    [1,0,0]
    #        [0,0,1]    [1,1,0]
    III_v1 = torch.tensor(np.zeros(shape=(3, 3)))
    III_v1[:, 2] = 1
    III_v1[0, 1] = 1
    # print(self.III_v1)
    III_v2 = torch.tensor(np.zeros(shape=(3, 3)))
    III_v2[:, 0] = 1
    III_v2[2, 1] = 1
    # print(self.III_v2)
    # IV型卷积核
    #     v1:[0,0,1] v2:[1,1,0]
    #        [0,0,1]    [1,0,0]
    #        [0,1,1]    [1,0,0]
    IV_v1 = torch.tensor(np.zeros(shape=(3, 3)))
    IV_v1[:, 2] = 1
    IV_v1[2, 1] = 1
    # print(self.IV_v1)
    IV_v2 = torch.tensor(np.zeros(shape=(3, 3)))
    IV_v2[:, 0] = 1
    IV_v2[0, 1] = 1
    # print(self.IV_v2)

    V_V1 = torch.tensor(np.zeros(shape=(5, 5))) #中心点是-1，其他权重为0
    V_V1[2, 2] = -1

    V_V2 = torch.where(V_V1==-1,0,1)  # 中心点是0，其他权重为1



    return center_zero,I_v1,I_v2,II_v1,II_v2,III_v1,III_v2,IV_v1,IV_v2,V_V1,V_V2

class Init_Conv(nn.Module):   #32个的正常卷积核，32个BayerConv2D卷积核
    '''32层正常卷积，32层BayerConv2D，分为四个分类（I、II、III和IV型）'''

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        # if not mid_channels:
        #     mid_channels = out_channels
        self.center_zero,self.I_v1,self.I_v2,self.II_v1,self.II_v2,self.III_v1,self.III_v2,self.IV_v1,self.IV_v2,self.V_V1,self.V_V2=mask_gen()
        self.init_conv = nn.Conv2d(in_channels, 32, 3, 1, 1) #输入通道数，输出通道数，卷积核大小，跨步，padding
        self.bayer_conv_1 = nn.Conv2d(in_channels, 8, 3, 1, 1)
        self.bayer_conv_2 = nn.Conv2d(in_channels, 8, 3, 1, 1)
        self.bayer_conv_3 = nn.Conv2d(in_channels, 8, 3, 1, 1)
        self.bayer_conv_4 = nn.Conv2d(in_channels, 8, 3, 1, 1)

        # print(self.bayer_conv_1.weight.data)
        # print(self.center_zero)
        self.bayer_conv_1.weight.data *= self.center_zero
        self.bayer_conv_2.weight.data *= self.center_zero
        self.bayer_conv_3.weight.data *= self.center_zero
        self.bayer_conv_4.weight.data *= self.center_zero

        #32层特殊卷积赋予权重约束
        batch_size = self.bayer_conv_1.weight.data.size(0)
        V1, V2 = self.I_v1, self.I_v2
        # V1 = V1.unsqueeze(0).unsqueeze(0)  #不能升维，升维之后再乘以相同的矩阵变为[[0.]]
        # V2 = V2.unsqueeze(0).unsqueeze(0)
        temp = self.bayer_conv_1.weight.data * V2
        self.bayer_conv_1.weight.data *= V1
        self.bayer_conv_1.weight.data *= torch.pow(self.bayer_conv_1.weight.data.sum(axis=(2, 3)).view(batch_size, 3, 1, 1),
                                                   -1)
        temp *= torch.pow(temp.sum(axis=(2, 3)).view(batch_size, 3, 1, 1), -1)
        temp *= -1
        self.bayer_conv_1.weight.data += temp
        # print(self.bayer_conv_1.weight.data)

        V1, V2 = self.II_v1, self.II_v2
        temp = self.bayer_conv_2.weight.data * V2
        self.bayer_conv_2.weight.data *= V1
        self.bayer_conv_2.weight.data *= torch.pow(
            self.bayer_conv_2.weight.data.sum(axis=(2, 3)).view(batch_size, 3, 1, 1), -1)
        temp *= torch.pow(temp.sum(axis=(2, 3)).view(batch_size, 3, 1, 1), -1)
        temp *= -1
        self.bayer_conv_2.weight.data += temp

        V1, V2 = self.III_v1, self.III_v2
        temp = self.bayer_conv_3.weight.data * V2
        self.bayer_conv_3.weight.data *= V1
        self.bayer_conv_3.weight.data *= torch.pow(
            self.bayer_conv_3.weight.data.sum(axis=(2, 3)).view(batch_size, 3, 1, 1), -1)
        temp *= torch.pow(temp.sum(axis=(2, 3)).view(batch_size, 3, 1, 1), -1)
        temp *= -1
        self.bayer_conv_3.weight.data += temp

        V1, V2 = self.IV_v1, self.IV_v2
        temp = self.bayer_conv_4.weight.data * V2
        self.bayer_conv_4.weight.data *= V1
        self.bayer_conv_4.weight.data *= torch.pow(
            self.bayer_conv_4.weight.data.sum(axis=(2, 3)).view(batch_size, 3, 1, 1), -1)
        temp *= torch.pow(temp.sum(axis=(2, 3)).view(batch_size, 3, 1, 1), -1)
        temp *= -1
        self.bayer_conv_4.weight.data += temp


    def forward(self, x):
        # x = x.type(torch.DoubleTensor)
        conv_normal = self.init_conv(x)
        bayer_conv_1 = self.bayer_conv_1(x)
        bayer_conv_2 = self.bayer_conv_2(x)
        bayer_conv_3 = self.bayer_conv_3(x)
        bayer_conv_4 = self.bayer_conv_4(x)

        return conv_normal,bayer_conv_1,bayer_conv_2,bayer_conv_3,bayer_conv_4

class Init_Conv_2(nn.Module):   #32个的正常卷积核，32个BayerConv2D卷积核——该卷积核是5*5大小，并且是受限卷积那篇文章用到的

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        # if not mid_channels:
        #     mid_channels = out_channels
        self.center_zero,self.I_v1,self.I_v2,self.II_v1,self.II_v2,self.III_v1,self.III_v2,self.IV_v1,self.IV_v2,self.V_v1,self.V_v2=mask_gen()
        self.init_conv = nn.Conv2d(in_channels, 32, 3, 1, 1) #输入通道数，输出通道数，卷积核大小，跨步，padding
        self.bayer_conv = nn.Conv2d(in_channels, 32, 5, 1, 2)
        V1, V2 = self.V_v1, self.V_v2
        batch_size = self.bayer_conv.weight.data.size(0)
        temp = self.bayer_conv.weight.data * V2
        self.bayer_conv.weight.data *= V1
        temp *= torch.pow(temp.sum(axis=(2, 3)).view(batch_size, 3, 1, 1), -1)
        self.bayer_conv.weight.data += temp


    def forward(self, x):
        conv_normal = self.init_conv(x)
        bayer_conv = self.bayer_conv(x)

        return conv_normal,bayer_conv

class OnceConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        self.once_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.once_conv(x)

class Conv_3(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv3(x)

class Conv_5(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv5(x)

class Conv_7(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv7(x)

class Conv_9(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super(Conv_9,self).__init__()
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=5, dilation=4, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv9(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class MaxPool(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self):
        super().__init__()
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.maxpool(x)


class Down_no_pool(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class RelationMap(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RelationMap, self).__init__()
        self.relation_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.relation_conv(x)
        return x


class RelationFuse(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RelationFuse, self).__init__()
        self.final = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, out_channels, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.final(x)
        return x


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class FuseStageOut(nn.Module):
    def __init__(self, in_channels, out_channles):
        super(FuseStageOut, self).__init__()
        self.fuse_stage_out = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channles, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channles),
            nn.ReLU(),
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.fuse_stage_out(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.Sigmoid()(x)
        return x
class TwoStageOut(nn.Module):
    def __init__(self, in_channels, out_channles):
        super(TwoStageOut, self).__init__()
        self.fuse_stage_out = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channles, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channles),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channles, out_channels=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.fuse_stage_out(x)
        return x
