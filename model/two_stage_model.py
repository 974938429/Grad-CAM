import torch.nn.functional as F
import sys

sys.path.append('.')
import torch.nn as nn
from model.unet_parts import *



class UNetStage1(nn.Module):
    def __init__(self, n_channels=3, bilinear=False):
        super(UNetStage1, self).__init__()
        factor = 2 if bilinear else 1
        _factor = 1 if bilinear else 2
        # print('factor is : ',_factor)
        self.n_channels = n_channels
        self.n_classes = 1
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 1)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        stage_x1 = self.up1(x5, x4)
        stage_x2 = self.up2(stage_x1, x3)
        stage_x3 = self.up3(stage_x2, x2)
        stage_x4 = self.up4(stage_x3, x1)
        logits = self.outc(stage_x4)
        return [logits, stage_x1, stage_x2, stage_x3]

class Postprocess(nn.Module):
    def __init__(self,in_channels=4):
        super(Postprocess, self).__init__()
        self.in_channels = in_channels
        self.conv_3 = Conv_3(in_channels,15)
        self.conv_5 = Conv_5(in_channels,15)
        self.conv_7 = Conv_7(in_channels, 15)
        self.dilated_9_conv = Conv_9(in_channels, 20)
        self.follow1 = DoubleConv(65, 64)
        self.follow2 = DoubleConv(64, 64)
        self.follow3 = DoubleConv(64, 64)
        self.follow4 = DoubleConv(64, 64)
        self.outc = OutConv(64, 1)
    def forward(self,cat_img):
        _post_3 = self.conv_3(cat_img)
        _post_5 = self.conv_5(cat_img)
        _post_7 = self.conv_7(cat_img)
        _post_9 = self.dilated_9_conv(cat_img)
        # print(_post_3.shape)
        # print(_post_5.shape)
        # print(_post_7.shape)
        # print(_post_9.shape)
        post2 = torch.cat([_post_3,_post_5,_post_7,_post_9], axis=1)
        post3 = self.follow1(post2)
        post4 = self.follow2(post3)
        post5 = self.follow3(post4)
        post6 = self.follow3(post5)
        post7 = self.outc(post6)
        return [post7]





class UNetStage2(nn.Module):
    def __init__(self, n_channels=4, bilinear=False):
        super(UNetStage2, self).__init__()
        factor = 2 if bilinear else 1
        _factor = 1 if bilinear else 2


        print('factor is : ',_factor)
        self.n_channels = n_channels
        self.n_classes = 1
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.maxpool = MaxPool()
        self.down1 = Down_no_pool(64, 128)
        self.down2 = Down_no_pool(128, 256)
        self.down3 = Down_no_pool(256, 512)

        self.down4 = Down_no_pool(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        # self.outc = OutConv(64, 1)

        self.fuse2 = FuseStageOut(in_channels=64*_factor + 64, out_channles=64)
        self.fuse3 = FuseStageOut(in_channels=128*_factor + 128, out_channles=128)
        self.fuse4 = FuseStageOut(in_channels=256*_factor + 256, out_channles=256)
        self.relation1 = RelationMap(in_channels=64, out_channels=2)
        self.relation2 = RelationMap(in_channels=64, out_channels=2)
        self.relation3 = RelationMap(in_channels=64, out_channels=2)
        self.relation4 = RelationMap(in_channels=64, out_channels=2)

        self.relation5 = RelationMap(in_channels=64, out_channels=2)
        self.relation6 = RelationMap(in_channels=64, out_channels=2)
        self.relation7 = RelationMap(in_channels=64, out_channels=2)
        self.relation8 = RelationMap(in_channels=64, out_channels=2)

        self.with_relation = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.final = RelationFuse(in_channels=16 + 8, out_channels=1)

    def forward(self, x, stage3, stage2, stage1):

        x1 = self.inc(x)


        # fuse stage
        x2 = self.maxpool(x1)
        x2 = self.fuse2(x2, stage1)
        x2 = self.down1(x2)

        x3 = self.maxpool(x2)
        x3 = self.fuse3(x3, stage2)
        x3 = self.down2(x3)

        x4 = self.maxpool(x3)
        x4 = self.fuse4(x4, stage3)
        x4 = self.down3(x4)

        x5 = self.maxpool(x4)
        x5 = self.down4(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        r1 = self.relation1(x)
        r2 = self.relation2(x)

        r3 = self.relation3(x)
        r4 = self.relation4(x)

        r5 = self.relation5(x)
        r6 = self.relation6(x)

        r7 = self.relation7(x)
        r8 = self.relation8(x)

        with_r = self.with_relation(x)
        x = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8, with_r], dim=1)
        x = self.final(x)

        return [x, r1, r2, r3, r4, r5, r6, r7, r8]


if __name__ == '__main__':
    model1 = UNetStage1(3,bilinear=False).cpu()
    model2 = UNetStage2(4, bilinear=False).cpu()
    model3 = Postprocess(4, bilinear=False).cpu()
    # in_size = 321
    # summary(model=model1,(3,320,320),device='cpu',batch_size=2)
    # summary(model2, [(4, in_size, in_size),(512, in_size // 8, in_size // 8),(256, in_size // 4, in_size // 4)
    #                 , (128, in_size // 2, in_size // 2) ], device='cpu', batch_size=2)
    # summary(model1, (3, 321, 321), device='cpu', batch_size=2)
