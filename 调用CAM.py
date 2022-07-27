from model.two_stage_model import UNetStage1 as Net
import os
import numpy as np
import torch
import argparse
from PIL import Image
from torchvision import transforms
from Grad_CAM.cam_utils import *
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='全局变量')
parser.add_argument('--resume',default=r'D:\21.路径\CTS-Net模型参数\general\the_coarse_stage.pth',help='加载的模型路径')
parser.add_argument('--img_src',default='./test_img.tif',help='单张图片src路径')
parser.add_argument('--img_gt',default='./gt.bmp',help='单张图片gt路径')
# 其实该方法中并没有用到gt，是直接将预测图想作为loss回传，得到的梯度图像
args = parser.parse_args()

def main():
    # 1)建立模型、加载预训练参数
    model = Net()
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    if os.path.exists(args.resume) and torch.cuda.is_available():
        print("=> 载入checkpoint'{}'".format(args.resume))
        checkpoint=torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> checkpoint'{}'已载入".format(args.resume))
    elif os.path.exists(args.resume) and not torch.cuda.is_available():
        print("=> 载入checkpoint'{}'".format(args.resume))
        checkpoint=torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        print("=> checkpoint'{}'已载入".format(args.resume))
    else:
        print("=> 预训练模型路径出错'{}'".format(args.resume))
    # 2)传入图片，这里后期可以注释掉改为文件夹地址
    # 展示预测结果，展示gt，预测结果与gt差值回传的grad_cam图，gt回传的grad_cam图
    if os.path.exists(args.img_src) and os.path.exists(args.img_gt):
        src = Image.open(args.img_src).convert('RGB')
        src = np.array(src, dtype=np.uint8)
        gt = Image.open(args.img_gt).convert('L')
        gt = np.array(gt, dtype=np.uint8)
        gt = np.where((gt == 255)|(gt == 100), 1, 0) #变为双边缘
        look(gt)
    else:
        print("src地址：'{}'或gt地址：'{}'".format(args.img_src, args.img_gt))
    # 3)对图片进行预处理
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27))])
    # src和gt从ndarry变为tensor后，形状由[H，W，C]变为[C，H，W]
    src_tensor = data_transform(src)
    gt_tensor = transforms.ToTensor()(gt)
    # src和gt都增加一个维度
    src_tensor = torch.unsqueeze(src_tensor, dim=0) #[B_S, C, H, W]
    gt_tensor  = torch.unsqueeze(gt_tensor, dim=0)

    # 4)指定需要计算CAM的网络结构
    # target_layers = [model.up4]
    target_layers = [model.down4]

    # 5)调用Grad-CAM方法
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    grayscale_cam = cam(input_tensor=src_tensor, target=gt_tensor)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(src.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()



if __name__=='__main__':
    main()















