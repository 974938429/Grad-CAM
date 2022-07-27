写在前面：Grad-CAM其实是通过将直接将预测值当作最终的Loss反向传播（而不是预测值与gt做Loss计算后反向传播的），
并用钩子函数获取梯度信息，与原始图像堆叠在一起并可视化呈现的。
下面将对文件夹下的*调用CAM.py*和*cam_utls.py*文件进行整理。
---
### 调用CAM.py
在步骤4）之前都是申明模型、加载模型参数、加载数据之类的操作，步骤4）用于申明需要查看的网络结构
~~~
target_layers = [model.down4]
~~~
其中网络结构在__init__()定义如下：
~~~
...
self.down3 = Down(256, 512)
self.down4 = Down(512, 1024 // factor)
self.up1 = Up(1024, 512 // factor, bilinear)
...
~~~
也就是说，在__init__()声明的self结构可以被拿来查看当前结构的“注意力图”。
步骤5）中，先申请*cam_utils.py*中定义的**GradCAM**类，其中传入的参数有模型、需要查看的网络结构和是否使用cuda加速:
~~~
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
~~~
grayscale_cam为在*model.down4*结构中，模型的注意力图：
~~~
grayscale_cam = cam(input_tensor=src_tensor, target=gt_tensor) #ndarray:(1,320,320)
~~~
只有通过*show_cam_on_image()* 方法与原图叠加，生成注意力的热力图，可视化展示或者保存。

---
### cam_utils.py
*ActivationsAndGradients* 类申请钩子函数，在正向传播过程中获取特征值，
存储在*self.activations* 列表中，在反向传播获取梯度图，存储在*self.gradients* 中。注意，
在*GradCAM* 中申请了实例化了*ActivationsAndGradients*类，
会自动调用self.model(x)开始正向传播。

*GradCAM*构造的方法介绍如下：
+ __init__：传入参数，并实例化一个*ActivationsAndGradients*类，开始正向传播；
+ **get_loss**：获取Loss（可以直接将预测图作为loss回传）
+ **get_cam_weights**：进行全局平均池化（GAP），获得权重参数；
+ **get_target_width_height**：获取特征层h和w；
+ **get_cam_image**：将特征图与回传的权重（经过GAP的梯度图）进行融合，突出梯度变化更大的特征层；
+ **scale_cam_img**：缩放得到的cam图（也就是经过权重叠加的特征层）；
