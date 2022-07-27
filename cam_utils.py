import cv2 as cv
import numpy as np
import torch
import matplotlib.pyplot as plt

def look(src):
    plt.imshow(src)
    plt.show()


class ActivationsAndGradients:
    # 自动调用__call__()函数，获取正向传播的特征层A和反向传播的梯度A'
    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers: # 这是设计者传入多层的卷积层，我们传入一层也可以
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation
                )
            )
        if hasattr(target_layer, 'register_full_backward_hook'):
        #hasattr(object,name)返回值:如果对象有该属性返回True,否则返回False
            self.handles.append(
                target_layer.register_full_backward_hook(self.save_gradient))
        else:
            self.handles.append(
                target_layer.register_backward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, model, grad_input, grad_output):
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients #反向传播的梯度A’放在最前，与上文的特征层排序相反

    def __call__(self, x):
        #自动调用的__call__方法
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()
            # handle要及时移除掉，不然会占用过多内存


class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model = self.model.cuda()
        else:
            pass
        self.activations_and_grads = ActivationsAndGradients(self.model, target_layers, reshape_transform)
        # 取得正向传播的特征层A和反向传播的梯度A'
    @staticmethod
    # @staticmethod和@classmethod只是将一个函数绑定在一个类别下，调用时无需再创建一个对象调用（创建一个对象在调用也可以）
    # 比如class A下有一个@staticmethod方法——f()，调用时就可以直接A.f()即可，当然也可以A().f()。
    # @classmethod与@staticmethod区别在于第一个传进来的参数是cls（类）。
    def get_loss(output, target):
        loss = 0
        # 从下方三种情况中选出一种
        loss = output.mul(target) # 1)查看模型预测正确的部分-->张量的对应点相乘
        _loss = loss.detach().cpu()
        _loss = _loss.squeeze(0).squeeze(0)
        # look(_loss)

        # 另外张量的内积、外积-->.mm()
        # loss = output # 2)查看预测的部分;
        # loss = output.mul(torch.where(target==1, 0, 1)) # 3)查看预测错误的部分;
        # loss = target #不能回传gt_tensor，因为此时它既没有grad，也没有grad_fn，是因为requires_grad是False
        return loss

    @staticmethod
    def get_cam_weights(grads): #GAP全局平均池化
        return np.mean(grads, axis=(2,3), keepdims=True)

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2) #这种写法很有趣，可以记录一下
        return width, height

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads) #对梯度图进行全局平均池化
        weighted_activations = weights * activations #和原特整层加权乘
        cam = weighted_activations.sum(axis=1)
        return cam

    @staticmethod
    def scale_cam_img(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img) #减去最小值
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv.resize(img, target_size) #注意：cv2.resize(src, (width, height))，width在height前，格式应注意
            result.append(img)
        result = np.float32(result)
        return result

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations] #在这里全部改为ndarray格式
        grads_list = [a.cpu().data.numpy() for a in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)
        cam_per_target_layer = []

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            # 一张一张特征图和梯度对应着处理
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam<0] = 0 #ReLU
            scaled = self.scale_cam_img(cam, target_size) #将CAM图缩放到原图大小，然后与原图叠加，这考虑到特征图可能小于或大于原图情况
            cam_per_target_layer.append(scaled[:, None, :]) # 在None标注的位置加入一个维度，相当于scaled.unsqueeze(1)
        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_layer):
        cam_per_layer = np.concatenate(cam_per_layer, axis=1) #在channels维度进行堆叠，并没有做相加的处理
        cam_per_layer = np.maximum(cam_per_layer, 0) #np.maximum：(a, b) a、b矩阵逐位比较取其大者
        result = np.mean(cam_per_layer, axis=1) #在channels维度求平均，压缩这个维度，该维度返回为1
        return self.scale_cam_img(result)

    def __call__(self, input_tensor, target): #创建该类后自动调用__call__()方法
        # 这里的target就是目标的gt（双边缘）
        if self.use_cuda:
            input_tensor = input_tensor.cuda()
        # 正向传播的输出结果，创建ActivationsAndGradients类后调用__call__()方法，执行self.model(x)
        # 注意这里的output未经softmax，所以调用这个包的时候一定要把网络结构中的最后一层激活函数给注释掉
        # 一定要注释掉！！！
        output = self.activations_and_grads(input_tensor)[0]
        _output = output.detach().cpu()
        _output=_output.squeeze(0).squeeze(0)

        self.model.zero_grad()
        # 这个loss回传是Grad-CAM文章的核心思想，源码中是分类任务，所以将未经softmax的该类预测结果作为loss反向传播，
        # 然后将回传得到的梯度A'排序，梯度最大的说明在该分类中该层特征层结构起到的作用最大，预测的部分展示出来就是整个网络预测时的注意力
        loss = self.get_loss(output, target)
        loss.backward(torch.ones_like(target), retain_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_tensor) #得到每一层的CAM，返回的是一个列表
        return self.aggregate_multi_layers(cam_per_layer) #将一整个列表的在channels维度求平均压缩，该维度处理后为1

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv.COLORMAP_JET) -> np.ndarray:
    heatmap = cv.applyColorMap(np.uint8(255 * mask), colormap) #将cam的结果转成伪彩色图片
    if use_rgb:
        heatmap = cv.cvtColor(heatmap, cv.COLOR_BGR2RGB) #使用opencv方法后，得到的一般都是BGR格式，还要转化为RGB格式
        # OpenCV中图像读入的数据格式是numpy的ndarray数据格式。是BGR格式，取值范围是[0,255].
    heatmap = np.float32(heatmap) / 255. #缩放到[0,1]之间

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255*cam)




















