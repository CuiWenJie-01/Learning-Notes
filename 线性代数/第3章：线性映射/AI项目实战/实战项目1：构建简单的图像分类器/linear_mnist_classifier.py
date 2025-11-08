#用线性映射构建一个真实的图像分类器
'''
torchvision
这是 PyTorch 的计算机视觉库
主要用于处理图像和视觉任务相关的数据集、模型和转换
在这个项目中，它被用来加载 MNIST 数据集
torchvision.transforms
这是 torchvision 中的一个模块
提供了常见的图像变换操作
可以对图像进行预处理，如调整大小、裁剪、归一化等
在这个项目中，它被用来将图像转换为张量并进行标准化处理
'''
try:
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
except ImportError:
    print("这些库未安装，请先安装：pip install torch torchvision matplotlib")

# 实战项目：MNIST手写数字分类（仅使用线性映射）
def linear_mnist_classifier():
    """
    使用纯线性映射（无隐藏层）进行MNIST分类
    """
    #数据预处理
    '''
    代码解释
    transforms.Compose([...]): 将多个图像变换操作组合成一个序列，按顺序执行
    transforms.ToTensor(): 将 PIL 图像或 NumPy 数组格式的图像转换为 PyTorch 张量格式，并将像素值从 [0, 255] 范围缩放到 [0, 1] 范围
    transforms.Normalize((0.5,), (0.5,)): 对图像张量进行标准化处理
        第一个 (0.5,) 是均值
        第二个 (0.5,) 是标准差
        对于灰度图像，这个操作将像素值从 [0, 1] 映射到 [-1, 1] 范围
    作用
    这个 transform 配置会在加载 MNIST 数据集时自动应用到每张图像上，确保所有输入数据都经过统一的预处理，为后续的线性分类器提供标准化的输入。
    '''
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])

    #加载数据
    trainset=torchvision.datasets.MNIST(root='./data',train=True,
                                        download=True,transform=transform)
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=64,
                                            shuffle=True)

    #定义模型：单个线性层（784输入->10输出）
    class LinearClassifier(nn.Module):#继承自nn.Module，这是PyTorch中所有神经网络模块的基类
        def __init__(self):#初始化模型
            super().__init__()#调用父类 nn.Module 的初始化方法
            self.linear=nn.Linear(28*28,10)# 创建一个线性层，将 28×28=784 个输入特征映射到 10 个输出类别（对应数字0-9）

        #前向传播函数
        def forward(self,x):# 定义前向传播过程
            x=x.view(x.size(0),-1)#将输入图像展平为一维向量，x.size(0) 保持批次大小不变，-1 自动计算剩余维度
            return self.linear(x)#对展平后的数据应用线性变换，得到分类结果
        '''
        这个模型实现了最简单的线性分类器，没有隐藏层，直接将图像像素映射到类别概率。
        '''

