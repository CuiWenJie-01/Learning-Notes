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
    '''
    torchvision.datasets.MNIST: MNIST 数据集加载器
    train=True: 加载训练集（而非测试集）
    download=True: 如果本地没有数据则自动下载
    transform=transform: 应用之前定义的数据预处理操作
    '''
    trainset=torchvision.datasets.MNIST(root='./data',train=True,
                                        download=True,transform=transform)
    '''
    torch.utils.data.DataLoader: 数据加载器
    trainset: 传入已加载的数据集
    batch_size=64: 每批处理64张图像
    shuffle=True: 每个epoch随机打乱数据顺序
    '''
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=64,
                                            shuffle=True)
    #trainloader 可以批量提供经过预处理的 MNIST 训练数据，每次迭代返回64张随机打乱的图像及其对应标签，供后续训练使用。

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

    #创建了模型实例并配置了训练所需的损失函数和优化器
    model=LinearClassifier()#模型实例化:创建 LinearClassifier 类的实例，即实际的线性分类器模型
    '''
    损失函数
        criterion=nn.CrossEntropyLoss(): 定义交叉熵损失函数
        适用于多分类任务
        结合了 nn.LogSoftmax() 和 nn.NLLLoss()
        用于计算模型预测结果与真实标签之间的损失
    '''
    criterion=nn.CrossEntropyLoss()
    '''
    优化器
        optimizer=torch.optim.SGD(model.parameters(),lr=0.01): 定义随机梯度下降优化器
        model.parameters(): 传入模型的所有可训练参数
        lr=0.01: 设置学习率为0.01
        用于更新模型参数以最小化损失函数
    '''
    optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
    #这三行代码完成了模型训练前的必要配置，为后续的训练循环做好准备。

    print("=== 训练线性MNIST分类器 ===")
    print(f"模型结构: 784维输入 → 线性映射 → 10维输出")
    '''
    model.parameters() 返回模型的所有可训练参数
    p.numel() 计算每个参数张量中的元素个数
    sum(...) 将所有参数的数量累加，从而得到总参数量。
    '''
    print(f"参数数量：{sum(p.numel() for p in model.parameters())}")

    #训练循环
    '''
    for epoch in range(5): 表示整个训练过程将重复进行5个epoch
    epoch: 遍历完整个训练数据集一次称为一个epoch
    这里设置为5，意味着模型将看到完整的MNIST训练数据集5次
    多个epoch有助于模型更好地学习数据中的模式
    在每个epoch中，模型会遍历所有的训练数据批次，通过前向传播、计算损失、反向传播和参数更新来逐步优化模型性能。
    '''
    for epoch in range(5):
        running_loss=0.0
        '''
        在每个新的epoch开始时，将累积损失重置为0.0
        在训练过程中，每个批次的损失会被累加到这个变量中
        通过这种方式可以计算整个epoch的平均损失，用于监控训练进度和模型性能
        '''
        runing_loss=0.0#声明并初始化一个浮点型变量，用于累计当前epoch中所有批次的损失值
        '''
        for i,(images,labels) in enumerate(trainloader,0): 遍历 trainloader 数据加载器
        enumerate(trainloader,0) 为每个迭代项提供索引，从0开始计数
        images：当前批次的图像数据，形状为 [batch_size, 1, 28, 28]
        labels：当前批次对应的标签数据，形状为 [batch_size]，包含每个图像的真实数字类别
        通过这种方式，模型可以逐批次处理训练数据，每次处理64张图像（根据之前设置的 batch_size=64）
        '''
        for i,(images,labels) in enumerate(trainloader,0):
            #前向传播
            '''
            调用 LinearClassifier 模型的 forward 方法
            输入 images 经过模型处理，输出形状为 [batch_size, 10] 的预测结果
            每个样本对应10个类别的得分
            '''
            outputs=model(images)
            '''
            使用 CrossEntropyLoss 计算预测结果 outputs 与真实标签 labels 之间的交叉熵损失
            损失值反映了模型当前预测的准确性，值越小表示预测越准确
            这个损失值将用于后续的反向传播和参数更新
            '''
            loss=criterion(outputs,labels)


            '''
            这是PyTorch中训练神经网络时的标准反向传播和参数更新步骤，让我逐行解释：
            1. optimizer.zero_grad()
            作用: 清零梯度
            原理: 在每次反向传播之前，需要将模型参数的梯度清零，因为PyTorch会累积梯度而不是自动清零
            必要性: 如果不清零，梯度会在多个batch之间累积，导致训练不稳定
            2. loss.backward()
            作用: 执行反向传播计算梯度
            原理: 从损失函数开始，通过链式法则自动计算网络中每个参数相对于损失函数的梯度
            结果: 每个可训练参数都会获得相应的梯度值，存储在.grad属性中
            3. optimizer.step()
            作用: 更新模型参数
            原理: 根据之前计算得到的梯度，使用优化器(这里是SGD)更新模型参数
            公式: 对于SGD，参数更新公式为: param = param - learning_rate * gradient
            这三行代码构成了深度学习训练的核心循环，通过不断迭代使模型逐渐学习到数据中的模式。
            '''
            #反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            '''
            1. running_loss+=loss.item()
            作用: 累积当前批次的损失值
            原理: 将当前 loss（tensor类型）通过 item() 方法转换为Python数值，并累加到 running_loss 中
            目的: 用于计算平均每100个批次的损失值
            2. if i%100==99:
            作用: 每100个批次输出一次训练状态
            条件解释: 当批次索引 i 除以100的余数等于99时执行（即第100、200、300...批次）
            注意: 因为 enumerate 从0开始计数，所以第100个批次的索引是99
            3. print(f'Epoch[{epoch+1}/5],Step[{i+1}],Loss:{running_loss/100:.4f}')
            作用: 输出训练进度和平均损失
            格式说明:
            Epoch[{epoch+1}/5]: 显示当前是第几个epoch（从1开始计数）
            Step[{i+1}]: 显示当前是第几个批次（从1开始计数）
            Loss:{running_loss/100:.4f}: 显示最近100个批次的平均损失，保留4位小数
            4. running_loss=0.0
            作用: 重置累积损失，为下一个100批次的平均损失计算做准备
            5. print("训练完成！")
            作用: 在每个epoch结束后输出提示信息
            这样设计可以让开发者实时监控模型训练过程中的损失变化，判断模型是否在正常学习。
            '''
            running_loss+=loss.item()
            if i%100==99:
                print(f'Epoch[{epoch+1}/5],Step[{i+1}],Loss:{running_loss/100:.4f}')
                running_loss=0.0
        print("训练完成！")


        '''
        ### 1. `weight_matrix=model.linear.weight.data`
        - **作用**: 获取线性层的权重数据
        - **说明**: 从 `LinearClassifier` 模型中提取 `linear` 层的权重参数，`.weight.data` 返回实际的权重张量
        
        ### 2. `print(f"\n学习到的权重矩形形状：{weight_matrix.shape}")`
        - **作用**: 输出权重矩阵的形状信息
        - **预期输出**: `torch.Size([10, 784])`，表示10个输出类别，每个类别对应784个输入特征（28×28像素）
        
        ### 3. `print(f"权重范数：\n{torch.norm(weight_matrix):.4f}")`
        - **作用**: 计算并输出权重矩阵的范数（模长）
        - **意义**: 
          - 范数大小反映了权重的整体规模
          - 可用于判断模型是否出现梯度爆炸或梯度消失问题
          - 较大的范数可能表明模型过拟合风险
        
        ## 补充说明
        
        从代码上下文看，这部分应该是在训练循环结束后添加的代码，用于分析训练后模型参数的特征。通过检查权重矩阵的形状和范数，可以了解：
        - 模型参数是否正确初始化和更新
        - 权重值的规模是否在合理范围内
        - 模型的学习效果如何
        '''
        #分析学习到线性映射
        weight_matrix=model.linear.weight.data
        print(f"\n学习到的权重矩形形状：{weight_matrix.shape}")
        print(f"权重范数：\n{torch.norm(weight_matrix):.4f}")

        '''
        这段代码用于可视化线性分类器学习到的权重模板，帮助理解模型是如何识别不同数字的：
        可视化代码解释
        1. fig,axes=plt.subplots(2,5,figsize=(12,5))
        作用: 创建一个2行5列的子图布局
        目的: 为10个数字类别(0-9)分别创建一个显示位置
        尺寸: 整体图形大小为12×5英寸
        2. for i in range(10):
        作用: 遍历10个输出类别(数字0-9)
        处理: 为每个数字类别可视化其对应的权重模板
        3. ax=axes[i//5,i%5]
        作用: 计算当前数字应放置在哪个子图位置
        逻辑:
        i//5 计算行号(0或1)
        i%5 计算列号(0-4)
        4. digit_template=weight_matrix[i].view(28,28)
        作用: 将第i个类别的权重向量重塑为28×28的图像格式
        意义: 权重向量原本是784维，重塑后可以作为图像显示
        5. ax.imshow(digit_template,cmap='RdBu_r')
        作用: 在子图中显示数字模板图像
        颜色映射: 使用'RedBu_r'色彩方案(红蓝反转)，突出正负权重差异
        6. ax.set_title(f'数字{i}的模版')
        作用: 为每个子图设置标题，标明对应数字
        7. ax.axis('off')
        作用: 关闭坐标轴显示，使图像更清晰
        8. 后续设置
        plt.suptitle(...): 添加整体标题
        plt.tight_layout(): 自动调整子图间距
        plt.show(): 显示图形
        这种可视化展示了线性分类器学到的"理想数字模板"，每个模板代表模型认为该数字应该具有的像素模式。
        '''
        #可视化权重（每个输出神经元对应的模版）
        fig,axes=plt.subplots(2,5,figsize=(12,5))
        for i in range(10):
            ax=axes[i//5,i%5]
            digit_template=weight_matrix[i].view(28,28)
            ax.imshow(digit_template,cmap='RdBu_r')
            ax.set_title(f'数字{i}的模版')
            ax.axis('off')
        plt.suptitle('线性分类器学习到的数字模版',fontsize=16)
        plt.tight_layout()
        plt.show()

        return model
#运行MNIST分类器
model=linear_mnist_classifier()
