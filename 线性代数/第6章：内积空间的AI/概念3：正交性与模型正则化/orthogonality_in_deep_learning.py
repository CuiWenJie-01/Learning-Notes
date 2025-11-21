# 数学概念：正交向量内积为零，正交矩阵保持向量长度和角度
# AI对应：正交初始化、正交正则化、防止梯度爆炸/消失
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# 实践3：正交性在深度学习中的应用
def orthogonality_in_deep_learning():
    """
    探索正交性在神经网络中的重要作用
    """
    print("=== 正交性在深度学习中的应用 ===")

    # 1. 正交初始化
    def analyze_orthogonal_initialization():
        print("\n1. 正交初始化分析:")

        #在实际应用中，这可能代表全连接层（fully connected layers）或者卷积层的输出通道数。
        layer_sizes=[128,256,512]#定义了一个列表 layer_sizes，其中包含了三个不同的神经网络层大小：128、256 和 512 个神经元。

        fig,axes=plt.subplots(2,3,figsize=(15,10))#创建了 2 行 3 列共 6 个子图（axes），整个图表尺寸为宽 15 英寸、高 10 英寸
        #这些代码通常会出现在分析不同初始化策略对神经网络训练影响的情景中

        for i,size in enumerate(layer_sizes):#获取索引 i 和值 size
            # 随机初始化
            W_random=torch.rand(size,size)
            # 正交初始化
            #正交初始化能够保持输入信号的范数不变，有助于缓解深度神经网络中的梯度消失/爆炸问题
            W_ortho=torch.empty(size,size)
            torch.nn.init.orthogonal_(W_ortho)#通过 torch.nn.init.orthogonal_ 函数将矩阵初始化为正交矩阵

            # 分析性质
            # 随机初始化
            WWT_random=W_random@W_random.T#计算随机初始化矩阵 W_random 的转置与自身的乘积，得到 WWT_random
            identity=torch.eye(size)#创建一个单位矩阵 identity，大小为 size x size
            random_deviation=torch.norm(WWT_random-identity)#计算随机初始化矩阵 WWT_random 与单位矩阵 identity 之间的范数，得到随机初始化的偏差 random_deviation

            # 正交初始化
            WWT_ortho=W_ortho@W_ortho.T#计算正交初始化矩阵 W_ortho 的转置与自身的乘积，得到 WWT_ortho
            ortho_deviation=torch.norm(WWT_ortho-identity)#计算正交初始化矩阵 WWT_ortho 与单位矩阵 identity 之间的范数，得到正交初始化的偏差 ortho_deviation

            print(f"尺寸 {size}×{size}:")
            print(f"  随机初始化 - 与单位矩阵偏差: {random_deviation:.4f}")
            print(f"  正交初始化 - 与单位矩阵偏差: {ortho_deviation:.4f}")

            # 特征值分析
            eigvals_random=torch.linalg.eigvals(WWT_random).abs()#计算随机初始化矩阵 WWT_random 的特征值，并取绝对值，得到 eigvals_random
            eigvals_ortho=torch.linalg.eigvals(WWT_ortho).abs()#计算正交初始化矩阵 WWT_ortho 的特征值，并取绝对值，得到 eigvals_ortho

            # print(f"  随机初始化 - 特征值范围: [{eigvals_random.min():.4f}, {eigvals_random.max():.4f}]")
            # print(f"  正交初始化 - 特征值范围: [{eigvals_ortho.min():.4f}, {eigvals_ortho.max():.4f}]")

            # 可视化特征值分布
            axes[0, i].hist(eigvals_random.numpy(), bins=20, alpha=0.7, label='随机初始化')
            axes[0, i].hist(eigvals_ortho.numpy(), bins=20, alpha=0.7, label='正交初始化')
            axes[0, i].set_xlabel('特征值模长')
            axes[0, i].set_ylabel('频数')
            axes[0, i].set_title(f'权重矩阵特征值分布\n({size}×{size})')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)

            # 可视化WWT
            axes[1, i].imshow(WWT_ortho.numpy(), cmap='RdBu_r', aspect='auto')
            axes[1, i].set_title(f'正交初始化 W @ W^T\n(近似单位矩阵)')
            axes[1, i].set_xlabel('列索引')
            axes[1, i].set_ylabel('行索引')

        plt.tight_layout()
        plt.show()

    #analyze_orthogonal_initialization()

    # 2. 正交正则化
    def orthogonal_regularization_demo():
        print("\n2. 正交正则化演示:")

        class OrthogonalRegularizedNet(nn.Module):#实现了带有正交正则化的神经网络
            def __init__(self,input_size=100,hidden_size=100,output_size=10):
                super().__init__()#调用父类 nn.Module 的构造函数，完成PyTorch模块的初始化
                self.fc1=nn.Linear(input_size,hidden_size)#创建第一个全连接层，将输入维度 input_size 映射到隐藏层维度 hidden_size
                self.fc2=nn.Linear(hidden_size, hidden_size)#创建第二个全连接层，保持隐藏层维度不变
                self.fc3=nn.Linear(hidden_size, output_size)#创建第三个全连接层，将隐藏层维度 hidden_size 映射到输出维度

            #定义了前向传播方法 forward，用于计算神经网络的输出
            #在 forward 方法中，输入 x 首先通过第一个全连接层 fc1，然后通过 ReLU 激活函数，接着通过第二个全连接层 fc2 再次通过 ReLU 激活函数，最后通过第三个全连接层 fc3 输出结果
            def forward(self,x):#定义前向传播函数，接收输入张量 x
                x=torch.relu(self.fc1(x))#将输入 x 通过第一个全连接层 fc1，然后应用 ReLU 激活函数
                x=torch.relu(self.fc2(x))#将结果通过第二个全连接层 fc2，再次应用 ReLU 激活函数
                return self.fc3(x)#最后通过第三个全连接层 fc3，并返回结果（注意这里没有激活函数，通常用于输出层）
            #这是一个典型的前馈神经网络结构，包含两个隐藏层，每层都使用 ReLU 激活函数，除了最后一层作为输出层。

            def orthogonal_regularization(self,lambda_ortho=0.001):#接收正则化系数 lambda_ortho 作为参数，默认值为 0.001
                """计算正交正则化损失"""
                reg_loss=0#初始化正则化损失为 0
                for layer in [self.fc1,self.fc2,self.fc3]:#遍历网络中的所有全连接层 (fc1, fc2, fc3)
                    W=layer.weight#获取当前层的权重矩阵 W
                    WWT=torch.mm(W,W.T)#计算权重矩阵与其转置的乘积 W @ W^T
                    I=torch.eye(WWT.size(0),device=WWT.device)#创建与 WWT 同维度的单位矩阵 I
                    reg_loss+=torch.norm(WWT-I,p='fro')**2#计算 WWT 与单位矩阵 I 的弗罗贝尼乌斯范数的平方，并累加到总损失
                return lambda_ortho*reg_loss#返回加权后的正则化损失
            #该方法通过强制权重矩阵接近正交矩阵来实现正交正则化，有助于改善网络训练的稳定性。

        # 创建网络实例，这个 model 对象是一个具有正交正则化功能的神经网络模型，包含三层全连接层结构。
        '''
        使用了类的默认参数：
            input_size=100
            hidden_size=100
            output_size=10
        '''
        model=OrthogonalRegularizedNet()

        # 模拟训练过程
        n_epochs=100#设置训练轮数为100轮
        '''
        0.0：无正则化（基准对照）
        0.001：弱正则化
        0.01：强正则化
        '''
        reg_strengths=[0.0,0.001,0.01]
        #这样设置是为了对比不同正则化强度对模型训练效果的影响。

        plt.figure(figsize=(15,5))

        for reg_strength in reg_strengths:
            # 重置模型，在循环中为不同正则化强度的实验重置模型和优化器
            model=OrthogonalRegularizedNet()#重置模型，确保每次训练都是从相同的初始状态开始
            optimizer=torch.optim.Adam(model.parameters(),lr=0.001)#) 创建新的Adam优化器，绑定到当前模型参数，学习率设为0.001
            #这样的重置机制保证了不同正则化强度下的对比实验具有公平性，避免了模型参数在不同实验间的相互影响。

            # 跟踪正交性度量
            orthogonality_measures=[]#初始化一个空列表，用于存储每个训练轮次的正交性度量值

            for epoch in range(n_epochs):
                # 模拟训练步骤
                optimizer.zero_grad()#清零梯度

                # 模拟输入和损失
                dummy_input=torch.randn(32,100)#创建一个随机输入张量，维度为 32×100
                dummy_target=torch.randn(32,10)#创建一个随机目标张量，维度为 32×10

                output=model(dummy_input)#通过模型计算输出
                main_loss=F.mse_loss(output,dummy_target)#计算均方误差损失，将模型输出与目标张量 dummy_target 进行比较

                # 添加正交正则化
                ortho_loss=model.orthogonal_regularization(reg_strength)#计算正交正则化损失，使用当前正则化强度 reg_strength
                total_loss=main_loss+ortho_loss#将主要损失和正交损失相加，得到总损失

                total_loss.backward()#反向传播，计算梯度
                optimizer.step()#更新模型参数

                # 计算当前的正交性度量
                with torch.no_grad():
                    W=model.fc2.weight#获取第二个全连接层 fc2 的权重矩阵 W
                    WWT=torch.mm(W,W.T)#计算权重矩阵 W 与其转置的乘积 W @ W^T
                    I=torch.eye(WWT.size(0))#创建与 WWT 同维度的单位矩阵 I
                    ortho_measure=torch.norm(WWT-I,p='fro')#计算 WWT 与单位矩阵 I 的弗罗贝尼乌斯范数，作为当前的正交性度量
                    orthogonality_measures.append(ortho_measure)#将当前正交性度量值添加到列表中

            plt.plot(orthogonality_measures,
                     label=f'正则化强度 λ={reg_strength}')

        plt.xlabel('训练步数')
        plt.ylabel('正交性偏差 (||W@W^T - I||)')
        plt.title('正交正则化对权重矩阵的影响')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        print("正交正则化效果:")
        print("λ=0.0: 无约束，权重可能偏离正交性")
        print("λ=0.001: 适度约束，保持较好的正交性")
        print("λ=0.01: 强约束，强制权重接近正交")

    #orthogonal_regularization_demo()

    # 3. 正交性在RNN中的重要性
    def orthogonality_in_RNN():
        print("\n3. RNN中的正交性:")

        # 创建RNN单元
        input_size=32#输入层向量的维数
        hidden_size=64#隐藏层向量的维数

        # 普通RNN vs 正交初始化的RNN
        rnn_regular=nn.RNNCell(input_size,hidden_size)#创建一个普通的RNN单元，使用默认的权重初始化方式
        rnn_ortho=nn.RNNCell(input_size,hidden_size)#创建另一个RNN单元，后续可能会使用正交初始化

        # 对正交RNN使用正交初始化
        nn.init.orthogonal_(rnn_ortho.weight_hh)

        # 模拟RNN展开（多个时间步）
        n_timesteps=50 #设置展开的步数
        hidden_regular=torch.zeros(hidden_size)#初始化一个隐藏向量，维数为 hidden_size
        hidden_ortho=torch.zeros(hidden_size)#初始化另一个隐藏向量，维数为 hidden_size

        hidden_norms_regular=[]#初始化一个空列表，用于存储每个时间步的普通RNN隐藏向量的范数
        hidden_norms_ortho=[]#初始化一个空列表，用于存储每个时间步的正交初始化RNN隐藏向量的范数

        for t in range(n_timesteps):
            # 随机输入
            input_t=torch.randn(input_size)#创建一个随机输入向量，维度为 input_size

            # 普通RNN前向传播
            hidden_regular=rnn_regular(input_t,hidden_regular)#将当前时间步的输入向量 input_t 和上一个时间步的隐藏向量 hidden_regular 输入到普通RNN中，更新当前时间步的隐藏向量 hidden_regular
            hidden_ortho=rnn_ortho(input_t,hidden_ortho)#将当前时间步的输入向量 input_t 和上一个时间步的隐藏向量 hidden_ortho 输入到正交初始化RNN中，更新当前时间步的隐藏向量 hidden_ortho

            # 记录隐藏向量范数
            hidden_norms_regular.append(torch.norm(hidden_regular).item())#将当前时间步的普通RNN隐藏向量的范数添加到列表中
            hidden_norms_ortho.append(torch.norm(hidden_ortho).item())#将当前时间步的正交初始化RNN隐藏向量的范数添加到列表中

        plt.figure(figsize=(10, 6))
        plt.plot(hidden_norms_regular, label='普通RNN', linewidth=2)
        plt.plot(hidden_norms_ortho, label='正交初始化RNN', linewidth=2)
        plt.xlabel('时间步')
        plt.ylabel('隐藏状态范数')
        plt.title('RNN隐藏状态范数随时间变化')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        print("RNN正交初始化分析:")
        print("普通RNN: 隐藏状态范数可能指数增长或消失（梯度爆炸/消失）")
        print("正交初始化RNN: 保持隐藏状态范数相对稳定")

        # 分析隐藏状态的变化
        final_change_regular=abs(hidden_norms_regular[-1]-hidden_norms_regular[0])#计算普通RNN隐藏状态范数在最后一个时间步与第一个时间步的变化量
        final_change_ortho=abs(hidden_norms_ortho[-1]-hidden_norms_ortho[0])#计算正交初始化RNN隐藏状态范数在最后一个时间步与第一个时间步的变化量

        print(f"普通RNN范数总变化: {final_change_regular:.4f}")
        print(f"正交RNN范数总变化: {final_change_ortho:.4f}")

    orthogonality_in_RNN()
orthogonality_in_deep_learning()

