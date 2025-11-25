# 数学概念：正规算子满足 $T^T = TT^$，包括自伴算子、酉算子等
# AI对应：稳定训练的权重矩阵、梯度流分析
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# 实践2：正规算子在深度学习训练稳定性中的应用
def normal_operators_training_stability():
    """
    分析正规算子性质如何影响神经网络训练的稳定性
    """
    print("=== 正规算子与训练稳定性 ===")

    # 比较不同类型的权重矩阵
    matrix_size = 100#矩阵大小
    n_iterations = 500#迭代次数

    # 创建不同类型的矩阵
    matrices = {
        '正规矩阵（正交）': torch.randn(matrix_size, matrix_size),
        '正规矩阵（对称）': torch.randn(matrix_size, matrix_size),
        '非正规矩阵': torch.randn(matrix_size, matrix_size),
        '病态非正规矩阵': torch.randn(matrix_size, matrix_size)
    }

    # 初始化矩阵使其具有特定性质
    with torch.no_grad():
        # 正交矩阵（正规）
        U, _, V = torch.svd(matrices['正规矩阵（正交）'])
        matrices['正规矩阵（正交）'] = U @ V.T

        # 对称矩阵（正规），为什么是0.5：这是平均权重，保证结果是对称的，即 $M = M^T$
        matrices['正规矩阵（对称）'] = 0.5 * (matrices['正规矩阵（对称）'] +
                                            matrices['正规矩阵（对称）'].T)

        # 病态非正规矩阵：使条件数很大且非正规
        U, S, V = torch.svd(matrices['病态非正规矩阵'])
        S[0] = 100.0  # 最大奇异值
        S[-1] = 0.01  # 最小奇异值
        matrices['病态非正规矩阵'] = U @ torch.diag(S) @ V.T
        # 添加非正规性
        '''
        为什么是0.1：
            控制非正规程度的强度
            太大会完全掩盖原矩阵特性，太小不足以产生显著的非正规性
            这里作为适中的扰动系数使用
        '''
        matrices['病态非正规矩阵'] = matrices['病态非正规矩阵'] + 0.1 * torch.randn(matrix_size, matrix_size)

    # 分析每个矩阵的性质
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))#创建子图
    # 2行4列，每个子图的大小为20x10英寸

    stability_metrics = {}#存储结果

    for idx, (name, matrix) in enumerate(matrices.items()):
        # 检查是否正规
        A = matrix
        A_star_A = A.T @ A  # 实数的共轭转置就是转置
        A_A_star = A @ A.T

        is_normal = torch.allclose(A_star_A, A_A_star, atol=1e-4)#判断矩阵是否正规

        # 计算条件数
        singular_values = torch.linalg.svdvals(A)
        '''
        计算矩阵 A 的奇异值分解（SVD）
        返回按降序排列的奇异值列表
        奇异值反映了矩阵在各个方向上的"拉伸"程度
        '''
        condition_number = singular_values[0] / singular_values[-1]
        '''
        计算矩阵的条件数
        使用最大奇异值除以最小奇异值
        条件数衡量矩阵的数值稳定性
        数学意义
            条件数小: 矩阵是良态的（well-conditioned），数值计算稳定
            条件数大: 矩阵是病态的（ill-conditioned），数值计算不稳定
            极端情况: 当最小奇异值为0时，条件数为无穷大，矩阵不可逆
        '''
        #这个条件数用于评估不同类型矩阵的数值稳定性，帮助理解正规矩阵与非正规矩阵在深度学习训练中的稳定性差异。条件数越大，表示矩阵越不稳定，在迭代计算中可能导致梯度爆炸或消失等问题。
        print(f"\n{name}:")
        print(f"  是否正规: {is_normal}")
        print(f"  条件数: {condition_number:.2f}")

        # 模拟矩阵幂的稳定性（类似于深度网络的前向传播）
        vector = torch.randn(matrix_size)
        vector = vector / torch.norm(vector)  # 归一化

        powers_norms = []#存储矩阵幂的范数
        for power in range(1, n_iterations + 1):
            powered_vector = torch.matrix_power(A, power) @ vector#计算矩阵的第power次幂，这个操作模拟了矩阵在多次变换下对向量的影响
            powered_norm = torch.norm(powered_vector)#计算变换后向量的欧几里得范数(长度)
            powers_norms.append(powered_norm.item())#添加到列表中
            ''''
            这部分代码模拟了神经网络中深层传播的数学过程：
                矩阵的幂运算对应于多层网络的前向传播
                向量范数的变化反映了信号在传播过程中的放大或衰减情况
                通过观察不同矩阵（正规/非正规）的范数变化趋势，可以评估训练稳定性
                正规矩阵通常表现出更稳定的范数变化，而非正规矩阵可能导致梯度爆炸或消失
            '''

        stability_metrics[name] = {
            'is_normal': is_normal,
            'condition_number': condition_number.item(),
            #这行代码计算的是变异系数(Coefficient of Variation)，用于衡量矩阵幂稳定性的一个重要指标。
            # 变异系数 = 标准差 / 均值
            # 用于衡量数据分布的离散程度与中心趋势的关系
            # 较大值表示数据分布较为分散，较小值表示数据分布较为集中
            'power_stability': np.std(powers_norms) / np.mean(powers_norms)
        }

        # 可视化矩阵性质
        # 矩阵热图
        ax1 = axes[0, idx]
        im = ax1.imshow(A.numpy(), cmap='RdBu_r', aspect='auto')
        ax1.set_title(f'{name}\n正规: {is_normal}\n条件数: {condition_number:.1f}')
        plt.colorbar(im, ax=ax1)

        # 矩阵幂的稳定性
        ax2 = axes[1, idx]
        ax2.plot(range(1, n_iterations + 1), powers_norms)
        ax2.set_xlabel('幂次')
        ax2.set_ylabel('向量范数')
        ax2.set_title(f'矩阵幂稳定性\n变异系数: {stability_metrics[name]["power_stability"]:.4f}')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 在神经网络训练中的应用
    print(f"\n正规性在神经网络训练中的意义:")

    class NormalidadRegularizedNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = torch.tanh(self.fc1(x))#将输入x通过第一个全连接层fc1，应用双曲正切激活函数tanh
            x = torch.tanh(self.fc2(x))#将输入x通过第二个全连接层fc2，应用双曲正切激活函数tanh
            return self.fc3(x)#最后通过第三个全连接层fc3，不使用激活函数，直接返回输出（适用于分类任务的最后一层）
            # 其中最后一层不加激活函数是为了配合后续的损失函数（如交叉熵损失）。

        def normality_regularization(self, lambda_norm=0.001):
            """正规性正则化：鼓励权重矩阵接近正规"""
            reg_loss = 0
            for layer in [self.fc1, self.fc2]:
                W = layer.weight
                W_T_W = W.T @ W
                W_W_T = W @ W.T
                # 正规性损失：||W*W - WW*||_F
                # 确保两个矩阵维度相同
                if W_T_W.shape == W_W_T.shape:
                    normality_loss = torch.norm(W_T_W - W_W_T, p='fro') ** 2
                else:
                    # 如果维度不同，使用 Frobenius 范数的平方作为替代
                    normality_loss = torch.norm(W_T_W, p='fro') ** 2 + torch.norm(W_W_T, p='fro') ** 2
                reg_loss += normality_loss
            return lambda_norm * reg_loss

    # 训练比较
    torch.manual_seed(42)#设置随机数种子，保证每次运行结果一致
    input_size, hidden_size, output_size = 20, 50, 2
    n_samples = 1000

    # 生成数据
    X_train = torch.randn(n_samples, input_size)
    y_train = (X_train.sum(dim=1) > 0).long()

    models = {}#存储模型
    training_curves = {}#存储训练曲线

    for reg_strength in [0.0, 0.001, 0.01]:
        model = NormalidadRegularizedNet(input_size, hidden_size, output_size)
        optimizer = optim.Adam(model.parameters(), lr=0.01)#优化器使用 Adam 算法，学习率为 0.01
        criterion = nn.CrossEntropyLoss()#损失函数使用 交叉熵损失，用于多分类任务

        losses = []#存储损失
        normality_scores = []#存储正规性分数

        for epoch in range(100):
            optimizer.zero_grad()

            outputs = model(X_train)
            main_loss = criterion(outputs, y_train)

            # 添加正规性正则化
            norm_reg = model.normality_regularization(reg_strength)
            total_loss = main_loss + norm_reg#将主要损失和正交损失相加，得到总损失

            total_loss.backward()
            optimizer.step()

            losses.append(total_loss.item())#添加到列表中

            # 计算当前的正规性分数
            with torch.no_grad():
                normality_score = 0
                for layer in [model.fc1, model.fc2]:
                    W = layer.weight
                    W_T_W = W.T @ W
                    W_W_T = W @ W.T

                    # 检查维度是否一致
                    if W_T_W.shape == W_W_T.shape:
                        # 维度一致时，计算正规性损失
                        normality_loss = torch.norm(W_T_W - W_W_T, p='fro')
                    else:
                        # 维度不一致时，分别计算Frobenius范数并求和
                        normality_loss = torch.norm(W_T_W, p='fro') + torch.norm(W_W_T, p='fro')

                    #normality_score += torch.norm(W_T_W - W_W_T, p='fro').item()
                    normality_score+=normality_loss.item()
                normality_scores.append(normality_score)

        models[reg_strength] = model #将模型添加到字典中
        training_curves[reg_strength] = {
            'losses': losses,
            'normality_scores': normality_scores
        }

    # 可视化训练结果
    plt.figure(figsize=(15, 5))

    # 损失曲线
    plt.subplot(1, 3, 1)
    for reg_strength, curves in training_curves.items():
        plt.plot(curves['losses'], label=f'λ={reg_strength}')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('训练损失曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 正规性分数
    plt.subplot(1, 3, 2)
    for reg_strength, curves in training_curves.items():
        plt.plot(curves['normality_scores'], label=f'λ={reg_strength}')
    plt.xlabel('Epoch')
    plt.ylabel('Normality Score')
    plt.title('权重矩阵正规性')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 最终权重矩阵的奇异值分布
    plt.subplot(1, 3, 3)
    for reg_strength, model in models.items():
        W = model.fc1.weight.detach()
        singular_values = torch.linalg.svdvals(W)
        plt.plot(singular_values.numpy(), 'o-', label=f'λ={reg_strength}', alpha=0.7)
    plt.xlabel('奇异值索引')
    plt.ylabel('奇异值大小')
    plt.title('权重矩阵奇异值分布')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n正规性正则化效果总结:")
    print("λ=0.0: 无约束，权重可能偏离正规性，训练可能不稳定")
    print("λ=0.001: 适度约束，保持较好的训练稳定性和正规性")
    print("λ=0.01: 强约束，可能过度限制模型表达能力")


normal_operators_training_stability()