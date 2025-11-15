# 数学概念：特征值反映矩阵变换的稳定性
# AI对应：梯度下降的收敛性、训练稳定性
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def analyze_neural_network_stability():
    """
    分析神经网络权重矩阵的特征值与训练动态的关系
    """
    #创建不同初始化的神经网络层
    layers={
        '好的初始化（Xavier）':nn.Linear(100,100),
        '大的权重':nn.Linear(100,100),
        '病态矩阵（接近奇异）':nn.Linear(100,100)
    }

    #不同的初始化策略
    nn.init.xavier_normal_(layers['好的初始化（Xavier）'].weight)#这是一种标准的神经网络权重初始化方法，能够保持前向传播和反向传播时方差的稳定
    nn.init.constant_(layers['大的权重'].weight, 0.1) # 将所有权重初始化为较大的常数值0.1

    #创建病态矩阵：使矩阵接近奇异
    W_sick=torch.randn(100,100)# 生成一个100x100的随机矩阵，元素服从标准正态分布(均值为0，标准差为1)
    '''
    U: 左奇异向量矩阵 - 包含矩阵 W_sick 的左奇异向量，是一个正交矩阵
    S: 奇异值向量 - 包含 W_sick 的奇异值，按降序排列的1D张量
    V: 右奇异向量矩阵 - 包含矩阵 W_sick 的右奇异向量，也是一个正交矩阵
    '''
    U,S,V=torch.svd(W_sick)# 执行SVD分解
    S[-50:]=0.01# 将后50个较小的奇异值设为极小值
    # 重构病态矩阵
    layers['病态矩阵（接近奇异）'].weight.data=U@torch.diag(S)@V.T

    '''
    fig: 图形对象，代表整个绘图窗口
    axes: 子图数组，包含多个坐标轴对象用于绘制不同图表
    plt.subplots(2, 3): 创建2行3列的子图网格，总共6个子图位置
    figsize=(15, 10): 设置图形大小为15英寸宽、10英寸高
    '''
    fig,axes=plt.subplots(2,3,figsize=(15,10))

    for idx,(name,layer) in enumerate(layers.items()):
        W=layer.weight.data
        '''
        特征值反映了矩阵变换的重要性质：
        绝对值大于1: 表示该方向上的变换会放大
        绝对值小于1: 表示该方向上的变换会缩小
        接近零: 表示矩阵接近奇异，可能导致数值不稳定
        分布范围过大: 表示矩阵条件数大，属于病态矩阵
        在神经网络中，权重矩阵的特征值分布直接影响：
        梯度传播的稳定性
        训练收敛速度
        模型的泛化能力
        通过分析不同初始化策略下特征值的分布情况，可以评估各种初始化方法对神经网络训练稳定性的影响。
        '''
        eigenvalues=torch.linalg.eigvals(W).real#torch.linalg.eigvals(W): 使用PyTorch的线性代数库计算矩阵 W 的所有特征值，.real: 取特征值的实部部分

        # 计算条件数(最大特征值/最小特征值的绝对值)
        cond_number=torch.max(eigenvalues)/torch.min(torch.abs(eigenvalues))

        print(f"\n{name}")
        print(f"  权重矩阵形状：{W.shape}")
        print(f"  特征值范围：[{eigenvalues.min():.3f},{eigenvalues.max():.3f}]")
        print(f"  条件数：{cond_number:.3f}")

        # 特征值分布直方图
        ax1 = axes[0, idx]
        ax1.hist(eigenvalues.numpy(), bins=50, alpha=0.7)
        ax1.set_title(f'{name}\n条件数: {cond_number:.1f}')
        ax1.set_xlabel('特征值')
        ax1.set_ylabel('频数')
        ax1.grid(True, alpha=0.3)

        # 特征值排序图
        ax2 = axes[1, idx]
        sorted_eigenvalues = torch.sort(eigenvalues).values.numpy()
        ax2.plot(range(len(sorted_eigenvalues)), sorted_eigenvalues, 'o-')
        ax2.set_xlabel('排序索引')
        ax2.set_ylabel('特征值大小')
        ax2.set_title('特征值谱')
        ax2.grid(True, alpha=0.3)

        #训练稳定性分析
        if cond_number>1000:
            print("  ⚠️ 高条件数：训练可能不稳定，梯度可能爆炸或消失")
        elif cond_number<10:
            print("  ✅ 良好条件数：训练应该稳定")
        else:
            print("  ⚠️ 中等条件数：需要注意学习率设置")
    plt.tight_layout()
    plt.show()

analyze_neural_network_stability()



