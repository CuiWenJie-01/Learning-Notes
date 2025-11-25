# 数学概念：自伴算子满足 $T^* = T$，在实数域中对应对称矩阵
# AI对应：协方差矩阵、Hessian矩阵、对称权重矩阵

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA


# 实践1：自伴算子在协方差分析和PCA中的应用
def self_adjoint_operators_in_ai():
    """
    自伴算子在AI中的核心应用：协方差分析和谱分解
    """
    print("=== 自伴算子在AI中的应用 ===")

    # 生成高维数据
    n_samples = 1000 # 数据样本数量
    n_features = 50 # 特征维度
    '''
    n_informative=10: 其中只有10个特征是真正有用的（informative features）
    n_redundant=10: 有10个特征是冗余的（由有用特征的线性组合生成）
    n_clusters_per_class=1: 每个类别由1个簇组成
    random_state=42: 设置随机种子，保证结果可重现
    '''
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=10, n_redundant=10,
                               n_clusters_per_class=1, random_state=42)

    print(f"数据形状: {X.shape}")
    print(f"标签分布: {np.unique(y, return_counts=True)}")

    # 计算协方差矩阵 - 典型的自伴算子
    '''
    将数据矩阵 X 按列（特征维度）减去各自均值
    这一步实现了数据中心化，使得每个特征的均值为0
    中心化后的数据 X_centered 更适合进行协方差分析
    '''
    X_centered = X - np.mean(X, axis=0)
    '''
    使用 np.cov() 函数计算协方差矩阵
    输入 X_centered.T 是因为 np.cov() 默认按行处理样本
    由于数据已被转置，实际上是在计算特征间的协方差
    '''
    covariance_matrix = np.cov(X_centered.T)
    '''
    数学意义
    协方差矩阵具有以下重要性质：
        对称性：covariance_matrix[i,j] = covariance_matrix[j,i]
        自伴性：在实数域中，满足 T* = T 的线性算子
        正定性：对于实际数据，通常是半正定矩阵
    '''

    print(f"\n协方差矩阵性质分析:")
    print(f"矩阵形状: {covariance_matrix.shape}")
    print(f"是否对称: {np.allclose(covariance_matrix, covariance_matrix.T)}")
    '''
    .conj() 是 NumPy 数组的一个方法，用于计算数组中每个元素的复共轭（complex conjugate）。
    主要作用
        对于实数：.conj() 返回原数值本身
        对于复数 a + bi：.conj() 返回 a - bi
    数学背景
        自伴算子的定义是满足 T* = T 的算子
        对于实数矩阵，自伴就是对称：T = T.T
        对于复数矩阵，自伴意味着 T = T.conj().T
    '''
    print(f"是否自伴: {np.allclose(covariance_matrix, covariance_matrix.conj().T)}")

    # 特征分解（谱分解）
    '''
    eigh 是 NumPy 中用于计算厄米特矩阵（Hermitian matrix）或实对称矩阵特征值和特征向量的函数。
    主要特点
        专门针对对称矩阵优化：比通用的 eig 函数更高效,协方差矩阵是对称矩阵（满足 covariance_matrix == covariance_matrix.T）
        保证实数特征值：对于对称矩阵，特征值一定是实数
        返回正交特征向量：特征向量自动满足正交性
    与普通 eig 的区别
        numpy.linalg.eig：适用于一般矩阵，可能返回复数特征值
        numpy.linalg.eigh：专为对称/厄米特矩阵设计，保证实数特征值和更好的性能
    '''
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)  # 对对称矩阵使用eigh

    # 验证特征向量的正交性，这是对称矩阵（自伴算子）特征分解的重要性质
    #eigenvectors.T @ eigenvectors: 计算特征向量矩阵的乘积,对于正交矩阵，这个结果应该等于单位矩阵 np.eye(n_features)
    orthogonality_check = np.abs(eigenvectors.T @ eigenvectors - np.eye(n_features))#计算与单位矩阵的偏差绝对值
    max_orthogonal_error = np.max(orthogonality_check)#找出最大的正交性误差
    '''
    数学原理
        对于对称矩阵（自伴算子）的特征分解：
        不同特征值对应的特征向量彼此正交
        相同特征值的特征向量可以通过正交化过程使其正交
        因此所有特征向量构成的矩阵应该是正交矩阵
    验证意义
    这个验证确保了：
        np.linalg.eigh 计算结果的正确性
        特征向量确实满足正交性（V^T V = I）
        后续基于正交性的PCA等操作的可靠性
    '''

    print(f"\n特征分解验证:")
    print(f"特征值范围: [{eigenvalues.min():.3f}, {eigenvalues.max():.3f}]")
    print(f"特征向量正交性误差: {max_orthogonal_error:.2e}")
    print(f"特征值实数性: {np.allclose(eigenvalues, eigenvalues.real)}")

    # 可视化特征谱和主成分
    plt.figure(figsize=(15, 10))

    # 特征值谱（Scree图）
    plt.subplot(2, 3, 1)
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    plt.plot(range(1, len(sorted_eigenvalues) + 1), sorted_eigenvalues, 'bo-')
    plt.xlabel('主成分索引')
    plt.ylabel('特征值（方差）')
    plt.title('特征值谱 - Scree图')
    plt.grid(True, alpha=0.3)

    # 累积解释方差
    plt.subplot(2, 3, 2)
    cumulative_variance = np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
    plt.axhline(y=0.95, color='green', linestyle='--', label='95%方差线')
    plt.xlabel('主成分数量')
    plt.ylabel('累积解释方差比例')
    plt.title('累积解释方差')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 前两个主成分的可视化
    plt.subplot(2, 3, 3)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.xlabel('第一主成分')
    plt.ylabel('第二主成分')
    plt.title('PCA降维可视化')
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)

    # 协方差矩阵的热图
    plt.subplot(2, 3, 4)
    im = plt.imshow(covariance_matrix, cmap='RdBu_r', aspect='auto')
    plt.colorbar(im)
    plt.title('协方差矩阵热图')
    plt.xlabel('特征索引')
    plt.ylabel('特征索引')

    # 特征向量的热图
    plt.subplot(2, 3, 5)
    # 显示前10个特征向量
    im = plt.imshow(eigenvectors[:, :10], cmap='RdBu_r', aspect='auto')
    plt.colorbar(im)
    plt.title('前10个特征向量')
    plt.xlabel('特征向量索引')
    plt.ylabel('原始特征索引')

    # 数据在主成分上的重建误差
    plt.subplot(2, 3, 6)
    n_components_range = range(1, min(20, n_features))
    reconstruction_errors = []

    for n_comp in n_components_range:
        # 使用前n_comp个主成分重建数据
        components = eigenvectors[:, :n_comp]
        projected_data = X_centered @ components
        reconstructed_data = projected_data @ components.T + np.mean(X, axis=0)

        error = np.mean((X - reconstructed_data) ** 2)
        reconstruction_errors.append(error)

    plt.plot(n_components_range, reconstruction_errors, 'go-')
    plt.xlabel('使用的主成分数量')
    plt.ylabel('重建MSE')
    plt.title('重建误差 vs 主成分数量')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 自伴算子在神经网络中的应用
    print(f"\n自伴算子在神经网络中的应用:")

    # 这段代码定义了一个具有对称权重矩阵的神经网络层，体现了自伴算子在深度学习中的应用
    class SymmetricLinear(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            assert in_features == out_features, "对称层要求输入输出维度相同"
            self.weight = nn.Parameter(torch.randn(in_features, in_features))#创建可训练的权重参数
            # 通过初始化使权重接近对称
            with torch.no_grad():
                self.weight.data = 0.5 * (self.weight.data + self.weight.data.T)#将权重矩阵初始化为对称矩阵

        def forward(self, x):#前向传播函数，输入x通过对称权重矩阵进行线性变换
            # 强制对称性
            symmetric_weight = 0.5 * (self.weight + self.weight.T)#在每次前向传播时强制保证权重矩阵的对称性
            return torch.matmul(x, symmetric_weight)#执行矩阵乘法运算
        '''
        数学原理
            这种方法确保了权重矩阵 W 满足 W = W.T，即自伴算子的条件。在神经网络中使用对称权重矩阵的好处包括：
            数学性质良好：特征值都是实数，特征向量正交
            参数效率：只需要存储和更新上三角或下三角部分
            物理意义明确：在某些应用场景中，对称性具有特定的物理含义
        '''

    # 测试对称层
    symmetric_layer = SymmetricLinear(10, 10)
    test_input = torch.randn(5, 10)
    output = symmetric_layer(test_input)

    print(f"对称层权重矩阵是否对称: {torch.allclose(symmetric_layer.weight, symmetric_layer.weight.T)}")
    print(f"对称层输出形状: {output.shape}")

    return covariance_matrix, eigenvalues, eigenvectors


cov_matrix, eigvals, eigvecs = self_adjoint_operators_in_ai()