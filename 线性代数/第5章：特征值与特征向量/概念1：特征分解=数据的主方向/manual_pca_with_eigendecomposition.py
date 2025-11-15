#数学概念：矩阵的特征向量表示变换的主方向，特征值表示在这些方向上的缩放因子。
#AI对应：PCA主成分分析、数据降维、特征提取

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import explained_variance_score

'''
load_iris: 加载经典的鸢尾花数据集，这是一个内置的机器学习标准数据集
make_blobs: 生成用于聚类的样本数据，可创建具有明显簇结构的人工数据集
'''
from sklearn.datasets import load_iris,make_blobs
'''
导入 scikit-learn 库中的主成分分析(PCA)类
提供现成的 PCA 实现，用于数据降维和特征提取
'''
from sklearn.decomposition import PCA

#实践1：手动实现PCA - 特征分解的直接应用
def manual_pca_with_eigendecomposition():
    """
    手动实现PCA：通过特征分解找到数据的主成分
    """
    print("=== 手动PCA：特征分解实践 ===")

    #生成示例数据
    np.random.seed(42)
    n_samples=1000
    #创建有相关性的二维数据
    mean=[0,0]# 数据的均值向量，表示二维数据每个维度的平均值
    cov=[[2,1.5],
         [1.5,1]]# 协方差矩阵，定义数据各维度间的相关性
    X=np.random.multivariate_normal(mean,cov,n_samples)# 生成符合指定均值和协方差的多元正态分布数据

    print(f"原始数据形状：{X.shape}")

    #1.中心化数据：这行代码执行的是数据中心化操作，是PCA算法中的关键预处理步骤
    '''
    np.mean(X, axis=0): 计算数据矩阵 X 每列（每个特征维度）的均值
    X - np.mean(X, axis=0): 将数据矩阵 X 的每列减去其对应的均值
    '''
    X_centered=X-np.mean(X,axis=0)

    #2.计算协方差矩阵，这是PCA算法中的核心计算步骤
    '''
    np.cov 函数要求：
    输入数据的每一行代表一个变量（特征）
    每一列代表一个观测值（样本）
    因此需要对原始数据进行转置操作
    '''
    cov_matrix=np.cov(X_centered.T)#X_centered.T: 对中心化后的数据进行转置
    print(f"协方差矩阵：\n{cov_matrix}")

    #3.特征分解---核心步骤！
    eigenvalues,eigenvectors=np.linalg.eig(cov_matrix)

    print(f"\n特征值（方差）：{eigenvalues}")
    print(f"特征向量（主成分）：\n{eigenvectors}")

    #4.按特征值大小排序
    sorted_indices=np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted=eigenvalues[sorted_indices]
    eigenvectors_sorted=eigenvectors[:,sorted_indices]

    print(f"\n排序后的特征值：{eigenvalues_sorted}")
    print(f"排序后的特征向量：\n{eigenvectors_sorted}")

    #5.选择主成分并投影
    n_components=1
    principal_components=eigenvectors_sorted[:,:n_components]
    X_pca=X_centered@principal_components

    print(f"\n降维后的数据：\n{X_pca.shape}")

    #可视化
    plt.figure(figsize=(15,5))

    #原始数据
    plt.subplot(1, 3, 1)#创建一个1行3列的子图布局，当前激活第1个子图（最左边的位置）
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6)#绘制散点图，X[:, 0]：所有样本的第一个特征值作为x坐标，所有样本的第二个特征值作为y坐标，alpha=0.6：设置点的透明度为0.6，避免重叠点过于密集
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('原始数据')
    plt.grid(True, alpha=0.3)#显示网格线，透明度设为0.3以便观察数据点

    #特征向量方向
    plt.subplot(1,3,2)#在1行3列的子图布局中，激活第2个子图（中间位置）
    '''
    绘制中心化数据的散点图
    X_centered[:,0]：中心化后数据的第一列特征作为x坐标
    X_centered[:1]：这里存在代码错误，应该是 X_centered[:,1] 才对，表示中心化后数据的第二列特征作为y坐标
    alpha=0.6：设置点的透明度
    '''
    plt.scatter(X_centered[:,0],X_centered[:,1],alpha=0.6)

    '''
    eigvec[0]：表示特征向量的第一个分量（x坐标分量）
    eigvec[1]：表示特征向量的第二个分量（y坐标分量）
    这里的 eigvec 是一个二维特征向量，表示数据的一个主成分方向。
    eigvec[0] * eigval：特征向量x分量乘以对应的特征值
    eigvec[1] * eigval：特征向量y分量乘以对应的特征值
    '''
    # 绘制特征向量（主成分方向）
    for i, (eigval, eigvec) in enumerate(zip(eigenvalues_sorted, eigenvectors_sorted.T)):
        plt.arrow(0, 0, eigvec[0] * eigval, eigvec[1] * eigval,
                  head_width=0.1, head_length=0.1, fc='red', ec='red',
                  label=f'PC{i + 1} (λ={eigval:.2f})', linewidth=3)
    plt.xlabel('Feature 1 (中心化)')
    plt.ylabel('Feature 2 (中心化)')
    plt.title('特征向量方向 = 主成分')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    # 降维结果
    plt.subplot(1, 3, 3)#在1行3列的子图布局中，激活第3个子图（最右边的位置）
    plt.scatter(X_pca, np.zeros_like(X_pca), alpha=0.6)
    plt.xlabel('第一主成分')
    plt.title(f'降维到{n_components}维')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()#会自动调整画布中各个子图的位置和大小
    plt.show()

    #解释方差比例
    '''
    计算方差解释比例：每个特征值除以所有特征值的总和
    评估主成分重要性：数值越大表示该主成分包含的信息越多
    降维决策依据：帮助决定保留多少个主成分
    '''
    explained_variance_ratio=eigenvalues_sorted/np.sum(eigenvalues)
    print(f"\n解释方差比例: {explained_variance_ratio}")
    print(f"第一主成分保留 {explained_variance_ratio[0]:.1%} 的信息")

manual_pca_with_eigendecomposition()


