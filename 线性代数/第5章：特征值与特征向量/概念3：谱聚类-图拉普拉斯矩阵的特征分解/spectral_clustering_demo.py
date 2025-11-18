# 数学概念：图拉普拉斯矩阵的特征向量揭示图的聚类结构
# AI对应：谱聚类、社区发现、图神经网络
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# 实践3：谱聚类 - 特征向量用于数据聚类
def spectral_clustering_demo():
    """
    谱聚类：使用拉普拉斯矩阵的特征向量进行聚类
    """
    print("=== 谱聚类：特征向量的聚类能力 ===")

    # 生成三个团状数据
    '''
    代码功能分析
        数据生成: 使用 make_blobs 函数创建用于聚类分析的合成数据集
    参数设置:
        n_samples=300: 生成300个数据样本点
        centers=3: 创建3个聚类中心
        cluster_std=0.6: 每个聚类的标准差为0.6，控制聚类的紧密程度
        random_state=42: 固定随机种子以保证结果可重现
    返回值说明
        X: 包含300个二维坐标点的数据矩阵，每个点代表一个数据样本
        y_true: 对应的真实聚类标签数组，标记了每个点所属的聚类
    '''
    #这行代码为后续的谱聚类算法提供了实验数据。谱聚类将利用这些数据点构建相似度图，计算图拉普拉斯矩阵，并通过特征分解找到最优的聚类划分。
    X,y_true=make_blobs(n_samples=300,centers=3,cluster_std=0.6,random_state=42)

    # 构建相似度图（使用k近邻）
    from sklearn.neighbors import kneighbors_graph #导入用于构建k近邻图的函数
    '''
    基于输入数据 X 构建邻接矩阵 A
    n_neighbors=10：每个点与其最近的10个邻居建立连接
    mode='connectivity'：返回连通性矩阵，表示点与点之间是否有连接关系
    include_self=True：包含自身节点的连接
    '''
    #这个邻接矩阵 A 描述了数据点之间的相似性关系，为后续计算图拉普拉斯矩阵 L 和特征分解提供了基础，是实现谱聚类的核心数据结构。
    A=kneighbors_graph(X,n_neighbors=10,mode='connectivity',include_self=True)
    A=A.toarray() # 将稀疏矩阵转换为普通的numpy数组格式

    # 构建拉普拉斯矩阵
    D=np.diag(np.sum(A,axis=1)) # 度矩阵，axis=1表示按行求和
    L=D-A # 非标准化拉普拉斯矩阵

    print(f"相似度矩阵形状：{A.shape}")
    print(f"拉普拉斯矩阵形状：{L.shape}")

    # 计算拉普拉斯矩阵的特征值和特征向量，这行代码执行了图拉普拉斯矩阵的特征分解操作
    '''
    eigenvalues: 包含拉普拉斯矩阵 L 所有特征值的一维数组
    eigenvectors: 包含对应特征向量的二维矩阵，每一列是一个特征向量
    在谱聚类中的意义
    这是谱聚类算法的核心步骤：
        通过特征分解获得拉普拉斯矩阵的特征信息
        特征值和特征向量揭示了图的聚类结构
        后续通常会选择前k个最小特征值对应的特征向量用于聚类
    数学背景
        对于图拉普拉斯矩阵 L，特征分解满足：L × v = λ × v
        其中 λ 是特征值，v 是对应的特征向量
        最小的几个特征值对应的特征向量能够很好地表示图的聚类结构
    '''
    eigenvalues,eigenvectors=np.linalg.eig(L) #使用NumPy的线性代数模块计算矩阵 L 的特征值和特征向量

    # 取前k个最小的非零特征值对应的特征向量，这段代码实现了谱聚类算法中的关键步骤——选择用于聚类的特征向量
    k=3 #设置期望的聚类数量为3个
    sorted_indices=np.argsort(eigenvalues) #对特征值进行升序排序，返回排序后的索引
    # 跳过第一个特征值（通常为0）
    selected_indices=sorted_indices[1:k+1]#选择第2到第k+1个特征值对应的索引（跳过第一个接近0的特征值）
    spectral_features=eigenvectors[:,selected_indices].real # 提取选定特征值对应的特征向量作为谱聚类的特征
    #spectral_features 是一个 (n_samples, k) 的矩阵，每一行代表一个数据点在新特征空间中的坐标，这些坐标将用于最终的聚类分析。
    '''
    谱聚类原理
        跳过零特征值: 图拉普拉斯矩阵的第一个特征值通常为0，对应常数特征向量，不包含聚类信息
        选择最小非零特征值: 最小的几个非零特征值对应的特征向量能够最好地表示图的聚类结构
        构建新的特征空间: 将原始高维数据映射到由这些特征向量构成的低维空间中
    '''

    print(f"前{k}个最小特征值：{eigenvalues[selected_indices]}")
    print(f"谱特征形状：{spectral_features.shape}")

    # 在谱特征空间中进行k-means聚类
    from sklearn.cluster import KMeans #导入scikit-learn库中的K均值聚类算法
    '''
    n_clusters=k: 设置聚类数量为3个（与之前设定的k值一致）
    random_state=42: 固定随机种子确保结果可重现
    '''
    kmeans=KMeans(n_clusters=k,random_state=42)#创建K均值聚类器实例
    '''
    fit_predict: 对 spectral_features 进行拟合并预测聚类标签
    返回值 y_pred_spectral 是每个数据点的预测聚类标签
    '''
    y_pred_spectral=kmeans.fit_predict(spectral_features)#在谱特征空间中进行聚类
    '''
    谱聚类流程总结
        原始数据 X 通过k近邻构建相似度图得到邻接矩阵 A
        计算图拉普拉斯矩阵 L=D-A
        对 L 进行特征分解得到特征值和特征向量
        选取前k个最小非零特征值对应的特征向量构成 spectral_features
        在 spectral_features 空间中使用 KMeans 进行最终聚类
        这种做法的优势在于，谱特征空间中的数据更容易被线性分离，从而提高聚类效果。
    '''

    # 对比传统k-means
    '''
    使用 KMeans 算法直接对原始数据 X 进行聚类分析
    fit_predict(X): 对输入数据 X 进行拟合并预测每个数据点的聚类标签
    返回值 y_pred_kmeans 是一个数组，包含了每个数据点被分配到的聚类标签
    不过需要注意的是，这行代码在当前的代码文件中似乎是一个对比实验的部分，用来与谱聚类的结果进行比较。
    它直接在原始数据空间中进行K均值聚类，而谱聚类是在通过图拉普拉斯矩阵特征分解得到的 spectral_features 空间中进行聚类，
    通常能获得更好的聚类效果，特别是当数据具有复杂的非凸结构时。
    '''
    y_pred_kmeans=kmeans.fit_predict(X)

    # 可视化比较
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    # 原始数据
    axes[0, 0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis')
    axes[0, 0].set_title('真实标签')
    axes[0, 0].grid(True, alpha=0.3)

    # 传统k-means
    axes[0, 1].scatter(X[:, 0], X[:, 1], c=y_pred_kmeans, cmap='viridis')
    axes[0, 1].set_title('传统K-means')
    axes[0, 1].grid(True, alpha=0.3)

    # 谱特征空间
    if spectral_features.shape[1] >= 2:
        axes[0, 2].scatter(spectral_features[:, 0], spectral_features[:, 1],
                           c=y_true, cmap='viridis')
        axes[0, 2].set_title('谱特征空间')
    else:
        axes[0, 2].scatter(spectral_features[:, 0], np.zeros_like(spectral_features[:, 0]),
                           c=y_true, cmap='viridis')
        axes[0, 2].set_title('谱特征空间（1D）')
    axes[0, 2].grid(True, alpha=0.3)

    # 谱聚类结果
    axes[1, 0].scatter(X[:, 0], X[:, 1], c=y_pred_spectral, cmap='viridis')
    axes[1, 0].set_title('谱聚类结果')
    axes[1, 0].grid(True, alpha=0.3)
    # 通过对比三张图（真实标签、传统K-means、谱聚类结果），可以直观地看出谱聚类相较于传统K-means在处理复杂数据结构时的优势。

    # 特征值谱
    axes[1, 1].plot(range(1, 11), eigenvalues[sorted_indices[:10]], 'bo-')
    axes[1, 1].set_xlabel('特征值索引')
    axes[1, 1].set_ylabel('特征值')
    axes[1, 1].set_title('拉普拉斯矩阵特征值谱')
    axes[1, 1].grid(True, alpha=0.3)

    # 特征向量可视化
    for i in range(min(3, spectral_features.shape[1])):
        axes[1, 2].scatter(range(len(spectral_features)),
                           spectral_features[:, i],
                           label=f'特征向量 {i + 1}', alpha=0.6)
    axes[1, 2].set_xlabel('数据点索引')
    axes[1, 2].set_ylabel('特征向量值')
    axes[1, 2].set_title('前几个特征向量')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 计算聚类准确率，这段代码使用 adjusted_rand_score 来评估和比较两种聚类方法的性能
    from sklearn.metrics import adjusted_rand_score#adjusted_rand_score: 这是一个聚类评估指标，用于衡量两个聚类结果之间的相似度
    '''
    输入：真实标签(y_true)和预测标签(如y_pred_kmeans或y_pred_spectral)
    输出：调整后的兰德指数，范围通常在0到1之间，值越大表示聚类结果越接近真实情况
    '''
    kmeans_score=adjusted_rand_score(y_true, y_pred_kmeans)#比较真实标签与直接在原始数据上应用K-means的结果
    spectral_score=adjusted_rand_score(y_true, y_pred_spectral)#比较真实标签与通过谱聚类方法得到的结果
    '''
    这种评估方式可以帮助我们量化地比较：
        哪种聚类方法更接近真实的聚类结构
        谱聚类相对于传统K-means的优势程度
    通过数值化的评分，我们可以客观地验证谱聚类在处理复杂数据结构时是否优于传统的K-means方法。
    '''

    print(f"\n聚类效果比较:")
    print(f"传统K-means ARI: {kmeans_score:.3f}")
    print(f"谱聚类 ARI: {spectral_score:.3f}")

spectral_clustering_demo()
