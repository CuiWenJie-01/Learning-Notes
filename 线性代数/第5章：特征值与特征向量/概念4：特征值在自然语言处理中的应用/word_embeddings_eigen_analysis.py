# 实践4：词嵌入与特征分解
import numpy as np
import matplotlib.pyplot as plt


def word_embeddings_eigen_analysis():
    """
    分析词嵌入矩阵的特征值，理解语义空间的几何结构
    """
    print("=== 词嵌入矩阵的特征分析 ===")

    '''
    vocab_size：词汇表大小为1000，表示这个词汇集合包含1000个不同的
    embedding_dim：嵌入维度为100，表示每个词用100维的向量来表示
    '''
    # 使用预训练的词向量（这里用随机生成模拟）
    #这部分代码是为了后续进行特征值分析做准备，通过分析词嵌入矩阵的特征值和特征向量，可以理解词向量空间的几何结构和语义分布特性。
    vocab_size,embedding_dim=1000,100
    '''
    创建一个词嵌入矩阵 word_embeddings
    使用 numpy.random.randn() 生成一个1000×100的随机矩阵
    每一行代表一个词的向量表示，每一列代表一个特定的语义维度
    这里用随机数模拟预训练的词向量，实际应用中会使用真实训练好的词嵌入模型
    '''
    word_embeddings=np.random.randn(vocab_size,embedding_dim)

    # 模拟一些语义关系
    # 让某些词在潜在空间中形成聚类
    for i in range(0,100,10):#循环遍历索引0到90，步长为10，这意味着将处理10个不同的词簇（clusters）
        '''
        为每个词簇生成一个随机的中心点 cluster_center
        这个中心点是 embedding_dim(100维)的向量，代表该词簇在语义空间中的核心位置
        '''
        cluster_center=np.random.randn(embedding_dim)
        '''
        将连续10个词向量（从索引i到i+9）都加上同一个 cluster_center
        这样使得每组10个词在语义空间中聚集在一起，模拟具有相似语义的词群
        '''
        word_embeddings[i:i+10]+=cluster_center
        #通过这种方式模拟词向量的语义聚类现象，为后续的特征值分析提供更接近真实情况的数据结构，帮助理解词嵌入空间的几何特性。

    print(f"词嵌入矩阵形状: {word_embeddings.shape}")

    # 分析词嵌入矩阵的协方差矩阵
    '''
    计算词嵌入矩阵的协方差矩阵 embedding_cov
    使用 word_embeddings.T（转置）是因为 np.cov 函数需要按列计算变量间的协方差
    协方差矩阵反映了各个维度之间的线性相关性，揭示词向量空间的统计特性
    '''
    embedding_cov=np.cov(word_embeddings.T)
    '''
    对协方差矩阵 embedding_cov 进行特征分解
    返回两个结果：
        eigenvalues：特征值数组，表示各个主成分的重要性程度
        eigenvectors：特征向量矩阵，表示主成分的方向
    '''
    eigenvalues,eigenvectors=np.linalg.eig(embedding_cov)
    '''
    通过特征值分解可以：
        识别词向量空间中最重要的语义方向（主成分）
        理解语义空间的维度重要性分布
        为降维和语义压缩提供数学基础
    '''

    # 按特征值大小排序
    '''
    这种排序操作的目的是：
        将最重要的主成分（对应最大特征值）放在前面
        便于后续的主成分分析（PCA）和降维处理
        方便选择前k个最重要成分进行语义空间分析
    '''
    '''
    格式为 [start:stop:step]，其中：
        第一个 : 表示起始位置（省略，表示从头开始）
        第二个 : 表示结束位置（省略，表示到末尾结束）
        :: 后面的 -1 表示步长为 -1（即反向步进）
    '''
    sorted_indices=np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted=eigenvalues[sorted_indices]#一维，不需要:
    '''
    : 的作用
        在 eigenvectors[:,sorted_indices] 中，: 表示选择 eigenvectors 矩阵的所有行
        这是 NumPy 数组切片语法的一部分，格式为 [行选择, 列选择]
        : 在行位置表示"所有行"，sorted_indices 在列位置表示按索引选择特定列
    '''
    eigenvectors_sorted=eigenvectors[:,sorted_indices]#二维，需要:

    print(f"特征值范围: [{eigenvalues.min():.3f}, {eigenvalues.max():.3f}]")

    # 计算有效维度（特征值大于阈值的数量）
    threshold=0.01*eigenvalues_sorted[0] # 最大特征值（因为已按降序排序）的1%
    effective_dim=np.sum(eigenvalues_sorted>threshold)#计算大于阈值的特征值数量
    print(f"有效维度: {effective_dim}/{embedding_dim}")#输出有效维度信息，格式为"有效维度数/总维度数"
    #这种方法用于确定词嵌入空间中真正承载语义信息的维度数量，小于原始维度数说明存在冗余，可以进行降维处理。
    '''
    比较对象：
        effective_dim：通过特征值分析得出的有效维度数
        embedding_dim：原始的嵌入维度（这里是100）
    判断方法：
        如果 effective_dim < embedding_dim，说明存在冗余维度
        如果 effective_dim ≈ embedding_dim，说明大部分维度都有用
        如果 effective_dim > embedding_dim，这种情况理论上不会出现
    '''

    # 可视化
    plt.figure(figsize=(15, 5))

    # 特征值谱
    plt.subplot(1, 3, 1)
    plt.semilogy(range(1, len(eigenvalues_sorted) + 1), eigenvalues_sorted, 'bo-')
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'阈值 ({threshold:.3f})')
    plt.xlabel('成分索引')
    plt.ylabel('特征值（对数尺度）')
    plt.title('词嵌入协方差矩阵特征值谱')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 累积解释方差
    plt.subplot(1, 3, 2)
    '''
    np.sum(eigenvalues_sorted) - 计算所有特征值的总和，代表总方差
    np.cumsum(eigenvalues_sorted) - 计算特征值的累积和
    两者相除得到每个主成分累积贡献的方差比例
    
    实际意义：
        方差贡献率：每个特征值代表对应主成分包含的信息量
        累积贡献率：前k个主成分总共包含的原始数据信息比例
        降维决策：通常选择累积贡献率达到80%-90%的前k个主成分即可
    '''
    #这个指标可以帮助确定保留多少个主成分能够在最大程度保留原始语义信息的同时实现有效降维。
    cumulative_variance = np.cumsum(eigenvalues_sorted) / np.sum(eigenvalues_sorted)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
    plt.axvline(x=effective_dim, color='green', linestyle='--',
                label=f'有效维度 ({effective_dim})')
    plt.xlabel('成分数量')
    plt.ylabel('累积解释方差比例')
    plt.title('累积解释方差')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 词嵌入在主要成分上的投影
    plt.subplot(1, 3, 3)
    # 投影到前两个主成分
    principal_components = eigenvectors_sorted[:, :2]
    words_2d = word_embeddings @ principal_components

    # 随机选择一些词来显示
    n_words_to_show = 100
    '''
    replace=False：每个词汇索引只能被选中一次，确保选出的词汇不重复
    replace=True：允许重复抽样，同一个词汇索引可能被多次选中
    '''
    selected_indices = np.random.choice(vocab_size, n_words_to_show, replace=False)

    plt.scatter(words_2d[selected_indices, 0], words_2d[selected_indices, 1], alpha=0.6)
    plt.xlabel('第一主成分')
    plt.ylabel('第二主成分')
    plt.title('词嵌入在主成分空间的投影')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 语义分析
    print(f"\n语义空间分析:")
    print(f"  前2个主成分解释 {cumulative_variance[1]:.1%} 的方差")
    print(f"  前10个主成分解释 {cumulative_variance[9]:.1%} 的方差")
    print(f"  这表明语义信息集中在相对较少的维度上")

word_embeddings_eigen_analysis()