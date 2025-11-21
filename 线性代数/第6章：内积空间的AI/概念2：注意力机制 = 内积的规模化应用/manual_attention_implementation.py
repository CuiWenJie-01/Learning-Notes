# 数学概念：缩放点积注意力使用查询和键的内积计算注意力权重
# AI对应：Transformer架构的核心
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def manual_attention_implementation():
    """
    从零实现Transformer的缩放点积注意力机制
    """
    print("=== 缩放点积注意力：内积的规模化应用 ===")

    # 模拟Transformer参数
    batch_size=2 #定义了在单次训练迭代中同时处理的样本数量，表示每次处理2个样本（或序列）
    seq_len=5 # 序列长度（比如5个词）
    d_model=64  # 嵌入维度
    d_k=8 # 注意力头的维度

    # 随机生成输入（模拟词嵌入）
    X=torch.rand(batch_size,seq_len,d_model)
    print(f"输入形状: (batch_size, seq_len, d_model) = {X.shape}")

    # 定义线性投影层（Q, K, V的投影矩阵）
    '''
    这些是Transformer注意力机制中的核心参数矩阵：
        所有变换都不使用偏置项（bias=False）
        输入维度都是 d_model=64（词嵌入维度）
        输出维度都是 d_k=8（注意力头的维度）
        这些权重矩阵在训练过程中会被学习和优化
    '''
    W_Q=nn.Linear(d_model,d_k,bias=False)#定义查询（Query）变换矩阵，将 d_model 维的输入映射到 d_k 维的查询空间
    W_K=nn.Linear(d_model,d_k,bias=False)#定义键（Key）变换矩阵，将 d_model 维的输入映射到 d_k 维的键空间
    W_V=nn.Linear(d_model,d_k,bias=False)#定义值（Value）变换矩阵，将 d_model 维的输入映射到 d_k 维的值空间

    # 初始化权重（为了可重现性）
    torch.manual_seed(42)
    nn.init.xavier_uniform_(W_Q.weight)#使用 Xavier 初始化方法初始化查询变换矩阵 W_Q 的权重，确保在训练开始时权重分布是合理的
    nn.init.xavier_uniform_(W_K.weight)#使用 Xavier 初始化方法初始化键变换矩阵 W_K 的权重，确保在训练开始时权重分布是合理的
    nn.init.xavier_uniform_(W_V.weight)#使用 Xavier 初始化方法初始化值变换矩阵 W_V 的权重，确保在训练开始时权重分布是合理的

    # 计算Q, K, V
    Q=W_Q(X)#将输入 X 映射到查询空间，得到查询矩阵 Q，形状为 (batch_size, seq_len, d_k)
    K=W_K(X)#将输入 X 映射到键空间，得到键矩阵 K，形状为 (batch_size, seq_len, d_k)
    V=W_V(X)#将输入 X 映射到值空间，得到值矩阵 V，形状为 (batch_size, seq_len, d_k)

    print(f"Q形状: {Q.shape}")
    print(f"K形状: {K.shape}")
    print(f"V形状: {V.shape}")

    # 手动实现缩放点积注意力
    def scaled_dot_product_attention(Q, K, V):
        """
        手动实现缩放点积注意力
        """
        # 步骤1: 计算Q和K^T的内积（矩阵乘法）
        # 这行代码实现了注意力机制中的关键计算步骤
        # Q: (batch_size, seq_len, d_k)
        # K: (batch_size, seq_len, d_k),将 K 的最后两个维度进行转置
        # 内积: Q @ K^T -> (batch_size, seq_len, seq_len)
        '''
        这个计算的本质是：
            对于每个位置的查询向量，计算它与所有位置的键向量的内积
            结果矩阵中的每个元素 attention_scores[i][j] 表示第 i 个位置的查询对第 j 个位置的键的相关性得分
            这正是Transformer中缩放点积注意力的核心操作，体现了内积在计算向量相似度方面的作用
        '''
        attention_scores=torch.matmul(Q,K.transpose(-2,-1))#计算查询矩阵 Q 和键矩阵 K 的转置之间的矩阵乘法

        print(f"注意力分数形状 (Q @ K^T): {attention_scores.shape}")
        print(f"注意力分数范围: [{attention_scores.min():.3f}, {attention_scores.max():.3f}]")

        # 步骤2: 缩放（防止softmax梯度消失）
        # softmax是一种数学函数，用于将任意实数向量转换为概率分布
        '''
        为什么要进行缩放？
            防止softmax饱和：当 d_k 较大时，Q 和 K 的内积值会变得很大，导致 softmax 函数进入梯度很小的饱和区域
            稳定梯度：通过除以 √d_k，使注意力分数的方差保持在合理范围内，有利于训练稳定
            理论依据：假设 Q 和 K 的元素是独立随机变量，均值为0，方差为1，那么它们的点积的方差就是 d_k，因此除以 √d_k 可以将方差归一化为1
        这是Transformer中"缩放点积注意力"名称的由来，也是其相比于原始点积注意力的重要改进。
        '''
        d_k=Q.size(-1)#：获取查询向量的最后一个维度大小，即注意力头的维度 d_k=8
        attention_scores=attention_scores/torch.sqrt(torch.tensor(d_k,dtype=torch.float32))#将注意力分数除以 √d_k 进行缩放

        # 步骤3: 应用softmax得到注意力权重
        #dim=-1 指定在序列维度上进行softmax操作，确保每个位置的注意力权重和为1
        attention_weights=F.softmax(attention_scores,dim=-1)#对 attention_scores 张量在最后一个维度上应用softmax函数

        print(f"注意力权重形状: {attention_weights.shape}")
        print(f"注意力权重每行和: {attention_weights[0].sum(dim=-1)}")  # 应该都是1

        # 步骤4: 对值向量加权求和
        output=torch.matmul(attention_weights,V)

        return  output, attention_weights

    # 应用注意力机制
    '''
    这个函数内部完成了Transformer注意力机制的四个核心步骤：
        计算查询和键的内积得到注意力分数
        对注意力分数进行缩放
        应用softmax得到注意力权重
        使用注意力权重对值向量进行加权求和
    最终输出的output就是经过注意力机制处理后的结果，每个位置都融合了序列中其他位置的信息。
    '''
    output,attention_weights=scaled_dot_product_attention(Q,K,V)
    print(f"输出形状: {output.shape}")

    # 与PyTorch内置实现对比
    pytorch_output=F.scaled_dot_product_attention(Q,K,V)
    print(f"PyTorch内置输出形状: {pytorch_output.shape}")
    print(f"手动实现与PyTorch实现是否一致: {torch.allclose(output, pytorch_output, atol=1e-6)}")

    # 可视化注意力模式
    plt.figure(figsize=(15, 5))

    # 第一个batch的注意力权重
    attention_map = attention_weights[0].detach().numpy()

    plt.subplot(1, 3, 1)
    plt.imshow(attention_map, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.xlabel('Key位置')
    plt.ylabel('Query位置')
    plt.title('注意力权重热图')

    # 特定query的注意力分布
    plt.subplot(1, 3, 2)
    query_idx = 2  # 查看第三个词的注意力分布
    plt.bar(range(seq_len), attention_map[query_idx])
    plt.xlabel('Key位置')
    plt.ylabel('注意力权重')
    plt.title(f'Query位置 {query_idx} 的注意力分布')
    plt.grid(True, alpha=0.3)

    # 内积值的分布
    plt.subplot(1, 3, 3)
    attention_scores = torch.matmul(Q, K.transpose(-2, -1))[0].detach().numpy()
    plt.hist(attention_scores.flatten(), bins=50, alpha=0.7)
    plt.xlabel('内积值')
    plt.ylabel('频数')
    plt.title('Q-K内积值分布')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 数学原理分析
    print(f"\n注意力机制的数学原理:")
    print("1. Q @ K^T: 计算每个查询与所有键的内积（相似度）")
    print("2. 缩放: 除以√d_k 防止内积过大导致softmax梯度消失")
    print("3. Softmax: 将内积转换为概率分布")
    print("4. 加权和: 用注意力权重对值向量进行加权求和")

    # 验证内积的线性性质
    print(f"\n验证内积的线性性质:")
    q1, q2 = Q[0, 0], Q[0, 1]  # 两个查询向量
    k = K[0, 0]  # 一个键向量

    # 验证: <q1 + q2, k> = <q1, k> + <q2, k>
    left_side = torch.dot(q1 + q2, k)
    right_side = torch.dot(q1, k) + torch.dot(q2, k)
    print(f"<q1 + q2, k> = {left_side:.4f}")
    print(f"<q1, k> + <q2, k> = {right_side:.4f}")
    print(f"线性性质成立: {torch.allclose(left_side, right_side, atol=1e-6)}")

manual_attention_implementation()
