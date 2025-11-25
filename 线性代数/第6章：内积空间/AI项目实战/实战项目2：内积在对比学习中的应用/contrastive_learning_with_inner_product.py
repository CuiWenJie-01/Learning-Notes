# 实战项目2：对比学习中的内积应用
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def contrastive_learning_with_inner_product():
    '''
    展示内积在对比学习中的核心作用
    '''
    print("=== 对比学习中的内积应用 ===")

    # 模拟对比学习场景：学习有意义的嵌入空间
    n_classes = 5  # 模拟5个类别
    n_samples_per_class = 100  # 每个类别100个样本
    embedding_dim = 16  # 16维嵌入空间

    # 生成模拟数据：每个类在高维空间中的一个簇
    torch.manual_seed(42)  # 设置随机数种子为42，确保每次运行代码时生成的随机数序列相同，保证实验结果可重现
    embeddings = []  # 存储所有样本的嵌入向量
    labels = []  # 存储所有样本的标签

    # 为每个类创建一个中心点
    class_centers = torch.randn(n_classes, embedding_dim)  # 创建一个n_classes x embedding_dim的随机矩阵，作为每个类的中心点

    for class_id in range(n_classes):
        center = class_centers[class_id]
        # 围绕中心点生成样本
        class_embeddings = center + torch.randn(n_samples_per_class, embedding_dim) * 0.3
        embeddings.append(class_embeddings)
        labels.extend([class_id] * n_samples_per_class)  # 为每个样本分配标签

    embeddings = torch.cat(embeddings, dim=0)  # 将所有样本的嵌入向量连接起来
    labels = torch.tensor(labels)  # 将所有样本的标签转换为张量

    print(f"嵌入矩阵形状: {embeddings.shape}")
    print(f"标签形状: {labels.shape}")

    # 对比学习损失函数（使用内积）
    class ContrastiveLoss(nn.Module):  # 继承自 PyTorch 的 nn.Module 基类。这使得它成为一个可训练的神经网络模块
        def __init__(self, temperature=1.0):  # 在对比学习中，temperature 是一个重要的超参数，用于控制相似度分数的分布锐度
            super().__init__()  # 调用父类的构造函数
            self.temperature = temperature  # 将传入的 temperature 参数保存为实例变量，供后续计算使用

        '''
        self: 类实例本身的引用
        embeddings: 输入的嵌入向量（特征表示）
        labels: 对应的标签信息
        '''

        # 前向传播：计算对比学习损失
        def forward(self, embeddings, labels):
            """
            计算对比学习损失
            embeddings: (batch_size, embedding_dim)
            labels: (batch_size,)
            """
            # 添加数值检查
            if torch.isnan(embeddings).any():
                print("警告：嵌入中检测到NaN")
                embeddings = torch.where(torch.isnan(embeddings), torch.zeros_like(embeddings), embeddings)

            # 计算内积相似度矩阵
            similarity_matrix = torch.mm(embeddings, embeddings.T) / self.temperature # 计算内积相似度矩阵

            # 限制相似度矩阵的范围，防止指数运算溢出
            similarity_matrix = torch.clamp(similarity_matrix, -50, 50)

            # similarity_matrix: (batch_size, batch_size)

            # 创建正样本对掩码（相同类别）
            labels=labels.unsqueeze(1) # ) 将一维的标签张量从形状 (batch_size,) 扩展为二维 (batch_size, 1)
            positive_mask = torch.eq(labels, labels.T).float() # 创建正样本对掩码（相同类别），用 .float() 将布尔值转换为浮点数（1.0 表示正样本对，0.0 表示负样本对）
            # 排除自身
            positive_mask.fill_diagonal_(0)#) 将对角线元素设为 0，因为每个样本与自身的相似度在对比学习中不参与计算，需要排除掉

            # 计算对比学习损失
            exp_sim=torch.exp(similarity_matrix)# 计算相似度矩阵的指数，将相似度转换为概率分布
            # 分母：所有样本的相似度（包括正负样本）
            sum_exp_sim=torch.sum(exp_sim,dim=1,keepdim=True)# 计算所有样本的相似度，dim=1 表示按行求和，得到每个样本的相似度总和，keepdim=True 保持维度，确保后续计算的广播兼容性

            # 分子：正样本对的相似度
            positive_exp_sim=torch.sum(exp_sim*positive_mask,dim=1,keepdim=True)# 计算所有正样本对的相似度，dim=1 表示按行求和，得到每个样本的正样本对相似度总和，keepdim=True 保持维度，确保后续计算的广播兼容性

            # 对比损失
            loss = -torch.log(positive_exp_sim / sum_exp_sim)# 计算对比损失，使用对数函数将相似度转换为概率分布，并取负数，得到对比损失
            return loss.mean()# 返回平均对比损失

    # 模拟对比学习训练过程，
    def simulate_contrastive_learning():
        # 初始化一个简单的编码器
        encoder = nn.Sequential(
            nn.Linear(embedding_dim, 32),# 编码器, 将输入的嵌入向量映射到 32 维
            nn.ReLU(),# 激活函数, 引入非线性, 使模型能够学习复杂的特征表示
            nn.Linear(32, 16),  # 编码器, 将 32 维映射到 16 维
            nn.LayerNorm(16)# 层归一化, 加速训练并提高模型的泛化能力
        )
        #这两行代码分别初始化了对比学习损失函数和优化器：
        contrastive_loss = ContrastiveLoss(temperature=0.1)#设置温度参数 temperature=0.1，这是一个较小的值，会使相似度分布更加尖锐，增强模型对正负样本的区分能力
        optimizer = torch.optim.Adam(encoder.parameters(), lr=0.0001)#使用 Adam 优化算法，encoder.parameters() 指定要优化的参数为 encoder 模型的所有可训练参数，设置学习率 lr=0.001

        n_epochs = 100 #设置训练轮数为100轮
        losses= [] # 存储训练过程中每一轮的损失
        intra_class_distances = [] # 存储训练过程中每一轮的类内距离
        inter_class_distances = [] # 存储训练过程中每一轮的类间距离

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # 前向传播
            projected_embeddings = encoder(embeddings)

            # 检查是否有NaN值
            if torch.isnan(projected_embeddings).any():
                print(f"Epoch {epoch}: 嵌入检测到NaN，跳过更新")
                continue

            # 计算对比损失
            loss = contrastive_loss(projected_embeddings, labels)

            # 检查损失是否为NaN
            if torch.isnan(loss):
                print(f"Epoch {epoch}: 嵌入检测到NaN，跳过更新")
                continue

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(loss.item())

            # 计算类内和类间距离（每10个epoch），epoch:完整遍历一次训练数据集的过程,在当前代码中: 表示对整个数据集进行一次完整的前向传播和反向传播训练
            if epoch % 10 == 0:
                with torch.no_grad():
                    # 计算类内平均距离
                    intra_dist = 0
                    inter_dist = 0
                    intra_count = 0
                    inter_count = 0

                    for i in range(len(projected_embeddings)):
                        for j in range(i + 1, len(projected_embeddings)):#这样确保每对样本只计算一次，避免重复
                            dist = torch.norm(projected_embeddings[i] - projected_embeddings[j])#计算两个嵌入向量之间的L2范数（欧氏距离）
                            # 类内距离统计
                            if labels[i] == labels[j]:
                                intra_dist += dist.item()#累加同类样本间的距离到 intra_dist
                                intra_count += 1 #计数器 intra_count 增加
                            else:
                                # 类间距离统计
                                inter_dist += dist.item()#累加不同类样本间的距离到 inter_dist
                                inter_count += 1

                    intra_class_distances.append(intra_dist / intra_count if intra_count > 0 else 0)#类内平均距离
                    inter_class_distances.append(inter_dist / inter_count if inter_count > 0 else 0)#类间平均距离

        # 这些指标用于监控训练过程中特征表示质量的改善情况。
        return encoder, losses, intra_class_distances, inter_class_distances

    # 运行对比学习
    '''
    函数调用
        simulate_contrastive_learning()：执行对比学习的模拟训练过程
    返回值解包
    函数返回四个值，分别赋给对应的变量：
        encoder：训练完成的编码器模型
            这是一个 nn.Sequential 神经网络模型
            用于将输入嵌入映射到新的特征空间
        losses：训练过程中的损失值列表
            记录每轮训练的 ContrastiveLoss 值
            用于监控训练收敛情况
        intra_dists：类内距离列表
            存储训练过程中每10个epoch计算的类内平均距离
            反映同类样本的聚集程度
        inter_dists：类间距离列表
            存储训练过程中每10个epoch计算的类间平均距离
            反映不同类别间的分离程度
    '''
    encoder, losses, intra_dists, inter_dists = simulate_contrastive_learning()

    # 可视化训练过程
    plt.figure(figsize=(15, 5))

    # 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.xlabel('训练步数')
    plt.ylabel('对比损失')
    plt.title('对比学习损失曲线')
    plt.grid(True, alpha=0.3)

    # 类内类间距离
    plt.subplot(1, 3, 2)
    epochs = list(range(0, 100, 10))
    plt.plot(epochs, intra_dists, 'bo-', label='类内平均距离')
    plt.plot(epochs, inter_dists, 'ro-', label='类间平均距离')
    plt.xlabel('训练周期')
    plt.ylabel('距离')
    plt.title('类内 vs 类间距离')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 最终嵌入空间可视化
    plt.subplot(1, 3, 3)
    with torch.no_grad():
        final_embeddings = encoder(embeddings)

        # 检查并处理NaN值
        if torch.isnan(final_embeddings).any():
            print("错误: NaN 值在最终嵌入中被发现，并用零替换")
            final_embeddings = torch.where(torch.isnan(final_embeddings),
                                           torch.zeros_like(final_embeddings),
                                           final_embeddings)

        # 使用PCA降维可视化
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(final_embeddings.numpy())

        for class_id in range(n_classes):
            class_mask = (labels == class_id)
            plt.scatter(embeddings_2d[class_mask, 0], embeddings_2d[class_mask, 1],
                        label=f'Class {class_id}', alpha=0.7)

        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('对比学习后的嵌入空间')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 分析对比学习效果
    print(f"\n对比学习效果分析:")
    print(f"最终损失: {losses[-1]:.4f}")
    print(f"类内距离: {intra_dists[-1]:.4f}")
    print(f"类间距离: {inter_dists[-1]:.4f}")
    print(f"分离度 (类间/类内): {inter_dists[-1] / intra_dists[-1]:.2f}")

    # 验证内积相似度的区分能力
    with torch.no_grad():
        final_embeddings = encoder(embeddings)

        # 计算一些样本对的内积相似度
        same_class_pairs = []
        diff_class_pairs = []

        for i in range(100):  # 检查100个样本对
            idx1, idx2 = torch.randint(0, len(final_embeddings), (2,))
            similarity = torch.dot(final_embeddings[idx1], final_embeddings[idx2])

            if labels[idx1] == labels[idx2]:
                same_class_pairs.append(similarity.item())
            else:
                diff_class_pairs.append(similarity.item())

        print(f"\n内积相似度分析:")
        print(f"同类样本平均相似度: {np.mean(same_class_pairs):.4f}")
        print(f"异类样本平均相似度: {np.mean(diff_class_pairs):.4f}")
        print(f"区分度: {np.mean(same_class_pairs) - np.mean(diff_class_pairs):.4f}")

contrastive_learning_with_inner_product()
