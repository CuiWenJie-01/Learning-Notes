# 实战项目1：基于SVD和正规算子的高级推荐系统
import matplotlib.pyplot as plt
import torch
import torch.optim as optim


def advanced_svd_recommendation_system():
    """
    使用正规算子和SVD的高级推荐系统优化
    """
    print("=== 基于正规算子和SVD的高级推荐系统 ===")

    # 创建更真实的用户-物品交互矩阵
    n_users = 200
    n_items = 100
    n_factors = 15 #潜在因子数量

    # 生成有结构的用户和物品嵌入
    torch.manual_seed(42)

    # 用户特征：假设有3个用户群体
    user_cluster_centers = torch.randn(3, n_factors)
    user_embeddings = []#创建一个列表，用于存储用户嵌入向量
    for i in range(n_users):
        cluster_id = i % 3#根据用户索引生成用户所属的群体ID
        center = user_cluster_centers[cluster_id]
        user_embed = center + torch.randn(n_factors) * 0.3
        user_embeddings.append(user_embed)
    user_embeddings = torch.stack(user_embeddings)

    # 物品特征：假设有2个物品类别
    item_cluster_centers = torch.randn(2, n_factors)
    item_embeddings = []
    for i in range(n_items):
        cluster_id = i % 2
        center = item_cluster_centers[cluster_id]
        item_embed = center + torch.randn(n_factors) * 0.3
        item_embeddings.append(item_embed)
    item_embeddings = torch.stack(item_embeddings)

    # 生成评分矩阵：用户嵌入和物品嵌入的内积
    true_ratings = torch.matmul(user_embeddings, item_embeddings.T)
    # 添加噪声和偏置
    true_ratings += torch.randn(n_users, n_items) * 0.5
    user_biases = torch.randn(n_users).unsqueeze(1)
    item_biases = torch.randn(n_items).unsqueeze(0)
    true_ratings += user_biases + item_biases

    # 二值化评分（1表示喜欢，0表示不喜欢）
    rating_threshold = true_ratings.median()
    binary_ratings = (true_ratings > rating_threshold).float()

    # 添加稀疏性（很多交互缺失）
    mask = torch.rand(n_users, n_items) > 0.7#生成0-1之间的随机数,通过 > 0.7 的条件判断，约有30%的元素为True，70%为False，得到一个布尔型掩码矩阵 mask
    observed_ratings = binary_ratings * mask.float()#创建一个掩码矩阵，用于指示哪些评分是观测到的

    print(f"用户数量: {n_users}")
    print(f"物品数量: {n_items}")
    print(f"观测到的交互数量: {torch.sum(observed_ratings > 0).item()}")
    print(f"稀疏度: {(observed_ratings == 0).sum() / observed_ratings.numel():.1%}")

    # 高级SVD分解考虑偏置
    def biased_svd(ratings_matrix, n_factors, n_epochs=100, learning_rate=0.01, reg=0.02):
        """
        带偏置的SVD分解
        """
        n_users, n_items = ratings_matrix.shape

        # 初始化参数
        user_factors = torch.randn(n_users, n_factors, requires_grad=True)
        item_factors = torch.randn(n_items, n_factors, requires_grad=True)
        user_biases = torch.randn(n_users, 1, requires_grad=True)
        item_biases = torch.randn(1, n_items, requires_grad=True)
        global_bias = torch.tensor([ratings_matrix[ratings_matrix > 0].mean()],
                                   requires_grad=True)

        optimizer = optim.Adam([user_factors, item_factors, user_biases, item_biases, global_bias],
                               lr=learning_rate, weight_decay=reg)

        # 只对观测到的评分进行训练
        observed_mask = (ratings_matrix > 0)

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # 预测评分
            predicted = (global_bias + user_biases + item_biases +
                         torch.matmul(user_factors, item_factors.T))

            # 计算损失（只对观测到的评分）
            loss = torch.sum((predicted[observed_mask] - ratings_matrix[observed_mask]) ** 2)

            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        return user_factors.detach(), item_factors.detach(), user_biases.detach(), item_biases.detach(), global_bias.detach()

    # 执行带偏置的SVD
    print("\n训练带偏置的SVD模型...")
    user_factors, item_factors, user_biases, item_biases, global_bias = biased_svd(
        observed_ratings, n_factors=10, n_epochs=100)

    # 重建评分矩阵
    predicted_ratings = (global_bias + user_biases + item_biases +
                         torch.matmul(user_factors, item_factors.T))

    # 评估模型
    def evaluate_recommendation_model(true_ratings, predicted_ratings, observed_ratings):
        """评估推荐模型性能"""
        # 只在测试集（未观测到的部分）上评估
        test_mask = (observed_ratings == 0) & (true_ratings > 0)  # 未观测但实际有交互

        if test_mask.sum() > 0:
            test_true = true_ratings[test_mask]
            test_pred = predicted_ratings[test_mask]

            # 计算RMSE
            rmse = torch.sqrt(torch.mean((test_true - test_pred) ** 2))

            # 计算AUC（二分类）
            from sklearn.metrics import roc_auc_score
            import numpy as np

            # 检查是否存在两个类别
            unique_labels = np.unique(test_true.numpy())#获取标签的唯一值
            if len(unique_labels) > 1:#存在两个类别
                auc = roc_auc_score(test_true.numpy(), torch.sigmoid(test_pred).numpy())
            else:
                auc = None # 或者设置为特定值如0.5
            #这是推荐系统评估中的常见情况，特别是在数据稀疏或分割不均匀时会出现此类问题。

            return rmse.item(), auc
        return None, None

    rmse, auc = evaluate_recommendation_model(binary_ratings, predicted_ratings, observed_ratings)#评估模型

    print(f"\n模型评估结果:")
    print(f"测试集RMSE: {rmse:.4f}" if rmse else "无测试数据")
    print(f"测试集AUC: {auc:.4f}" if auc else "无测试数据")

    # 可视化结果
    plt.figure(figsize=(15, 10))

    # 原始评分矩阵
    plt.subplot(2, 3, 1)
    plt.imshow(observed_ratings.numpy(), cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('观测到的用户-物品交互')
    plt.xlabel('物品索引')
    plt.ylabel('用户索引')

    # 预测评分矩阵
    plt.subplot(2, 3, 2)
    plt.imshow(predicted_ratings.numpy(), cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('预测的用户-物品评分')
    plt.xlabel('物品索引')
    plt.ylabel('用户索引')

    # 用户潜在特征可视化
    plt.subplot(2, 3, 3)
    from sklearn.manifold import TSNE
    user_tsne = TSNE(n_components=2, random_state=42)
    user_2d = user_tsne.fit_transform(user_factors.numpy())

    # 根据用户群体着色
    user_clusters = torch.arange(n_users) % 3
    scatter = plt.scatter(user_2d[:, 0], user_2d[:, 1], c=user_clusters.numpy(), cmap='tab10')
    plt.colorbar(scatter)
    plt.title('用户潜在空间 (t-SNE)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')

    # 物品潜在特征可视化
    plt.subplot(2, 3, 4)
    item_tsne = TSNE(n_components=2, random_state=42)
    item_2d = item_tsne.fit_transform(item_factors.numpy())

    # 根据物品类别着色
    item_categories = torch.arange(n_items) % 2
    scatter = plt.scatter(item_2d[:, 0], item_2d[:, 1], c=item_categories.numpy(), cmap='Set2')
    plt.colorbar(scatter)
    plt.title('物品潜在空间 (t-SNE)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')

    # 评分预测分布
    plt.subplot(2, 3, 5)
    plt.hist(predicted_ratings[observed_ratings > 0].flatten().numpy(),
             bins=30, alpha=0.7, label='观测交互的预测', color='blue')
    plt.hist(predicted_ratings[observed_ratings == 0].flatten().numpy(),
             bins=30, alpha=0.7, label='未观测交互的预测', color='red')
    plt.xlabel('预测评分')
    plt.ylabel('频数')
    plt.title('评分预测分布')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 推荐质量分析
    plt.subplot(2, 3, 6)

    def precision_at_k(user_id, k=10):
        """计算用户的前K精度"""
        user_ratings = observed_ratings[user_id]#从原始评分矩阵中提取指定用户的所有评分记录，返回该用户对应的行向量
        user_predictions = predicted_ratings[user_id]#从预测评分矩阵中提取指定用户所有的预测评分记录，返回该用户对应的行向量

        # 获取用户没有交互的物品
        unrated_items = torch.where(user_ratings == 0)[0]

        if len(unrated_items) == 0:
            return 0.0

        # 根据预测评分排序
        pred_scores = user_predictions[unrated_items]#获取用户没有交互的物品的预测评分
        top_k_indices = unrated_items[torch.argsort(pred_scores, descending=True)[:k]]#获取用户没有交互的物品的预测评分的Top-K索引

        # 检查这些物品是否在真实偏好中
        true_preferences = torch.where(binary_ratings[user_id] == 1)[0]#获取用户真实偏好的索引,即用户喜欢的物品索引
        hits = len(set(top_k_indices.numpy()) & set(true_preferences.numpy()))#计算用户没有交互的物品中真实偏好的物品的个数

        return hits / k #计算前K精度

    # 计算多个用户的精度
    k_values = [5, 10, 20]
    precision_results = {k: [] for k in k_values}

    for user_id in range(min(50, n_users)):  # 评估前50个用户
        for k in k_values:
            prec = precision_at_k(user_id, k)
            precision_results[k].append(prec)

    for k, precisions in precision_results.items():
        plt.plot([k] * len(precisions), precisions, 'o', alpha=0.6, label=f'P@{k}')

    plt.xlabel('K值')
    plt.ylabel('精度')
    plt.title('不同K值的推荐精度')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 推荐系统的高级特性
    print(f"\n高级推荐系统特性:")
    print("1. 带偏置的SVD考虑了个体差异和物品流行度")
    print("2. 正规算子理论保证了分解的数值稳定性")
    print("3. 谱方法提供了对推荐系统的理论理解")
    print("4. 潜在空间可视化揭示了用户和物品的聚类结构")

    # 生成个性化推荐
    def generate_personalized_recommendations(user_id, n_recommendations=10):
        """为特定用户生成个性化推荐"""
        user_embedding = user_factors[user_id]
        user_bias = user_biases[user_id]

        # 计算与所有物品的预测评分
        item_scores = (global_bias + user_bias + item_biases.squeeze(0) +
                       torch.matmul(item_factors, user_embedding))

        # 获取用户已经交互过的物品
        interacted_items = torch.where(observed_ratings[user_id] > 0)[0]

        # 排除已经交互过的物品
        mask = torch.ones(n_items, dtype=torch.bool)
        mask[interacted_items] = False
        available_items = torch.where(mask)[0]

        # 选择评分最高的物品
        available_scores = item_scores[available_items]
        top_indices = torch.argsort(available_scores, descending=True)[:n_recommendations]
        recommended_items = available_items[top_indices]
        recommendation_scores = available_scores[top_indices]

        return recommended_items, recommendation_scores

    # 为示例用户生成推荐
    example_user = 10
    recommended, scores = generate_personalized_recommendations(example_user)

    print(f"\n为用户 {example_user} 的个性化推荐:")
    for i, (item, score) in enumerate(zip(recommended, scores)):
        print(f"  推荐 {i + 1}: 物品 {item.item()}, 预测评分 {score.item():.3f}")


advanced_svd_recommendation_system()