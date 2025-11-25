# 实战项目1：基于内积的协同过滤推荐系统
import numpy as np
import torch
import matplotlib.pyplot as plt


def inner_product_recommendation_system():
    """
    使用内积构建简单的推荐系统
    """
    print("=== 基于内积的推荐系统 ===")

    # 创建模拟的用户-物品交互矩阵
    n_users=100#用户数量
    n_items=50#物品数量
    n_factors=10#潜在因子数量

    # 生成用户和物品的潜在因子（嵌入）
    torch.manual_seed(42)#设置随机数种子为42，确保每次运行代码时生成的随机数序列相同，保证实验结果可重现
    user_embeddings=torch.randn(n_users,n_factors)#生成用户嵌入矩阵，行数是用户数量，列数是潜在因子数量
    item_embeddings=torch.randn(n_items,n_factors)#生成物品嵌入矩阵，行数是物品数量，列数是潜在因子数量

    print(f"用户嵌入形状: {user_embeddings.shape}")
    print(f"物品嵌入形状: {item_embeddings.shape}")

    # 计算所有用户-物品对的内积（预测评分）
    predicted_ratings=torch.matmul(user_embeddings,item_embeddings.T)#计算用户嵌入矩阵和物品嵌入矩阵的转置的矩阵乘积，得到所有用户-物品对的内积（预测评分）
    print(f"预测评分形状: {predicted_ratings.shape}")

    # 模拟真实评分（二值化：1表示喜欢，0表示无交互）
    real_ratings=(torch.sigmoid(predicted_ratings)>0.6).float()#将预测评分通过sigmoid函数映射到[0,1]区间，大于0.6的被认为是喜欢（1），否则为无交互（0）
    print(f"真实交互矩阵稀疏度：{(real_ratings==0).sum()/real_ratings.numel():.1%}") # 计算真实评分矩阵的稀疏度，两者相除得到稀疏度，即未交互的比例
    #这行代码展示了推荐系统中典型的稀疏性问题：大多数用户-物品对没有交互记录。

    # 为特定用户做推荐
    def recommend_for_user(user_id,n_recommendations=5):#定义一个函数，用于为指定用户(user_id)做推荐,参数n_recommendations表示要返回的推荐数量
        user_embedding=user_embeddings[user_id]#从用户嵌入矩阵中提取指定用户的嵌入向量

        # 计算与所有物品的内积（相似度）
        similarities=torch.matmul(user_embedding,item_embeddings.T)#计算用户嵌入向量与物品嵌入矩阵的转置的矩阵乘积，得到所有物品的相似度（内积）

        # 获取用户已经交互过的物品
        interacted_items=torch.where(real_ratings[user_id]==1)[0]#从真实评分矩阵中提取指定用户(user_id)交互过的物品索引，即评分矩阵中值为1的位置（即用户喜欢的物品）

        # 排除已经交互过的物品
        mask=torch.ones(n_items,dtype=torch.bool)#创建一个布尔类型的掩码张量，长度是物品数量，所有元素初始化为True
        mask[interacted_items]=False#将用户已经交互过的物品在掩码中对应位置设为False，即排除这些物品
        available_items=torch.where(mask)[0]#获取剩余的可用物品索引

        # 在可用物品中选择最相似的
        available_similarities=similarities[available_items]#从相似度张量中提取剩余的可用物品的相似度
        top_indices=torch.argsort(available_similarities,descending=True)[:n_recommendations]#对可用物品的相似度进行排序，取Top-N个索引，即相似度最高的N个物品索引
        recommended_items=available_items[top_indices]#根据索引获取推荐的物品索引
        recommendation_scores=available_similarities[top_indices]#根据索引获取推荐物品的相似度（评分）
        return recommended_items,recommendation_scores#返回推荐的物品索引列表和对应的相似度（评分）

    # 测试推荐
    test_user=0#测试用户ID
    recommended_items,scores=recommend_for_user(test_user)#为测试用户(test_user)做推荐，返回推荐的物品索引列表和对应的相似度（评分）
    print(f"\n为用户 {test_user} 的推荐:")
    for i, (item, score) in enumerate(zip(recommended_items, scores)):
        print(f"  推荐物品 {item}: 预测得分 {score:.3f}")

    # 可视化嵌入空间
    from sklearn.manifold import TSNE

    # 使用t-SNE降维可视化
    combined_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(combined_embeddings.detach().numpy())

    user_embeddings_2d = embeddings_2d[:n_users]
    item_embeddings_2d = embeddings_2d[n_users:]

    plt.figure(figsize=(15, 5))

    # 用户和物品的嵌入分布
    plt.subplot(1, 3, 1)
    plt.scatter(user_embeddings_2d[:, 0], user_embeddings_2d[:, 1],
                alpha=0.6, label='用户', color='blue')
    plt.scatter(item_embeddings_2d[:, 0], item_embeddings_2d[:, 1],
                alpha=0.6, label='物品', color='red')
    plt.xlabel('t-SNE维度1')
    plt.ylabel('t-SNE维度2')
    plt.title('用户和物品的嵌入空间')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 特定用户与物品的相似度
    plt.subplot(1, 3, 2)
    user_id = 0
    user_similarities = torch.matmul(user_embeddings[user_id], item_embeddings.T)

    plt.hist(user_similarities.detach().numpy(), bins=30, alpha=0.7)
    plt.axvline(x=user_similarities[recommended_items].mean().item(),
                color='red', linestyle='--', label='推荐物品平均相似度')
    plt.xlabel('用户-物品内积相似度')
    plt.ylabel('频数')
    plt.title(f'用户 {user_id} 与所有物品的相似度分布')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 推荐解释：用户嵌入与物品嵌入的关系
    plt.subplot(1, 3, 3)
    # 选取几个推荐物品，展示它们与用户嵌入的关系
    n_to_show = min(3, len(recommended_items))

    user_vec = user_embeddings[user_id].detach().numpy()
    for i in range(n_to_show):
        item_idx = recommended_items[i].item()
        item_vec = item_embeddings[item_idx].detach().numpy()

        # 绘制向量
        plt.quiver(0, 0, user_vec[0], user_vec[1], color='blue',
                   scale=1, scale_units='xy', angles='xy', width=0.01,
                   label='用户嵌入' if i == 0 else "")
        plt.quiver(0, 0, item_vec[0], item_vec[1], color='red',
                   scale=1, scale_units='xy', angles='xy', width=0.01,
                   label=f'物品 {item_idx}' if i == 0 else f'物品 {item_idx}')

    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.title('用户嵌入与推荐物品嵌入')
    plt.legend()
    plt.axis('equal')

    plt.tight_layout()
    plt.show()

    # 推荐系统评估
    def evaluate_recommendation_quality():
        """评估推荐质量"""
        precisions=[]#存储准确率
        recalls=[]#存储召回率

        for user_id in range(min(20,n_users)):#评估前20个用户
            recommended,_=recommend_for_user(user_id,n_recommendations=5)#为用户做推荐，返回推荐物品索引列表和对应的相似度（评分）,n_recommendations=5表示返回前5个推荐物品

            # 模拟真实偏好（在实际系统中，这需要真实的测试数据）
            # 这里我们假设内积高的物品确实是用户喜欢的
            user_preferences=torch.where(real_ratings[user_id]==1)[0]#获取用户(user_id)喜欢的物品索引列表

            if len(user_preferences)>0:
                # 计算精度和召回率
                hits=len(set(recommended.numpy())&set(user_preferences.numpy()))#计算推荐物品中用户喜欢的物品数量（即命中数量）
                precision=hits/len(recommended)#计算准确率=命中数量/推荐物品数量
                recall=hits/len(user_preferences)if len(user_preferences)>0 else 0#计算召回率=命中数量/用户喜欢的物品数量

                precisions.append(precision)#将准确率添加到准确率列表中
                recalls.append(recall)#将召回率添加到召回率列表中

        avg_precision=np.mean(precisions)#计算平均准确率
        avg_recall=np.mean(recalls)#计算平均召回率

        print(f"\n推荐质量评估 (前20个用户):")
        print(f"平均精度: {avg_precision:.3f}")
        print(f"平均召回率: {avg_recall:.3f}")
        # 修改 F1 分数计算，避免除以零
        f1_score = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
        print(f"F1分数: {f1_score:.3f}")


    evaluate_recommendation_quality()
inner_product_recommendation_system()