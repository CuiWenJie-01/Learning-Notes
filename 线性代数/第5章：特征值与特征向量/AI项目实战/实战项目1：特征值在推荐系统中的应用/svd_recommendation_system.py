# 实战项目1：SVD在推荐系统中的应用
import numpy as np
import matplotlib.pyplot as plt

def svd_recommendation_system():
    """
    奇异值分解（SVD）是特征分解的推广，在推荐系统中广泛应用
    """
    print("=== SVD在推荐系统中的应用 ===")

    # 创建用户-物品评分矩阵（稀疏）
    # 行：用户，列：物品，值：评分（1-5）
    #这个评分矩阵将作为SVD算法的输入数据，用来演示如何基于奇异值分解实现推荐功能。
    np.random.seed(2)#设置随机数种子为2，确保每次运行代码时生成的随机数序列相同，保证实验结果可重现
    n_users,n_items=100,50#：定义用户数量为100，物品数量为50
    ratings=np.random.randint(1,6,size=(n_users,n_items))#生成一个100×50的评分矩阵，矩阵中的每个元素都是1到5之间的随机整数（模拟1-5星评分）

    # 添加稀疏性（很多用户没有评分）
    # 生成0-1之间的随机数,通过 > 0.7 的条件判断，约有30%的元素为True，70%为False，得到一个布尔型掩码矩阵 mask
    mask=np.random.random((n_users,n_items))>0.7
    # ~mask 表示对掩码矩阵取反（原来是True的地方变为False，False变为True）
    # 将评分矩阵 ratings 中对应掩码为False的位置设为0
    # 这样就模拟了现实情况：大部分用户只对少数物品进行了评分，其余位置为空（用0表示未评分）
    ratings[~mask]=0 # 0表示没有评分

    print(f"评分矩阵形状: {ratings.shape}")
    print(f"稀疏度：{(ratings==0).sum()/ratings.size:.1%}")
    print(f"平均评分（非零）：{ratings[ratings>0].mean():.2f}")# mean() 是 NumPy 数组的方法，计算数组中所有元素的算术平均值

    # 使用SVD进行矩阵分解
    # 注意：实际推荐系统会使用更复杂的方法处理缺失值
    from scipy.sparse.linalg import svds

    # 只对非零评分进行分解（简化处理）
    ratings_filled=ratings.copy()#创建 ratings 矩阵的一个副本，避免修改原始数据，保留原始的稀疏矩阵结构
    mean_rating=ratings[ratings>0].mean()#计算所有非零评分的平均值，这个均值将用于填充缺失的评分数据
    ratings_filled[ratings_filled==0]=mean_rating#将 ratings_filled 矩阵中所有为0的位置（未评分项），替换为计算得到的平均评分 mean_rating
    #这种处理方式是为了让SVD算法能够处理完整的矩阵数据，因为在实际应用中，SVD需要完整的数值矩阵才能进行分解计算。通过用均值填充缺失值，可以在一定程度上保持数据的整体分布特征。

    # 执行SVD
    k=10 # 潜在因子数量
    U,sigma,Vt=svds(ratings_filled,k=k)

    print(f"\nSVD结果:")
    print(f"U矩阵形状（用户潜在特征）: {U.shape}")
    print(f"奇异值: {sigma}")
    print(f"Vt矩阵形状（物品潜在特征）: {Vt.shape}")

    # 重建评分矩阵,是SVD分解后的关键步骤
    '''
    将一维的奇异值数组 sigma 转换为对角矩阵
    np.diag() 函数创建一个对角矩阵，对角线上的元素为奇异值
    这样可以满足矩阵乘法的维度要求
    '''
    sigma_matrix=np.diag(sigma)
    '''
    执行矩阵乘法运算，重构预测评分矩阵
    @ 是Python中的矩阵乘法运算符
    根据SVD的原理：原始矩阵 ≈ U × Σ × Vt
    通过这三个矩阵的乘积得到近似的完整评分矩阵 ratings_pred
    这个预测矩阵可以用于推荐，填补原始稀疏矩阵中的缺失值
    '''
    ratings_pred=U@sigma_matrix@Vt
    #这个重构过程是SVD推荐系统的核心，通过降维的潜在因子来预测用户对未评分物品的偏好

    print(f"\n重建矩阵形状: {ratings_pred.shape}")
    print(f"原始评分范围: [{ratings.min()}, {ratings.max()}]")
    print(f"预测评分范围: [{ratings_pred.min():.2f}, {ratings_pred.max():.2f}]")

    # 为用户做推荐
    # 这个函数的目的是基于SVD预测结果，为指定用户找出他们尚未评分但可能感兴趣的物品。后续代码会根据这两个向量来确定具体的推荐列表。
    def recommend_for_user(user_id,n_recommendations=5):#定义一个函数，接收用户ID和推荐数量作为参数，设置默认推荐5个物品
        user_ratings=ratings[user_id]#从原始评分矩阵 ratings 中提取指定用户的所有评分记录，返回该用户对应的行向量
        user_pred=ratings_pred[user_id]#从预测评分矩阵 ratings_pred 中提取该用户的预测评分，返回该用户对所有物品的预测评分向量

        # 找出用户没有评分的物品
        '''
        使用 np.where() 函数查找 user_ratings 数组中值为0的元素位置
        user_ratings == 0 生成一个布尔数组，标记哪些物品未被用户评分
        np.where() 返回满足条件的索引，由于是一维数组，所以取 [0] 获取索引数组
        结果 unrated_items 包含了该用户所有未评分物品的索引列表
        '''
        #找出指定用户还没有评分的物品，为后续基于预测评分进行推荐做准备
        unrated_items=np.where(user_ratings==0)[0]

        # 根据预测评分排序
        '''
        pred_scores = user_pred[unrated_items]：
            从该用户的所有预测评分 user_pred 中
            提取出未评分物品 unrated_items 对应的预测评分
            得到一个只包含未评分物品预测分数的数组
        np.argsort(pred_scores)[::-1][:n_recommendations]：
            np.argsort(pred_scores) 对预测分数进行升序排序，返回排序后的索引
            [::-1] 将索引数组反转，变成降序排列
            [:n_recommendations] 取前 n_recommendations 个最高分的索引
        recommend_indices = unrated_items[np.argsort(pred_scores)[::-1][:n_recommendations]]：
            将排序后的预测分数索引映射回原始物品索引
            得到推荐物品在原矩阵中的实际索引
        return recommend_indices, pred_scores[np.argsort(pred_scores)[::-1][:n_recommendations]]：
            返回推荐物品的索引和对应的预测分数
            用于展示给用户看推荐结果和推荐理由（预测分数）
        '''
        pred_scores=user_pred[unrated_items]
        recommend_indices=unrated_items[np.argsort(pred_scores)[::-1][:n_recommendations]]
        return recommend_indices,pred_scores[np.argsort(pred_scores)[::-1][:n_recommendations]]

    # 为示例用户做推荐
    example_user=0 #设置示例用户的ID为0（即第一个用户），设置示例用户的ID为0（即第一个用户）
    '''
    调用之前定义的 recommend_for_user 函数
    为 example_user 用户生成推荐
    返回两个值：
        recommended_items：推荐物品的索引列表
        scores：对应的预测评分分数
    '''
    #这部分代码展示了如何使用训练好的SVD模型为具体用户生成个性化推荐列表。
    recommended_items,scores=recommend_for_user(example_user)

    print(f"\n为用户 {example_user} 的推荐:")
    for i, (item, score) in enumerate(zip(recommended_items, scores)):
        print(f"  推荐物品 {item}: 预测评分 {score:.2f}")

    # 可视化潜在空间
    plt.figure(figsize=(15, 5))

    # 用户潜在特征
    plt.subplot(1, 3, 1)
    plt.scatter(U[:, 0], U[:, 1], alpha=0.6)
    plt.xlabel('潜在维度 1')
    plt.ylabel('潜在维度 2')
    plt.title('用户潜在特征空间')
    plt.grid(True, alpha=0.3)

    # 物品潜在特征
    plt.subplot(1, 3, 2)
    plt.scatter(Vt[0, :], Vt[1, :], alpha=0.6)
    plt.xlabel('潜在维度 1')
    plt.ylabel('潜在维度 2')
    plt.title('物品潜在特征空间')
    plt.grid(True, alpha=0.3)

    # 奇异值（特征值的平方根）
    '''
    range(1, len(sigma) + 1)：生成x轴坐标，从1开始到奇异值个数结束
    sigma：y轴坐标，即计算得到的奇异值数组
    'ro-'：绘图样式参数
        r：红色(red)
        o：圆形标记(circle marker)
        -：实线连接(line)
    该图展示了各个奇异值的大小变化，反映了不同潜在因子的重要性程度。通常情况下，前面几个奇异值较大，包含更多信息，而后面的奇异值较小，代表的信息较少。这种可视化有助于理解SVD降维的效果和各成分的信息贡献度。
    '''
    plt.subplot(1, 3, 3)
    plt.plot(range(1, len(sigma) + 1), sigma, 'ro-')
    plt.xlabel('成分索引')
    plt.ylabel('奇异值')
    plt.title('奇异值谱（信息含量）')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 解释SVD与特征分解的关系
    print(f"\nSVD与特征分解的关系:")
    print(f"  A^T * A 的特征值 = 奇异值的平方")
    print(f"  A^T * A 的特征向量 = V 矩阵的列")
    print(f"  A * A^T 的特征向量 = U 矩阵的列")

svd_recommendation_system()
