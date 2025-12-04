# 数学概念：条件概率、全概率公式、贝叶斯定理
# AI对应：朴素贝叶斯分类器、贝叶斯网络、概率图模型

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import torch
import torch.nn as nn
import torch.distributions as dist


# 实践1：贝叶斯定理与朴素贝叶斯分类器
def bayes_theorem_naive_bayes():
    """
    贝叶斯定理在朴素贝叶斯分类器中的应用
    """
    print("=== 贝叶斯定理与朴素贝叶斯分类器 ===")

    # 加载鸢尾花数据集
    iris = load_iris()
    X = iris.data # 特征数据（输入变量）
    y = iris.target # 标签数据（目标变量）
    feature_names = iris.feature_names # 特征名称
    target_names = iris.target_names # 类别名称

    print(f"数据集信息:")
    print(f"  样本数量: {X.shape[0]}")
    print(f"  特征数量: {X.shape[1]}")
    print(f"  特征名称: {feature_names}")
    print(f"  类别名称: {target_names}")
    print(f"  类别分布: {np.bincount(y)}")

    '''
    参数解释：
        X: 特征数据（输入变量）
        y: 标签数据（目标变量）
        test_size=0.3: 指定测试集占总数据的30%
        random_state=42: 设置随机种子，保证每次运行结果一致
        stratify=y: 分层抽样，确保训练集和测试集中各类别的比例相同
    返回值：
        X_train: 训练集的特征数据
        X_test: 测试集的特征数据
        y_train: 训练集的标签数据
        y_test: 测试集的标签数据
    为什么要这样做？
        模型训练: 使用 X_train 和 y_train 来训练模型
        模型评估: 使用 X_test 进行预测，并与 y_test 对比来评估模型性能
        避免过拟合: 确保模型在未见过的数据上也能有良好表现
    '''
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 训练朴素贝叶斯分类器
    nb_classifier = GaussianNB() # 创建朴素贝叶斯分类器
    nb_classifier.fit(X_train, y_train) # 训练模型

    # 预测
    y_pred = nb_classifier.predict(X_test)#使用训练好的朴素贝叶斯分类器对测试集进行预测
    # 准确率 = 正确预测的样本数 / 总样本数
    accuracy = accuracy_score(y_test, y_pred)#计算模型预测的准确率，即正确分类的样本数占总样本数的比例
    '''
    工作流程
    1.预测阶段:
        模型根据学到的参数，对每个测试样本计算各类别的后验概率
        选择概率最高的类别作为预测结果
    2.评估阶段:
        将预测结果 y_pred 与真实标签 y_test 进行比较
        计算整体准确率来衡量模型性能
    '''

    print(f"\n朴素贝叶斯分类器性能:")
    print(f"  训练集大小: {X_train.shape[0]}")
    print(f"  测试集大小: {X_test.shape[0]}")
    print(f"  准确率: {accuracy:.4f}")

    # 手动实现贝叶斯定理
    def manual_bayes_theorem_demo():
        """手动计算贝叶斯定理示例"""
        print(f"\n贝叶斯定理手动计算示例:")

        # 假设一个简单的二分类问题
        # 特征: 花瓣长度 < 2.5cm 为特征F
        # 类别: Setosa (0) vs 非Setosa (1,2)

        # 创建简化数据集
        is_setosa = (y == 0) # 创建一个布尔数组，用于标标签为0的样本
        petal_length = X[:, 2]  # 从特征矩阵 X 中提取第3列（索引为2）的所有数据，花瓣长度，在鸢尾花数据集中，第3列（索引2）对应花瓣长度（petal length）

        # 定义特征: 花瓣长度 < 2.5
        feature_F = (petal_length < 2.5)

        # 计算先验概率 P(Setosa)
        P_Setosa = np.mean(is_setosa)
        P_notSetosa = 1 - P_Setosa

        # 计算似然度 P(F|Setosa) 和 P(F|notSetosa)
        P_F_given_Setosa = np.mean(feature_F[is_setosa])
        P_F_given_notSetosa = np.mean(feature_F[~is_setosa])

        # 计算证据 P(F)
        P_F = np.mean(feature_F) # 概率 P(F) = 特征F为True的样本数占总样本数的比例

        # 使用贝叶斯定理计算后验概率 P(Setosa|F)
        P_Setosa_given_F = (P_F_given_Setosa * P_Setosa) / P_F

        print(f"  先验概率 P(Setosa) = {P_Setosa:.4f}")
        print(f"  似然度 P(F|Setosa) = {P_F_given_Setosa:.4f}")
        print(f"  似然度 P(F|¬Setosa) = {P_F_given_notSetosa:.4f}")
        print(f"  证据 P(F) = {P_F:.4f}")
        print(f"  后验概率 P(Setosa|F) = {P_Setosa_given_F:.4f}")

        # 可视化
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 类别分布
        axes[0, 0].bar(range(3), np.bincount(y) / len(y))
        axes[0, 0].set_xlabel('类别')
        axes[0, 0].set_ylabel('先验概率')
        axes[0, 0].set_title('类别先验概率分布')
        axes[0, 0].set_xticks(range(3))
        axes[0, 0].set_xticklabels(target_names)

        # 特征分布（花瓣长度）
        axes[0, 1].hist([petal_length[is_setosa], petal_length[~is_setosa]],
                        bins=20, alpha=0.7, label=['Setosa', '非Setosa'], density=True)
        axes[0, 1].axvline(x=2.5, color='red', linestyle='--', label='阈值 2.5cm')
        axes[0, 1].set_xlabel('花瓣长度 (cm)')
        axes[0, 1].set_ylabel('概率密度')
        axes[0, 1].set_title('花瓣长度分布')
        axes[0, 1].legend()

        # 贝叶斯定理可视化
        axes[0, 2].bar(['P(Setosa)', 'P(¬Setosa)'], [P_Setosa, P_notSetosa], alpha=0.7)
        axes[0, 2].set_ylabel('概率')
        axes[0, 2].set_title('先验概率')

        # 似然度
        axes[1, 0].bar(['P(F|Setosa)', 'P(F|¬Setosa)'],
                       [P_F_given_Setosa, P_F_given_notSetosa], alpha=0.7, color='orange')
        axes[1, 0].set_ylabel('概率')
        axes[1, 0].set_title('似然度')

        # 后验概率
        axes[1, 1].bar(['P(Setosa|F)', 'P(¬Setosa|F)'],
                       [P_Setosa_given_F, 1 - P_Setosa_given_F], alpha=0.7, color='green')
        axes[1, 1].set_ylabel('概率')
        axes[1, 1].set_title('后验概率')

        # 贝叶斯公式展示
        axes[1, 2].text(0.1, 0.8, '贝叶斯定理:', fontsize=14, fontweight='bold')
        axes[1, 2].text(0.1, 0.6, f'P(Setosa|F) =', fontsize=12)
        axes[1, 2].text(0.1, 0.5, f'P(F|Setosa) × P(Setosa)', fontsize=12)
        axes[1, 2].text(0.1, 0.4, f'─────────────────────', fontsize=12)
        axes[1, 2].text(0.1, 0.3, f'        P(F)', fontsize=12)
        axes[1, 2].text(0.1, 0.1, f'= {P_F_given_Setosa:.3f} × {P_Setosa:.3f}', fontsize=12)
        axes[1, 2].text(0.1, 0.0, f'───────────────────── = {P_Setosa_given_F:.3f}', fontsize=12)
        axes[1, 2].text(0.1, -0.1, f'      {P_F:.3f}', fontsize=12)
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()

    manual_bayes_theorem_demo()

    # 朴素贝叶斯的概率解释
    def naive_bayes_probability_explanation():
        """解释朴素贝叶斯如何计算概率"""
        print(f"\n朴素贝叶斯概率计算:")

        # 获取一个测试样本
        test_sample = X_test[0]
        true_label = y_test[0]

        print(f"测试样本特征: {test_sample}")
        print(f"真实类别: {target_names[true_label]}")


        '''
        各参数含义：
        class_priors: 类别的先验概率
            存储每个类别的先验概率 P(类别)
            通过 nb_classifier.class_prior_ 属性获取
        means: 各特征在各类别下的均值
            存储每个特征在每个类别中的平均值
            通过 nb_classifier.theta_ 属性获取
            形状为 [类别数, 特征数]
        variances: 各特征在各类别下的方差
            存储每个特征在每个类别中的方差
            通过 nb_classifier.var_ 属性获取
            形状为 [类别数, 特征数]
        用途
            这些参数用于后续的概率计算：
            在朴素贝叶斯分类中，假设特征服从高斯分布
            通过均值和方差来描述每个特征在各类别下的分布
            结合先验概率计算后验概率，进行分类预测
        '''
        # 获取朴素贝叶斯的参数
        class_priors = nb_classifier.class_prior_# 每个类别的先验概率 P(类别)
        means = nb_classifier.theta_# 每个特征在每个类别下的均值
        variances = nb_classifier.var_# 每个特征在每个类别下的方差

        # 计算每个类别的后验概率（未归一化）
        log_probs = []
        for i in range(len(target_names)):
            # 计算对数概率
            log_prior = np.log(class_priors[i])

            # 假设特征独立，计算每个特征的对数似然
            log_likelihood = 0
            for j in range(len(feature_names)):

                '''
                使用场景：
                    这些参数用于计算高斯分布的概率密度，在朴素贝叶斯分类中：
                    假设每个特征在各类别下服从高斯分布
                    通过均值(mean)和方差(var)定义分布
                    使用测试样本的实际特征值(x)计算概率密度
                    最终用于计算后验概率进行分类决策
                '''
                # 使用高斯分布的概率密度
                mean = means[i, j]#第i个类别下第j个特征的均值，用于高斯分布的概率计算
                var = variances[i, j]#第i个类别下第j个特征的方差，用于高斯分布的概率计算
                x = test_sample[j]# 测试样本的第j个特征值，用于计算该特征值在各分布下的概率密度

                # 高斯分布的对数概率密度
                log_p = -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)
                log_likelihood += log_p

            #log_prior: 当前类别的对数先验概率，log_likelihood: 当前类别下所有特征的对数似然度之和
            log_prob = log_prior + log_likelihood
            log_probs.append(log_prob)#将计算得到的对数后验概率添加到列表中，每个类别都会计算并存储其对应的对数后验概率
            '''
            工作原理
                这体现了朴素贝叶斯分类器的核心计算过程：
                对每个类别计算其对数后验概率
                通过比较不同类别的对数后验概率来进行分类决策
                选择具有最高对数后验概率的类别作为预测结果
                后续会通过 softmax 函数将这些对数概率转换为实际的概率值。
            '''


        # 转换为概率（softmax），这三行代码用于将对数概率转换为实际概率值，并通过softmax函数进行归一化处理。
        log_probs = np.array(log_probs)#将Python列表转换为NumPy数组，便于后续进行向量化运算
        probs = np.exp(log_probs - np.max(log_probs))  #首先减去最大值避免指数运算溢出（数值稳定性技巧），然后进行指数运算将对数概率转换回原始概率空间，这实际上实现了softmax函数的核心部分
        probs = probs / probs.sum()# 将概率值标准化，使所有类别的概率之和等于1，确保输出的是合法的概率分布
        '''
        重要性
        这个过程完成了从对数后验概率到实际概率的转换：
            解决了直接计算指数可能导致的数值溢出问题
            保证了最终输出的概率值在[0,1]范围内
            确保所有类别的概率和为1，形成标准的概率分布
            便于后续的分类决策和概率解释
        '''

        print(f"每个类别的预测概率:")
        for i, name in enumerate(target_names):
            print(f"  {name}: {probs[i]:.4f}")

        predicted_class = np.argmax(probs)
        print(f"预测类别: {target_names[predicted_class]}")
        print(f"预测正确: {predicted_class == true_label}")

    naive_bayes_probability_explanation()


    '''
    这行代码用于计算混淆矩阵，是机器学习中评估分类模型性能的重要工具。
    功能说明：
        confusion_matrix：Scikit-learn 提供的函数，用于生成混淆矩阵
        y_test：测试集的真实标签
        y_pred：模型对测试集的预测结果
    混淆矩阵作用：
        可视化分类结果：展示每个类别的正确和错误分类情况
        详细性能分析：不仅看整体准确率，还能了解具体错误类型
        识别问题模式：发现哪些类别容易被混淆
    输出结果：
        一个二维数组，行表示真实类别，列表示预测类别
        对角线元素：正确分类的样本数
        非对角线元素：错误分类的样本数
        在后续代码中，这个混淆矩阵会被可视化为热力图，直观显示模型的分类表现。
    '''
    # 混淆矩阵和性能评估
    conf_mat = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title('朴素贝叶斯混淆矩阵')
    plt.show()

    return nb_classifier


nb_model = bayes_theorem_naive_bayes()