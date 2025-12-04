# 数学概念：期望、方差、协方差、相关系数、矩
# AI对应：特征工程、数据预处理、降维技术
import numpy as np
import matplotlib.pyplot as plt


# 实践3：数字特征在特征工程中的应用
def numerical_features_feature_engineering():
    """
    期望、方差、协方差等数字特征在特征工程中的应用
    """
    print("=== 数字特征与特征工程 ===")

    # 1. 加载复杂数据集
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    # 加载加州房价数据集
    housing = fetch_california_housing()  # 使用sklearn获取加州房价数据集
    X_housing = housing.data  # 提取特征数据（输入变量）
    y_housing = housing.target  # 提取目标变量（房价）
    feature_names = housing.feature_names  # 获取特征名称列表

    print(f"加州房价数据集:")
    print(f"  样本数量: {X_housing.shape[0]}")
    print(f"  特征数量: {X_housing.shape[1]}")
    print(f"  特征名称: {feature_names}")

    # 计算基本统计量
    def calculate_basic_statistics(X, feature_names):
        """计算基本统计量"""
        n_features = X.shape[1]#获取特征数量

        stats = {
            '均值': np.mean(X, axis=0),
            '方差': np.var(X, axis=0),
            '标准差': np.std(X, axis=0),
            '偏度': np.zeros(n_features),
            '峰度': np.zeros(n_features),
            '最小值': np.min(X, axis=0),
            '最大值': np.max(X, axis=0),
            '中位数': np.median(X, axis=0),
            'Q1': np.percentile(X, 25, axis=0),
            'Q3': np.percentile(X, 75, axis=0),
            'IQR': np.zeros(n_features)
        }

        # 计算偏度和峰度
        from scipy.stats import skew, kurtosis
        stats['偏度'] = skew(X, axis=0)  # 计算每个特征的偏度（三阶中心矩），衡量分布的不对称性
        stats['峰度'] = kurtosis(X, axis=0)  # 计算每个特征的峰度（四阶中心矩），衡量分布的尖锐程度
        stats['IQR'] = stats['Q3'] - stats['Q1']  # 计算四分位距（IQR），即第三四分位数与第一四分位数的差值

        return stats

    stats = calculate_basic_statistics(X_housing, feature_names)#计算基本统计量,返回统计量字典

    # 可视化统计量
    fig, axes = plt.subplots(3, 4, figsize=(15, 12))
    axes = axes.flatten()

    stat_names = ['均值', '方差', '标准差', '偏度', '峰度',
                  '最小值', '最大值', '中位数', 'Q1', 'Q3', 'IQR']

    for idx, stat_name in enumerate(stat_names[:11]):
        ax = axes[idx]
        values = stats[stat_name]

        bars = ax.bar(range(len(feature_names)), values, alpha=0.7)
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45)
        ax.set_ylabel(stat_name)
        ax.set_title(f'{stat_name}分布')
        ax.grid(True, alpha=0.3)

        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01 * max(values),
                    f'{value:.2f}', ha='center', va='bottom', fontsize=8)

    axes[-1].axis('off')
    plt.tight_layout()
    plt.show()

    # 2. 协方差与相关系数分析
    def covariance_correlation_analysis(X, feature_names):
        """协方差与相关系数分析"""
        print("\n2. 协方差与相关系数分析:")

        # 计算协方差矩阵和相关系数矩阵
        cov_matrix = np.cov(X.T) #计算协方差矩阵，X.T表示将特征矩阵转置，使每个特征成为一个样本
        corr_matrix = np.corrcoef(X.T) #计算相关系数矩阵，相关系数矩阵是一个对称矩阵，每个元素的范围在[-1, 1]之间

        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 协方差矩阵热图
        im1 = axes[0].imshow(cov_matrix, cmap='RdBu_r', aspect='auto')
        axes[0].set_xticks(range(len(feature_names)))
        axes[0].set_yticks(range(len(feature_names)))
        axes[0].set_xticklabels(feature_names, rotation=45)
        axes[0].set_yticklabels(feature_names)
        axes[0].set_title('协方差矩阵')
        plt.colorbar(im1, ax=axes[0])

        # 添加数值
        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
                text = axes[0].text(j, i, f'{cov_matrix[i, j]:.2f}',
                                    ha="center", va="center", color="black", fontsize=8)

        # 相关系数矩阵热图
        im2 = axes[1].imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        axes[1].set_xticks(range(len(feature_names)))
        axes[1].set_yticks(range(len(feature_names)))
        axes[1].set_xticklabels(feature_names, rotation=45)
        axes[1].set_yticklabels(feature_names)
        axes[1].set_title('相关系数矩阵')
        plt.colorbar(im2, ax=axes[1])

        # 添加数值
        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
                text = axes[1].text(j, i, f'{corr_matrix[i, j]:.2f}',
                                    ha="center", va="center", color="black", fontsize=8)

        plt.tight_layout()
        plt.show()

        # 找出高度相关的特征对
        print("\n高度相关的特征对 (|r| > 0.7):")
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                corr = corr_matrix[i, j]
                if abs(corr) > 0.7:
                    print(f"  {feature_names[i]} - {feature_names[j]}: r = {corr:.3f}")

        return cov_matrix, corr_matrix

    cov_matrix, corr_matrix = covariance_correlation_analysis(X_housing, feature_names)

    # 3. 基于统计量的特征工程
    def feature_engineering_techniques(X, y, feature_names):
        """基于统计量的特征工程技术"""
        print("\n3. 基于统计量的特征工程:")

        # 原始特征
        X_original = X.copy()

        # 特征工程技术
        engineered_features = {}

        # 1. 标准化 (Z-score归一化)
        scaler_standard = StandardScaler()#创建一个StandardScaler对象，用于对特征进行Z-score归一化
        X_standardized = scaler_standard.fit_transform(X)#对特征矩阵X进行Z-score归一化，返回归一化后的特征矩阵
        engineered_features['标准化'] = X_standardized

        # 2. 最小-最大归一化
        scaler_minmax = MinMaxScaler()#创建一个MinMaxScaler对象，用于对特征进行最小-最大归一化
        X_minmax = scaler_minmax.fit_transform(X)#对特征矩阵X进行最小-最大归一化，返回归一化后的特征矩阵
        engineered_features['最小-最大归一化'] = X_minmax

        # 3. 基于方差的特征选择 (移除低方差特征)
        from sklearn.feature_selection import VarianceThreshold
        var_threshold = VarianceThreshold(threshold=0.1)  # 移除方差小于0.1的特征
        X_var_selected = var_threshold.fit_transform(X) #对特征矩阵X进行基于方差的特征选择，返回选择后的特征矩阵
        selected_mask = var_threshold.get_support() #获取选择后的特征索引
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]] #根据选择后的特征索引获取选择的特征名称列表

        print(f"基于方差的特征选择:")
        print(f"  原始特征数: {len(feature_names)}")
        print(f"  选择后特征数: {len(selected_features)}")
        print(f"  选择的特征: {selected_features}")

        engineered_features['方差选择'] = X_var_selected

        # 4. 创建交互特征
        # 选择两个高度相关的特征创建交互项
        high_corr_pairs = []
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                if abs(corr_matrix[i, j]) > 0.5:
                    high_corr_pairs.append((i, j))

        # 创建交互特征
        X_with_interaction = np.hstack([X_original])
        interaction_names = list(feature_names)

        for idx, (i, j) in enumerate(high_corr_pairs[:3]):  # 只创建前3个交互特征
            interaction = X_original[:, i] * X_original[:, j] #创建交互项，即两个特征的乘积
            X_with_interaction = np.column_stack([X_with_interaction, interaction]) #将交互项添加到特征矩阵中
            interaction_names.append(f"{feature_names[i]}×{feature_names[j]}") #将交互项的名称添加到特征名称列表中

        engineered_features['交互特征'] = X_with_interaction

        # 5. 创建多项式特征 (二次项)
        X_poly = np.hstack([X_original]) #创建一个特征矩阵，将原始特征添加到特征矩阵中
        poly_names = list(feature_names) #创建一个特征名称列表，将原始特征名称添加到列表中

        for i in range(min(3, len(feature_names))):  # 为前3个特征创建二次项
            squared = X_original[:, i] ** 2 #创建二次项，即特征的平方
            X_poly = np.column_stack([X_poly, squared]) #将二次项添加到特征矩阵中
            poly_names.append(f"{feature_names[i]}²") #将二次项的名称添加到特征名称列表中

        engineered_features['多项式特征'] = X_poly

        # 可视化不同特征工程技术的效果
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        techniques = ['原始特征', '标准化', '最小-最大归一化', '方差选择', '交互特征', '多项式特征']

        for idx, (technique, ax) in enumerate(zip(techniques, axes.flatten())):
            if technique == '原始特征':
                data_to_show = X_original
                title = '原始特征'
            elif technique == '方差选择':
                data_to_show = engineered_features['方差选择']
                title = f'方差选择\n{len(selected_features)}个特征'
            elif technique == '交互特征':
                data_to_show = engineered_features['交互特征']
                title = f'交互特征\n{len(interaction_names)}个特征'
            elif technique == '多项式特征':
                data_to_show = engineered_features['多项式特征']
                title = f'多项式特征\n{len(poly_names)}个特征'
            else:
                data_to_show = engineered_features[technique]
                title = technique

            # 显示前两个特征的散点图
            if data_to_show.shape[1] >= 2:
                scatter = ax.scatter(data_to_show[:, 0], data_to_show[:, 1],
                                     c=y_housing, cmap='viridis', alpha=0.6, s=10)
                ax.set_xlabel('特征1' if technique == '原始特征' else '转换后特征1')
                ax.set_ylabel('特征2' if technique == '原始特征' else '转换后特征2')
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'特征维度: {data_to_show.shape[1]}',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
                ax.axis('off')

        plt.tight_layout()
        plt.show()

        # 评估特征工程技术对模型性能的影响
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score

        print("\n特征工程技术对线性回归性能的影响:")
        print("技术".ljust(20) + "特征数量".ljust(15) + "R²得分".ljust(15) + "提升%")
        print("-" * 60)

        baseline_score = None  # 初始化基准R²得分为None

        # 遍历每种特征工程技术，评估其对模型性能的影响
        for technique, X_engineered in engineered_features.items():
            model = LinearRegression()  # 创建线性回归模型
            scores = cross_val_score(model, X_engineered, y_housing,
                                     cv=5, scoring='r2')  # 使用5折交叉验证计算R²得分
            mean_score = np.mean(scores)  # 计算平均R²得分

            # 计算相对于基准模型的性能提升百分比
            if baseline_score is None:
                baseline_score = mean_score  # 将第一个技术作为基准
                improvement = 0.0
            else:
                improvement = ((mean_score - baseline_score) / abs(baseline_score)) * 100  # 计算提升百分比

            # 打印结果：技术名称、特征数量、R²得分和相对提升
            print(f"{technique.ljust(20)} {str(X_engineered.shape[1]).ljust(15)} "
                  f"{mean_score:.4f}".ljust(15) + f"{improvement:+.1f}%")

        return engineered_features

    engineered_features = feature_engineering_techniques(X_housing, y_housing, feature_names)

    # 4. 主成分分析 (PCA) - 基于协方差矩阵的降维
    def pca_dimensionality_reduction(X, feature_names):
        """基于协方差矩阵的主成分分析"""
        print("\n4. 基于协方差矩阵的主成分分析:")

        from sklearn.decomposition import PCA

        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 应用PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)

        # PCA结果分析
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        print("主成分解释方差比例:")
        for i, (var_ratio, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
            print(f"  PC{i + 1}: {var_ratio:.3f} ({cum_var:.3f} 累计)")

        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 碎石图
        axes[0, 0].plot(range(1, len(explained_variance_ratio) + 1),
                        explained_variance_ratio, 'bo-')
        axes[0, 0].set_xlabel('主成分')
        axes[0, 0].set_ylabel('解释方差比例')
        axes[0, 0].set_title('碎石图')
        axes[0, 0].grid(True, alpha=0.3)

        # 累计解释方差
        axes[0, 1].plot(range(1, len(cumulative_variance) + 1),
                        cumulative_variance, 'ro-')
        axes[0, 1].axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='95%')
        axes[0, 1].axhline(y=0.90, color='orange', linestyle='--', alpha=0.5, label='90%')
        axes[0, 1].set_xlabel('主成分数量')
        axes[0, 1].set_ylabel('累计解释方差比例')
        axes[0, 1].set_title('累计解释方差')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 主成分载荷（前两个主成分）
        axes[1, 0].bar(range(len(feature_names)), pca.components_[0, :])
        axes[1, 0].set_xticks(range(len(feature_names)))
        axes[1, 0].set_xticklabels(feature_names, rotation=45)
        axes[1, 0].set_ylabel('载荷')
        axes[1, 0].set_title('第一主成分载荷')
        axes[1, 0].grid(True, alpha=0.3)

        # 前两个主成分的散点图
        scatter = axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1],
                                     c=y_housing, cmap='viridis', alpha=0.6, s=10)
        axes[1, 1].set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%})')
        axes[1, 1].set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%})')
        axes[1, 1].set_title('前两个主成分')
        plt.colorbar(scatter, ax=axes[1, 1], label='房价')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 分析主成分的意义
        print("\n第一主成分载荷分析 (最重要的方向):")
        for i, (feature, loading) in enumerate(zip(feature_names, pca.components_[0, :])):
            print(f"  {feature}: {loading:.3f}")

        return pca, X_pca

    pca_model, X_pca = pca_dimensionality_reduction(X_housing, feature_names)

    return engineered_features, pca_model, X_pca


engineered_features, pca_model, X_pca = numerical_features_feature_engineering()