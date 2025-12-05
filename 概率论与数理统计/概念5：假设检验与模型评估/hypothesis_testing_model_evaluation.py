# 数学概念：假设检验、p值、显著性水平、统计功效
# AI对应：模型评估、特征重要性检验、A/B测试
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from scipy.stats import t, chi2 as chi2_dist
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest, proportion_confint


# 实践5：假设检验在AI模型评估中的应用
def hypothesis_testing_model_evaluation():
    """
    假设检验在机器学习模型评估中的应用
    """
    print("=== 假设检验与模型评估 ===")

    # 1. 模型性能的统计显著性检验
    def model_performance_significance(X_train, y_train,X_test, y_test):
        """模型性能的统计显著性检验"""
        print("\n1. 模型性能的统计显著性检验:")

        # 训练和评估
        models = {
            '逻辑回归': LogisticRegression(max_iter=1000),
            '随机森林': RandomForestClassifier(n_estimators=100, random_state=42),
            '支持向量机': SVC(probability=True, random_state=42)
        }

        predictions = {} # 保存预测结果
        accuracies = {} # 保存模型准确率

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)  # 使用测试集进行预测
            predictions[name] = y_pred
            accuracies[name] = np.mean(y_pred == y_test)

        print(f"模型准确率:")
        for name, acc in accuracies.items():
            print(f"  {name}: {acc:.4f}")

        # McNemar检验：比较两个分类器的性能
        def mcnemar_test(y_true, y_pred1, y_pred2):
            """McNemar检验比较两个分类器"""
            # 构建列联表
            n00 = np.sum((y_pred1 == y_true) & (y_pred2 == y_true))
            n01 = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
            n10 = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
            n11 = np.sum((y_pred1 != y_true) & (y_pred2 != y_true))

            contingency_table = np.array([[n00, n01], [n10, n11]]) # 构建列联表

            # McNemar统计量（带连续性校正），只考虑模型预测不一致的情况（b和c）
            chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)

            p_value = 1 - chi2_dist.cdf(chi2, df=1) # 计算p值，只考虑模型预测不一致的情况（b和c）

            return chi2, p_value, contingency_table

        # 比较逻辑回归和随机森林
        y_pred_lr = predictions['逻辑回归']
        y_pred_rf = predictions['随机森林']

        chi2, p_value, cont_table = mcnemar_test(y_test, y_pred_lr, y_pred_rf)

        print(f"\nMcNemar检验 (逻辑回归 vs 随机森林):")
        print(f"  列联表:")
        print(f"           随机森林正确 | 随机森林错误")
        print(f"  逻辑回归正确: {cont_table[0, 0]:4d} | {cont_table[0, 1]:4d}")
        print(f"  逻辑回归错误: {cont_table[1, 0]:4d} | {cont_table[1, 1]:4d}")
        print(f"  χ²统计量: {chi2:.4f}")
        print(f"  p值: {p_value:.4f}")

        alpha = 0.05
        if p_value < alpha:
            print(f"  结论: 在α={alpha}水平下，两个模型性能有显著差异")
        else:
            print(f"  结论: 在α={alpha}水平下，两个模型性能没有显著差异")

        # 可视化
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 准确率比较
        model_names = list(accuracies.keys())
        acc_values = list(accuracies.values())

        bars = axes[0, 0].bar(model_names, acc_values)
        axes[0, 0].set_ylabel('准确率')
        axes[0, 0].set_title('模型准确率比较')
        axes[0, 0].set_ylim(0.5, 1.0)
        axes[0, 0].grid(True, alpha=0.3)

        for bar, acc in zip(bars, acc_values):
            # bar.get_x() + bar.get_width() / 2: 计算条形中心的x坐标
            # bar.get_height() + 0.01: 计算文本的y坐标(条形高度+0.01的偏移量)
            axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f'{acc:.3f}', ha='center', va='bottom')

        # 混淆矩阵热图（逻辑回归）
        conf_mat_lr = confusion_matrix(y_test, y_pred_lr)

        im1 = axes[0, 1].imshow(conf_mat_lr, cmap='Blues')
        axes[0, 1].set_xticks([0, 1])
        axes[0, 1].set_yticks([0, 1])
        axes[0, 1].set_xticklabels(['预测0', '预测1'])
        axes[0, 1].set_yticklabels(['真实0', '真实1'])
        axes[0, 1].set_title('逻辑回归混淆矩阵')

        for i in range(2):
            for j in range(2):
                axes[0, 1].text(j, i, f'{conf_mat_lr[i, j]}',
                                ha='center', va='center', color='black', fontsize=12)

        # McNemar检验列联表
        im2 = axes[0, 2].imshow(cont_table, cmap='RdBu_r')
        axes[0, 2].set_xticks([0, 1])
        axes[0, 2].set_yticks([0, 1])
        axes[0, 2].set_xticklabels(['RF正确', 'RF错误'])
        axes[0, 2].set_yticklabels(['LR正确', 'LR错误'])
        axes[0, 2].set_title('McNemar检验列联表')

        for i in range(2):
            for j in range(2):
                axes[0, 2].text(j, i, f'{cont_table[i, j]}',
                                ha='center', va='center', color='black', fontsize=12)

        # p值解释
        axes[1, 0].barh(['p值'], [p_value])
        axes[1, 0].axvline(x=0.05, color='red', linestyle='--', label='α=0.05')
        axes[1, 0].set_xlabel('p值')
        axes[1, 0].set_title('显著性检验')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 不同样本量下的p值变化（模拟）
        axes[1, 1].plot(range(100, 1001, 100),
                        [0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001],
                        'bo-')
        axes[1, 1].axhline(y=0.05, color='red', linestyle='--', label='α=0.05')
        axes[1, 1].set_xlabel('样本量')
        axes[1, 1].set_ylabel('p值')
        axes[1, 1].set_title('样本量对p值的影响')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 统计功效分析
        axes[1, 2].plot([0.1, 0.2, 0.3, 0.4, 0.5],
                        [0.3, 0.5, 0.7, 0.85, 0.95], 'ro-')
        axes[1, 2].set_xlabel('效应大小 (准确率差异)')
        axes[1, 2].set_ylabel('统计功效 (1-β)')
        axes[1, 2].set_title('统计功效分析')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return models, predictions, accuracies, chi2, p_value

    # 2. 特征重要性的假设检验
    def feature_importance_hypothesis_test(X_train, y_train, true_weights, models):
        """特征重要性的假设检验"""
        print("\n2. 特征重要性的假设检验:")

        # 使用逻辑回归模型
        lr_model = models['逻辑回归']

        # 获取系数
        coefficients = lr_model.coef_[0]

        # 计算系数的标准误差（使用自助法）
        n_bootstrap = 1000
        n_samples = len(X_train)
        bootstrap_coefs = np.zeros((n_bootstrap, len(coefficients))) # 创建一个数组，用于存储自举结果

        for i in range(n_bootstrap):
            # 自助采样
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X_train[indices] # 自举数据
            y_boot = y_train[indices] # 自举标签

            # 训练模型
            # 创建逻辑回归模型实例，设置最大迭代次数为1000
            lr_boot = LogisticRegression(max_iter=1000)
            # 在自助采样的数据上训练逻辑回归模型
            lr_boot.fit(X_boot, y_boot)
            # 将训练好的模型的系数(权重)存储到bootstrap_coefs数组的第i行
            # coef_[0]表示获取第一个(也是唯一一个)类别对应的系数数组
            bootstrap_coefs[i] = lr_boot.coef_[0]

        # 计算标准误差和置信区间
        coef_se = np.std(bootstrap_coefs, axis=0)
        coef_ci_lower = np.percentile(bootstrap_coefs, 2.5, axis=0)
        coef_ci_upper = np.percentile(bootstrap_coefs, 97.5, axis=0)

        # t检验：检验系数是否显著不为0
        t_stats = coefficients / coef_se # 计算t统计量
        p_values = 2 * (1 - t.cdf(np.abs(t_stats), df=n_samples - 1)) # 计算p值

        print(f"特征重要性检验:")
        print("特征".ljust(15) + "系数".ljust(15) + "标准误".ljust(15) +
              "t统计量".ljust(15) + "p值".ljust(15) + "显著")
        print("-" * 90)

        significant_features = []
        for i in range(len(coefficients)):
            is_significant = p_values[i] < 0.05
            if is_significant:
                significant_features.append(i)

            print(f"特征{i}".ljust(15) +
                  f"{coefficients[i]:.4f}".ljust(15) +
                  f"{coef_se[i]:.4f}".ljust(15) +
                  f"{t_stats[i]:.4f}".ljust(15) +
                  f"{p_values[i]:.4f}".ljust(15) +
                  f"{'是' if is_significant else '否'}")

        print(f"\n显著特征数量: {len(significant_features)}")
        print(f"显著特征索引: {significant_features}")
        print(f"真实重要特征索引: {np.where(true_weights != 0)[0]}")

        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 系数估计与置信区间
        axes[0, 0].errorbar(range(len(coefficients)), coefficients,
                            yerr=1.96 * coef_se, fmt='o', capsize=5, alpha=0.7)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('特征索引')
        axes[0, 0].set_ylabel('系数估计')
        axes[0, 0].set_title('逻辑回归系数估计与95%置信区间')
        axes[0, 0].grid(True, alpha=0.3)

        # p值分布
        axes[0, 1].hist(p_values, bins=20, alpha=0.7)
        axes[0, 1].axvline(x=0.05, color='red', linestyle='--', label='α=0.05')
        axes[0, 1].set_xlabel('p值')
        axes[0, 1].set_ylabel('频数')
        axes[0, 1].set_title('特征重要性检验的p值分布')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 自助法系数分布（前几个特征）
        axes[1, 0].boxplot(bootstrap_coefs[:, :5].T)
        axes[1, 0].set_xticks(range(1, 6))
        axes[1, 0].set_xticklabels([f'特征{i}' for i in range(5)])
        axes[1, 0].set_ylabel('系数值')
        axes[1, 0].set_title('前5个特征的自助法系数分布')
        axes[1, 0].grid(True, alpha=0.3)

        # 真实系数 vs 估计系数
        axes[1, 1].scatter(true_weights[:len(coefficients)], coefficients, alpha=0.7)

        # 添加置信椭圆
        from matplotlib.patches import Ellipse
        for i in range(min(5, len(coefficients))):
            ell = Ellipse(xy=(true_weights[i], coefficients[i]),
                          width=2 * 1.96 * coef_se[i], height=2 * 1.96 * coef_se[i],
                          angle=0, alpha=0.2)
            axes[1, 1].add_patch(ell)

        # 对角线
        lims = [
            min(axes[1, 1].get_xlim()[0], axes[1, 1].get_ylim()[0]),
            max(axes[1, 1].get_xlim()[1], axes[1, 1].get_ylim()[1])
        ]
        axes[1, 1].plot(lims, lims, 'k--', alpha=0.5)

        axes[1, 1].set_xlabel('真实系数')
        axes[1, 1].set_ylabel('估计系数')
        axes[1, 1].set_title('真实系数 vs 估计系数')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return coefficients, coef_se, p_values, significant_features

    # 3. A/B测试在推荐系统中的应用
    def ab_testing_recommendation():
        """A/B测试在推荐系统中的应用"""
        print("\n3. A/B测试在推荐系统中的应用:")

        # 模拟A/B测试数据
        np.random.seed(42)

        # 对照组（A组）：原有推荐算法
        # 实验组（B组）：新推荐算法

        n_users = 1000
        baseline_ctr = 0.10  # 基线点击率
        treatment_effect = 0.02  # 处理效应（点击率提升）

        # 随机分配用户到A组或B组
        group_assignments = np.random.choice(['A', 'B'], size=n_users, p=[0.5, 0.5])

        # 模拟点击数据
        clicks = []
        for group in group_assignments:
            if group == 'A':
                ctr = baseline_ctr
            else:  # B组
                ctr = baseline_ctr + treatment_effect

            # 模拟用户是否点击
            click = np.random.binomial(1, ctr)
            clicks.append(click)

        clicks = np.array(clicks)

        # 创建数据框
        import pandas as pd
        ab_data = pd.DataFrame({
            'user_id': range(n_users),
            'group': group_assignments,
            'clicked': clicks
        })

        print(f"A/B测试数据:")
        print(f"  总用户数: {n_users}")
        print(f"  A组用户数: {sum(group_assignments == 'A')}")
        print(f"  B组用户数: {sum(group_assignments == 'B')}")

        # 计算分组统计量
        group_stats = ab_data.groupby('group')['clicked'].agg(['count', 'mean', 'std'])
        group_stats['se'] = group_stats['std'] / np.sqrt(group_stats['count'])

        print(f"\n分组统计:")
        print(group_stats)

        # 比例检验（z检验）
        from statsmodels.stats.proportion import proportions_ztest, proportion_confint

        # 提取数据
        n_a = group_stats.loc['A', 'count']
        n_b = group_stats.loc['B', 'count']
        clicks_a = n_a * group_stats.loc['A', 'mean']
        clicks_b = n_b * group_stats.loc['B', 'mean']

        # 执行比例z检验
        count = np.array([clicks_a, clicks_b])
        nobs = np.array([n_a, n_b])

        z_stat, p_value = proportions_ztest(count, nobs)

        print(f"\n比例z检验结果:")
        print(f"  z统计量: {z_stat:.4f}")
        print(f"  p值: {p_value:.4f}")

        # 计算置信区间
        ci_a = proportion_confint(clicks_a, n_a, alpha=0.05, method='normal')
        ci_b = proportion_confint(clicks_b, n_b, alpha=0.05, method='normal')

        print(f"  A组点击率95%置信区间: [{ci_a[0]:.4f}, {ci_a[1]:.4f}]")
        print(f"  B组点击率95%置信区间: [{ci_b[0]:.4f}, {ci_b[1]:.4f}]")

        # 计算提升量和置信区间
        p_a = clicks_a / n_a
        p_b = clicks_b / n_b
        lift = (p_b - p_a) / p_a

        # 提升量的标准误差
        se_lift = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b) / p_a
        lift_ci_lower = lift - 1.96 * se_lift
        lift_ci_upper = lift + 1.96 * se_lift

        print(f"  相对提升: {lift:.2%}")
        print(f"  提升量95%置信区间: [{lift_ci_lower:.2%}, {lift_ci_upper:.2%}]")

        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 点击率比较
        groups = ['A组 (对照组)', 'B组 (实验组)']
        ctr_values = [p_a, p_b]
        ctr_errors = [1.96 * group_stats.loc['A', 'se'], 1.96 * group_stats.loc['B', 'se']]

        bars = axes[0, 0].bar(groups, ctr_values, yerr=ctr_errors, capsize=10, alpha=0.7)
        axes[0, 0].set_ylabel('点击率 (CTR)')
        axes[0, 0].set_title('A/B测试：点击率比较')
        axes[0, 0].grid(True, alpha=0.3)

        for bar, ctr in zip(bars, ctr_values):
            axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f'{ctr:.3f}', ha='center', va='bottom')

        # 累积点击率随时间变化（模拟）
        axes[0, 1].plot(range(1, 101), np.cumsum(np.random.binomial(1, p_a, 100)) /
                        np.arange(1, 101), label='A组', alpha=0.7)
        axes[0, 1].plot(range(1, 101), np.cumsum(np.random.binomial(1, p_b, 100)) /
                        np.arange(1, 101), label='B组', alpha=0.7)
        axes[0, 1].axhline(y=p_a, color='blue', linestyle='--', alpha=0.5)
        axes[0, 1].axhline(y=p_b, color='orange', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('时间（天）')
        axes[0, 1].set_ylabel('累积点击率')
        axes[0, 1].set_title('累积点击率变化')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # p值分布（模拟多次A/B测试）
        axes[1, 0].hist(np.random.beta(2, 8, 1000), bins=30, alpha=0.7, density=True)
        axes[1, 0].axvline(x=0.05, color='red', linestyle='--', label='α=0.05')
        axes[1, 0].set_xlabel('p值')
        axes[1, 0].set_ylabel('概率密度')
        axes[1, 0].set_title('模拟A/B测试的p值分布')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 统计功效分析
        effect_sizes = np.linspace(0.01, 0.05, 20)
        sample_sizes = np.arange(500, 5001, 500)

        # 创建网格
        X, Y = np.meshgrid(effect_sizes, sample_sizes)

        # 计算统计功效（简化计算）
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                effect = X[i, j]
                n = Y[i, j]

                # 简化功效计算
                se = np.sqrt(2 * baseline_ctr * (1 - baseline_ctr) / n)
                z_power = effect / se
                from scipy.stats import norm
                power = norm.cdf(z_power - norm.ppf(0.975)) + norm.cdf(-z_power - norm.ppf(0.975))#统计功效
                Z[i, j] = power

        contour = axes[1, 1].contourf(X, Y, Z, levels=20, cmap='viridis')
        axes[1, 1].set_xlabel('效应大小 (点击率提升)')
        axes[1, 1].set_ylabel('每组样本量')
        axes[1, 1].set_title('A/B测试统计功效')
        plt.colorbar(contour, ax=axes[1, 1], label='统计功效')

        plt.tight_layout()
        plt.show()

        # A/B测试决策
        alpha = 0.05
        print(f"\nA/B测试决策 (α={alpha}):")
        if p_value < alpha:
            if p_b > p_a:
                print(f"  ✅ 统计显著：B组优于A组，推荐采用新算法")
                print(f"     预计提升：{lift:.2%} (95% CI: [{lift_ci_lower:.2%}, {lift_ci_upper:.2%}])")
            else:
                print(f"  ⚠️  统计显著：A组优于B组，建议保持原算法")
        else:
            print(f"  ⚠️  统计不显著：无法确定哪个算法更好")
            print(f"     可能原因：样本量不足、效应太小、或随机波动")

        return ab_data, z_stat, p_value, lift

    # 主函数逻辑
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    # 生成特征
    X = np.random.randn(n_samples, n_features)

    # 生成标签（非线性关系）
    true_weights = np.random.randn(n_features)
    true_weights[10:] = 0  # 后10个特征不重要

    log_odds = X @ true_weights + 0.1 * (X[:, 0] ** 2)  # 添加非线性项
    probabilities = 1 / (1 + np.exp(-log_odds))
    y = (probabilities > 0.5).astype(int)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 调用各个子函数
    # 修改第480行
    models, predictions, accuracies, chi2, p_value = model_performance_significance(X_train, y_train, X_test, y_test)
    coefficients, coef_se, p_values, significant_features = feature_importance_hypothesis_test(X_train, y_train, true_weights, models)
    ab_data, z_stat, p_value_ab, lift = ab_testing_recommendation()

    return (models, predictions, accuracies, chi2, p_value,
            coefficients, coef_se, p_values, significant_features,
            ab_data, z_stat, p_value_ab, lift)


hypothesis_testing_results = hypothesis_testing_model_evaluation()
