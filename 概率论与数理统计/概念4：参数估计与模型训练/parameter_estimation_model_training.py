# 数学概念：点估计、区间估计、最大似然估计、贝叶斯估计
# AI对应：模型参数学习、正则化、贝叶斯优化
import numpy as np
import matplotlib.pyplot as plt


# 实践4：参数估计在模型训练中的应用
def parameter_estimation_model_training():
    """
    参数估计方法在机器学习模型训练中的应用
    """
    print("=== 参数估计与模型训练 ===")

    # 1. 最大似然估计在线性回归中的应用
    def mle_linear_regression():
        """最大似然估计在线性回归中的应用"""
        print("\n1. 最大似然估计在线性回归中的应用:")

        # 生成模拟数据
        np.random.seed(42)
        n_samples = 100
        true_slope = 2.5
        true_intercept = 1.0
        noise_std = 0.5

        X = np.linspace(0, 1, n_samples)
        true_y = true_slope * X + true_intercept
        y = true_y + np.random.normal(0, noise_std, n_samples)

        # 最大似然估计推导
        # 假设 y_i ~ N(wx_i + b, σ²)
        # 对数似然函数: log L(w, b, σ²) = -n/2 log(2πσ²) - 1/(2σ²) Σ(y_i - wx_i - b)²
        # 对w, b求导等于最小二乘法
        # 对σ²求导得到 σ²_MLE = 1/n Σ(y_i - wx_i - b)²

        # 使用最小二乘法估计w, b（等价于MLE）
        X_design = np.column_stack([X, np.ones_like(X)])  # 添加截距项
        w_b_hat = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
        w_mle = w_b_hat[0]
        b_mle = w_b_hat[1]

        # 估计噪声方差
        y_pred = w_mle * X + b_mle
        residuals = y - y_pred
        sigma2_mle = np.mean(residuals ** 2)

        print(f"真实参数: w={true_slope:.4f}, b={true_intercept:.4f}, σ={noise_std:.4f}")
        print(f"MLE估计: w={w_mle:.4f}, b={b_mle:.4f}, σ²={sigma2_mle:.4f}")

        # 可视化似然函数
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 数据点和拟合线
        axes[0, 0].scatter(X, y, alpha=0.6, label='数据点')
        axes[0, 0].plot(X, true_y, 'g-', label='真实关系', linewidth=2)
        axes[0, 0].plot(X, y_pred, 'r-', label='MLE拟合', linewidth=2)
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('y')
        axes[0, 0].set_title('数据与MLE拟合')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 对数似然函数曲面
        w_range = np.linspace(1.5, 3.5, 50)
        b_range = np.linspace(0, 2, 50)
        W, B = np.meshgrid(w_range, b_range)

        # 计算对数似然（忽略常数项）
        log_likelihood = np.zeros_like(W)
        n = len(X)

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                w = W[i, j]
                b = B[i, j]
                residuals = y - (w * X + b)
                # 忽略常数项的对数似然
                log_likelihood[i, j] = -0.5 * n * np.log(np.mean(residuals ** 2))

        contour = axes[0, 1].contour(W, B, log_likelihood, levels=20)
        axes[0, 1].clabel(contour, inline=True, fontsize=8)
        axes[0, 1].scatter(true_slope, true_intercept, color='green', s=100,
                           label='真实参数', marker='*')
        axes[0, 1].scatter(w_mle, b_mle, color='red', s=100,
                           label='MLE估计', marker='o')
        axes[0, 1].set_xlabel('权重 w')
        axes[0, 1].set_ylabel('偏置 b')
        axes[0, 1].set_title('对数似然函数曲面')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 残差分布
        axes[1, 0].hist(residuals, bins=20, density=True, alpha=0.7,
                        label='残差直方图')

        # 拟合的正态分布
        x_norm = np.linspace(-1.5, 1.5, 100)
        from scipy.stats import norm
        y_norm = norm.pdf(x_norm, 0, np.sqrt(sigma2_mle))
        axes[1, 0].plot(x_norm, y_norm, 'r-', label=f'N(0, {sigma2_mle:.3f})')

        axes[1, 0].set_xlabel('残差')
        axes[1, 0].set_ylabel('概率密度')
        axes[1, 0].set_title('残差分布与正态拟合')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 似然函数随σ²的变化
        axes[1, 1].plot(range(n), residuals, 'o-', alpha=0.6)
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].fill_between(range(n), -np.sqrt(sigma2_mle), np.sqrt(sigma2_mle),
                                alpha=0.2, color='gray', label='±σ区间')
        axes[1, 1].set_xlabel('样本索引')
        axes[1, 1].set_ylabel('残差')
        axes[1, 1].set_title('残差序列与MLE估计的σ')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 置信区间估计
        print(f"\n参数估计的置信区间 (95%):")

        # 计算标准误差
        X_design = np.column_stack([X, np.ones_like(X)])
        sigma_hat = np.sqrt(sigma2_mle)
        var_cov_matrix = sigma2_mle * np.linalg.inv(X_design.T @ X_design)

        # w的置信区间
        se_w = np.sqrt(var_cov_matrix[0, 0])
        w_ci_lower = w_mle - 1.96 * se_w
        w_ci_upper = w_mle + 1.96 * se_w

        # b的置信区间
        se_b = np.sqrt(var_cov_matrix[1, 1])
        b_ci_lower = b_mle - 1.96 * se_b
        b_ci_upper = b_mle + 1.96 * se_b

        print(f"w的95%置信区间: [{w_ci_lower:.4f}, {w_ci_upper:.4f}]")
        print(f"b的95%置信区间: [{b_ci_lower:.4f}, {b_ci_upper:.4f}]")
        print(f"真实w={true_slope:.4f}在区间内: {w_ci_lower <= true_slope <= w_ci_upper}")
        print(f"真实b={true_intercept:.4f}在区间内: {b_ci_lower <= true_intercept <= b_ci_upper}")

        return w_mle, b_mle, sigma2_mle

    w_mle, b_mle, sigma2_mle = mle_linear_regression()

    # 2. 贝叶斯估计与正则化
    def bayesian_estimation_regularization():
        """贝叶斯估计与正则化的关系"""
        print("\n2. 贝叶斯估计与正则化:")

        # 生成高维数据（特征比样本多）
        np.random.seed(42)
        n_samples = 50
        n_features = 100  # 特征比样本多

        # 真实权重：只有前5个特征非零
        true_w = np.zeros(n_features)
        true_w[:5] = [2.0, -1.5, 1.0, 0.5, -0.8]

        # 生成数据
        X = np.random.randn(n_samples, n_features)
        noise_std = 0.5
        y = X @ true_w + np.random.normal(0, noise_std, n_samples)

        print(f"高维数据:")
        print(f"  样本数量: {n_samples}")
        print(f"  特征数量: {n_features}")
        print(f"  真实非零权重数量: {np.sum(true_w != 0)}")

        # 不同的估计方法
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge

        models = {
            '最小二乘法 (MLE)': LinearRegression(),
            '岭回归 (L2正则)': Ridge(alpha=1.0),
            'Lasso (L1正则)': Lasso(alpha=0.1),
            '贝叶斯岭回归': BayesianRidge()
        }

        # 训练模型
        fitted_models = {}

        for name, model in models.items():
            model.fit(X, y)
            fitted_models[name] = model

        # 可视化比较
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 真实权重
        axes[0, 0].stem(range(n_features), true_w, basefmt=" ")
        axes[0, 0].set_xlabel('特征索引')
        axes[0, 0].set_ylabel('权重值')
        axes[0, 0].set_title('真实权重')
        axes[0, 0].grid(True, alpha=0.3)

        # 不同方法估计的权重
        for idx, (name, model) in enumerate(fitted_models.items()):
            ax = axes[(idx + 1) // 3, (idx + 1) % 3]

            if hasattr(model, 'coef_'):
                weights = model.coef_
            else:
                weights = model.coef_[0] if hasattr(model.coef_, '__len__') else model.coef_

            ax.stem(range(n_features), weights, basefmt=" ")
            ax.set_xlabel('特征索引')
            ax.set_ylabel('权重值')
            ax.set_title(f'{name}')
            ax.grid(True, alpha=0.3)

            # 计算稀疏性
            sparsity = np.sum(np.abs(weights) < 1e-3) / n_features
            ax.text(0.05, 0.95, f'稀疏度: {sparsity:.1%}',
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()

        # 贝叶斯解释
        print(f"\n贝叶斯估计的解释:")
        print("岭回归 (L2正则) ↔ 权重先验为高斯分布 N(0, 1/α)")
        print("Lasso (L1正则) ↔ 权重先验为拉普拉斯分布 Laplace(0, 1/α)")
        print("贝叶斯岭回归 ↔ 权重先验为高斯分布，同时估计超参数")

        # 贝叶斯岭回归的后验分布
        bayesian_model = fitted_models['贝叶斯岭回归']
        if hasattr(bayesian_model, 'sigma_'):
            # 后验方差
            posterior_var = bayesian_model.sigma_

            print(f"\n贝叶斯岭回归的后验分析:")
            print(f"  估计的噪声方差: {bayesian_model.alpha_:.4f}")
            print(f"  估计的权重先验方差: {bayesian_model.lambda_:.4f}")

            # 可视化后验不确定性
            plt.figure(figsize=(10, 6))

            # 前20个权重的后验分布
            n_to_show = min(20, n_features)

            for i in range(n_to_show):
                # 后验均值和标准差
                posterior_mean = bayesian_model.coef_[i]
                posterior_std = np.sqrt(posterior_var[i, i])

                # 绘制置信区间
                x = i
                y = posterior_mean
                y_err = 1.96 * posterior_std  # 95%置信区间

                plt.errorbar(x, y, yerr=y_err, fmt='o', capsize=5, alpha=0.7)
                plt.axhline(y=true_w[i], color='red', linestyle='--', alpha=0.5)

            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.xlabel('特征索引')
            plt.ylabel('权重估计')
            plt.title('贝叶斯估计：后验均值与95%置信区间')
            plt.grid(True, alpha=0.3)
            plt.show()

        return fitted_models

    fitted_models = bayesian_estimation_regularization()

    # 3. EM算法在高斯混合模型中的应用
    def em_algorithm_gmm():
        """EM算法在高斯混合模型参数估计中的应用"""
        print("\n3. EM算法在高斯混合模型参数估计:")

        # 生成混合高斯数据
        np.random.seed(42)
        n_samples = 500

        # 真实参数
        true_weights = np.array([0.3, 0.5, 0.2])
        true_means = np.array([[-2.0, -2.0], [0.0, 2.0], [2.0, -1.0]])
        true_covs = np.array([
            [[0.5, 0.2], [0.2, 0.5]],
            [[0.8, -0.3], [-0.3, 0.8]],
            [[0.4, 0.0], [0.0, 0.4]]
        ])

        # 生成数据
        X_gmm = []
        true_labels = []

        for i in range(n_samples):
            # 选择组件
            comp = np.random.choice(3, p=true_weights)

            # 从选定的高斯分布生成样本
            mean = true_means[comp]
            cov = true_covs[comp]

            sample = np.random.multivariate_normal(mean, cov)

            X_gmm.append(sample)
            true_labels.append(comp)

        X_gmm = np.array(X_gmm)
        true_labels = np.array(true_labels)

        print(f"生成数据:")
        print(f"  样本数量: {n_samples}")
        print(f"  真实混合权重: {true_weights}")

        # EM算法实现
        def em_gmm(X, n_components=3, max_iter=100, tol=1e-4):
            """EM算法估计GMM参数"""
            n_samples, n_features = X.shape

            # 初始化参数
            # 使用K-means初始化
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_components, random_state=42)
            kmeans.fit(X)

            weights = np.ones(n_components) / n_components
            means = kmeans.cluster_centers_

            # 初始化协方差矩阵
            covs = np.zeros((n_components, n_features, n_features))
            for k in range(n_components):
                cluster_points = X[kmeans.labels_ == k]
                if len(cluster_points) > 1:
                    covs[k] = np.cov(cluster_points.T)
                else:
                    covs[k] = np.eye(n_features) * 0.1

            # EM迭代
            log_likelihoods = []

            for iteration in range(max_iter):
                # E步：计算后验概率（责任）
                responsibilities = np.zeros((n_samples, n_components))

                for k in range(n_components):
                    # 多元高斯分布的概率密度
                    diff = X - means[k]
                    inv_cov = np.linalg.inv(covs[k] + np.eye(n_features) * 1e-6)  # 添加正则化
                    exp_term = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)

                    # 对数概率密度（忽略常数项）
                    log_det = np.linalg.slogdet(covs[k])[1]
                    log_prob = -0.5 * (n_features * np.log(2 * np.pi) + log_det) + exp_term

                    responsibilities[:, k] = np.log(weights[k]) + log_prob

                # 对数-求和-指数技巧（数值稳定性）
                max_log = np.max(responsibilities, axis=1, keepdims=True)
                exp_log = np.exp(responsibilities - max_log)
                responsibilities = exp_log / np.sum(exp_log, axis=1, keepdims=True)

                # 计算对数似然
                log_likelihood = np.sum(max_log + np.log(np.sum(exp_log, axis=1)))
                log_likelihoods.append(log_likelihood)

                # M步：更新参数
                Nk = np.sum(responsibilities, axis=0)

                # 更新权重
                weights = Nk / n_samples

                # 更新均值
                for k in range(n_components):
                    means[k] = np.sum(responsibilities[:, k:k + 1] * X, axis=0) / Nk[k]

                # 更新协方差
                for k in range(n_components):
                    diff = X - means[k]
                    weighted_diff = responsibilities[:, k:k + 1] * diff
                    covs[k] = (weighted_diff.T @ diff) / Nk[k] + np.eye(n_features) * 1e-6

                # 检查收敛
                if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
                    print(f"EM在第{iteration}次迭代收敛")
                    break

            return weights, means, covs, responsibilities, log_likelihoods

        # 运行EM算法
        print("运行EM算法估计GMM参数...")
        weights_est, means_est, covs_est, responsibilities, log_likelihoods = em_gmm(X_gmm)

        print(f"\n估计结果:")
        print(f"  估计的混合权重: {weights_est}")
        print(f"  真实的混合权重: {true_weights}")

        # 可视化EM过程
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 数据点
        axes[0, 0].scatter(X_gmm[:, 0], X_gmm[:, 1], c=true_labels,
                           cmap='viridis', alpha=0.6, s=10)
        axes[0, 0].set_xlabel('特征1')
        axes[0, 0].set_ylabel('特征2')
        axes[0, 0].set_title('真实数据（带标签）')
        axes[0, 0].grid(True, alpha=0.3)

        # EM估计的聚类
        pred_labels = np.argmax(responsibilities, axis=1)
        axes[0, 1].scatter(X_gmm[:, 0], X_gmm[:, 1], c=pred_labels,
                           cmap='viridis', alpha=0.6, s=10)

        # 绘制估计的高斯分布
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X_grid, Y_grid = np.meshgrid(x, y)
        grid_points = np.stack([X_grid.ravel(), Y_grid.ravel()], axis=1)

        Z_total = np.zeros(grid_points.shape[0])
        for k in range(3):
            diff = grid_points - means_est[k]
            inv_cov = np.linalg.inv(covs_est[k])
            exp_term = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
            log_det = np.linalg.slogdet(covs_est[k])[1]
            log_prob = -0.5 * (2 * np.log(2 * np.pi) + log_det) + exp_term
            prob = np.exp(log_prob) * weights_est[k]
            Z_total += prob

            # 绘制等高线
            Z = prob.reshape(X_grid.shape)
            axes[0, 1].contour(X_grid, Y_grid, Z, levels=3, alpha=0.5, linestyles='--')

        axes[0, 1].set_xlabel('特征1')
        axes[0, 1].set_ylabel('特征2')
        axes[0, 1].set_title('EM估计的GMM')
        axes[0, 1].grid(True, alpha=0.3)

        # 对数似然收敛
        axes[0, 2].plot(log_likelihoods, 'bo-')
        axes[0, 2].set_xlabel('迭代次数')
        axes[0, 2].set_ylabel('对数似然')
        axes[0, 2].set_title('EM算法收敛')
        axes[0, 2].grid(True, alpha=0.3)

        # 责任（后验概率）矩阵
        im = axes[1, 0].imshow(responsibilities[:50, :].T, aspect='auto', cmap='viridis')
        axes[1, 0].set_xlabel('样本索引')
        axes[1, 0].set_ylabel('组件')
        axes[1, 0].set_title('责任矩阵（前50个样本）')
        plt.colorbar(im, ax=axes[1, 0])

        # 参数估计误差
        param_errors = []
        param_names = ['权重', '均值1', '均值2', '均值3', '协方差1', '协方差2', '协方差3']

        # 计算参数误差
        weight_error = np.mean(np.abs(weights_est - true_weights))
        param_errors.append(weight_error)

        for k in range(3):
            mean_error = np.mean(np.abs(means_est[k] - true_means[k]))
            param_errors.append(mean_error)

        for k in range(3):
            cov_error = np.mean(np.abs(covs_est[k] - true_covs[k]))
            param_errors.append(cov_error)

        axes[1, 1].bar(range(len(param_errors)), param_errors)
        axes[1, 1].set_xticks(range(len(param_errors)))
        axes[1, 1].set_xticklabels(param_names, rotation=45)
        axes[1, 1].set_ylabel('平均绝对误差')
        axes[1, 1].set_title('参数估计误差')
        axes[1, 1].grid(True, alpha=0.3)

        # 责任向量的分布
        axes[1, 2].hist(responsibilities.ravel(), bins=50, alpha=0.7, density=True)
        axes[1, 2].set_xlabel('责任值')
        axes[1, 2].set_ylabel('概率密度')
        axes[1, 2].set_title('责任值分布')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # EM算法解释
        print(f"\nEM算法的概率解释:")
        print("E步（期望步）: 计算后验概率 P(z|x, θ)")
        print("M步（最大化步）: 最大化完全数据对数似然的期望 Q(θ|θ⁽ᵗ⁾)")
        print("收敛性: 每次迭代保证增加对数似然，收敛到局部最优")

        return weights_est, means_est, covs_est, responsibilities

    gmm_weights, gmm_means, gmm_covs, responsibilities = em_algorithm_gmm()

    return w_mle, b_mle, sigma2_mle, fitted_models, gmm_weights, gmm_means, gmm_covs


estimation_results = parameter_estimation_model_training()