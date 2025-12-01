# 数学概念：在约束条件下求函数极值的方法
# AI对应：支持向量机、带约束的优化问题
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans, Arial, sans-serif'#设置带有数学平方的字体，用于显示数学公式


# 实践4：拉格朗日乘数法在AI中的应用
def lagrange_multipliers_ai():
    """
    拉格朗日乘数法在支持向量机等AI算法中的应用
    """
    print("=== 拉格朗日乘数法与约束优化 ===")

    # 示例1：简单的约束优化问题
    # 最大化 f(x, y) = x + y，约束条件 g(x, y) = x² + y² - 1 = 0
    def example_constrained_optimization():
        """简单的约束优化示例"""
        print("示例1: 在单位圆上最大化 x + y")

        # 拉格朗日函数: L(x, y, λ) = f(x, y) - λ * g(x, y) = x + y - λ(x² + y² - 1)
        # 最优解条件:
        # ∂L/∂x = 1 - 2λx = 0  => x = 1/(2λ)
        # ∂L/∂y = 1 - 2λy = 0  => y = 1/(2λ)
        # ∂L/∂λ = x² + y² - 1 = 0

        # 代入约束: (1/(2λ))² + (1/(2λ))² = 1 => 2/(4λ²) = 1 => λ² = 1/2
        lambda_opt = 1 / np.sqrt(2)
        x_opt = 1 / (2 * lambda_opt)
        y_opt = 1 / (2 * lambda_opt)

        print(f"解析解:")
        print(f"  最优解: x={x_opt:.4f}, y={y_opt:.4f}")
        print(f"  最大值: f(x,y)={x_opt + y_opt:.4f}")
        print(f"  拉格朗日乘数: λ={lambda_opt:.4f}")

        # 可视化
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        objective = circle_x + circle_y

        plt.figure(figsize=(15, 5))

        # 约束和目标函数
        plt.subplot(1, 3, 1)
        plt.plot(circle_x, circle_y, 'b-', label='约束: x² + y² = 1')
        plt.scatter(x_opt, y_opt, color='red', s=100, label=f'最优解 ({x_opt:.3f}, {y_opt:.3f})')

        # 绘制梯度方向
        grad_f = np.array([1, 1])  # ∇f = (1, 1)
        grad_g = np.array([2 * x_opt, 2 * y_opt])  # ∇g = (2x, 2y)

        plt.quiver(x_opt, y_opt, grad_f[0], grad_f[1], color='green', scale=5,
                   label='∇f', alpha=0.7)
        plt.quiver(x_opt, y_opt, grad_g[0], grad_g[1], color='blue', scale=5,
                   label='∇g', alpha=0.7)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('约束优化: 在单位圆上最大化 x+y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')

        # 目标函数值沿约束的变化
        plt.subplot(1, 3, 2)
        plt.plot(theta, objective, 'r-')
        plt.axvline(x=np.pi / 4, color='green', linestyle='--', label='最优角度 π/4')
        plt.xlabel('角度 θ (弧度)')
        plt.ylabel('目标函数值 x+y')
        plt.title('目标函数沿约束的变化')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 拉格朗日函数
        plt.subplot(1, 3, 3)
        # 在最优解附近可视化拉格朗日函数
        x_vals = np.linspace(0.5, 0.8, 50)
        y_vals = np.linspace(0.5, 0.8, 50)
        X, Y = np.meshgrid(x_vals, y_vals)

        # 拉格朗日函数 L(x, y, λ) = x + y - λ(x² + y² - 1)，使用最优λ
        L = X + Y - lambda_opt * (X ** 2 + Y ** 2 - 1)

        contour = plt.contour(X, Y, L, levels=20)
        plt.clabel(contour, inline=True, fontsize=8)
        plt.scatter(x_opt, y_opt, color='red', s=100, label='最优解')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('拉格朗日函数等高线')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return x_opt, y_opt, lambda_opt

    x_opt, y_opt, lambda_opt = example_constrained_optimization()

    # 示例2：支持向量机的简化版
    def svm_lagrangian_example():
        """支持向量机的拉格朗日对偶问题简化示例"""
        print(f"\n示例2: 支持向量机（SVM）的拉格朗日对偶")

        # 生成简单的线性可分数据
        np.random.seed(42)
        n_samples = 50

        # 类别1的数据
        class1_mean = [2, 2]
        class1_cov = [[1, 0.5], [0.5, 1]]
        # //2 - 整数除法运算符（地板除法）,返回整数结果，向下取整到最接近的整数，结果类型取决于操作数类型
        # /2 - 浮点除法运算符,返回浮点数结果，即使两个操作数都是整数，结果也是浮点数类型
        X1 = np.random.multivariate_normal(class1_mean, class1_cov, n_samples // 2)#这行代码用于确定每个类别的样本数量，以便生成两类数据（例如正类和负类）。
        y1 = np.ones(n_samples // 2)#X1 和 y1 代表第一类数据（标记为 +1）

        # 类别-1的数据
        class2_mean = [-2, -2]
        class2_cov = [[1, -0.5], [-0.5, 1]]
        X2 = np.random.multivariate_normal(class2_mean, class2_cov, n_samples // 2)
        y2 = -np.ones(n_samples // 2)#X2 和 y2 代表第二类数据（标记为 -1）

        X = np.vstack([X1, X2])
        y = np.hstack([y1, y2])

        print(f"SVM数据:")
        print(f"  样本数量: {n_samples}")
        print(f"  特征维度: 2")
        print(f"  类别: +1 (红色), -1 (蓝色)")

        # 使用sklearn的SVM找到最优超平面
        from sklearn.svm import SVC
        svm = SVC(kernel='linear', C=1.0)
        svm.fit(X, y)

        # 获取SVM参数
        w = svm.coef_[0]
        b = svm.intercept_[0]
        support_vectors = svm.support_vectors_

        print(f"\nSVM训练结果:")
        print(f"  权重向量 w: [{w[0]:.4f}, {w[1]:.4f}]")
        print(f"  偏置 b: {b:.4f}")
        print(f"  支持向量数量: {len(support_vectors)}")

        # 可视化SVM结果
        plt.figure(figsize=(15, 10))

        # 数据点和决策边界
        plt.subplot(2, 3, 1)
        plt.scatter(X1[:, 0], X1[:, 1], c='red', alpha=0.6, label='类别 +1')
        plt.scatter(X2[:, 0], X2[:, 1], c='blue', alpha=0.6, label='类别 -1')
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1],
                    facecolors='none', edgecolors='black', s=100, linewidths=2, label='支持向量')

        # 绘制决策边界
        xx = np.linspace(-5, 5, 100)
        yy = (-w[0] * xx - b) / w[1]
        margin = 1 / np.linalg.norm(w)

        plt.plot(xx, yy, 'k-', label='决策边界')
        plt.plot(xx, yy + margin / w[1], 'k--', alpha=0.5, label='间隔边界')
        plt.plot(xx, yy - margin / w[1], 'k--', alpha=0.5)

        plt.xlabel('特征1')
        plt.ylabel('特征2')
        plt.title('SVM: 数据与决策边界')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # SVM的原始优化问题
        plt.subplot(2, 3, 2)
        # 原始问题: min ½||w||², s.t. y_i(w·x_i + b) ≥ 1

        text_lines = [
            'SVM原始优化问题:',
            'min ½||w||²',
            '约束: y_i(w·x_i + b) ≥ 1, ∀i',
            '',
            '拉格朗日函数:',
            'L(w, b, α) = ½||w||²',
            '    - Σα_i[y_i(w·x_i + b) - 1]',
            '    α_i ≥ 0',
            '',
            '对偶问题:',
            'max Σα_i - ½ΣΣα_iα_jy_iy_jx_i·x_j',
            '约束: Σα_iy_i = 0, α_i ≥ 0'
        ]

        for i, line in enumerate(text_lines):
            plt.text(0.1, 0.9 - i * 0.07, line, transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top')

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('SVM的优化问题形式')

        # 支持向量的拉格朗日乘数
        plt.subplot(2, 3, 3)
        alpha_values = np.abs(svm.dual_coef_[0])
        plt.bar(range(len(alpha_values)), alpha_values, color='orange', alpha=0.7)
        plt.xlabel('支持向量索引')
        plt.ylabel('拉格朗日乘数 |α_i|')
        plt.title('支持向量的拉格朗日乘数')
        plt.grid(True, alpha=0.3)

        # 间隔边界可视化
        plt.subplot(2, 3, 4)
        # 创建网格计算决策函数值
        xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
        Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        contour = plt.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'])
        plt.clabel(contour, inline=True, fontsize=8)
        plt.scatter(X1[:, 0], X1[:, 1], c='red', alpha=0.6)
        plt.scatter(X2[:, 0], X2[:, 1], c='blue', alpha=0.6)
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1],
                    facecolors='none', edgecolors='black', s=100, linewidths=2)
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        plt.title('SVM决策函数与间隔')
        plt.grid(True, alpha=0.3)

        # 权重向量的几何意义
        plt.subplot(2, 3, 5)
        plt.scatter(X1[:, 0], X1[:, 1], c='red', alpha=0.6, label='类别 +1')
        plt.scatter(X2[:, 0], X2[:, 1], c='blue', alpha=0.6, label='类别 -1')

        # 绘制权重向量方向（垂直于决策边界）
        center = [0, (-b) / w[1]]  # 决策边界上的一个点
        w_direction = w / np.linalg.norm(w)

        plt.quiver(center[0], center[1], w_direction[0], w_direction[1],
                   scale=5, color='green', label='权重向量方向')
        plt.plot(xx, yy, 'k-', label='决策边界')

        plt.xlabel('特征1')
        plt.ylabel('特征2')
        plt.title('权重向量与决策边界')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 不同C值的影响
        plt.subplot(2, 3, 6)
        C_values = [0.1, 1.0, 10.0]
        colors = ['blue', 'green', 'red']

        for C, color in zip(C_values, colors):
            svm_temp = SVC(kernel='linear', C=C)
            svm_temp.fit(X, y)
            w_temp = svm_temp.coef_[0]
            b_temp = svm_temp.intercept_[0]

            yy_temp = (-w_temp[0] * xx - b_temp) / w_temp[1]
            plt.plot(xx, yy_temp, color=color, linestyle='-',
                     label=f'C={C}, SV={len(svm_temp.support_vectors_)}')

        plt.scatter(X1[:, 0], X1[:, 1], c='red', alpha=0.3)
        plt.scatter(X2[:, 0], X2[:, 1], c='blue', alpha=0.3)
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        plt.title('不同C值的决策边界')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # SVM的KKT条件解释
        print(f"\nSVM的KKT条件:")
        print("1. 原始可行性: y_i(w·x_i + b) ≥ 1")
        print("2. 对偶可行性: α_i ≥ 0")
        print("3. 互补松弛性: α_i[y_i(w·x_i + b) - 1] = 0")
        print("4. 梯度条件: w = Σα_iy_ix_i, Σα_iy_i = 0")

        print(f"\n支持向量的性质:")
        print("• 支持向量满足: y_i(w·x_i + b) = 1")
        print("• 对应的 α_i > 0")
        print("• 非支持向量的 α_i = 0")

        return svm, w, b, support_vectors

    svm_model, w_svm, b_svm, support_vectors = svm_lagrangian_example()

    # 数学原理总结
    print(f"\n拉格朗日乘数法在AI中的重要意义:")
    print("1. 支持向量机: 通过拉格朗日对偶将原问题转化为更易求解的形式")
    print("2. 约束优化: 处理模型复杂度控制、公平性约束等问题")
    print("3. 对偶理论: 为理解模型提供了不同的视角")
    print("4. KKT条件: 最优解的充要条件，指导算法设计")


lagrange_multipliers_ai()