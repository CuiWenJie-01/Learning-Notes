# 数学概念：多元函数对每个变量的偏导数表示在其他变量不变时，该变量变化对函数值的影响
# AI对应：神经网络中损失函数对每个权重的偏导数，用于梯度下降更新
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_classification


# 实践1：偏导数的几何直观与梯度下降
def partial_derivatives_gradient_descent():
    """
    偏导数的几何意义及其在梯度下降中的应用
    """
    print("=== 偏导数与梯度下降 ===")

    # 定义一个简单的二元函数：f(x, y) = x² + y² + sin(2x) + cos(2y)
    def f(x, y):
        return x ** 2 + y ** 2 + np.sin(2 * x) + np.cos(2 * y)

    # 手动计算偏导数
    def df_dx(x, y):
        return 2 * x + 2 * np.cos(2 * x)  # ∂f/∂x

    def df_dy(x, y):
        return 2 * y - 2 * np.sin(2 * y)  # ∂f/∂y

    # 梯度向量
    def gradient(x, y):
        return np.array([df_dx(x, y), df_dy(x, y)])

    # 创建网格数据用于可视化
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    # 手动实现梯度下降
    def gradient_descent_manual(start_point, learning_rate=0.1, n_iterations=50):
        """手动实现梯度下降"""
        path = [start_point]
        current_point = start_point.copy()

        for i in range(n_iterations):
            grad = gradient(current_point[0], current_point[1])
            current_point = current_point - learning_rate * grad #更新当前点
            path.append(current_point.copy())

            if i % 10 == 0:
                current_value = f(current_point[0], current_point[1])
                print(f"迭代 {i}: 位置 ({current_point[0]:.3f}, {current_point[1]:.3f}), "
                      f"函数值 {current_value:.3f}, 梯度范数 {np.linalg.norm(grad):.3f}")

        return np.array(path)

    # 运行梯度下降
    start_point = np.array([2.5, 2.0])
    path = gradient_descent_manual(start_point) #运行梯度下降

    # 可视化结果
    fig = plt.figure(figsize=(20, 10))

    # 3D曲面图
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.plot(path[:, 0], path[:, 1], f(path[:, 0], path[:, 1]),
             'ro-', linewidth=2, markersize=4, label='梯度下降路径')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('f(X, Y)')
    ax1.set_title('函数曲面与梯度下降路径')
    fig.colorbar(surf, ax=ax1, shrink=0.5)

    # 等高线图
    ax2 = fig.add_subplot(2, 3, 2)
    contour = ax2.contour(X, Y, Z, levels=20)
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2, markersize=4)
    ax2.quiver(path[:-1, 0], path[:-1, 1],
               -df_dx(path[:-1, 0], path[:-1, 1]),
               -df_dy(path[:-1, 0], path[:-1, 1]),
               color='red', scale=10, alpha=0.6, label='负梯度方向')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('等高线图与梯度下降')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 偏导数场（梯度场）
    ax3 = fig.add_subplot(2, 3, 3)
    # 在稀疏网格上计算梯度
    x_coarse = np.linspace(-3, 3, 15)
    y_coarse = np.linspace(-3, 3, 15)
    Xc, Yc = np.meshgrid(x_coarse, y_coarse)

    U = df_dx(Xc, Yc)  # x方向的偏导数
    V = df_dy(Xc, Yc)  # y方向的偏导数

    ax3.quiver(Xc, Yc, U, V, color='blue', alpha=0.6, scale=20)
    ax3.contour(X, Y, Z, levels=10, alpha=0.5)
    ax3.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2, markersize=4)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('梯度场（偏导数向量场）')
    ax3.grid(True, alpha=0.3)

    # 函数值随迭代的变化
    ax4 = fig.add_subplot(2, 3, 4)
    function_values = [f(point[0], point[1]) for point in path]
    ax4.plot(range(len(function_values)), function_values, 'bo-')
    ax4.set_xlabel('迭代次数')
    ax4.set_ylabel('函数值 f(x, y)')
    ax4.set_title('梯度下降：函数值收敛')
    ax4.grid(True, alpha=0.3)

    # 梯度范数随迭代的变化
    ax5 = fig.add_subplot(2, 3, 5)
    grad_norms = [np.linalg.norm(gradient(point[0], point[1])) for point in path]
    ax5.plot(range(len(grad_norms)), grad_norms, 'go-')
    ax5.set_xlabel('迭代次数')
    ax5.set_ylabel('梯度范数 ||∇f||')
    ax5.set_title('梯度下降：梯度范数收敛')
    ax5.grid(True, alpha=0.3)

    # 学习率影响比较
    ax6 = fig.add_subplot(2, 3, 6)
    learning_rates = [0.01, 0.1, 0.5]
    colors = ['r', 'b', 'g']

    for lr, color in zip(learning_rates, colors):
        test_path = gradient_descent_manual(start_point, learning_rate=lr, n_iterations=30)
        test_values = [f(point[0], point[1]) for point in test_path]
        ax6.plot(range(len(test_values)), test_values, color + '-', label=f'LR={lr}')

    ax6.set_xlabel('迭代次数')
    ax6.set_ylabel('函数值 f(x, y)')
    ax6.set_title('不同学习率的收敛比较')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 与PyTorch自动微分的比较
    print(f"\n与PyTorch自动微分的比较:")

    # 使用PyTorch计算相同的梯度
    #requires_grad=True：关键参数，设置为 True 表示需要计算这个张量的梯度，PyTorch 会跟踪所有对该张量的操作以便后续自动求导
    #requires_grad=False时，PyTorch不会为该张量记录任何操作历史，也不会计算其梯度，节省内存：由于不需要保存计算图信息，可以显著减少内存使用，提高计算效率：对于不需要求导的张量，计算速度会更快
    #只有明确设置为True的张量才会参与自动微分过程。
    x_torch = torch.tensor([2.5], dtype=torch.float32, requires_grad=True)
    y_torch = torch.tensor([2.0], dtype=torch.float32, requires_grad=True)

    f_torch = x_torch ** 2 + y_torch ** 2 + torch.sin(2 * x_torch) + torch.cos(2 * y_torch)
    f_torch.backward()

    print(f"手动计算梯度: [{df_dx(2.5, 2.0):.4f}, {df_dy(2.5, 2.0):.4f}]")
    print(f"PyTorch计算梯度: [{x_torch.grad.item():.4f}, {y_torch.grad.item():.4f}]")
    print(f"梯度计算一致性: {np.allclose([df_dx(2.5, 2.0), df_dy(2.5, 2.0)],
                                         [x_torch.grad.item(), y_torch.grad.item()])}")


partial_derivatives_gradient_descent()