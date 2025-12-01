# 数学概念：方向导数表示函数在某个方向的变化率，梯度指向函数增长最快的方向
# AI对应：优化算法中的梯度方向、学习率调整
import numpy as np
import matplotlib.pyplot as plt


# 实践3：方向导数与梯度在优化算法中的应用
def directional_derivative_optimization():
    """
    方向导数和梯度在AI优化算法中的关键作用
    """
    print("=== 方向导数与梯度优化 ===")

    # 定义测试函数
    def f(x, y):
        return np.sin(x) * np.cos(y) + 0.1 * (x ** 2 + y ** 2)#0.1 是一个系数，用于控制二次项 0.1 * (x ** 2 + y ** 2) 对整个函数 f(x, y) 的影响程度。

    # 梯度计算
    def gradient(x, y):
        df_dx = np.cos(x) * np.cos(y) + 0.2 * x
        df_dy = -np.sin(x) * np.sin(y) + 0.2 * y
        return np.array([df_dx, df_dy])

    # 方向导数计算
    def directional_derivative(x, y, direction):
        """在给定方向的方向导数"""
        grad = gradient(x, y)
        '''
        这段代码的作用是将向量归一化为单位向量。
        具体解释如下：
            direction 是一个方向向量
            np.linalg.norm(direction) 计算该向量的模长（欧几里得范数）
            通过除以模长，得到单位向量 unit_direction
        为什么要归一化？
            统一尺度：确保所有方向向量具有相同的长度（为1），这样比较不同方向的方向导数才有意义
            方向导数计算需要：方向导数的定义要求使用单位向量，公式为：D_u f = ∇f · u，其中 u 必须是单位向量
            避免长度影响：如果不归一化，向量长度会影响点积结果，无法真实反映函数在该方向上的变化率
        例如：
            向量 [2, 0] 归一化后变为 [1, 0]
            向量 [1, 1] 归一化后变为 [√2/2, √2/2] ≈ [0.707, 0.707]
            这保证了无论原始方向向量多长，都转换为长度为1的标准方向向量。
        '''
        unit_direction = direction / np.linalg.norm(direction)#求的方向向量
        # 方向导数=梯度向量与方向向量之间的点积
        return np.dot(grad, unit_direction)

    # 创建可视化数据
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    # 测试点
    test_point = np.array([1.0, 1.0])
    test_gradient = gradient(test_point[0], test_point[1])

    print(f"测试点: ({test_point[0]}, {test_point[1]})")
    print(f"函数值: {f(test_point[0], test_point[1]):.4f}")
    print(f"梯度向量: [{test_gradient[0]:.4f}, {test_gradient[1]:.4f}]")
    print(f"梯度方向: 角度 {np.degrees(np.arctan2(test_gradient[1], test_gradient[0])):.1f}°")

    # 比较不同方向的方向导数
    directions = [
        np.array([1, 0]),  # 正x方向
        np.array([0, 1]),  # 正y方向
        np.array([1, 1]),  # 45度方向
        test_gradient,  # 梯度方向
        np.array([-1, 1]),  # 135度方向
        -test_gradient  # 负梯度方向（梯度下降方向）
    ]

    direction_names = ['正X方向', '正Y方向', '45°方向', '梯度方向', '135°方向', '负梯度方向']

    fig=plt.figure(figsize=(15, 10))#创建一个figure对象

    # 函数曲面
    ax1 = plt.subplot(2, 3, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    ax1.scatter(test_point[0], test_point[1], f(test_point[0], test_point[1]),
                color='red', s=100, label='测试点')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('f(X, Y)')
    ax1.set_title('函数曲面')
    fig.colorbar(surf, ax=ax1, shrink=0.5)

    # 等高线和梯度场
    ax2 = plt.subplot(2, 3, 2)
    contour = ax2.contour(X, Y, Z, levels=15)
    ax2.clabel(contour, inline=True, fontsize=8)

    # 绘制梯度场
    x_coarse = np.linspace(-3, 3, 12)
    y_coarse = np.linspace(-3, 3, 12)
    Xc, Yc = np.meshgrid(x_coarse, y_coarse)

    U, V = gradient(Xc, Yc)
    ax2.quiver(Xc, Yc, U, V, color='blue', alpha=0.6, scale=15)

    ax2.scatter(test_point[0], test_point[1], color='red', s=100, label='测试点')
    ax2.quiver(test_point[0], test_point[1], test_gradient[0], test_gradient[1],
               color='red', scale=1, scale_units='xy', angles='xy', width=0.005, label='梯度')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('等高线与梯度场')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 方向导数比较
    ax3 = plt.subplot(2, 3, 3)
    dir_derivatives = [directional_derivative(test_point[0], test_point[1], dir) for dir in directions]

    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']
    bars = ax3.bar(range(len(directions)), dir_derivatives, color=colors, alpha=0.7)

    for bar, value in zip(bars, dir_derivatives):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom')

    ax3.set_xticks(range(len(directions)))
    ax3.set_xticklabels(direction_names, rotation=45)
    ax3.set_ylabel('方向导数值')
    ax3.set_title('不同方向的方向导数')
    ax3.grid(True, alpha=0.3)

    # 方向可视化
    ax4 = plt.subplot(2, 3, 4)
    ax4.contour(X, Y, Z, levels=10, alpha=0.5)
    ax4.scatter(test_point[0], test_point[1], color='red', s=100, zorder=5)

    # 绘制各个方向
    for i, (direction, name, color) in enumerate(zip(directions, direction_names, colors)):
        # 归一化方向向量
        unit_dir = direction / np.linalg.norm(direction)
        scale = 1.0

        ax4.quiver(test_point[0], test_point[1], unit_dir[0] * scale, unit_dir[1] * scale,
                   color=color, scale=1, scale_units='xy', angles='xy', width=0.005,
                   label=f'{name}', alpha=0.8)

    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_title('不同方向向量')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(test_point[0] - 1.5, test_point[0] + 1.5)
    ax4.set_ylim(test_point[1] - 1.5, test_point[1] + 1.5)

    # 沿梯度方向的函数变化
    ax5 = plt.subplot(2, 3, 5)
    grad_direction = test_gradient / np.linalg.norm(test_gradient)

    # 沿梯度方向采样点
    t_values = np.linspace(-1, 1, 100)
    points_along_grad = test_point + np.outer(t_values, grad_direction)
    values_along_grad = [f(p[0], p[1]) for p in points_along_grad]

    ax5.plot(t_values, values_along_grad, 'r-', linewidth=2, label='沿梯度方向')
    ax5.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='当前位置')
    ax5.set_xlabel('沿梯度方向的位移')
    ax5.set_ylabel('函数值')
    ax5.set_title('沿梯度方向的函数变化')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 梯度下降优化演示
    ax6 = plt.subplot(2, 3, 6)

    '''
    momentum=0.9：动量参数，用于带动量的梯度下降算法中。它控制着历史梯度信息在当前更新中的权重，取值范围通常在[0,1]之间。0.9表示保留90%的历史速度信息，有助于加速收敛并减少震荡。
    n_iter=30：迭代次数，表示梯度下降算法的迭代轮数。算法会执行30次参数更新步骤，以逐步接近最优解。
    '''
    def gradient_descent_with_momentum(start_point, learning_rate=0.1, momentum=0.9, n_iter=30):
        """带动量的梯度下降"""
        path = [start_point]#创建一个路径列表，用于记录优化过程中经过的所有点，初始只包含起始点
        velocity = np.zeros_like(start_point)#初始化速度向量为零向量，维度与起始点相同，用于存储历史更新信息
        current_point = start_point.copy()#创建起始点的副本作为当前点，避免直接修改原始起始点

        for i in range(n_iter):
            grad = gradient(current_point[0], current_point[1])#计算当前点的梯度向量，表示函数在该点增长最快的方向
            velocity = momentum * velocity - learning_rate * grad#更新速度向量，结合了两部分：momentum * velocity：保留之前的速度信息（惯性项），- learning_rate * grad：当前梯度的负方向，乘以学习率
            current_point = current_point + velocity #根据速度更新当前位置，向最优解移动
            path.append(current_point.copy()) #将新位置添加到路径记录中，用于后续可视化

        return np.array(path)

    # 添加标准梯度下降函数
    def gradient_descent_manual(start_point, learning_rate=0.1, n_iterations=30):
        """标准梯度下降"""
        path = [start_point]
        current_point = start_point.copy()

        for i in range(n_iterations):
            grad = gradient(current_point[0], current_point[1])
            current_point = current_point - learning_rate * grad#为了减小函数值，必须沿着梯度的反方向（负梯度方向）移动
            path.append(current_point.copy())

        return np.array(path)

    # 比较不同优化算法
    gd_path = gradient_descent_manual(test_point, learning_rate=0.1, n_iterations=30)
    momentum_path = gradient_descent_with_momentum(test_point, learning_rate=0.1, momentum=0.9, n_iter=30)

    ax6.contour(X, Y, Z, levels=15, alpha=0.5)
    ax6.plot(gd_path[:, 0], gd_path[:, 1], 'bo-', label='普通梯度下降', alpha=0.7)
    ax6.plot(momentum_path[:, 0], momentum_path[:, 1], 'ro-', label='带动量梯度下降', alpha=0.7)
    ax6.scatter(test_point[0], test_point[1], color='green', s=100, label='起点')
    ax6.scatter(gd_path[-1, 0], gd_path[-1, 1], color='blue', s=100, marker='s', label='GD终点')
    ax6.scatter(momentum_path[-1, 0], momentum_path[-1, 1], color='red', s=100, marker='s', label='动量GD终点')

    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_title('不同优化算法比较')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 数学原理总结
    print(f"\n方向导数与梯度的数学原理:")
    print("1. 方向导数: 函数在特定方向的变化率")
    print("2. 梯度方向: 函数增长最快的方向")
    print("3. 梯度大小: 最大方向导数的值")
    print("4. 负梯度方向: 函数下降最快的方向（梯度下降）")

    # 在深度学习中的实际应用
    print(f"\n在深度学习中的实际应用:")
    print("• 梯度下降使用负梯度方向作为更新方向")
    print("• 动量方法结合当前梯度和历史梯度方向")
    print("• 自适应学习率方法（如Adam）考虑梯度方向的历史信息")
    print("• 梯度裁剪防止梯度方向变化过大")


directional_derivative_optimization()