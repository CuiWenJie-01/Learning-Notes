import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
plt.rcParams['font.family'] = 'DejaVu Sans, Arial, sans-serif'#设置带有数学平方的字体，用于显示数学公式

# 实战项目1：从零实现线性回归（偏导数的直接应用）
def linear_regression_from_scratch():
    """
    使用偏导数从零实现线性回归，展示梯度下降的实际应用
    """
    print("=== 从零实现线性回归 ===")

    # 生成合成数据
    np.random.seed(42)
    n_samples = 100
    true_slope = 2.5 # 真实斜率
    true_intercept = 1.0 # 真实截距
    noise_std = 0.5 # 噪声标准差

    # 生成数据: y = true_slope * x + true_intercept + noise
    X = np.linspace(0, 1, n_samples)
    true_y = true_slope * X + true_intercept
    y = true_y + np.random.normal(0, noise_std, n_samples)

    print(f"生成数据统计:")
    print(f"  样本数量: {n_samples}")
    print(f"  真实斜率: {true_slope}")
    print(f"  真实截距: {true_intercept}")
    print(f"  噪声标准差: {noise_std}")

    # 线性回归模型：y_pred = w * x + b
    def predict(X, w, b):
        return w * X + b

    # 损失函数：均方误差
    def mse_loss(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)#y_true: 真实值（实际观测值）, y_pred: 预测值（模型预测的结果），(y_true - y_pred): 计算每个样本的真实值与预测值之间的差值（即误差）

    # 手动计算梯度
    def compute_gradients(X, y, w, b):
        """计算损失函数对w和b的偏导数"""
        n = len(X)
        y_pred = predict(X, w, b)

        # ∂L/∂w = (2/n) * Σ(y_pred - y) * x
        dw = (2 / n) * np.sum((y_pred - y) * X)

        # ∂L/∂b = (2/n) * Σ(y_pred - y)
        db = (2 / n) * np.sum(y_pred - y)

        return dw, db

    # 梯度下降训练
    def train_linear_regression(X, y, learning_rate=0.1, n_epochs=100):
        """使用梯度下降训练线性回归模型"""
        # 初始化参数
        w = np.random.randn()
        b = np.random.randn()

        losses = []
        weights = [w]
        biases = [b]

        print(f"\n开始训练:")
        print(f"  初始参数: w={w:.3f}, b={b:.3f}")
        print(f"  学习率: {learning_rate}")
        print(f"  训练轮数: {n_epochs}")

        for epoch in range(n_epochs):
            # 前向传播
            y_pred = predict(X, w, b)
            loss = mse_loss(y, y_pred)
            losses.append(loss)

            # 计算梯度
            dw, db = compute_gradients(X, y, w, b)

            # 更新参数
            w = w - learning_rate * dw
            b = b - learning_rate * db

            weights.append(w)
            biases.append(b)

            if epoch % 20 == 0:
                print(f"  轮次 {epoch}: w={w:.3f}, b={b:.3f}, 损失={loss:.4f}, "
                      f"梯度=({dw:.4f}, {db:.4f})")

        return w, b, losses, weights, biases

    # 训练模型
    final_w, final_b, losses, weights, biases = train_linear_regression(X, y)

    # 使用PyTorch验证
    X_torch = torch.tensor(X, dtype=torch.float32).reshape(-1, 1)
    y_torch = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    torch_model = nn.Linear(1, 1) # 输入维度为1，输出维度为1
    criterion = nn.MSELoss() # 损失函数
    optimizer = optim.SGD(torch_model.parameters(), lr=0.1)

    torch_losses = []
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = torch_model(X_torch)
        loss = criterion(outputs, y_torch)
        loss.backward()
        optimizer.step() # 更新参数
        torch_losses.append(loss.item())

    torch_w = torch_model.weight.data.item()
    torch_b = torch_model.bias.data.item()

    print(f"\n训练结果比较:")
    print(f"  手动实现: w={final_w:.4f}, b={final_b:.4f}")
    print(f"  PyTorch实现: w={torch_w:.4f}, b={torch_b:.4f}")
    print(f"  真实参数: w={true_slope:.4f}, b={true_intercept:.4f}")

    # 可视化结果
    plt.figure(figsize=(15, 10))

    # 数据点和回归线
    plt.subplot(2, 3, 1)
    plt.scatter(X, y, alpha=0.6, label='数据点')
    plt.plot(X, true_y, 'g-', linewidth=2, label='真实关系')
    plt.plot(X, predict(X, final_w, final_b), 'r-', linewidth=2, label='学习到的关系')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('线性回归结果')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 损失函数下降
    plt.subplot(2, 3, 2)
    plt.plot(losses, 'b-', label='手动实现')
    plt.plot(torch_losses, 'r--', label='PyTorch实现')
    plt.xlabel('训练轮次')
    plt.ylabel('均方误差损失')
    plt.title('损失函数下降')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 参数空间轨迹
    plt.subplot(2, 3, 3)
    # 创建参数空间的损失函数等高线
    w_range = np.linspace(0, 4, 50)
    b_range = np.linspace(-1, 3, 50)
    W, B = np.meshgrid(w_range, b_range)

    Z = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Z[i, j] = mse_loss(y, predict(X, W[i, j], B[i, j]))

    contour = plt.contour(W, B, Z, levels=20)
    plt.clabel(contour, inline=True, fontsize=8)
    plt.plot(weights, biases, 'ro-', linewidth=2, markersize=4, label='参数轨迹')
    plt.plot([true_slope], [true_intercept], 'g*', markersize=15, label='真实参数')
    plt.plot([final_w], [final_b], 'bs', markersize=10, label='最终参数')
    plt.xlabel('权重 w')
    plt.ylabel('偏置 b')
    plt.title('参数空间与优化轨迹')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 梯度变化
    plt.subplot(2, 3, 4)
    # 计算每个点的梯度大小
    grad_magnitudes = []
    for w, b in zip(weights[:-1], biases[:-1]):
        dw, db = compute_gradients(X, y, w, b)
        grad_magnitude = np.sqrt(dw ** 2 + db ** 2)
        grad_magnitudes.append(grad_magnitude)

    plt.plot(grad_magnitudes, 'g-')
    plt.xlabel('训练轮次')
    plt.ylabel('梯度范数 ||∇L||')
    plt.title('梯度范数变化')
    plt.grid(True, alpha=0.3)

    # 参数变化
    plt.subplot(2, 3, 5)
    plt.plot(weights, 'b-', label='权重 w')
    plt.plot(biases, 'r-', label='偏置 b')
    plt.axhline(y=true_slope, color='blue', linestyle='--', alpha=0.5, label='真实 w')
    plt.axhline(y=true_intercept, color='red', linestyle='--', alpha=0.5, label='真实 b')
    plt.xlabel('训练轮次')
    plt.ylabel('参数值')
    plt.title('参数学习过程')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 残差分析
    plt.subplot(2, 3, 6)
    y_pred_final = predict(X, final_w, final_b)
    residuals = y - y_pred_final

    plt.scatter(y_pred_final, residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('残差图')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 偏导数的具体计算演示
    print(f"\n偏导数计算演示 (在初始参数点):")
    initial_w, initial_b = weights[0], biases[0]
    dw_initial, db_initial = compute_gradients(X, y, initial_w, initial_b)

    print(f"初始参数: w={initial_w:.3f}, b={initial_b:.3f}")
    print(f"偏导数计算:")
    print(f"  ∂L/∂w = (2/{n_samples}) * Σ((w*x + b - y) * x)")
    print(f"        = {dw_initial:.4f}")
    print(f"  ∂L/∂b = (2/{n_samples}) * Σ(w*x + b - y)")
    print(f"        = {db_initial:.4f}")

    # 数值梯度验证
    def numerical_gradient(X, y, w, b, epsilon=1e-7):#epsilon=ε:表示一个极小的数值增量，用于计算数值导数。
        """数值方法计算梯度（验证用）"""
        # ∂L/∂w ≈ [L(w+ε, b) - L(w-ε, b)] / (2ε)
        loss_plus = mse_loss(y, predict(X, w + epsilon, b))
        loss_minus = mse_loss(y, predict(X, w - epsilon, b))
        dw_numerical = (loss_plus - loss_minus) / (2 * epsilon)

        # ∂L/∂b ≈ [L(w, b+ε) - L(w, b-ε)] / (2ε)
        loss_plus = mse_loss(y, predict(X, w, b + epsilon))
        loss_minus = mse_loss(y, predict(X, w, b - epsilon))
        db_numerical = (loss_plus - loss_minus) / (2 * epsilon)

        return dw_numerical, db_numerical

    dw_numerical, db_numerical = numerical_gradient(X, y, initial_w, initial_b)

    print(f"\n数值梯度验证:")
    print(f"  解析梯度: (∂L/∂w, ∂L/∂b) = ({dw_initial:.6f}, {db_initial:.6f})")
    print(f"  数值梯度: (∂L/∂w, ∂L/∂b) = ({dw_numerical:.6f}, {db_numerical:.6f})")
    print(f"  相对误差: {np.abs(dw_initial - dw_numerical) / (np.abs(dw_initial) + 1e-8):.2e}, "
          f"{np.abs(db_initial - db_numerical) / (np.abs(db_initial) + 1e-8):.2e}")


linear_regression_from_scratch()