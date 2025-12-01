# 数学概念：多元复合函数的偏导数可以通过链式法则计算
# AI对应：神经网络中误差反向传播的理论基础
import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams['font.family'] = 'DejaVu Sans, Arial, sans-serif'#设置带有数学平方的字体，用于显示数学公式
import torch
import torch.nn as nn


# 实践2：链式法则与神经网络反向传播
def chain_rule_backpropagation():
    """
    链式法则在神经网络反向传播中的核心作用
    """
    print("=== 链式法则与反向传播 ===")

    # 定义一个简单的计算图：f(x, y, z) = (x × y + z)²
    # 分解为中间变量：
    # u = x × y
    # v = u + z
    # f = v²

    def forward_pass(x, y, z):
        """前向传播计算函数值"""
        u = x * y
        v = u + z
        f = v ** 2
        return f, u, v

    def manual_backward(x, y, z):
        """手动反向传播计算偏导数"""
        # 前向传播
        f, u, v = forward_pass(x, y, z)

        # 反向传播（链式法则）
        # ∂f/∂v = 2v
        df_dv = 2 * v

        # ∂f/∂u = ∂f/∂v × ∂v/∂u = 2v × 1
        df_du = df_dv * 1

        # ∂f/∂z = ∂f/∂v × ∂v/∂z = 2v × 1
        df_dz = df_dv * 1

        # ∂f/∂x = ∂f/∂u × ∂u/∂x = 2v × y
        df_dx = df_du * y

        # ∂f/∂y = ∂f/∂u × ∂u/∂y = 2v × x
        df_dy = df_du * x

        return df_dx, df_dy, df_dz, f

    # 测试点
    x_test, y_test, z_test = 2.0, 3.0, 1.0
    df_dx_manual, df_dy_manual, df_dz_manual, f_value = manual_backward(x_test, y_test, z_test)

    print(f"测试点: x={x_test}, y={y_test}, z={z_test}")
    print(f"函数值 f(x,y,z) = {f_value}")
    print(f"手动计算偏导数:")
    print(f"  ∂f/∂x = {df_dx_manual}")
    print(f"  ∂f/∂y = {df_dy_manual}")
    print(f"  ∂f/∂z = {df_dz_manual}")

    # 使用PyTorch验证
    x_pt = torch.tensor([x_test], dtype=torch.float32, requires_grad=True)
    y_pt = torch.tensor([y_test], dtype=torch.float32, requires_grad=True)
    z_pt = torch.tensor([z_test], dtype=torch.float32, requires_grad=True)

    f_pt = (x_pt * y_pt + z_pt) ** 2
    f_pt.backward()

    print(f"\nPyTorch自动微分验证:")
    print(f"  ∂f/∂x = {x_pt.grad.item()}")
    print(f"  ∂f/∂y = {y_pt.grad.item()}")
    print(f"  ∂f/∂z = {z_pt.grad.item()}")

    # 可视化计算图
    plt.figure(figsize=(15, 8))

    # 计算图结构
    plt.subplot(2, 3, 1)
    nodes = ['x', 'y', 'z', 'u = x×y', 'v = u+z', 'f = v²']
    positions = {
        'x': (0, 2), 'y': (0, 1), 'z': (0, 0),
        'u = x×y': (2, 1.5), 'v = u+z': (4, 1), 'f = v²': (6, 1)
    }

    # 绘制节点
    for node, (x, y) in positions.items():
        plt.scatter(x, y, s=500, c='lightblue', edgecolors='black')
        plt.text(x, y, node, ha='center', va='center', fontweight='bold')

    # 绘制边
    edges = [('x', 'u = x×y'), ('y', 'u = x×y'),
             ('u = x×y', 'v = u+z'), ('z', 'v = u+z'),
             ('v = u+z', 'f = v²')]

    for start, end in edges:
        start_pos = positions[start]
        end_pos = positions[end]
        plt.arrow(start_pos[0], start_pos[1],
                  end_pos[0] - start_pos[0], end_pos[1] - start_pos[1],
                  head_width=0.1, head_length=0.1, fc='black', ec='black', alpha=0.6)

    plt.xlim(-1, 7)
    plt.ylim(-1, 3)
    plt.title('计算图结构')
    plt.axis('off')

    # 前向传播值
    plt.subplot(2, 3, 2)#参数含义：2行3列网格中的第2个位置，对应位置：第一行的第二个格子，用途：显示前向传播的节点数值
    values = {
        'x': x_test, 'y': y_test, 'z': z_test,
        'u = x×y': x_test * y_test,
        'v = u+z': x_test * y_test + z_test,
        'f = v²': f_value
    }

    for i, (node, value) in enumerate(values.items()):
        plt.bar(i, value, color='lightgreen', alpha=0.7)
        plt.text(i, value + 0.1, f'{value:.1f}', ha='center', va='bottom')

    plt.xticks(range(len(values)), list(values.keys()), rotation=45)
    plt.ylabel('数值')
    plt.title('前向传播：节点数值')
    plt.grid(True, alpha=0.3)

    # 反向传播梯度
    plt.subplot(2, 3, 3)#参数含义：2行3列网格中的第3个位置，对应位置：第一行的第三个格子，用途：显示反向传播的梯度计算
    gradients = {
        '∂f/∂x': df_dx_manual,
        '∂f/∂y': df_dy_manual,
        '∂f/∂z': df_dz_manual,
        '∂f/∂u': 2 * (x_test * y_test + z_test),  # ∂f/∂v × ∂v/∂u
        '∂f/∂v': 2 * (x_test * y_test + z_test),  # ∂f/∂f × ∂f/∂v
        '∂f/∂f': 1.0  # ∂f/∂f = 1
    }

    for i, (node, grad) in enumerate(gradients.items()):
        plt.bar(i, grad, color='lightcoral', alpha=0.7)
        plt.text(i, grad + (1 if grad > 0 else -1), f'{grad:.1f}',
                 ha='center', va='bottom' if grad > 0 else 'top')

    plt.xticks(range(len(gradients)), list(gradients.keys()), rotation=45)
    plt.ylabel('梯度值')
    plt.title('反向传播：梯度计算')
    plt.grid(True, alpha=0.3)

    # 链式法则分解
    plt.subplot(2, 3, 4)
    chain_rule_steps = [
        '∂f/∂v = 2v',
        '∂f/∂u = ∂f/∂v × ∂v/∂u = 2v × 1',
        '∂f/∂z = ∂f/∂v × ∂v/∂z = 2v × 1',
        '∂f/∂x = ∂f/∂u × ∂u/∂x = 2v × y',
        '∂f/∂y = ∂f/∂u × ∂u/∂y = 2v × x'
    ]

    for i, step in enumerate(chain_rule_steps):
        plt.text(0.1, 0.9 - i * 0.15, f'{i + 1}. {step}', transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('链式法则分解步骤')

    # 梯度流向图
    plt.subplot(2, 3, 5)
    # 重新绘制计算图，这次显示梯度流向
    for node, (x, y) in positions.items():
        plt.scatter(x, y, s=500, c='lightyellow', edgecolors='black')
        plt.text(x, y, node, ha='center', va='center', fontweight='bold')

    # 绘制梯度反向流动
    grad_edges = [('f = v²', 'v = u+z'), ('v = u+z', 'u = x×y'),
                  ('v = u+z', 'z'), ('u = x×y', 'x'), ('u = x×y', 'y')]

    for start, end in grad_edges:
        start_pos = positions[start]
        end_pos = positions[end]
        plt.arrow(start_pos[0], start_pos[1],
                  end_pos[0] - start_pos[0], end_pos[1] - start_pos[1],
                  head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.8, linestyle='--')

    plt.xlim(-1, 7)
    plt.ylim(-1, 3)
    plt.title('梯度反向传播路径')
    plt.axis('off')

    # 数值验证
    plt.subplot(2, 3, 6)
    manual_grads = [df_dx_manual, df_dy_manual, df_dz_manual]
    pytorch_grads = [x_pt.grad.item(), y_pt.grad.item(), z_pt.grad.item()]
    labels = ['∂f/∂x', '∂f/∂y', '∂f/∂z']

    x_pos = np.arange(len(labels))
    width = 0.35

    plt.bar(x_pos - width / 2, manual_grads, width, label='手动计算', alpha=0.7)
    plt.bar(x_pos + width / 2, pytorch_grads, width, label='PyTorch', alpha=0.7)

    plt.ylabel('梯度值')
    plt.title('手动计算 vs PyTorch自动微分')
    plt.xticks(x_pos, labels)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 扩展到神经网络场景
    print(f"\n在神经网络中的应用:")

    # 简单的神经网络层示例
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)  # 2输入，1输出
            nn.init.constant_(self.linear.weight, 0.5)#权重初始化为0.5
            nn.init.constant_(self.linear.bias, 0.2)#偏置初始化为0.2

        def forward(self, x):
            return self.linear(x)

    # 创建网络和测试数据
    net = SimpleNet()
    criterion = nn.MSELoss()

    # 测试输入
    x_input = torch.tensor([[1.0, 2.0]], requires_grad=True)
    target = torch.tensor([[2.0]])

    # 前向传播
    output = net(x_input)
    loss = criterion(output, target)

    print(f"网络前向传播:")
    print(f"  输入: {x_input.detach().numpy()}")
    print(f"  权重: {net.linear.weight.detach().numpy()}")
    print(f"  偏置: {net.linear.bias.detach().numpy()}")
    print(f"  输出: {output.detach().numpy()}")
    print(f"  目标: {target.detach().numpy()}")
    print(f"  损失: {loss.item()}")

    # 反向传播
    loss.backward()

    print(f"\n反向传播梯度:")
    print(f"  ∂loss/∂input: {x_input.grad.numpy()}")
    print(f"  ∂loss/∂weight: {net.linear.weight.grad.numpy()}")
    print(f"  ∂loss/∂bias: {net.linear.bias.grad.numpy()}")


chain_rule_backpropagation()