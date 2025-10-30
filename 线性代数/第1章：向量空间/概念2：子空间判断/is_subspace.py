# 数学概念：判断一个子集是否是向量空间的子空间。
# 代码实践：验证R²中的各种子集是否为子空间。
import numpy as np
# 绘图库，子空间可视化
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def is_subspace(vectors,tolerance=1e-10):
    """
    判断向量集合是否构成子空间
    简化版：检查零向量存在，且任意线性组合仍在集合中
    """
    vectors = np.array(vectors)

    # 检查零向量是否存在集合中（近似）
    # 使用np.allclose()逐个检查向量集合中的每个向量是否接近零向量
    # any()函数判断是否存在至少一个零向量
    # 如果集合中没有零向量，则直接返回False和相应错误信息
    # 这是判断向量空间子空间的第一步检验，因为子空间必须包含零向量。
    zero_in_set=any(np.allclose(v,0) for v in vectors)
    if not zero_in_set:
        return False,"零向量不在集合中"

    # 简化检查：随机线性组合是否仍在张成空间中
    # 注意：这在实际中需要更严谨的检查
    span_vector=vectors
    for i in range(len(vectors)):#range()用于生成一个数字序列
        for j in range(i+1,len(vectors)):
            linear_comb=vectors[i]+vectors[j]#子空间的加法封闭性：如果集合是子空间，那么任意两个向量的和也必须在该集合中
            # 向量加法示例
            # [1,0] + [2,0] = [3,0]  # 第一个和第二个向量相加
            # [1,0] + [0,0] = [1,0]  # 第一个和第三个向量相加
            # [2,0] + [0,0] = [2,0]  # 第二个和第三个向量相加
            # 通过这些向量运算来检验：
            # 加法封闭性：两个x轴向量的和是否仍在x轴上
            # 零向量存在性：集合是否包含零向量 [0,0]
            # 检查线性组合是否可以用原向量线性表示
            # 这里简化处理，实际应该解线性方程组
            if not can_be_represented(linear_comb, vectors, tolerance):
                return False, f"向量 {linear_comb} 不能由原向量组线性表示"
            pass
    # 在现有加法封闭性检查之后添加标量乘法封闭性验证
    # 选择一些典型标量进行测试
    scalars = [0, 1, -1, 2, -2, 0.5]
    for i in range(len(vectors)):
        for scalar in scalars:
            scaled_vector = scalar * vectors[i]
            # 检查缩放后的向量是否仍在张成空间中
            # 这里需要实现判断向量是否在张成空间的逻辑
            # 例如通过求解线性方程组判断scaled_vector是否能被vectors线性表示
            # 检查缩放后的向量是否仍在张成空间中
            if not can_be_represented(scaled_vector, vectors, tolerance):
                return False, f"向量 {scaled_vector} 不能由原向量组线性表示"
            pass
    return  True,"可能是子空间"

def can_be_represented(target_vector, basis_vectors, tolerance=1e-10):
    """
    判断目标向量是否能由基向量组线性表示
    通过求解线性方程组 Ax = b 来判断
    """
    try:
        # 构造系数矩阵A（转置以适应numpy的求解要求）
        A = basis_vectors.T
        b = target_vector

        # 使用最小二乘法求解
        coeffs, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

        # 检查解的有效性
        reconstructed = A @ coeffs
        if np.allclose(reconstructed, b, atol=tolerance):
            return True
        else:
            return False
    except Exception:
        return False

# 测试例子
print("=== 子空间验证 ===")

# 例子1: x轴上的所有向量 (是子空间)
x_axis_vectors=[np.array([1,0]),np.array([2,0]),np.array([0,0])]
result,reason=is_subspace(x_axis_vectors)
print(f"x轴向量集合：{result} - {reason}")

# 例子2: 第一象限的所有向量 (不是子空间)
first_quadrant = [np.array([1, 1]), np.array([2, 3]), np.array([1, 2])]
result, reason = is_subspace(first_quadrant)
print(f"第一象限向量集合: {result} - {reason}")

# 可视化子空间

def plot_subspace_example():
    # 创建左右子空间
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # x轴子空间
    x = np.linspace(-3, 3, 100)
    y = np.zeros_like(x)
    ax1.plot(x, y, 'b-', linewidth=2, label='x轴子空间')
    ax1.quiver(0, 0, 2, 0, color='r', scale=1, scale_units='xy', angles='xy', label='向量 [2,0]')
    ax1.quiver(0, 0, -1, 0, color='g', scale=1, scale_units='xy', angles='xy', label='向量 [-1,0]')
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.grid(True)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax1.set_title('子空间示例: x轴')
    ax1.legend()

    # 第一象限（非子空间）
    x_pos = np.linspace(0.1, 3, 50)
    y_pos = np.linspace(0.1, 3, 50)
    X, Y = np.meshgrid(x_pos, y_pos)
    ax2.scatter(X, Y, alpha=0.6, label='第一 象限点')
    ax2.quiver(0, 0, 1, 1, color='r', scale=1, scale_units='xy', angles='xy', label='向量 [1,1]')
    ax2.quiver(0, 0, 2, 2, color='r', scale=1, scale_units='xy', angles='xy')
    ax2.set_xlim(-1, 3)
    ax2.set_ylim(-1, 3)
    ax2.grid(True)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_title('非子空间示例: 第一象限')
    ax2.legend()

    plt.tight_layout()
    plt.show()


plot_subspace_example()