# 数学概念：在不同基下表示同一个向量。
# 代码实践：计算向量在给定基下的坐标。
import numpy as np

def find_coordinates(vector, basis):
    """
    找到向量在给定基下的坐标
    解方程: vector = c1*b1 + c2*b2 + ...
    """
    # 构造系数矩阵
    A = np.array(basis).T
    b = np.array(vector)

    #解线性方程组A*x=b
    coordinates=np.linalg.solve(A, b)

    print(f"向量: {vector}")
    print(f"基: {basis}")
    print(f"在基下的坐标: {coordinates}")

    # 验证：用坐标重构原向量
    #reconstructed = sum(c * np.array(b_vec) for c, b_vec in zip(coordinates, basis))
    reconstructed=A @ coordinates# 使用矩阵乘法
    print(f"重构的向量: {reconstructed}")
    print(f"重构是否准确: {np.allclose(vector, reconstructed)}")

    return coordinates

print("=== 坐标变换 ===")
# 标准基
standard_basis = [[1, 0], [0, 1]]
vector = [3, 4]
coords_standard = find_coordinates(vector, standard_basis)

print("\n" + "="*50)

# 非标准基
non_standard_basis = [[1, 1], [1, -1]]  # 45度旋转的基
coords_non_std = find_coordinates(vector, non_standard_basis)

#对于向量 vector = [3, 4] 在标准基 standard_basis = [[1, 0], [0, 1]] 下的坐标：
#   [3, 4] = c₁[1, 0] + c₂[0, 1]
# [3, 4] = [c₁, c₂]
# 3 = c₁ × 1 + c₂ × 0  →  c₁ = 3
# 4 = c₁ × 0 + c₂ × 1  →  c₂ = 4




