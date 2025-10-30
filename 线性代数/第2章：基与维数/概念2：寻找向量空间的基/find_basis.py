# 数学概念：给定一组向量，找到它们张成空间的基。
# 代码实践：使用行简化阶梯形找到极大线性无关组。
import numpy as np

def find_basis(vectors):
    """
    找到向量组的基（极大线性无关组）
    原理：使用行简化阶梯形方法
    """
    matrix = np.array(vectors).T  # 构造矩阵，每列是一个向量
    print(f"原始向量组：{vectors}")
    print(f"构造的矩阵：\n{matrix}")

    # 计算行简化阶梯形
    # 注意：np没有直接的行简化函数，我们用QR分解的Q矩阵的列空间作为基
    Q,R=np.linalg.qr(matrix)
    rank=np.linalg.matrix_rank(matrix)

    print(f"\n矩阵的秩（维数）: {rank}")

    basis=Q[:,:rank].T
    print(f"\n找到的基向量:")
    for i,vec in enumerate(basis):
        print(f"基向量{i+1}: {vec}")
    return  basis
print("=== 寻找基 ===")
# 例子1: 二维空间的基
vectors_2d = [[1, 2], [3, 4], [2, 4]]  # 第三个是前两个的线性组合
basis_2d = find_basis(vectors_2d)

print("\n" + "="*50)

# 例子2: 三维空间的基
vectors_3d = [[1, 0, 0], [0, 1, 0], [1, 1, 0]]  # 都在xy平面
basis_3d = find_basis(vectors_3d)

# ### 代码解释
#
# 这行代码 `basis=Q[:,:rank].T` 是从QR分解结果中提取基向量的关键步骤：
#
# ### 操作分解
#
# 1. **`Q[:,:rank]`**：
#    - 从 `Q` 矩阵中取出前 `rank` 列
#    - `rank` 是矩阵的秩，表示线性无关向量的个数
#    - 这些列向量构成了原向量组张成空间的一组标准正交基
#
# 2. **`.T` 转置操作**：
#    - 将列向量转换为行向量的形式
#    - 使输出格式更符合通常的向量表示习惯
#
# ### 数学原理
#
# - QR分解中的 `Q` 矩阵是正交矩阵，其列向量线性无关
# - 前 `rank` 个列向量恰好构成原向量组所张成空间的一组基
# - 由于 `Q` 的列向量是标准正交的，所以这组基具有良好的数值性质
#
# ### 结果
# 最终得到的 `basis` 变量包含了原向量组张成空间的一组标准正交基。