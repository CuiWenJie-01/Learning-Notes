# 数学概念：判断一组向量是否线性相关。
# 代码实践：通过矩阵秩来判断线性相关性。
import numpy as np

def check_linear_independence(vectors):
    """
    判断向量组是否线性无关
    原理: 如果矩阵的秩 = 向量个数，则线性无关
    """
    matrix=np.array(vectors).T #构造矩阵，每列是一个向量
    rank=np.linalg.matrix_rank(matrix)#使用NumPy的matrix_rank函数计算矩阵的秩
    num_vectors=len(vectors)#获取向量组中向量的总个数

    print(f"向量组：{vectors}")
    print(f"构造的矩阵：\n{matrix}")
    print(f"矩阵的秩：{rank}")
    print(f"向量个数：{num_vectors}")

    # 如果矩阵的秩等于向量个数，则这些向量线性无关
    # 如果矩阵的秩小于向量个数，则这些向量线性相关
    #秩：将矩阵化成最简，非0行数是否<向量个数，小于的则线性相关，等于的则线性无关
    if rank==num_vectors:
        print("结论：向量组线性无关")
        return True
    else:
        print("结论：向量组线性相关")
        return False

print("=== 线性相关性验证 ===")
# 线性无关的例子
vectors1 = [[1, 0], [0, 1]]
check_linear_independence(vectors1)

print("\n" + "="*50)

# 线性相关的例子
vectors2 = [[1, 2], [2, 4]]  # 第二个向量是第一个的2倍
check_linear_independence(vectors2)
print("\n" + "="*50)

# 3维空间中的例子
vectors3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # 标准基
check_linear_independence(vectors3)
print("\n" + "="*50)

vectors4 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # 线性相关（第三行是第一第二行的和）
check_linear_independence(vectors4)