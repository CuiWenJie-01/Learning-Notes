# 数学概念：验证一个集合是否构成向量空间（封闭性、结合律、交换律等）。
# 代码实践：验证二维向量的集合是否满足向量空间公理。

import numpy as np
#numpy库的作用
# 提供高效的数值计算功能：NumPy是Python中进行科学计算的基础库，特别适用于处理向量、矩阵等线性代数运算。
# 实现向量空间的相关操作：根据函数注释，这里要判断向量组是否线性无关，需要用到矩阵的秩等概念，这些都可以通过NumPy高效地实现。
# 简化数学计算：NumPy提供了丰富的线性代数函数，可以方便地进行向量加法、标量乘法、矩阵运算等向量空间基本操作。
def vector_space_verification():
    """验证二维实数向量构成向量空间"""
    # 随机生成几个二维向量
    u=np.array([1,2])
    v=np.array([3,4])
    w=np.array([5,6])

    print("向量u:",u)
    print("向量v:",v)
    print("向量w:",w)

    #1.加法交换律：u+v=v+u
    print("\n1.加法交换律验证：")
    print(f"u+v={u+v}")#f前缀表示这是一个格式化字符串，可以在字符串中直接嵌入变量和表达式
    print(f"v+u={v+u}")
    print(f"是否相等：{np.array_equal(u+v,v+u)}")

    #2.加法结合律：(u+v)+w=u+(v+w)
    print("\n2.加法结合律验证：")
    print(f"(u+v)+w={u+v+w}")
    print(f"u+(v+w)={u+(v+w)}")
    print(f"是否相等：{np.array_equal(u+v+w,u+(v+w))}")

    #3.零向量存在：u+0=u
    zero_vec=np.array([0,0])
    print("\n3.零向量存在验证：")
    print(f"u+0={u+zero_vec}")
    print(f"是否相等：{np.array_equal(u+zero_vec,u)}")

    #4.负向量存在：u+(-u)=0
    neg_u=-u
    print("\n4.负向量存在验证：")
    print(f"u+(-u)={u+neg_u}")
    print(f"是否相等：{np.array_equal(u+neg_u,zero_vec)}")

vector_space_verification()