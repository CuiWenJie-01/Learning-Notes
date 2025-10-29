import numpy as np


def vector_space_verification():
    """验证二维实数向量构成向量空间"""
    # 随机生成几个二维向量
    u = np.array([1, 2])
    v = np.array([3, 4])
    w = np.array([5, 6])

    print("向量u:", u)
    print("向量v:", v)
    print("向量w:", w)

    # 1. 加法交换律: u + v = v + u
    print("\n1. 加法交换律验证:")
    print(f"u + v = {u + v}")
    print(f"v + u = {v + u}")
    print(f"是否相等: {np.array_equal(u + v, v + u)}")

    # 2. 加法结合律: (u + v) + w = u + (v + w)
    print("\n2. 加法结合律验证:")
    print(f"(u + v) + w = {(u + v) + w}")
    print(f"u + (v + w) = {u + (v + w)}")
    print(f"是否相等: {np.array_equal((u + v) + w, u + (v + w))}")

    # 3. 零向量存在: u + 0 = u
    zero_vec = np.array([0, 0])
    print("\n3. 零向量验证:")
    print(f"u + 零向量 = {u + zero_vec}")
    print(f"是否等于u: {np.array_equal(u + zero_vec, u)}")

    # 4. 负向量存在: u + (-u) = 0
    neg_u = -u
    print("\n4. 负向量验证:")
    print(f"u + (-u) = {u + neg_u}")
    print(f"是否等于零向量: {np.array_equal(u + neg_u, zero_vec)}")


vector_space_verification()