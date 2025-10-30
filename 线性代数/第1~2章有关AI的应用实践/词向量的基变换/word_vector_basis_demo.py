import numpy as np

def word_vector_basis_demo():
    """
    演示在词向量空间中基的概念
    """
    # 假设我们有一个小的词向量空间（实际中会用Word2Vec或GloVe）
    # 词汇表: king, queen, man, woman, computer, science

    # 原始基下的词向量（3维简化示例）
    word_vectors_original={
        "king": np.array([1, 0, 0]),
        "queen": np.array([0, 1, 0]),
        "man": np.array([0, 0, 1]),
        "woman": np.array([1, 1, 0]),
        "computer": np.array([0, 1, 1]),
        "science": np.array([1, 0, 1])
    }

    print("原始基下的词向量:")
    for word,vec in word_vectors_original.items():
        print(f"{word}: {vec}")

    # 定义一个新的基（线性变换）
    new_basis=[
        [1, 1, 0],  # 新基向量1
        [1, -1, 0],  # 新基向量2
        [0, 0, 2]  # 新基向量3
    ]

    # 计算变换矩阵（从新基到标准基）
    P = np.array(new_basis).T  # 变换矩阵
    P_inv = np.linalg.inv(P)  # 逆变换

    print(f"\n变换矩阵 P:\n{P}")
    print(f"\n逆变换矩阵 P_inv:\n{P_inv}")

    # 将词向量变换到新基下
    print("\n新基下的词向量（坐标）:")
    word_vectors_new_basis={}
    for word,vec in word_vectors_original.items():
        # 在新基下的坐标: [x]_new = P_inv * [x]_standard
        new_coords =P_inv @ vec
        word_vectors_new_basis[word] = new_coords
        print(f"{word}: {new_coords}")

    # 验证向量在不同基下表示的是同一个概念
    print("\n=== 验证国王-男人+女人 ≈ 皇后 ===")
    king_old=word_vectors_original["king"]
    man_old=word_vectors_original["man"]
    woman_old=word_vectors_original["woman"]
    queen_old=word_vectors_original["queen"]

    # 在原始基下的经典类比
    analogy_old=king_old-man_old+woman_old
    print(f"原始基下: king - man + woman = {analogy_old}")
    print(f"与queen的相识度:{np.dot(analogy_old, queen_old)}")

    # 在新基下的类比
    king_new=word_vectors_new_basis["king"]
    man_new=word_vectors_new_basis["man"]
    woman_new=word_vectors_new_basis["woman"]
    queen_new=word_vectors_new_basis["queen"]

    analogy_new=king_new-man_new+woman_new
    print(f"新基下: king - man + woman = {analogy_new}")
    print(f"与queen的相识度:{np.dot(analogy_new, queen_new)}")

    print("\n结论: 虽然坐标表示不同，但向量关系在基变换下保持不变！")
    #理解为：方向相同，只是长度或者大小不同
word_vector_basis_demo()

#是的，**数值越大越相似**。

## 相似度说明

### 原理
# 这里使用的是**点积**（内积）来计算相似度：
# ```python
# np.dot(analogy_old, queen_old)
# np.dot(analogy_new, queen_new)
# ```
#
#
# ### 判断标准
# - **点积数值越大** → 两个向量越相似
# - **点积数值越小** → 两个向量越不相似
# - **点积为0** → 两个向量正交（完全不相关）
#
# ### 数学解释
# 点积的计算公式：
# ```
# a · b = |a| × |b| × cos(θ)
# ```
#
#
# 其中 `θ` 是两个向量之间的夹角：
# - 当 `θ = 0°` 时，`cos(θ) = 1`，点积最大，向量最相似
# - 当 `θ = 90°` 时，`cos(θ) = 0`，点积为0
# - 当 `θ = 180°` 时，`cos(θ) = -1`，点积最小（负值）
#
# ### 实际应用
# 在词向量应用中，通过比较 `king - man + woman` 的结果向量与 `queen` 向量的点积来验证语义关系的保持性。点积越大说明语义关系在基变换前后保持得越好。