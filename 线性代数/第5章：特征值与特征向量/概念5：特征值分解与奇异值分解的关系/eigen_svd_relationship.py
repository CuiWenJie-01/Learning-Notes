# 实践5：特征值分解与SVD的数学关系
import numpy as np
import matplotlib.pyplot as plt


def eigen_svd_relationship():
    """
    探索特征值分解与奇异值分解的深刻数学关系
    """
    print("=== 特征值分解 vs 奇异值分解 ===")

    # 创建一个随机矩阵
    A=np.random.rand(5,3)
    print(f"原始矩阵 A 形状: {A.shape}")
    print(f"A = \n{A}")

    # 方法1：直接计算A^T*A的特征分解
    ATA = A.T @ A
    print(f"\nA^T*A 形状: {ATA.shape}")

    # A^T*A的特征分解
    eigenvalues_ATA, eigenvectors_ATA  = np.linalg.eigh(ATA)
    print(f"\nA^T*A 的特征值: {eigenvalues_ATA}")
    print(f"A^T*A 的特征向量(V): \n{eigenvectors_ATA}")

    # 方法2：直接对A进行SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    print(f"\nSVD结果:")
    print(f"U 形状: {U.shape}")
    print(f"奇异值 S: {S}")
    print(f"Vt 形状: {Vt.shape}")
    print(f"Vt:\n{Vt}")

    # 验证关系
    print(f"\n数学关系验证:")
    print(f"A^T*A 的特征值 = 奇异值的平方")
    print(f"理论: λ_i = σ_i²")

    for i, (eigval, sigma) in enumerate(zip(eigenvalues_ATA, S)):
        print(f"  λ_{i + 1} = {eigval:.6f}, σ_{i + 1}² = {sigma ** 2:.6f}, 相等: {np.isclose(eigval, sigma ** 2)}")

    '''
    数学原理
        在SVD和特征值分解的关系中，对于矩阵 A 的SVD分解 A = U @ S @ Vt，其中的 Vt 矩阵的行向量应该等于 A.T @ A 的特征向量
        由于特征向量可能相差一个符号（方向相反但长度相同），所以使用 np.abs() 取绝对值
        如果两个单位向量相同（或仅相差符号），它们的点积绝对值应该等于1.0
    '''
    print(f"\nA^T*A 的特征向量 = V 矩阵的列")
    print("验证 V 和 eigenvectors_ATA 是否相等（可能差一个符号）:")
    for i in range(Vt.shape[0]):#遍历每个特征向量
        dot_product = np.abs(np.dot(Vt[i], eigenvectors_ATA[:, i]))#计算两个对应特征向量的点积绝对值
        print(f"  向量 {i + 1}: 点积绝对值 = {dot_product:.6f}, 是否单位向量: {np.isclose(dot_product, 1.0)}")
        #使用 np.isclose(dot_product, 1.0) 来判断点积是否接近1.0，从而确认两个向量是否相同（考虑数值计算精度误差）

    # 重建验证
    Sigma_matrix = np.diag(S)#将一维的奇异值数组 S 转换为对角矩阵形式
    '''
    S 是一个包含奇异值的一维数组 [σ₁, σ₂, σ₃]
    np.diag(S) 创建一个对角矩阵，对角线元素为奇异值，其余元素为0
    '''
    A_reconstructed=U@Sigma_matrix@Vt#根据SVD定义重新构建原始矩阵
    '''
    这是SVD分解的核心公式：A = U × Σ × Vᵀ
    @ 表示矩阵乘法运算
    如果SVD分解正确，A_reconstructed 应该等于原始矩阵 A
    '''

    print(f"\n重建验证:")
    print(f"原始 A 与重建 A 的最大差异: {np.max(np.abs(A - A_reconstructed)):.6e}")

    plt.subplot(1, 3, 1)
    plt.imshow(A, cmap='RdBu_r', aspect='auto')
    plt.colorbar()
    plt.title('原始矩阵 A')

    plt.subplot(1, 3, 2)
    plt.imshow(ATA, cmap='RdBu_r', aspect='auto')
    plt.colorbar()
    plt.title('A^T A')

    plt.subplot(1, 3, 3)
    plt.plot(range(1, len(S) + 1), S, 'ro-', label='奇异值')
    plt.plot(range(1, len(eigenvalues_ATA) + 1), np.sqrt(eigenvalues_ATA), 'bx--',
             label='sqrt(特征值)')
    plt.xlabel('索引')
    plt.ylabel('值')
    plt.title('奇异值 vs sqrt(特征值)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 在推荐系统中的应用解释
    print(f"\n在推荐系统中的应用理解:")
    print(f"  U: 用户潜在特征矩阵")
    print(f"  Σ: 重要性权重（奇异值）")
    print(f"  V^T: 物品潜在特征矩阵")
    print(f"  A ≈ U Σ V^T: 评分矩阵分解为用户和物品的潜在特征")

eigen_svd_relationship()