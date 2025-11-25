import matplotlib.pyplot as plt
import torch

# 实践4：奇异值分解与正规算子的深刻联系
def svd_normal_operator_relationship():
    """
    探索SVD与正规算子理论的深刻数学联系
    """
    print("=== 奇异值分解与正规算子的关系 ===")

    # 创建一个一般的矩阵（不一定是正规的）
    m, n = 40, 30
    A = torch.randn(m, n)

    print(f"矩阵A形状: {m}×{n}")

    # 方法1：直接对A进行SVD
    '''
    当 full_matrices=False 时：
    返回的 U 是一个 m × k 矩阵
    返回的 Vh 是一个 k × n 矩阵
    其中 k = min(m, n)
    这种形式称为「经济型」或「紧凑型」SVD（economy SVD）
    当 full_matrices=True 时（默认）：
    返回的 U 是一个 m × m 矩阵
    返回的 Vh 是一个 n × n 矩阵
    即返回完整的正交矩阵
    '''
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)

    print(f"\nA的SVD分解:")
    print(f"U形状: {U.shape} (左奇异向量)")
    print(f"S形状: {S.shape} (奇异值)")
    print(f"Vh形状: {Vh.shape} (右奇异向量)")

    # 方法2：通过正规矩阵A^T A和AA^T的特征分解得到SVD
    ATA = A.T @ A  # n×n矩阵
    AAT = A @ A.T  # m×m矩阵

    print(f"\n正规矩阵分析:")
    print(f"A^T A形状: {ATA.shape}")
    print(f"AA^T形状: {AAT.shape}")

    # A^T A和AA^T都是对称的，因此是正规矩阵
    print(f"A^T A是否对称: {torch.allclose(ATA, ATA.T)}")
    print(f"AA^T是否对称: {torch.allclose(AAT, AAT.T)}")

    # 对A^T A进行特征分解
    eigenvalues_ATA, V = torch.linalg.eigh(ATA)
    # 对AA^T进行特征分解
    eigenvalues_AAT, U_alt = torch.linalg.eigh(AAT)

    # 奇异值应该是特征值的平方根
    S_from_ATA = torch.sqrt(torch.abs(eigenvalues_ATA))
    S_from_AAT = torch.sqrt(torch.abs(eigenvalues_AAT))

    # 排序（特征分解可能不是按顺序的）
    sorted_indices_ATA = torch.argsort(eigenvalues_ATA, descending=True)#在使用 torch.argsort 函数时，descending=True 参数表示按照降序（从大到小）对元素进行排序。
    sorted_indices_AAT = torch.argsort(eigenvalues_AAT, descending=True)

    S_from_ATA = S_from_ATA[sorted_indices_ATA]
    S_from_AAT = S_from_AAT[sorted_indices_AAT]
    V_sorted = V[:, sorted_indices_ATA]
    U_alt_sorted = U_alt[:, sorted_indices_AAT]

    print(f"\nSVD与特征分解关系验证:")
    print(f"直接SVD的奇异值: {S[:5]}...")
    print(f"来自A^T A的奇异值: {S_from_ATA[:5]}...")
    print(f"来自AA^T的奇异值: {S_from_AAT[:5]}...")

    # 验证一致性
    svd_consistency_ATA = torch.allclose(S, S_from_ATA[:len(S)], atol=1e-5)
    #直接对矩阵 A 进行奇异值分解（SVD）得到的奇异值 S
    # 通过对 A^T A 进行特征分解得到的奇异值 S_from_ATA 的前 len(S) 个值
    svd_consistency_AAT = torch.allclose(S, S_from_AAT[:len(S)], atol=1e-5)
    # 直接对矩阵 A 进行奇异值分解（SVD）得到的奇异值 S
    # 通过对 AA^T 进行特征分解得到的奇异值 S_from_AAT 的前 len(S) 个值

    print(f"SVD与A^T A特征分解一致: {svd_consistency_ATA}")
    print(f"SVD与AA^T特征分解一致: {svd_consistency_AAT}")

    # 可视化比较
    plt.figure(figsize=(15, 10))

    # 原始矩阵
    plt.subplot(2, 3, 1)
    plt.imshow(A.numpy(), cmap='RdBu_r', aspect='auto')
    plt.colorbar()
    plt.title('原始矩阵 A')

    # 直接SVD重建
    plt.subplot(2, 3, 2)
    A_reconstructed = U @ torch.diag(S) @ Vh
    plt.imshow(A_reconstructed.numpy(), cmap='RdBu_r', aspect='auto')
    plt.colorbar()
    plt.title('SVD重建矩阵')
    reconstruction_error = torch.norm(A - A_reconstructed) / torch.norm(A)
    plt.text(0.5, -0.1, f'相对误差: {reconstruction_error:.2e}',
             transform=plt.gca().transAxes, ha='center')

    # 奇异值谱比较
    plt.subplot(2, 3, 3)
    plt.semilogy(S.numpy(), 'bo-', label='直接SVD奇异值', alpha=0.7)
    plt.semilogy(S_from_ATA.numpy()[:len(S)], 'rx--', label='来自A^T A', alpha=0.7)
    plt.semilogy(S_from_AAT.numpy()[:len(S)], 'g^--', label='来自AA^T', alpha=0.7)
    plt.xlabel('索引')
    plt.ylabel('奇异值（对数尺度）')
    plt.title('奇异值比较')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # A^T A的特征值谱
    plt.subplot(2, 3, 4)
    plt.semilogy(eigenvalues_ATA[sorted_indices_ATA].numpy(), 'ro-')
    plt.xlabel('索引')
    plt.ylabel('特征值（对数尺度）')
    plt.title('A^T A的特征值谱')
    plt.grid(True, alpha=0.3)

    # AA^T的特征值谱
    plt.subplot(2, 3, 5)
    plt.semilogy(eigenvalues_AAT[sorted_indices_AAT].numpy(), 'go-')
    plt.xlabel('索引')
    plt.ylabel('特征值（对数尺度）')
    plt.title('AA^T的特征值谱')
    plt.grid(True, alpha=0.3)

    # 正规算子的性质总结
    plt.subplot(2, 3, 6)
    properties = [
        '* A^T A是对称矩阵',
        '* AA^T是对称矩阵',
        '* 对称矩阵是正规矩阵',
        '* 正规矩阵可酉对角化',
        '* SVD可通过正规矩阵获得'
    ]

    for i, prop in enumerate(properties):
        plt.text(0.1, 0.9 - i * 0.15, f'• {prop}', transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('正规算子与SVD的关系')

    plt.tight_layout()
    plt.show()

    # 当A是正规矩阵时的特殊情况
    print(f"\n当A是正规矩阵时的特殊情况:")

    # 创建一个正规矩阵
    n_square = 20
    A_normal = torch.randn(n_square, n_square)
    # 使其正规：A^T A = A A^T
    A_normal = 0.5 * (A_normal + A_normal.T)  # 对称矩阵是正规的

    print(f"正规矩阵A的形状: {A_normal.shape}")
    print(f"A是否正规: {torch.allclose(A_normal.T @ A_normal, A_normal @ A_normal.T)}")

    # 对正规矩阵进行特征分解和SVD
    eigenvalues_normal, eigenvectors_normal = torch.linalg.eigh(A_normal)
    U_normal, S_normal, Vh_normal = torch.linalg.svd(A_normal)

    print(f"\n正规矩阵的特殊性质:")
    print(f"特征值: {eigenvalues_normal[:5]}...")
    print(f"奇异值: {S_normal[:5]}...")

    # 对于正规矩阵，奇异值等于特征值的模
    singular_values_from_eigen = torch.abs(eigenvalues_normal)
    sorted_singular_values = torch.sort(singular_values_from_eigen, descending=True).values#使用 torch.sort 函数对特征值进行排序，descending=True 表示按降序排序
    # 对排序后的特征值取绝对值，得到奇异值

    normal_matrix_property = torch.allclose(S_normal, sorted_singular_values, atol=1e-5)#使用 torch.allclose 函数比较两个张量的元素是否相等，atol=1e-5 表示允许的误差范围
    print(f"正规矩阵的奇异值=特征值模: {normal_matrix_property}")

    # 数学意义总结
    print(f"\n数学意义总结:")
    print("1. SVD可以看作是对两个正规矩阵A^T A和AA^T的同时对角化")
    print("2. 正规矩阵的SVD与其特征分解有直接关系")
    print("3. 这解释了为什么对称矩阵（自伴算子）的SVD特别简单")
    print("4. 正规算子理论为SVD提供了深刻的数学基础")


svd_normal_operator_relationship()