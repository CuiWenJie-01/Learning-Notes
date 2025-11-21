# 数学概念：内积度量两个向量的相似程度，与夹角余弦成正比
# AI对应：注意力机制、相似性搜索、推荐系统
import numpy as np
import matplotlib.pyplot as plt


def inner_product_similarity_demo():
    '''
    展示内积如何度量向量相似性
    '''

    print("=== 内积与向量相似性 ===")

    # 创建示例向量
    vectors={
        '相似向量1':np.array([3,2]),
        '相似向量2':np.array([2.8,2.1]),# 与向量1相似
        '正交向量':np.array([-2,3]), # 与向量1接近正交
        '相反向量':np.array([-3,-2]), # 与向量1相似，但方向相反
        '长向量':np.array([6,4]), # 与向量1相似，但长度不同
    }

    '''
    这行代码的作用是：
        设定参考向量：从预先定义好的向量字典 vectors 中取出名为 '相似向量1' 的向量作为参考向量
        用于后续比较：这个参考向量将作为基准，与其他向量计算内积，从而度量它们之间的相似性
        具体值：根据上面的代码定义，'相似向量1' 是一个二维向量 [3, 2]
    在数学上，内积（点积）可以用来衡量两个向量的相似程度：
        内积为正：两向量方向相近（夹角小于90度）
        内积为零：两向量正交（垂直，夹角90度）
        内积为负：两向量方向相反（夹角大于90度）
    '''
    reference_vector=vectors['相似向量1']

    print(f"参考向量: {reference_vector}")
    print("\n相似性分析:")
    '''
    使用 ljust() 方法对每个列标题进行左对齐并指定宽度
    创建一个格式化的表格标题行和分隔线，用于清晰地展示向量相似性分析的结果
    '''
    print("向量名称".ljust(15) + "向量值".ljust(20) + "内积".ljust(12) + "余弦相似度".ljust(15) + "夹角(度)")
    print("-" * 78)#输出一条长度为78个字符的横线

    for name,vec in vectors.items():
        inner_prod=np.dot(reference_vector,vec)#使用 np.dot 计算参考向量与当前向量的内积（点积）
        cos_sim=inner_prod/(np.linalg.norm(reference_vector)*np.linalg.norm(vec))#计算余弦相似度
        angle_deg=np.degrees(np.arccos(np.clip(cos_sim,-1,1)))#计算夹角（弧度转换为角度）
        print(f"{name.ljust(15)} {str(vec).ljust(20)} {inner_prod:10.2f} {cos_sim:12.2f} {angle_deg:10.1f}")


    # 可视化
    plt.figure(figsize=(12, 10))

    # 向量图
    plt.subplot(2, 2, 1)
    origin = np.array([[0, 0], [0, 0]])  # 原点

    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (name, vec) in enumerate(vectors.items()):
        plt.quiver(*origin[0], *vec, color=colors[i], scale=1, scale_units='xy',
                   angles='xy', label=name, alpha=0.7, width=0.01)

    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.title('向量空间中的向量')
    plt.legend()
    plt.axis('equal')

    # 内积值比较
    plt.subplot(2, 2, 2)
    names = list(vectors.keys())
    inner_prods = [np.dot(reference_vector, vec) for vec in vectors.values()]

    bars = plt.bar(names, inner_prods, color=colors, alpha=0.7)
    plt.ylabel('内积值')
    plt.title('各向量与参考向量的内积')
    plt.xticks(rotation=45)

    # 添加数值标签
    for bar, value in zip(bars, inner_prods):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{value:.1f}', ha='center', va='bottom')

    # 余弦相似度比较
    plt.subplot(2, 2, 3)
    cos_sims = [np.dot(reference_vector, vec) /
                (np.linalg.norm(reference_vector) * np.linalg.norm(vec))
                for vec in vectors.values()]

    bars = plt.bar(names, cos_sims, color=colors, alpha=0.7)
    plt.ylabel('余弦相似度')
    plt.title('各向量与参考向量的余弦相似度')
    plt.xticks(rotation=45)
    plt.ylim(-1.1, 1.1)

    # 添加数值标签
    for bar, value in zip(bars, cos_sims):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f'{value:.2f}', ha='center', va='bottom')

    # 夹角可视化
    plt.subplot(2, 2, 4)
    # 绘制单位圆上的点
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.3)

    for i, (name, vec) in enumerate(vectors.items()):
        vec_norm = vec / np.linalg.norm(vec)
        plt.quiver(0, 0, vec_norm[0], vec_norm[1], color=colors[i],
                   scale=1, scale_units='xy', angles='xy', alpha=0.7, width=0.01)
        plt.text(vec_norm[0] * 1.1, vec_norm[1] * 1.1, name, fontsize=9)

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.title('单位圆上的向量方向')
    plt.axis('equal')

    plt.tight_layout()
    plt.show()

    # AI应用启示
    print(f"\nAI应用启示:")
    print("1. 内积越大 → 向量越相似 → 在注意力机制中关注度越高")
    print("2. 余弦相似度消除了向量长度的影响，只关注方向")
    print("3. 正交向量内积为0 → 在注意力中完全不相关")

inner_product_similarity_demo()
