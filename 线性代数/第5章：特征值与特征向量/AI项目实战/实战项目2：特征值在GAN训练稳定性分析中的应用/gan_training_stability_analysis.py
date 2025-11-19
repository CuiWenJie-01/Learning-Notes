# 实战项目2：使用特征值分析GAN(生成对抗网络)的训练稳定性
import numpy as np
import matplotlib.pyplot as plt

def gan_training_stability_analysis():
    """
    使用特征值分析生成对抗网络（GAN）的训练稳定性
    """

    # 模拟GAN训练过程中的权重矩阵变化
    n_iterations=500#这表示GAN训练的迭代次数为500次。在GAN的训练过程中，生成器和判别器会进行多次交替训练，这个参数定义了整个训练过程的总迭代轮数。
    layer_size=50#这表示神经网络层的大小为50个神经元。在GAN的网络结构中，这通常指的是隐藏层的神经元数量，用于控制模型的复杂度和表达能力。

    # 存储训练过程中的特征值
    generator_eigenvalues=[] # 存储生成器在每次迭代中的特征值
    discriminator_eigenvalues=[] # 存储判别器在每次迭代中的特征值

    # 模拟训练过程
    '''
    整数除法说明
        n_iterations//3: 表示将 n_iterations（500）除以3后取整数部分，即 500÷3 = 166.67，取整后为 166
        4*n_iterations//5: 表示将 4*n_iterations（2000）除以5后取整数部分，即 2000÷5 = 400
    代码中的具体应用
        iterator < n_iterations//3: 判断当前迭代次数是否在前166次迭代内（训练初期）
        iterator < 2*n_iterations//3: 判断当前迭代次数是否在前333次迭代内（训练中期）
        iterator > 4*n_iterations//5: 判断当前迭代次数是否超过了400次（训练后期的最后阶段）
    '''
    for iterator in range(n_iterations):#循环模拟整个GAN训练过程，共进行500次迭代
        # 生成器权重矩阵（模拟训练过程中的变化）
        if iterator<n_iterations//3:
            # 初期：随机权重
            '''
            特点: 权重矩阵初始化为小幅度随机值
            意义: 模拟GAN训练开始时的随机初始化状态
            '''
            W_g=np.random.randn(layer_size,layer_size)*0.1
        elif iterator<2*n_iterations//3:
            # 中期：开始收敛
            '''
            特点: 以单位矩阵为基础加上小幅随机扰动
            意义: 模拟训练逐渐收敛的过程
            '''
            W_g=np.eye(layer_size)+np.random.randn(layer_size,layer_size)*0.05
        else:
            # 后期：接近收敛但可能出现模式崩溃
            '''
            特点: 扰动更小，权重趋于稳定
            特殊情况处理: 在最后1/5迭代中模拟模式崩溃
            '''
            W_g = np.eye(layer_size) + np.random.randn(layer_size, layer_size) * 0.02
            # 模拟模式崩溃：当迭代次数超过80%时,使矩阵接近奇异
            if iterator>4*n_iterations//5:
                U,S,V=np.linalg.svd(W_g)#对权重矩阵进行奇异值分解
                S[-10:]=0.001 #  将最小的10个奇异值设为极小值
                W_g=U@np.diag(S)@V.T#重新构建接近奇异的权重矩阵
            #这种设计是为了分析当GAN出现模式崩溃时，权重矩阵特征值的变化情况。

        # 判别器权重矩阵，以单位矩阵为中心，添加小幅随机扰动作为判别器权重
        #将随机矩阵的数值缩放至较小范围（标准差为0.03）
        '''
        设计意图：
            初始化策略: 采用接近单位矩阵的初始化方式，有助于训练稳定
            随机扰动: 添加小幅度随机噪声，避免权重完全对称
            固定模式: 与生成器不同，判别器在整个训练过程中都采用相同的权重更新模式
        这种设计简化了判别器权重变化的模拟，专注于分析生成器权重变化对训练稳定性的影响。
        '''
        W_d=np.eye(layer_size)+np.random.randn(layer_size,layer_size)*0.03

        # 计算特征值
        eigvals_g=np.linalg.eigvals(W_g)#计算生成器权重矩阵 W_g 的特征值
        eigvals_d=np.linalg.eigvals(W_d)#计算判别器权重矩阵 W_d 的特征值

        generator_eigenvalues.append(eigvals_g)#将当前迭代的生成器特征值添加到列表中
        discriminator_eigenvalues.append(eigvals_d)#将当前迭代的判别器特征值添加到列表中

    generator_eigenvalues=np.array(generator_eigenvalues)# 将生成器特征值列表转换为NumPy数组，便于后续分析
    discriminator_eigenvalues=np.array(discriminator_eigenvalues)#将判别器特征值列表转换为NumPy数组

    # 计算条件数（最大特征值/最小特征值）
    cond_g=np.max(np.abs(generator_eigenvalues),axis=1)/np.min(np.abs(generator_eigenvalues),axis=1)#计算生成器权重矩阵的cond_g
    cond_d=np.max(np.abs(discriminator_eigenvalues),axis=1)/np.min(np.abs(discriminator_eigenvalues),axis=1)#计算判别器权重矩阵的cond_d

    # 可视化训练过程
    plt.figure(figsize=(15, 10))

    # 生成器特征值变化
    plt.subplot(2, 2, 1)
    plt.plot(range(n_iterations), np.real(generator_eigenvalues[:, 0]),
             label='最大特征值', alpha=0.7)
    plt.plot(range(n_iterations), np.real(generator_eigenvalues[:, -1]),
             label='最小特征值', alpha=0.7)
    plt.xlabel('训练迭代')
    plt.ylabel('特征值')
    plt.title('生成器特征值变化')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 判别器特征值变化
    plt.subplot(2, 2, 2)
    plt.plot(range(n_iterations), np.real(discriminator_eigenvalues[:, 0]),
             label='最大特征值', alpha=0.7)
    plt.plot(range(n_iterations), np.real(discriminator_eigenvalues[:, -1]),
             label='最小特征值', alpha=0.7)
    plt.xlabel('训练迭代')
    plt.ylabel('特征值')
    plt.title('判别器特征值变化')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 条件数变化
    plt.subplot(2, 2, 3)
    plt.semilogy(range(n_iterations), cond_g, label='生成器条件数', alpha=0.8)
    plt.semilogy(range(n_iterations), cond_d, label='判别器条件数', alpha=0.8)
    plt.axhline(y=1000, color='red', linestyle='--', label='不稳定阈值')
    plt.xlabel('训练迭代')
    plt.ylabel('条件数（对数尺度）')
    plt.title('条件数变化 - GAN稳定性指标')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 特征值分布演化
    plt.subplot(2, 2, 4)
    iterations_to_show = [0, n_iterations // 3, 2 * n_iterations // 3, n_iterations - 1]
    colors = ['blue', 'green', 'orange', 'red']

    for i, iter_idx in enumerate(iterations_to_show):
        eigenvalues = np.real(generator_eigenvalues[iter_idx])
        plt.hist(eigenvalues, bins=30, alpha=0.5,
                 label=f'迭代 {iter_idx}', color=colors[i])

    plt.xlabel('特征值')
    plt.ylabel('频数')
    plt.title('生成器特征值分布演化')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 训练稳定性分析
    print(f"\nGAN训练稳定性分析:")
    final_cond_g = cond_g[-1]#获取 cond_g 数组的最后一个元素，即第500次迭代时生成器的条件数
    final_cond_d = cond_d[-1]#获取 cond_d 数组的最后一个元素，即第500次迭代时判别器的条件数
    #条件数越大表示矩阵越接近奇异状态，训练越不稳定；条件数接近1表示训练较为稳定。

    print(f"最终生成器条件数: {final_cond_g:.1f}")
    print(f"最终判别器条件数: {final_cond_d:.1f}")

    if final_cond_g > 1000 or final_cond_d > 1000:
        print("⚠️ 训练不稳定：可能出现模式崩溃或梯度问题")
    elif final_cond_g > 100 or final_cond_d > 100:
        print("⚠️ 训练相对稳定，但需要监控")
    else:
        print("✅ 训练稳定")

    # 特征值分析建议
    print(f"\n基于特征值的训练建议:")
    if np.max(np.abs(generator_eigenvalues[-1])) > 2:#如果超过，说明权重更新过大，建议降低学习率
        print("  - 考虑降低生成器的学习率")
    if np.min(np.abs(generator_eigenvalues[-1])) < 0.01:#如果过小，可能导致梯度消失，建议更换激活函数
        print("  - 生成器可能出现梯度消失，考虑使用不同的激活函数")
    if final_cond_g > final_cond_d * 10:#比较生成器和判别器条件数的差异，如果生成器条件数远大于判别器，说明两者训练不平衡，需要调整训练比例
        print("  - 生成器-判别器不平衡，考虑调整训练比例")
    #通过监控权重矩阵的特征值特性，为GAN训练提供实用的调参建议，帮助提升训练稳定性和模型性能。

gan_training_stability_analysis()