# 数学概念：正规算子可以通过谱分解，矩阵函数可以通过特征值计算
# AI对应：矩阵指数、图神经网络、动力系统建模
import matplotlib.pyplot as plt
import torch


# 实践3：谱定理在矩阵函数和图神经网络中的应用
def spectral_theorem_matrix_functions():# func 是一个可调用的对象，接收特征值作为输入，返回经过函数变换后的特征值
    """
    谱定理在矩阵函数计算和图神经网络中的应用
    """
    print("=== 谱定理与矩阵函数在AI中的应用 ===")

    # 1. 通过谱分解计算矩阵函数
    def matrix_function_via_spectral(A, func):
        """
        通过谱分解计算矩阵函数 f(A)
        """
        # 特征分解
        eigenvalues, eigenvectors = torch.linalg.eigh(A)  # A需要是正规矩阵

        # 应用函数到特征值
        f_eigenvalues = func(eigenvalues)

        # 重建矩阵：V @ diag(f(λ)) @ V^T
        '''
        torch.diag() 函数:
            当输入是一个向量时，它会创建一个对角矩阵，其中对角线上的元素是该向量的元素
            当输入是一个矩阵时，它会提取矩阵的对角线元素，返回一个向量
        数学含义:
            对角矩阵是一种特殊的方阵，只有主对角线上的元素是非零的
            在谱定理中，对角矩阵包含特征值，而特征向量组成正交矩阵
        '''
        f_A = eigenvectors @ torch.diag(f_eigenvalues) @ eigenvectors.T

        return f_A

    # 创建对称矩阵（正规矩阵）
    n = 50 #矩阵维度
    A = torch.randn(n, n)#创建一个随机的n*n矩阵
    A = 0.5 * (A + A.T)  # 使其对称

    print(f"矩阵A形状: {A.shape}")
    print(f"A是否对称: {torch.allclose(A, A.T)}")

    # 定义一些重要的矩阵函数
    matrix_functions = {
        '矩阵指数': lambda x: torch.exp(x),
        '矩阵平方根': lambda x: torch.sqrt(torch.abs(x)),  # 处理负特征值
        '矩阵对数': lambda x: torch.log(torch.abs(x) + 1e-8),  # 避免log(0)
        '矩阵Sigmoid': lambda x: torch.sigmoid(x)#Sigmoid 是一种常用的激活函数
    }

    # 计算各种矩阵函数
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for idx, (func_name, func) in enumerate(matrix_functions.items()):
        # 使用谱方法计算矩阵函数
        f_A_spectral = matrix_function_via_spectral(A, func)

        # 与直接方法比较（对于某些函数）
        if func_name == '矩阵指数':
            # 使用PyTorch的矩阵指数
            f_A_direct = torch.matrix_exp(A)
        else:
            f_A_direct = None

        # 可视化原矩阵和函数后的矩阵
        ax1 = axes[0, idx]
        im1 = ax1.imshow(A.numpy(), cmap='RdBu_r', aspect='auto')
        ax1.set_title(f'原矩阵A')
        plt.colorbar(im1, ax=ax1)

        ax2 = axes[1, idx]
        im2 = ax2.imshow(f_A_spectral.numpy(), cmap='RdBu_r', aspect='auto')
        ax2.set_title(f'{func_name}(A)\n(谱方法)')
        plt.colorbar(im2, ax=ax2)

        # 验证精度（对于有直接方法的函数）
        '''
        具体解释：
            torch.norm(f_A_spectral - f_A_direct): 计算两个矩阵差值的Frobenius范数（矩阵的欧几里得范数）
            torch.norm(f_A_direct): 计算直接计算方法结果的范数作为归一化因子
            整体公式: ||f_A_spectral - f_A_direct|| / ||f_A_direct||
        这是相对误差的计算公式，用来衡量通过谱分解方法(f_A_spectral)计算出的矩阵指数与PyTorch内置函数torch.matrix_exp直接计算的结果(f_A_direct)之间的差异程度。
        相对误差越小，说明谱分解方法的精度越高。在代码中，这个误差值会被打印出来，用于评估基于特征值分解的矩阵函数计算方法的准确性。
        '''
        if f_A_direct is not None:
            error = torch.norm(f_A_spectral - f_A_direct) / torch.norm(f_A_direct)
            print(f"{func_name}计算误差: {error.item():.6f}")

    plt.tight_layout()
    plt.show()

    # 2. 在图神经网络中的应用
    def graph_neural_network_spectral():
        """
        谱定理在图神经网络中的应用
        """
        print(f"\n谱定理在图神经网络中的应用:")

        # 创建图数据
        n_nodes = 30
        # 随机邻接矩阵
        A_graph = torch.rand(n_nodes, n_nodes)
        A_graph = (A_graph > 0.7).float()  # 二值化，大于0.7的元素设为1（表示存在边），小于等于0.7的元素设为0（表示不存在边）
        A_graph = A_graph - torch.diag(torch.diag(A_graph))  # 移除自环
        '''
        移除自环（self-loops）
        torch.diag(A_graph)提取矩阵对角线元素
        torch.diag(torch.diag(A_graph))将对角线元素重新构造成对角矩阵
        通过减法操作将对角线元素置为0，确保图中没有节点连接到自身
        '''
        #这样处理后得到的是一个没有自环的稀疏邻接矩阵，更适合用于图神经网络的相关计算。

        # 对称化（无向图）
        A_graph = 0.5 * (A_graph + A_graph.T)
        A_graph = (A_graph > 0).float()  # 再次二值化，将大于0的元素设为1（表示存在边），小于等于0的元素设为0（表示不存在边）

        # 图拉普拉斯矩阵
        '''
        计算图的度矩阵(Degree Matrix)
        torch.sum(A_graph, dim=1)计算每个节点的度数（即每个节点相连的边数）
        torch.diag()将度数向量转换为对角矩阵，非对角线元素为0
        '''
        D = torch.diag(torch.sum(A_graph, dim=1))
        '''
        构建无向图的拉普拉斯矩阵
        其中 D 是度矩阵，A_graph 是邻接矩阵
        拉普拉斯矩阵的性质：
            对角线元素：节点的度数
            非对角线元素：如果节点间有边则为-1，否则为0
        '''
        L = D - A_graph
        #图拉普拉斯矩阵在图神经网络中非常重要，它的特征值和特征向量提供了图的谱信息，可用于图信号处理和图卷积操作。

        print(f"图节点数: {n_nodes}")
        print(f"边数量: {torch.sum(A_graph).int().item()}")
        print(f"拉普拉斯矩阵形状: {L.shape}")

        # 拉普拉斯矩阵的特征分解
        eigenvalues, eigenvectors = torch.linalg.eigh(L)

        # 图傅里叶变换
        '''
        图傅里叶变换（Graph Fourier Transform, GFT）是将图信号从节点域转换到谱域的一种方法。
        它利用图拉普拉斯矩阵的特征向量将图信号表示为其在不同频率下的组合。
        图傅里叶变换的性质：
            线性性质：GFT(u+v) = GFT(u) + GFT(v)
            平移性质：GFT(u) = exp(-i2πλ) * GFT(u)，其中 λ 是拉普拉斯矩阵的特征值
        '''
        def graph_fourier_transform(signal, eigenvectors):
            """图傅里叶变换"""
            return eigenvectors.T @ signal

        '''
        逆图傅里叶变换（Inverse Graph Fourier Transform, iGFT）是将图信号从谱域转换回节点域的方法。
        它利用图拉普拉斯矩阵的特征向量将谱域信号表示为其在不同节点下的组合。
        逆图傅里叶变换的性质：
            线性性质：iGFT(GFT(u+v)) = iGFT(GFT(u)) + iGFT(GFT(v))
            平移性质：iGFT(GFT(u)) = u，其中 u 是节点域信号
        '''
        def inverse_graph_fourier_transform(spectral_signal, eigenvectors):
            """逆图傅里叶变换"""
            return eigenvectors @ spectral_signal

        # 创建图信号
        node_features = torch.randn(n_nodes)

        # 傅里叶变换
        '''
        图傅里叶变换将节点域信号(node_features)转换为谱域信号(spectral_features)。
        这使得我们可以在谱域对信号进行滤波操作，而不是直接在节点域操作。
        '''
        spectral_features = graph_fourier_transform(node_features, eigenvectors)

        # 在谱域进行滤波
        def spectral_filtering(spectral_signal, filter_func, eigenvalues):
            """在谱域进行滤波"""
            filter_weights = filter_func(eigenvalues)
            return filter_weights * spectral_signal
        '''
        这段代码实现了**谱域滤波**功能，具体解释如下：

            - `spectral_filtering` 函数在图信号的谱域（频率域）中进行滤波操作
            - `filter_func` 是滤波器函数，根据特征值（频率）计算滤波权重
            - `eigenvalues` 是图拉普拉斯矩阵的特征值，对应图的不同频率成分
            - `spectral_signal` 是已经变换到谱域的信号
            
            工作原理：
            1. 通过 `filter_func(eigenvalues)` 计算每个频率成分的滤波权重
            2. 将这些权重与谱域信号逐元素相乘 `filter_weights * spectral_signal`
            3. 实现对不同频率成分的增强或抑制
            
            这种方式类似于传统信号处理中的频域滤波，但针对的是图结构数据。不同的 `filter_func` 可以实现低通、高通或带通滤波效果。
        '''

        # 定义几种谱滤波器
        filters = {
            '低通滤波器': lambda x: torch.exp(-5 * x),
            '高通滤波器': lambda x: 1 - torch.exp(-5 * x),
            '带通滤波器': lambda x: torch.exp(-5 * (x - 0.5) ** 2)
        }

        # 应用不同的滤波器
        plt.figure(figsize=(15, 10))

        for i, (filter_name, filter_func) in enumerate(filters.items()):
            # 应用滤波器
            filtered_spectral = spectral_filtering(spectral_features, filter_func, eigenvalues)
            filtered_signal = inverse_graph_fourier_transform(filtered_spectral, eigenvectors)

            # 可视化
            plt.subplot(2, 3, i + 1)

            # 原信号和滤波后信号
            plt.plot(node_features.numpy(), 'bo-', label='原信号', alpha=0.7)
            plt.plot(filtered_signal.numpy(), 'ro-', label='滤波后信号', alpha=0.7)
            plt.xlabel('节点索引')
            plt.ylabel('信号值')
            plt.title(f'{filter_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)

        # 特征向量可视化
        plt.subplot(2, 3, 4)
        # 显示前几个特征向量
        n_eigenvectors_to_show = min(6, n_nodes)
        for i in range(n_eigenvectors_to_show):
            plt.plot(eigenvectors[:, i].numpy(), label=f'特征向量 {i + 1}')
        plt.xlabel('节点索引')
        plt.ylabel('特征向量值')
        plt.title('图拉普拉斯特征向量')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 特征值分布
        plt.subplot(2, 3, 5)
        plt.plot(eigenvalues.numpy(), 'go-')
        plt.xlabel('特征值索引')
        plt.ylabel('特征值')
        plt.title('图拉普拉斯特征值谱')
        plt.grid(True, alpha=0.3)

        # 图结构可视化
        plt.subplot(2, 3, 6)
        # 简单的图可视化
        import networkx as nx
        G = nx.from_numpy_array(A_graph.numpy())
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=300, font_size=8)
        plt.title('图结构')

        plt.tight_layout()
        plt.show()

        print(f"\n图神经网络中的谱方法意义:")
        print("1. 特征向量定义了图的傅里叶基")
        print("2. 特征值对应图的频率")
        print("3. 谱滤波可以在频域处理图信号")
        print("4. 为图卷积神经网络提供理论基础")

    graph_neural_network_spectral()

    # 3. 矩阵指数在动力系统中的应用
    def matrix_exponential_dynamical_systems():
        """
        矩阵指数在连续时间动力系统建模中的应用
        """
        print(f"\n矩阵指数在动力系统建模中的应用:")

        # 创建动力系统矩阵
        n_states = 4
        # 稳定的动力系统矩阵（特征值实部为负）
        A_dynamics = torch.randn(n_states, n_states)# 随机生成动力系统矩阵
        # 使系统稳定：确保特征值实部为负
        eigenvalues = torch.linalg.eigvals(A_dynamics)# 求矩阵的特征值
        stability_shift = -torch.max(eigenvalues.real) - 0.5# 计算稳定偏移
        A_dynamics = A_dynamics + stability_shift * torch.eye(n_states)

        print(f"动力系统矩阵A的特征值实部: {torch.linalg.eigvals(A_dynamics).real}")

        # 使用矩阵指数模拟连续时间系统
        '''
        A: 系统矩阵，描述系统的动力学特性
        x0: 初始状态向量
        time_points: 时间点序列
        核心计算：
            对于每个时间点 t，计算矩阵指数 exp(At)（通过 torch.matrix_exp(A * t) 实现）
            利用线性系统解的公式 x(t) = exp(At) * x0 计算该时刻的状态
            将所有时间点的状态收集到 trajectories 列表中
        输出结果：
            返回堆叠后的状态轨迹张量，形状为 (时间点数, 状态维度)
        这种方法基于线性系统理论，其中连续时间线性时不变系统的解可以通过矩阵指数来表达。矩阵指数 exp(At) 描述了系统从初始时间到任意时间 t 的状态转移过程。
        '''
        def continuous_dynamics(A, x0, time_points):
            """模拟连续时间线性系统 dx/dt = Ax"""
            trajectories = []
            for t in time_points:
                # x(t) = exp(At) * x0
                exp_At = torch.matrix_exp(A * t)
                x_t = exp_At @ x0
                trajectories.append(x_t)
            return torch.stack(trajectories)

        # 模拟多个初始条件
        n_trajectories = 5
        time_points = torch.linspace(0, 10, 100)

        plt.figure(figsize=(15, 5))

        for traj_idx in range(n_trajectories):
            # 随机初始状态
            x0 = torch.randn(n_states)
            x0 = x0 / torch.norm(x0)  # 归一化

            # 模拟轨迹
            trajectory = continuous_dynamics(A_dynamics, x0, time_points)

            # 可视化
            plt.subplot(1, 3, 1)
            plt.plot(time_points.numpy(), trajectory[:, 0].numpy(),
                     label=f'轨迹{traj_idx + 1}' if traj_idx < 3 else "")

            plt.subplot(1, 3, 2)
            if traj_idx < 3:  # 只显示前3个轨迹的状态空间
                plt.plot(trajectory[:, 0].numpy(), trajectory[:, 1].numpy(),
                         'o-', alpha=0.7, label=f'轨迹{traj_idx + 1}')

        plt.subplot(1, 3, 1)
        plt.xlabel('时间')
        plt.ylabel('状态1')
        plt.title('状态1的时间演化')
        if n_trajectories <= 3:
            plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.xlabel('状态1')
        plt.ylabel('状态2')
        plt.title('状态空间轨迹')
        if n_trajectories <= 3:
            plt.legend()
        plt.grid(True, alpha=0.3)

        # 矩阵指数的时间演化
        plt.subplot(1, 3, 3)
        exp_norms = []
        for t in time_points:
            exp_At = torch.matrix_exp(A_dynamics * t)
            exp_norms.append(torch.norm(exp_At).item())

        plt.plot(time_points.numpy(), exp_norms, 'r-', linewidth=2)
        plt.xlabel('时间')
        plt.ylabel('||exp(At)||')
        plt.title('矩阵指数范数的时间演化')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print(f"动力系统分析:")
        print("矩阵指数exp(At)描述了系统从时间0到t的演化")
        print("系统稳定性由A的特征值决定")
        print("所有轨迹收敛到原点表明系统稳定")

    matrix_exponential_dynamical_systems()


spectral_theorem_matrix_functions()