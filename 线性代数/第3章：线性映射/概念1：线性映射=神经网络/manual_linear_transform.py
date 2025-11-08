#AI对应：神经网络中的全连接层（不含激活函数）就是线性映射
#线性映射：将输入向量映射到输出向量空间，输出向量等于输入向量乘以权重矩阵，加上偏置向量。
import torch #导入 PyTorch 的主模块，提供了张量操作、自动微分等基础功能。
import torch.nn as nn #导入 PyTorch 的神经网络模块，并将其别名为 nn，方便后续使用神经网络层（如 Linear 层）。
import numpy as np


try:

    import matplotlib.pyplot as plt
    # # 使用中文显示
    # plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    # plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    print("请先安装 matplotlib 库: pip install matplotlib")





# 实践1：手动实现线性映射（全连接层,即神经网络Linear 层）
def manual_linear_transform(input_vec, weight_matrix, bias_vec):
    """
    手动实现线性映射: output = input @ W^T + b
    对应数学: T(x) = Ax + b
    :param input_vec: 输入向量
    :param weight_matrix: 权重矩阵
    :param bias_vec: 偏置向量
    """
    # 矩阵乘法: input_vec (1×n) @ weight_matrix.T (n×m) = (1×m)，相当与神经网络层（如 Linear 层）
    linecache_output=np.dot(input_vec, weight_matrix.T)
    #加上偏置向量
    output=linecache_output+bias_vec
    return output

# 实践2：对比PyTorch的线性映射
def compare_manual_vs_pytorch_linear():
    print("===线性映射:手动实现 vs PyTorch的Linear层===")

    #定义输入和参数
    '''
    batch_size: 批次大小，表示一次处理的数据样本数量（2个样本）  
    input_size: 输入特征维度，每个样本的输入向量长度（3维）
    output_size: 输出特征维度，每个样本的输出向量长度（2维）
    '''
    batch_size,input_size,output_size=2,3,2
    input_data=np.random.randn(batch_size,input_size)#生成随机输入数据，形状为 (2, 3) 的二维数组,每行代表一个样本，每列代表一个特征
    weight=np.random.randn(output_size,input_size)#生成随机权重矩阵，形状为 (2, 3),第 i 行对应输出第 i 维的权重系数
    bias=np.random.randn(output_size)#生成随机偏置向量，形状为 (2,),每个元素对应输出维度的偏置项

    print(f"输入数据：\n{input_data}")
    print(f"权重矩阵 w：\n{weight}")
    print(f"偏置向量 b：\n{bias}")

    #手动实现线性映射
    manual_output=[]#初始化一个空列表，用于存储手动实现的线性映射结果
    for i in range(batch_size):
        '''
        对每个输入样本调用 manual_linear_transform 函数，计算线性映射结果
        使用当前样本 input_data[i]、权重矩阵 weight 和偏置向量 bias
        '''
        output=manual_linear_transform(input_data[i], weight, bias)
        manual_output.append(output)
    manual_output=np.array(manual_output)
    #这段代码实现了对整个批次数据的手动线性映射计算，并将结果组织成数组格式。
    print(f"\n手动实现输入：\n{manual_output}")

    #使用PyTorch实现Linear 层
    pytorch_linear=nn.Linear(input_size,output_size,bias=True)#创建一个 nn.Linear 层，实现从 input_size 维到 output_size 维的线性映射
    #设置相同的权重和偏置
    pytorch_linear.weight.data=torch.tensor(weight,dtype=torch.float32)
    pytorch_linear.bias.data=torch.tensor(bias,dtype=torch.float32)
    pytorch_input=torch.tensor(input_data,dtype=torch.float32)
    pytorch_output=pytorch_linear(pytorch_input).detach().numpy()#将输入数据 input_data 转换为 PyTorch 张量，准备进行前向传播
    '''
    执行前向传播计算：pytorch_linear(pytorch_input)
    使用 .detach() 断开梯度计算图，避免内存占用 
    使用 .numpy() 将结果转换回 NumPy 数组，便于与手动实现的结果进行比较
    '''

    print(f"\nPyTorch的Linear层输入：\n{pytorch_output}")
    print(f"结果是否一致：{np.allclose(manual_output,pytorch_output,atol=1e-6)}")
    '''
    np.allclose()：
        检查两个数组是否在指定精度范围内相等
        atol=1e-6 表示绝对容差为 1e-6，允许微小的数值差异
    输出：
        如果结果一致返回 True
        如果存在显著差异返回 False
    目的：
        验证手动实现的数学原理与 PyTorch 的 nn.Linear 层计算结果的一致性
        确保两种实现方式在数值上是等价的
    '''

    # 添加可视化功能
    try:
        import matplotlib
        import seaborn as sns

        # 设置图形样式
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 绘制手动实现结果
        im1 = axes[0].imshow(manual_output, cmap='viridis', aspect='auto')
        axes[0].set_title('手动实现线性映射结果')
        axes[0].set_xlabel('输出维度')
        axes[0].set_ylabel('样本索引')
        plt.colorbar(im1, ax=axes[0])

        # 绘制PyTorch实现结果
        im2 = axes[1].imshow(pytorch_output, cmap='plasma', aspect='auto')
        axes[1].set_title('PyTorch Linear层结果')
        axes[1].set_xlabel('输出维度')
        axes[1].set_ylabel('样本索引')
        plt.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("matplotlib 或 seaborn 未安装，跳过可视化")

    # # 示例：添加可视化功能
    # if plt is not None:
    #     plt.figure(figsize=(10, 6))
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(manual_output, cmap='viridis')
    #     plt.title('手动实现输出')
    #
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(pytorch_output, cmap='viridis')
    #     plt.title('PyTorch输出')
    #
    #     plt.tight_layout()
    #     plt.show()

compare_manual_vs_pytorch_linear()


