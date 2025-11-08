#AI对应：模型的表达能力与信息损失
import numpy as np
import torch
import torch.nn as nn
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("请先安装 matplotlib 库: pip install matplotlib")

# 实践4：分析神经网络层的值域和零空间
def analyze_layer_capacity():
    """
    分析神经网络层的信息保持能力
    """
    #模拟不同容量的层
    layers_config=[
        ('宽层（信息保持）',nn.Linear(10,20)),#输出维度 > 输入维度
        ('等维层（信息保持）',nn.Linear(10,10)),#输出维度 = 输入维度
        ('窄层（信息丢失）',nn.Linear(10,5)),#输出维度 < 输入维度
        ('瓶颈层',nn.Linear(10,3))
    ]

    # 生成测试数据
    input_data=torch.randn(100,10)#100个样本，10维特征

    print("=== 神经网络层的信息保持能力分析 ===")

    for name,layer in layers_config:
        #初始化层
        '''
        代码解释
            isinstance(layer, nn.Linear): 检查当前层是否为线性层（全连接层）
            torch.nn.init.orthogonal_(layer.weight): 使用正交初始化方法初始化线性层的权重矩阵
        正交初始化的意义
            保持信息流动: 正交矩阵具有良好的数值性质，能够保持输入数据的范数，避免信息在传播过程中的过度放大或缩小
            避免梯度问题: 在深度网络中，正交初始化有助于缓解梯度消失或梯度爆炸问题
            均匀特征表示: 正交权重矩阵的各维度具有相同的尺度，确保各特征维度得到均匀处理
            实验一致性: 在分析不同层容量对信息保持能力的影响时，使用正交初始化可以排除权重初始化方式对结果的干扰，使比较更加公平
        这种初始化方式特别适合用于分析线性变换的本质特性，如值域和零空间的影响。
        '''
        if isinstance(layer,nn.Linear):
            torch.nn.init.orthogonal_(layer.weight)#使用正交初始化权重矩阵

        #层前向传播
        with torch.no_grad():
            output=layer(input_data)

        #分析信息保持
        input_cov=torch.cov(input_data.T)#计算输入数据的协方差矩阵
        output_cov=torch.cov(output.T)#计算输出数据的协方差矩阵

        intput_rank=torch.linalg.matrix_rank(input_cov)#计算输入协方差矩阵的秩
        output_rank=torch.linalg.matrix_rank(output_cov)#计算输出协方差矩阵的秩
        '''
        信息保持分析：通过比较输入和输出协方差矩阵的秩来衡量信息保持程度
        表达能力评估：秩的变化反映了线性层对信息的压缩或扩展能力
        理论验证：验证线性代数中值域和零空间理论在AI中的实际应用
        协方差矩阵的秩表示数据中有效信息的维度数量，通过比较输入输出的秩可以量化信息损失程度。
        '''

        print(f"\n{name}:")
        print(f"输入维度：10，输出维度：{layer.out_features}")
        print(f"输入协方差矩阵的秩：{intput_rank.item()}")
        print(f"输出协方差矩阵的秩：{output_rank.item()}")
        print(f"信息保持率：{output_rank.item()/intput_rank.item():.2%}")

        #如果输出维度小于输入维度，计算信息损失
        if layer.out_features < 10:
            #使用SVD分析信息损失
            u,s,v=torch.svd(layer.weight.data)
            print(f"权重矩阵的奇异值：{s.numpy()}")
            print(f"条件数：{s[0]/s[-1]:.2f}")

        '''
        这段代码用于分析当输出维度小于输入维度时的信息损失情况，具体解释如下：
        
        ## 代码功能
        
        - **条件判断**：`if layer.out_features < 10:` 检查是否为窄层（输出维度小于输入维度10）
        - **奇异值分解**：`torch.svd(layer.weight.data)` 对权重矩阵进行奇异值分解
        - **信息展示**：打印奇异值和条件数来分析信息损失
        
        ## 关键概念
        
        1. **奇异值分解(SVD)**：
           - 将权重矩阵分解为 `U`, `S`, [V](file://C:\Users\EDY\Desktop\code\Learning-Notes\线性代数\.venv\Lib\site-packages\pip\_internal\resolution\resolvelib\provider.py#L49-L49) 三个矩阵
           - 奇异值 `s` 反映了各个方向上的信息重要性
        
        2. **条件数**：
           - 最大奇异值与最小奇异值的比值
           - 反映矩阵的数值稳定性，条件数越接近1越好
        
        ## 在当前上下文中的作用
        
        当神经网络层是窄层时（如窄层和瓶颈层），通过SVD分析可以：
        - 了解权重矩阵在各方向上的信息保持能力
        - 评估信息压缩的均匀性
        - 判断是否存在严重的数值不稳定问题
        
        这是分析线性映射中值域和零空间特性的实用方法。
        '''

analyze_layer_capacity()