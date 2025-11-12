# 实践5：手动实现Transformer中的线性投影，Transformer是线性映射应用的集大成者
try:
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
except ImportError:
    print("这些库未安装，请先安装：pip install torch torchvision matplotlib")

def transformer_linear_projection():
    """
    核心概念：实现了Transformer中Query(Q)、Key(K)、Value(V)三个重要的线性投影变换
    数学本质：这些投影都是典型的线性映射，将高维空间映射到低维空间
    """
    # 模拟Transformer参数
    d_model=512 # 模型维度(原始词嵌入的维度大小)
    d_k=64 # 注意力维度(注意力机制中使用的投影维度)
    seq_len=10 # 序列长度(输入序列的长度)
    batch_size=2 # 批次大小

    #输入：词嵌入
    input_embeddings=torch.randn(batch_size,seq_len,d_model)

    #定义线性投影层（Q,K,V的投影）
    '''
    使用 nn.Linear 创建三个无偏置的线性变换层，分别对应查询（Q）、键（K）和值（V）的投影。
    '''
    W_Q=nn.Linear(d_model,d_k,bias=False) #将512维映射到64维，用于生成查询向量（投影）
    W_K=nn.Linear(d_model,d_k,bias=False) #将512维映射到64维，用于生成键向量（投影）
    W_V=nn.Linear(d_model,d_k,bias=False) #将512维映射到64维，用于生成值向量（投影）


    print("=== Transformer中的线性映射 ===")
    print(f"输入维度：{d_model} -> 输出维度：{d_k}")
    '''
    参数规模
        每个线性投影层都有 512 × 64 = 32,768 个可训练参数，体现了大型语言模型参数密集的特点。
        这种设计使得Transformer能够从不同角度对输入信息进行多重表示学习。
    '''
    print(f"每个投影层的参数数量：{d_model*d_k}")

    #应用线性投影
    Q=W_Q(input_embeddings) #线性映射得到查询
    K=W_K(input_embeddings) #线性映射得到键
    V=W_V(input_embeddings) #线性映射得到值

    # 输出维度验证
    print(f"查询Q的形状：{Q.shape}") # (2, 10, 64)
    print(f"键K的形状：{K.shape}") # (2, 10, 64)
    print(f"值V的形状：{V.shape}") # (2, 10, 64)

    #计算注意力分数（再次使用线性代数!）
    attention_scores=torch.matmul(Q,K.transpose(-2,-1)/(d_k**0.5))
    attention_weights=torch.softmax(attention_scores,dim=-1)

    #应用注意力权重（线性组合）
    context=torch.matmul(attention_weights,V)

    print(f"注意力分数形状：{attention_scores.shape}")
    print(f"上下文向量形状：{context.shape}")

    #可视化注意力权重（线性组合）
    plt.figure(figsize=(10,6))#创建一个10*6的窗口

    #这些变量通常用于绘制Transformer处理流程的示意图，帮助理解数据在各个阶段的维度变化。
    operations=['输入嵌入','Q投影','K投影','V投影','注意力计算','输出']
    dimensions=[d_model,d_k,d_k,d_k,f'{seq_len}x{seq_len}',d_k]

    #创建柱状图
    '''
    x轴：operations 列表中的操作名称（输入嵌入、Q投影、K投影、V投影、注意力计算、输出）
    y轴：手动指定的重要性数值 [1,1,1,1,0.5,1]
    颜色：为不同操作分配不同颜色
    '''
    plt.bar(operations,[1,1,1,1,0.5,1],color=['blue','red','red','red','green',
                                              'purple'])
    plt.ylabel('操作重要性')# 设置y轴标签为"操作重要性"
    plt.title('Transformer中的线性映射流程')

    #添加维度标注
    for i,(op,dim) in enumerate(zip(operations,dimensions)):
        plt.text(i,0.5,f'dim={dim}',ha='center',va='bottom')

    plt.xticks(rotation=45)#将x轴刻度标签旋转45度,防止操作名称过长时重叠，提高可读性
    plt.tight_layout()#自动调整图表布局，确保所有元素（标题、标签、刻度等）都能完整显示，避免被截断
    plt.show()#显示最终生成的图表，在图形窗口中展示前面所有绘图操作的结果

transformer_linear_projection()






