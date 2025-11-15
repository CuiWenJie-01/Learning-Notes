# 实践6：分析线性映射的可逆性与模型表达能力
try:
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
except ImportError:
    print("这些库未安装，请先安装：pip install torch torchvision matplotlib")

def analyze_invertibility():
    """
    分析线性层是否可逆，及其对模型能力的影响
    """
    #测试不同形状的线性层
    layers=[
        ('超定系统（m>n）',nn.Linear(5,10)),# 输出维度 > 输入维度
         ('方阵（m=n）',nn.Linear(8,8)),  # 输出维度 = 输入维度
        ('欠定系统（m<n）',nn.Linear(10,3))# 输出维度 < 输入维度
    ]

    print("=== 线性映射的可逆性分析 ===")

    for name,layer in layers:
        '''
        获取线性层(layer)的权重矩阵数据
        nn.Linear 层的核心是权重矩阵 W，它决定了线性变换的具体形式
        '''
        W=layer.weight.data

        '''
        使用 PyTorch 的线性代数函数计算权重矩阵 W 的秩
        矩阵的秩反映了矩阵中线性无关行/列的数量，是判断可逆性的重要指标
        '''
        #计算矩阵的秩
        rank=torch.linalg.matrix_rank(W)
        '''
        计算矩阵 W 可能达到的最大秩值
        对于任意矩阵，其秩不会超过行数和列数中的较小值
        W.shape[0] 表示矩阵的行数（输出维度）
        W.shape[1] 表示矩阵的列数（输入维度）
        '''
        max_rank=min(W.shape[0],W.shape[1])

        print(f"\n{name}")
        print(f" 权重矩阵的秩：{W.shape}")
        print(f" 矩阵的秩：{rank}/{max_rank}")
        print(f" 是否满秩：{rank==max_rank}")

        if W.shape[0]==W.shape[1]:#检查权重矩阵 W 是否为方阵（行数等于列数，只有方阵才能计算行列式和判断可逆性
            try:
                det=torch.det(W)#使用 PyTorch 计算方阵 W 的行列式（determinant），行列式是判断矩阵可逆性的重要指标
                print(f" 行列式：{det:.6f}")#输出行列式的值，保留6位小数
                print(f" 是否可逆：{abs(det)>1e-6}")#判断矩阵是否可逆：当行列式的绝对值大于一个很小的阈值（1e-6）时，认为矩阵可逆，如果行列式接近0，则矩阵不可逆（奇异矩阵
            except:
                print(" 行列式计算错误")

        #分析表达能力
        if rank<max_rank:
            print("  ⚠️  信息损失: 该层无法保持所有输入信息")
        elif W.shape[0] >= W.shape[1]:
            print("  ✅ 信息保持: 该层可以无损编码输入信息")
        else:
            # 欠定系统的情况：输出维度小于输入维度
            print("  ⚠️  信息损失: 欠定系统，输出维度小于输入维度，存在信息压缩")

analyze_invertibility()