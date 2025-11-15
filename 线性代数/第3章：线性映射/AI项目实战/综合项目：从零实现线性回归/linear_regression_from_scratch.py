# 终极实践：从数学定义到AI实现
from torch.ao.nn.quantized.functional import leaky_relu

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
except ImportError:
    print("这些库未安装，请先安装：pip install torch torchvision matplotlib")

def linear_regression_from_scratch():
   """
   从线性映射的角度实现线性回归
   y = Wx + b 就是最简单的线性映射
   """
   #生成合成数据
   np.random.seed(42)#设置随机数种子为42，确保每次运行代码时生成的随机数序列相同，保证实验结果可重现
   n_samples=100#定义样本数量为100个
   '''
    生成100个服从[0,1]均匀分布的随机数
    乘以2后变成[0,2]区间内的随机数
    x作为特征变量(feature)，形状为(100,1)
   '''
   X=2*np.random.rand(n_samples,1)
   '''
    构造目标变量(target)与特征变量的线性关系: y = 4 + 3x + 噪声
    4是偏置项(bias/intercept)
    3是权重系数(weight/slope)
    np.random.randn()添加高斯白噪声，使数据更接近真实情况
   '''
   y=4+3*X+np.random.randn(n_samples,1)#y=4+3x+噪声

   #手动实现线性回归（寻找最优的线性映射）
   print("=== 从零实现线性回归 ===")
   print("寻找最优线性映射: y = Wx + b")

   #初始化参数（我们要求的线性映射）
   W=np.random.randn(1,1)#权重
   b=np.zeros(1)  #偏置
   learning_rate=0.1 #学习率
   n_iterations=1000 #迭代次数

   #梯度下降
   for iteration in range(n_iterations):
       #前向传播：应用当前的线性映射
       y_pred=X@W.T+b #计算预测值

       #计算损失（MSE）
       loss=np.mean((y_pred-y)**2)

       #反向传播：计算梯度
       grad_W=(2/n_samples)*X.T@(y_pred-y)
       grad_b=(2/n_samples)*np.sum(y_pred-y)

       #更新参数（优化线性映射）
       W-=learning_rate*grad_W.T
       b-=learning_rate*grad_b

       if iteration%100==0:
           print(f"迭代{iteration}:W={W[0,0]:.3f}，b={b[0]:.3f},损失：{loss:.4f}")
   print(f"\n最终结果:y={W[0,0]:.3f}x+{b[0]:.3f}")
   print(f"真实关系:y=3.000x+4.000")

   #可视化结果
   plt.figure(figsize=(10,6))
   plt.scatter(X,y,alpha=0.7,label='真实数据')

   X_line=np.array([[0],[2]])
   y_line=X_line@W.T+b
   plt.plot(X_line,y_line,'r-',linewidth=3,label=f'学习到的线性映射：y={W[0,0]:.3f}x+{b[0]:.3f}')
   plt.xlabel('x')
   plt.ylabel('y')
   plt.title('线性回归：寻找最优线性映射')
   plt.legend()
   plt.grid(True,alpha=0.3)
   plt.show()
linear_regression_from_scratch()












