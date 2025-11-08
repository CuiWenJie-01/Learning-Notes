#AI对应：神经网络的权重矩阵就是线性映射的矩阵
import torch
import torch.nn as nn
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("请先安装 matplotlib 库: pip install matplotlib")

# 实践3：可视化线性映射的矩阵表示
def visualize_linear_mapping():
    """
    可视化线性映射如何变换空间
    """
    #创建原始网格点
    x=np.linspace(-2,2,10)
    y=np.linspace(-2,2,10)
    X,Y=np.meshgrid(x,y)#基于x和y数组创建二维网格坐标矩阵（虽然这里没有被后续使用）
    points=np.vstack([x.ravel(),y.ravel()]).T#将x和y展平后垂直堆叠，再转置成N×2的点坐标矩阵

    #定义不同的线性变换矩阵
    transformations={
        '恒等映射':np.eye(2),#创建一个2×2的单位矩阵，表示不改变空间的恒等变换
        '缩放':np.array([[1.5,0],[0,0.5]]),#沿x轴放大1.5倍，沿y轴缩小为0.5倍的缩放变换
        '旋转':np.array([[0.707,-0.707],[0.707,0.707]]),#实现约45度逆时针旋转的变换矩阵
        '剪切':np.array([[1,0.5],[0,1]]),#沿x方向的剪切变换，y坐标保持不变
        '神经网络层(随机)':np.random.randn(2,2)*0.5#生成一个随机的2×2矩阵并乘以0.5，模拟神经网络中的权重矩阵效果
    }

    fig,axes=plt.subplots(2,3,figsize=(15,10))#创建一个2行3列的子图网格，整体图形大小为15×10英寸
    axes=axes.ravel()#将2维的 axes 数组展平为1维数组，便于后续通过索引访问各个子图

    '''
    遍历之前定义的所有线性变换，idx是索引，name是变换名称，matrix是对应的变换矩阵
    '''
    for idx,(name,matrix) in enumerate(transformations.items()):
        #应用线性变换
        #使用矩阵乘法将所有原始网格点 points 应用当前的线性变换 matrix，得到变换后的点集 transformed_points
        transformed_points=points @ matrix
        ax=axes[idx]#获取当前子图的坐标轴对象
        #绘制原始点
        ax.scatter(points[:,0],points[:,1],alpha=0.6,color='blue',label='原始空间')#绘制原始网格点，蓝色表示，透明度0.6
        #绘制变换后的点
        ax.scatter(transformed_points[:,0],transformed_points[:,1],alpha=0.6,color='red',label='变换后的空间')
        #绘制经过变换后的点，红色表示，透明度0.6

        ax.set_title(f'{name}\n矩阵:{matrix[0]}\n{matrix[1]}')
        ax.axhline(y=0,color='k',linestyle='-',alpha=0.3)
        ax.axvline(x=0,color='k',linestyle='-',alpha=0.3)
        ax.legend()#显示图例
        ax.set_aspect('equal')#设置坐标轴比例相等，保证视觉效果准确

    axes[-1].axis('off')#隐藏最后一个子图的坐标轴
    plt.tight_layout()#自动调整子图间距
    plt.show()

visualize_linear_mapping()

