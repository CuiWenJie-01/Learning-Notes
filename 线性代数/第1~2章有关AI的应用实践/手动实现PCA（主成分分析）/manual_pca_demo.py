#PCA是基概念在AI中的经典应用——寻找数据的最优基（主成分）
#可视化结果（matplotlib.pyplot）
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("请先安装 matplotlib 库: pip install matplotlib")
import numpy as np

def manual_pca_demo():
    """
    手动实现PCA，展示基变换在降维中的应用
    """
    try:
        #加载数据（load_iris）
        from sklearn.datasets import load_iris

        #加载鸾尾花数据集
        iris=load_iris()
        x=iris.data #150个样本，4个特征
        y=iris.target #150个标签

        print("原始数据形状：",x.shape) #x.shape 是 NumPy 数组的一个属性，用于返回数组的维度信息，（150,4）
        print("前5个样本：\n",x[:5])

        #1.中心化数据（减去均值）
        x_centered=x-np.mean(x,axis=0)
        print(f"\n中心化后的数据形状：{x_centered.shape}")

        #2.计算协方差矩阵
        cov_matrix=np.cov(x_centered.T)
        print(f"\n协方差矩阵：{cov_matrix.shape}")

        #3.计算特征值和特征向量（这就是在找新基！）
        eigenvalues, eigenvectors=np.linalg.eigh(cov_matrix)
        print(f"\n特征值（方差）: {eigenvalues}")
        print(f"特征向量（主成分/新基）:\n{eigenvectors}")

        #4.选择前2个主成分（降维到2维）
        pc1=eigenvectors[:,0]#获取第一个主成分
        pc2=eigenvectors[:,1]#获取第二个主成分
        print(f"\n第一个主成分（PC1）:{pc1}")
        print(f"第二个主成分（PC2）:{pc2}")

        #5.投影到新基上
        projection_matrix=np.column_stack((pc1,pc2))
        x_pca=x_centered @ projection_matrix
        print(f"\n降维后的数据形状：{x_pca.shape}")

        #6.可视化结果
        plt.figure(figsize=(10,6))
        scatter=plt.scatter(x_pca[:,0],x_pca[:,1],c=y,cmap='viridis')
        plt.xlabel('第一主成分')
        plt.ylabel('第二主成分')
        plt.title('PCA降维结果')
        plt.colorbar(scatter)
        plt.grid(True,alpha=0.3)
        plt.show()

        print("\nPCA的本质：为数据找到了一个最优的新坐标系（基），")
        print("在这个坐标系下，前几个坐标轴包含了数据的大部分信息。")
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保已正确安装scikit-learn库")
        return
    except Exception as e:
        print(f"发生错误: {e}")
        return

manual_pca_demo()