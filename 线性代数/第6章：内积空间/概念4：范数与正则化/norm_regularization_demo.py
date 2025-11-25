# 数学概念：向量范数度量向量大小，用于正则化防止过拟合
# AI对应：L1/L2正则化、权重衰减、模型复杂度控制
import numpy as np # 导入NumPy库，用于数值计算
import matplotlib.pyplot as plt # 导入Matplotlib库，用于数据可视化
# plt.rcParams['font.family'] = 'DejaVu Sans, Arial, sans-serif'#设置带有数学平方的字体，用于显示数学公式


def norm_regularization_demo():
    """
    探索不同范数正则化对模型的影响
    """
    print("=== 范数正则化在深度学习中的应用 ===")

    # 创建简单的回归问题
    np.random.seed(42)#设置随机数种子，保证每次运行结果一致
    n_samples=1000#定义样本数量为1000个
    n_features=20#定义特征数量为20个
    n_redundant=15#定义冗余特征数量为15个

    # 生成数据：只有前5个特征真正有用，其他都是冗余特征
    X=np.random.randn(n_samples,n_features)#生成服从标准正态分布的随机数矩阵，大小为n_samples行n_features列
    true_weights=np.zeros(n_features)#定义真实系数向量，初始值为0
    true_weights[:5]=[1,-0.5,2,-1,0.8]#设置前5个特征系数为[1,-0.5,2,-1,0.8]
    y=X@true_weights+np.random.randn(n_samples)*0.1 #生成目标变量y，服从标准正态分布的随机数向量，大小为n_samples个，每个样本的真实值为X@true_weights+噪声

    print(f"数据形状: X {X.shape}, y {y.shape}")
    print(f"真实权重: {true_weights}")

    # 比较不同正则化方法
    from sklearn.linear_model import LinearRegression, Ridge, Lasso # 导入线性回归模型和正则化模型
    from sklearn.preprocessing import StandardScaler # 导入数据预处理模块
    from sklearn.model_selection import train_test_split ## 导入数据集划分模块

    # 标准化特征
    scaler=StandardScaler()#创建一个StandardScaler对象，用于标准化特征
    X_scaled=scaler.fit_transform(X)#对特征矩阵X进行标准化处理，返回标准化后的特征矩阵X_scaled

    # 分割数据
    X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42) # 分割数据集，80%用于训练，20%用于测试

    # 不同模型
    models = {
        '无正则化': LinearRegression(),# 创建一个线性回归模型，无正则化
        'L2正则化 (Ridge)': Ridge(alpha=1.0),# 创建一个Ridge模型，L2正则化，alpha为正则化强度
        'L1正则化 (Lasso)': Lasso(alpha=0.1)# 创建一个Lasso模型，L1正则化，alpha为正则化强度
    }

    results={}#创建一个字典，用于存储结果

    plt.figure(figsize=(15, 10))

    for i,(name,model) in enumerate(models.items()):
        # 训练模型
        model.fit(X_train,y_train)#使用训练数据X_train和目标变量y_train训练模型model

        # 预测
        y_pred=model.predict(X_test)#使用测试数据X_test对模型model进行预测，返回预测结果y_pred
        test_score=model.score(X_test,y_test)#计算模型在测试数据上的R^2分数，返回一个值，用于评估模型的拟合优度

        # 获取权重
        if hasattr(model,'coef_'):
            weights=model.coef_#获取模型model的系数向量，用于评估特征的重要性
        else:
            weights=model.coef_[0] if hasattr(model.coff_,'__len__') else model.coef_ #获取模型model的系数向量，用于评估特征的重要性

        results[name] = {
            'weights': weights,#模型的系数向量，用于评估特征的重要性
            'test_score': test_score,#评估模型在测试数据上的拟合优度
            'y_pred': y_pred#模型预测的标签向量
        }

        print(f"\n{name}:")
        print(f"  测试集 R²: {test_score:.4f}")
        print(f"  权重范数: L1={np.linalg.norm(weights, 1):.4f}, L2={np.linalg.norm(weights, 2):.4f}")

        # 绘制权重比较
        plt.subplot(2, 3, i + 1)
        x_pos = np.arange(len(weights))
        plt.bar(x_pos - 0.2, true_weights, width=0.4, label='真实权重', alpha=0.7)
        plt.bar(x_pos + 0.2, weights, width=0.4, label='学习权重', alpha=0.7)
        plt.xlabel('特征索引')
        plt.ylabel('权重值')
        plt.title(f'{name}\n测试R²: {test_score:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 标记真正有用的特征
        for j in range(5):
            plt.axvspan(j - 0.5, j + 0.5, alpha=0.1, color='green')

    # 权重分布比较
    plt.subplot(2, 3, 4)
    for name, result in results.items():
        plt.hist(result['weights'], bins=20, alpha=0.6, label=name)
    plt.xlabel('权重值')
    plt.ylabel('频数')
    plt.title('权重分布比较')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 预测结果比较
    plt.subplot(2, 3, 5)
    for name, result in results.items():
        plt.scatter(y_test, result['y_pred'], alpha=0.6, label=name)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', alpha=0.5)
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('预测结果比较')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 正则化路径分析
    plt.subplot(2, 3, 6)#创建一个子图，用于绘制正则化路径分析
    alphas = np.logspace(-3, 2, 50)#生成50个10的幂次方数，作为正则化强度的参数
    l2_norms = []#创建一个空列表，用于存储L2范数
    l1_norms = []#创建一个空列表，用于存储L1范数

    for alpha in alphas:#循环遍历正则化强度参数
        ridge = Ridge(alpha=alpha)#创建一个Ridge模型，正则化强度为alpha
        ridge.fit(X_train, y_train)#使用训练数据X_train和目标变量y_train训练模型ridge
        l2_norms.append(np.linalg.norm(ridge.coef_, 2))#计算模型ridge的L2范数，并添加到列表l2_norms中
        l1_norms.append(np.linalg.norm(ridge.coef_, 1))#计算模型ridge的L1范

    plt.semilogx(alphas, l2_norms, 'b-', label='L2范数', linewidth=2)
    plt.semilogx(alphas, l1_norms, 'r-', label='L1范数', linewidth=2)
    plt.xlabel('正则化强度 α')
    plt.ylabel('权重范数')
    plt.title('正则化路径')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 正则化效果总结
    print(f"\n正则化效果总结:")
    print("无正则化: 可能过拟合，权重值较大，对噪声敏感")
    print("L2正则化: 权重收缩均匀，保持所有特征但减小影响")
    print("L1正则化: 产生稀疏解，自动进行特征选择")

norm_regularization_demo()