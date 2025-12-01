
import matplotlib.pyplot as plt
print(plt.__version__)

#头文件模版1
try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
except ImportError:
    print("这些库未安装，请先安装：pip install torch torchvision matplotlib")
#头文件模版2
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    from sklearn.datasets import load_iris, make_blobs


#图表数学字体
import matplotlib.pyplot as plt

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 替换为你的系统中存在的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

#或者直接使用
plt.rcParams['font.family'] = 'DejaVu Sans, Arial, sans-serif'#设置带有数学平方的字体，用于显示数学公式


