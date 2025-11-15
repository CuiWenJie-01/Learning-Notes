
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
