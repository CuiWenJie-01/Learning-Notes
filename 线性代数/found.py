import matplotlib
from matplotlib import pyplot as plt
print(matplotlib.matplotlib_fname())
'''
font.family: sans-serif
font.sans-serif: SimHei, DejaVu Sans, Bitstream Vera Sans, ... 
axes.unicode_minus: False
'''

plt.rcParams['font.family'] = 'DejaVu Sans, Arial, sans-serif'#设置带有数学平方的字体，用于显示数学公式