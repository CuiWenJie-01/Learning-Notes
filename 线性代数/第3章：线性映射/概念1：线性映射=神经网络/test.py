# import sys
# print(sys.executable)  # 应该指向 .venv/Scripts/python.exe

# import sys
# print(sys.executable)
# print(sys.path)

import matplotlib.font_manager as fm

# 查看可用字体
fonts = [f.name for f in fm.findSystemFonts(fontext='ttf')]
print(fonts)