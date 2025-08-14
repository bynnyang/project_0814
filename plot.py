import os
import json
import matplotlib.pyplot as plt

# 设置文件夹路径
folder_path = "train/measurements"

# 初始化存储坐标值的列表
x_coords = []
y_coords = []

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):  # 确保只处理 JSON 文件
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            data = json.load(file)
            # 提取 x 和 y 坐标
            x = data.get("x")
            y = data.get("y")
            if x is not None and y is not None:
                x_coords.append(x)
                y_coords.append(y)

# 绘制坐标
plt.figure(figsize=(10, 6))
plt.scatter(x_coords, y_coords, color='blue', label='Coordinates')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('X and Y Coordinates from JSON Files')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()