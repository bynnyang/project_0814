import os
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 设置文件夹路径
folder_path = "e2e_dataset/train/"


# 遍历文件夹中的所有文件
for filename in tqdm(sorted(os.listdir(folder_path))):
    cnt = 0
    path_sring = os.path.join(folder_path, filename)
    for item in tqdm(os.listdir(path_sring)):
        file_path = os.path.join(path_sring, str(cnt))
        if not os.path.exists(file_path):
            continue
        plot_father_path = os.path.join(file_path, "pred")
        save_cluster_path = os.path.join(file_path, "pred_cluster")
        x_coords = []
        y_coords = []
        for json_name in sorted(os.listdir(plot_father_path)):
            if json_name.endswith(".json"):  # 确保只处理 JSON 文件
                plot_path = os.path.join(plot_father_path, json_name)
                with open(plot_path, 'r') as file:
                    data = json.load(file)
                 # 提取 x 和 y 坐标
                    x = data.get("x")
                    y = data.get("y")
                    if x is not None and y is not None:
                        x_coords.append(x)
                        y_coords.append(y)


        plot_cluster_path = os.path.join(file_path, "pre_cluster", "0000.json")
        with open(plot_cluster_path, 'r') as plot_cluster:
            segments = json.load(plot_cluster)
        
        plot_target_path = os.path.join(file_path, "pre_target", "0000.json")
        with open(plot_target_path, 'r') as plot_target:
            target_point = json.load(plot_target)
            
        if segments:      # 防止空列表
            plt.ioff()
            plt.figure(figsize=(6, 6))
            for seg in segments:                       # segments 就是刚才读到的 JSON 列表
                x0, y0 = seg["p0"]["x"], seg["p0"]["y"]
                x1, y1 = seg["p1"]["x"], seg["p1"]["y"]
                plt.plot([x0, x1], [y0, y1], color='red', linewidth=2, label='segments')
                mid_x = (x0 + x1) / 2
                mid_y = (y0 + y1) / 2
                plt.text(mid_x, mid_y,               # 中点坐标
                        str(int(seg["id"])),       # 转成 int 去掉 .0
                        color='black',
                        fontsize=7,
                        ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.2",
                        facecolor='white', alpha=0.6))
            plt.scatter(x_coords, y_coords, color='blue', s = 2, label='Coordinates')

            pose_x, pose_y, pose_theta = target_point['x'], target_point['y'], target_point['theta']

            plt.scatter(pose_x, pose_y, facecolors='none', edgecolors='red', s=20, zorder=3)

            dx = np.cos(pose_theta) * 1.0   # 长度可自行调
            dy = np.sin(pose_theta) * 1.0
            plt.arrow(pose_x, pose_y, dx, dy, head_width=0.3, head_length=0.3,fc='purple', ec='purple', zorder=3)

            plt.title(f'Scene {save_cluster_path}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.axis('equal')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(save_cluster_path)
            plt.close()        # 或 plt.savefig(...)

        if x_coords:
            plt.ioff()
            plt.figure(figsize=(6, 6))
            plt.scatter(x_coords, y_coords, color='blue', s = 5, label='Coordinates')
            plt.title(f'Scene {plot_father_path}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.axis('equal')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_father_path)
            plt.close()        # 或 plt.savefig(...)
        cnt = cnt + 1

# # 绘制坐标
# plt.figure(figsize=(10, 6))
# plt.scatter(x_coords, y_coords, color='blue', label='Coordinates')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.title('X and Y Coordinates from JSON Files')
# plt.legend()
# plt.grid(True)
# plt.axis('equal')
# plt.show()