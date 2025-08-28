import os
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

# 设置文件夹路径
folder_path = "e2e_dataset/test/"


# 遍历文件夹中的所有文件
for filename in tqdm(sorted(os.listdir(folder_path))):
    cnt = 0
    path_sring = os.path.join(folder_path, filename)
    if filename == "20250623T153621":
        continue
    for item in tqdm(os.listdir(path_sring)):
        file_path = os.path.join(path_sring, str(cnt))
        if not os.path.exists(file_path):
            continue
        plot_father_path = os.path.join(file_path, "test","test.png")
        data_pre_view_path = os.path.join(path_sring, "test")
        os.makedirs(data_pre_view_path, exist_ok=True)
        plt.ioff()
        img_pred = Image.open(plot_father_path)
        plt.imshow(img_pred)
        plt.savefig(os.path.join(data_pre_view_path, str(cnt)))          # 弹窗查看；若想保存：plt.savefig(os.path.join(root, sub, 'pred_view.png'))
        plt.close()
        cnt = cnt + 1
