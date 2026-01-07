import cv2
import numpy as np

from vehicle_config import *
import matplotlib.pyplot as plt
    
class Obs_Processor():
    def __init__(self) -> None:
        self.downsample_rate = 1
        self.n_channels = 3

    # def process_img(self, img):
    #     processed_img = self.change_bg_color(img)
    #     H, W = img.shape[:2]
    #     processed_img = cv2.resize(processed_img, (W//self.downsample_rate, H//self.downsample_rate))
    #     # plt.imshow(processed_img)  # 直接显示
    #     # plt.savefig('processed_img.png')  # 保存到当前目录
    #     processed_img = processed_img/255.0

    #     return processed_img
    
    def process_img(self, img):
        processed_img = self.change_bg_color(img)

        H, W = processed_img.shape[:2]
        if self.downsample_rate != 1:
            processed_img = cv2.resize(
                processed_img,
                (W // self.downsample_rate, H // self.downsample_rate),
                interpolation=cv2.INTER_AREA  # 下采样更合适
            )

        # 关键：保持 uint8，不做归一化
        if processed_img.dtype != np.uint8:
            processed_img = processed_img.astype(np.uint8, copy=False)

        return processed_img

    def change_bg_color(self, img):
        processed_img = img.copy()
        # bg_pos = img==BG_COLOR[:3]
        # bg_pos = (np.sum(bg_pos,axis=-1) == 3)
        # processed_img[bg_pos] = (0,0,0)
        return processed_img
    