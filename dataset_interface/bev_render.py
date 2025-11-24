

import sys
from typing import Optional, Union
import math
from typing import OrderedDict
import random
import cv2
import os

import numpy as np
import gym
from gym import spaces
from gym.error import DependencyNotInstalled
from shapely.geometry import Polygon
from shapely.affinity import affine_transform
import matplotlib.pyplot as plt
from utils.vec2d import Vec2d
from utils.box2d import Box2d
from utils.pose_utils import CustomizePose
from PIL import Image
try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
except ImportError:
    raise DependencyNotInstalled(
        "pygame is not installed, run `pip install pygame`"
    )

BG_COLOR = (0, 0, 0)
# START_COLOR = (0, 128, 255) 
DEST_COLOR = (0, 0, 255) 
OBSTACLE_COLOR = (255, 0, 0)
# TRAJ_COLOR_HIGH = (10, 10, 200)
# TRAJ_COLOR_LOW = (10, 10, 10)
EGO_CENTER_COLOR = (255, 255, 255)  
TARGET_CENTER_COLOR = (255, 255, 255) 
# TRAJ_COLORS = list(map(tuple,np.linspace(\
#     np.array(TRAJ_COLOR_LOW), np.array(TRAJ_COLOR_HIGH), TRAJ_RENDER_LEN, endpoint=True, dtype=np.uint8)))

BASE = np.array([0, 160, 0], dtype=np.uint8)  # 深绿
PEAK = np.array([0, 255, 0], dtype=np.uint8)  # 较亮绿
TRAJ_RENDER_LEN        = 12   # 最多回看多少帧历史（你之前就有类似参数）
TRAJ_COLORS = [
    tuple((BASE * (1 - t) + PEAK * t).astype(np.uint8))
    for t in np.linspace(0, 1, TRAJ_RENDER_LEN)
]
OBS_W = 512
OBS_H = 512
RENDER_TRAJ = True
FPS = 100
COLOR_POOL = [
    (0, 255, 0), # dodger blue
    (255, 127, 80), # coral
    (255, 215, 0) # gold
]

# 轨迹渲染相关
TRAJ_POINT_STEP        = 1    # 中心线点的步长（每 1 帧一个点）
TRAJ_BOX_STEP          = 5    # 小车框的步长（每 5 帧一个 box）
TRAJ_POINT_RADIUS      = 2    # 小点的像素半径
HISTORY_BOX_SCALE      = 0.4  # 历史车身 box 相对正式车身缩小比例（0.4 比较合适）

# 颜色：建议固定一个绿色通道
TRAJ_POINT_COLOR       = (0, 160, 0)   # 深绿：轨迹中心线点
TRAJ_BOX_COLOR         = (0, 200, 0)   # 亮绿：缩小小车 box

class Obs_Processor():
    def __init__(self) -> None:
        self.downsample_rate = 4
        self.n_channels = 3

    def process_img(self, img):
        processed_img = self.change_bg_color(img)
        processed_img = cv2.resize(processed_img, (img.shape[0]//self.downsample_rate, img.shape[1]//self.downsample_rate))
        # plt.imshow(processed_img)  # 直接显示
        # plt.savefig('processed_img.png')  # 保存到当前目录
        processed_img = processed_img/255.0

        return processed_img

    def change_bg_color(self, img):
        processed_img = img.copy()
        bg_pos = img==BG_COLOR[:3]
        bg_pos = (np.sum(bg_pos,axis=-1) == 3)
        processed_img[bg_pos] = (0,0,0)
        return processed_img

class BevRender(gym.Env):

    metadata = {
        "render_mode": [
            "human", 
            "rgb_array",
        ]
    }

    def __init__(
        self, 
        render_mode: str = None,
        fps: int = FPS,
        verbose: bool =True,
        meter_per_pixel = 0.04
    ):
        super().__init__()

        self.verbose = verbose
        self.render_mode = "hiden" if render_mode is None else render_mode
        self.fps = fps
        self.screen: Optional[pygame.Surface] = None
        self.matrix = None
        self.clock = None
        self.meter_per_pixel = meter_per_pixel
        self.is_open = True
        self.t = 0.0
        self.k = None
       
        # self.observation_space = {}
    
        # self.img_processor = Obs_Processor()
        # self.observation_space['img'] = spaces.Box(low=0, high=1, 
        #         shape=(OBS_W//self.img_processor.downsample_rate, OBS_H//self.img_processor.downsample_rate, 
        #         self.img_processor.n_channels), dtype=np.uint8
        #     )             # 数据类型不匹配，dtype是float类型
        # self.raw_img_shape = (OBS_W, OBS_H, 3)

    def _vcs_to_pixel(self, x_vcs, y_vcs):
        """
            现在的定义：
            x_vcs：前为正
            y_vcs：左为正

            像素坐标：
            u：向右为正
            v：向下为正

            映射关系（你想要的）：
            x 前 → u+
            y 左 → v-
        """
        pixel_per_meter = 1.0 / self.meter_per_pixel

        u = OBS_W / 2.0 + x_vcs * pixel_per_meter     # 前方 → 右
        v = OBS_H / 2.0 - y_vcs * pixel_per_meter     # 左侧 → 上

        return int(u), int(v)

    # def _process_img_observation(self, img):
    #     '''
    #     Process the img into channels of different information.

    #     Parameters
    #     ------
    #     img (np.ndarray): RGB image of shape (OBS_W, OBS_H, 3)

    #     Returns
    #     ------
    #     processed img (np.ndarray): shape (OBS_W//downsample_rate, OBS_H//downsample_rate, n_channels )
    #     '''
    #     processed_img = self.img_processor.process_img(img)
    #     return processed_img
    
    def compute_vehicle_pixel_corners(self, pose_vcs, yaw, scale = 1.0):
        """
        输入：
            pose_vcs: Vec2d(x, y) 车辆参考点（通常是 rear axle center）
            yaw:      float，车辆朝向
            vehicle_length: 车长（m）
            vehicle_width:  车宽（m）
            rear_overhang:  后悬（m）
        
        输出：
            pixel_corners: [(u,v), (u,v), (u,v), (u,v)]
                        已转换为 pygame 像素坐标
        """
        vehicle_width = 1.781 * scale
        vehicle_length = 3.99 * scale
        vehicle_rear_overhang = 0.704 * scale
        # 1. 计算车体几何中心（Box2d 使用的是几何中心）
        center_offset = vehicle_length / 2.0 - vehicle_rear_overhang
        vehicle_center = pose_vcs + Vec2d.create_unit_vec2d(yaw) * center_offset

        # 2. 创建 Box2d
        vehicle_box = Box2d(vehicle_center, yaw, vehicle_length, vehicle_width)

        # 3. 获取四个角点
        vehicle_corners = vehicle_box.GetAllCorners()  # 返回 4 个点

        # 4. 转为像素坐标
        pixel_corners = []
        for corner in vehicle_corners:
            cx = corner.x_
            cy = corner.y_
            u, v = self._vcs_to_pixel(cx, cy)
            pixel_corners.append((u, v))

        return pixel_corners
    
    def render_history_traj(self, history_traj_vcs, history_traj_len):
        if not RENDER_TRAJ:
            return
        if history_traj_len == 0:
            return

        # 1. 先截断，只看最近 TRAJ_RENDER_LEN 帧
        render_len = min(history_traj_len, TRAJ_RENDER_LEN)
        traj = history_traj_vcs[-render_len:]  # 最近 render_len 个 pose

        # 2. 每步画一个小点：轨迹中心线
        for pose in traj[::TRAJ_POINT_STEP]:
            x, y = pose.x, pose.y
            u, v = self._vcs_to_pixel(x, y)
            pygame.draw.circle(self.screen, TRAJ_POINT_COLOR, (u, v), TRAJ_POINT_RADIUS)

        # 3. 每 N 步画一个缩小版小车 box：表示姿态
        for idx, pose in enumerate(traj):
            # if idx % TRAJ_BOX_STEP != 0:
            #     continue

            history_pose = Vec2d(pose.x, pose.y)
            history_yaw = pose.yaw

            history_pixel_corners = self.compute_vehicle_pixel_corners(
                history_pose, history_yaw,
                scale=HISTORY_BOX_SCALE
            )

            pygame.draw.polygon(
                self.screen,
                TRAJ_COLORS[idx],
                history_pixel_corners,
                width=1   # 薄线就够了
            )

    def render(self, start_pose_vsc: CustomizePose, history_traj_vcs: list[CustomizePose], history_traj_len, target_point_vcs: list, cluster_info_vcs:dict, ego_index, filename):
       
        if self.render_mode == "human":
            display_flags = pygame.SHOWN
        else:
            display_flags = pygame.HIDDEN
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((OBS_W, OBS_H), flags = display_flags)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.screen.fill(BG_COLOR)

        # 3. 渲染自车位置
        vehicle_angle = 0.0
        vehicle_position = Vec2d(0.0, 0.0)
        vehicle_pixel_corners = self.compute_vehicle_pixel_corners(vehicle_position, vehicle_angle)
        pygame.draw.polygon(self.screen, COLOR_POOL[0], vehicle_pixel_corners)

        ego_u, ego_v = self._vcs_to_pixel(vehicle_position.x_, vehicle_position.y_)
        pygame.draw.circle(self.screen, EGO_CENTER_COLOR, (ego_u, ego_v), 3)

        # 4. 渲染起点位置
        start_pose = Vec2d(start_pose_vsc.x, start_pose_vsc.y)
        start_pose_yaw = start_pose_vsc.yaw

        start_pose_pixel_corners = self.compute_vehicle_pixel_corners(start_pose, start_pose_yaw)

        # pygame.draw.polygon(self.screen, START_COLOR, start_pose_pixel_corners, width=1)


        #5、渲染目标位置
        target_pose = Vec2d(target_point_vcs[0], target_point_vcs[1])
        target_pose_yaw = target_point_vcs[2]

        target_pose_pixel_corners = self.compute_vehicle_pixel_corners(target_pose, target_pose_yaw)

        pygame.draw.polygon(self.screen, DEST_COLOR, target_pose_pixel_corners, width=2)

        tx, ty = target_point_vcs[0], target_point_vcs[1]
        tu, tv = self._vcs_to_pixel(tx, ty)
        pygame.draw.circle(self.screen, TARGET_CENTER_COLOR, (tu, tv), 3)

    

        # 6. 画障碍物 / cluster（假设是多条折线）
        # cluster_info_vcs: List[List[(x, y)]]
        for cluster in cluster_info_vcs:
            p0 = cluster["p0"]
            p1 = cluster["p1"]

            # 自车坐标系 (x 前, y 左) → 像素坐标 (u, v)
            u0, v0 = self._vcs_to_pixel(p0.x, p0.y)
            u1, v1 = self._vcs_to_pixel(p1.x, p1.y)
            # 折线
            pygame.draw.line(self.screen, OBSTACLE_COLOR, (u0, v0), (u1, v1), width= 2)

        # # 4. 画历史轨迹（用小圆点表示）
        # if history_traj_vcs is not None and history_traj_len > 0:
        #     n = min(history_traj_len, len(history_traj_vcs))
        #     for i in range(n):
        #         pose = history_traj_vcs[i]
        #         if len(pose) == 3:
        #             x, y, _ = pose
        #         else:
        #             x, y = pose
        #         u, v = self._vcs_to_pixel(x, y)
        #         pygame.draw.circle(self.screen, TRAJ_COLOR, (u, v), 2)

        self.render_history_traj(history_traj_vcs, history_traj_len)

        # if RENDER_TRAJ and history_traj_len > 5:
        #     render_len = min(len(history_traj_vcs), TRAJ_RENDER_LEN)
        #     for i in range(render_len):
        #         history_pose = Vec2d(history_traj_vcs[-(render_len-i)].x, history_traj_vcs[-(render_len-i)].y)
        #         history_pose_yaw = history_traj_vcs[-(render_len-i)].yaw
        #         history_pose_pixel_corners = self.compute_vehicle_pixel_corners(history_pose, history_pose_yaw)
        #         pygame.draw.polygon(
        #             self.screen, TRAJ_COLORS[-(render_len-i)], history_pose_pixel_corners)


        pygame.display.update()
        self.clock.tick(self.fps)

        obs_str = pygame.image.tostring(self.screen, "RGB")
        img = np.frombuffer(obs_str, dtype=np.uint8)
        img = img.reshape((OBS_H, OBS_W, 3))

        measurements_path = os.path.join(filename, str(ego_index))
        os.makedirs(measurements_path, exist_ok=True)
        measurements_path_final = os.path.join(measurements_path, "cnn.png")

        # image = Image.fromarray(img)      # numpy → PIL
        # image.save(measurements_path_final)          # 保存为 png/jpg 都可以

        # observation = {"img": img}
        
        return img

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.is_open = False
            pygame.quit()

