import numpy as np
import torch
import os
#########################
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if "LOCAL_RANK" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
# vehicle
WHEEL_BASE = 2.615  # wheelbase
FRONT_HANG = 0.68  # front hang length
REAR_HANG = 0.704  # rear hang length
LENGTH = WHEEL_BASE+FRONT_HANG+REAR_HANG
WIDTH = 1.781  # width
STEP_TIME_AND_LENGHT = 0.2
MASK_STEP_TIME = 0.6

from shapely.geometry import LinearRing
VehicleBox = LinearRing([
    (-REAR_HANG, -WIDTH/2), 
    (FRONT_HANG + WHEEL_BASE, -WIDTH/2), 
    (FRONT_HANG + WHEEL_BASE,  WIDTH/2),
    (-REAR_HANG,  WIDTH/2)])
VehicleBoxRS = LinearRing([
    (-(REAR_HANG+0.1), -(WIDTH+0.2)/2), 
    ((FRONT_HANG+0.1) + WHEEL_BASE, -(WIDTH+0.2)/2), 
    ((FRONT_HANG+0.1) + WHEEL_BASE,  (WIDTH+0.2)/2),
    (-(REAR_HANG+0.1),  (WIDTH+0.2)/2)])

VALID_SPEED = [-2.5, 2.5]
VALID_STEER = [-0.62, 0.62]

NUM_STEP = 1 #这个记得改为匹配模型的0.2s  也就是NUM_STEP = 4
STEP_LENGTH = 0.2

USE_LIDAR = True
USE_ACTION_MASK = True
LIDAR_RANGE = 10.0
LIDAR_NUM = 120
USE_IMG = True
UPDATE_IMG_ENCODE = False

# action mask
PRECISION = 10
step_speed = 1
discrete_actions = []
for i in np.arange(VALID_STEER[-1], -(VALID_STEER[-1] + VALID_STEER[-1]/PRECISION), -VALID_STEER[-1]/PRECISION):
    discrete_actions.append([i, step_speed])
for i in np.arange(VALID_STEER[-1], -(VALID_STEER[-1] + VALID_STEER[-1]/PRECISION), -VALID_STEER[-1]/PRECISION):
    discrete_actions.append([i, -step_speed])
N_DISCRETE_ACTION = len(discrete_actions)


GAMMA = 0.994
BATCH_SIZE = 2048
LR = 5e-5 #LR = 5e-6
TAU = 0.1
MAX_TRAIN_STEP = 1e6
ORTHOGONAL_INIT = True
LR_DECAY = True
UPDATE_IMG_ENCODE = False

C_CONV = [4, 8]
SIZE_FC = [256]

ATTENTION_CONFIG = {
                'depth': 1,
                'heads': 4,
                'dim_head': 32,
                'mlp_dim': 512,
                'hidden_dim': 128,
    }
USE_ATTENTION = True
ACTOR_CONFIGS = {
    'n_modal':2+int(USE_IMG)+int(USE_ACTION_MASK),
    'lidar_shape':LIDAR_NUM,
    'target_shape':5,
    'action_mask_shape':N_DISCRETE_ACTION if USE_ACTION_MASK else None,
    'img_shape':(3,512,512) if USE_IMG else None,
    'output_size':2,
    'embed_size':128,
    'hidden_size':256,
    'n_hidden_layers':3,
    'n_embed_layers':2,
    'img_conv_layers':C_CONV,
    'img_linear_layers':SIZE_FC,
    'k_img_conv':3,
    'orthogonal_init':True,
    'use_tanh_output':False,  # 自己有解码器  先不要输出
    'use_tanh_activate':True,
    'attention_configs': ATTENTION_CONFIG if USE_ATTENTION else None,
}

CRITIC_CONFIGS = {
    'n_modal':2+int(USE_IMG)+int(USE_ACTION_MASK),
    'lidar_shape':LIDAR_NUM,
    'target_shape':5,
    'action_mask_shape':N_DISCRETE_ACTION if USE_ACTION_MASK else None,
    'img_shape':(3,512,512) if USE_IMG else None,
    'output_size':1,
    'embed_size':128,
    'hidden_size':256,
    'n_hidden_layers':3,
    'n_embed_layers':2,
    'img_conv_layers':C_CONV,
    'img_linear_layers':SIZE_FC,
    'k_img_conv':3,
    'orthogonal_init':True,
    'use_tanh_output':False,
    'use_tanh_activate':True,
    'attention_configs': ATTENTION_CONFIG if USE_ATTENTION else None,
}

REWARD_RATIO = 0.1
from typing import OrderedDict
REWARD_WEIGHT = OrderedDict({'time_cost':1,\
            'rs_dist_reward':0,\
            'dist_reward':10,\
            'angle_reward':15,\
            'box_union_reward':20,
            'gear_shift_reward':4.0,
            'abs_shape':0.0,
            'near_bonus':5.0,
            'low_speed':1.0,
            'risk_reward':5.0,
            'high_speed':1.0,
            'big_steer':1.0,
            'u_turn':1.0})

OUTBOUND_REWARD = -30
OUTTIME_REWARD = -30
ARRIVED_REWARD = 100
COLLIDED_REWARD = -100


FPS = 100

########################
# senerio
MAP_LEVEL = 'Normal' # ['Normal', 'Complex', 'Extrem', 'dlp']
MIN_PARK_LOT_LEN_DICT = {'Extrem':LENGTH+0.6,
                            'Complex':LENGTH+0.9,
                            'Normal':LENGTH*1.25,}
MAX_PARK_LOT_LEN_DICT = {'Extrem':LENGTH+0.9,
                            'Complex':LENGTH*1.25,
                            'Normal':LENGTH*1.25+0.5}
MIN_PARK_LOT_WIDTH_DICT = {
    'Extrem':WIDTH+0.4,
    'Complex':WIDTH+0.4,
    'Normal':WIDTH+0.85,
}
MAX_PARK_LOT_WIDTH_DICT = {
    'Extrem':WIDTH+0.85,
    'Complex':WIDTH+0.85,
    'Normal':WIDTH+1.2,
}
PARA_PARK_WALL_DIST_DICT = {
    'Extrem':3.5,
    'Complex':4.0,
    'Normal':4.5,
}
BAY_PARK_WALL_DIST_DICT = {
    'Extrem':8.0,
    'Complex':8.0,
    'Normal':8.0,
}
N_OBSTACLE_DICT = {
    'Extrem':8,
    'Complex':5,
    'Normal':3,
}

# Normal level
MIN_DIST_TO_OBST = 0.1
MAX_DRIVE_DISTANCE = 15.0
DROUP_OUT_OBST = 0.1

#########################

OBS_W = 512
OBS_H = 512
RENDER_TRAJ = True
MAX_DIST_TO_DEST = 14.15
ENV_COLLIDE = True


BG_COLOR = (0, 0, 0)

DEST_COLOR = (0, 0, 255) 
OBSTACLE_COLOR = (255, 0, 0)

EGO_CENTER_COLOR = (255, 255, 255)  
TARGET_CENTER_COLOR = (255, 255, 255) 


BASE = np.array([0, 160, 0], dtype=np.uint8)  # 深绿
PEAK = np.array([0, 255, 0], dtype=np.uint8)  # 较亮绿
TRAJ_RENDER_LEN        = 12   # 最多回看多少帧历史（你之前就有类似参数）
TRAJ_COLORS = [
    tuple((BASE * (1 - t) + PEAK * t).astype(np.uint8))
    for t in np.linspace(0, 1, TRAJ_RENDER_LEN)
]
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

RS_MAX_DIST = 15
TOLERANT_TIME = 200

TRAJXRANGE = 10.0
TRAJYRANGE = 10.0
LIDARRANGE = 10.0

REGRESSIVE_STEP = 10