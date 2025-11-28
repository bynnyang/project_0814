import numpy as np
#########################

# vehicle
WHEEL_BASE = 2.615  # wheelbase
FRONT_HANG = 0.68  # front hang length
REAR_HANG = 0.704  # rear hang length
LENGTH = WHEEL_BASE+FRONT_HANG+REAR_HANG
WIDTH = 1.781  # width
STEP_TIME_AND_LENGHT = 0.2

from shapely.geometry import LinearRing
VehicleBox = LinearRing([
    (-REAR_HANG, -WIDTH/2), 
    (FRONT_HANG + WHEEL_BASE, -WIDTH/2), 
    (FRONT_HANG + WHEEL_BASE,  WIDTH/2),
    (-REAR_HANG,  WIDTH/2)])

VALID_SPEED = [-1.2, 1.2]
VALID_STEER = [-0.62, 0.62]

NUM_STEP = 10
STEP_LENGTH = 5e-2

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




C_CONV = [4, 8,]
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