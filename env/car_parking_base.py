'''
This is a collision-free env which only includes one parking case with random start position.
'''


import sys
sys.path.append("../")
from typing import Optional, Union
import math
from typing import OrderedDict
import random

import numpy as np
import gym
from gym import spaces
from gym.error import DependencyNotInstalled
from shapely.geometry import Polygon
from shapely.affinity import affine_transform
import matplotlib.pyplot as plt

from env.vehicle import *
from env.lidar_simulator import LidarSimlator
from env.action_mask import ActionMask
from vehicle_config import *
from shapely.geometry import LineString

class CarParking(gym.Env):


    def __init__(
        self, 
        use_lidar_observation: bool =USE_LIDAR,
        use_action_mask: bool=USE_ACTION_MASK,
    ):
        super().__init__()

        self.use_lidar_observation = use_lidar_observation
        self.use_action_mask = use_action_mask
        self.matrix = None
        self.tgt_repr_size = 5 # relative_distance, cos(theta), sin(theta), cos(phi), sin(phi)

        self.vehicle = Vehicle(n_step=NUM_STEP, step_len=STEP_LENGTH)
        self.lidar = LidarSimlator(LIDAR_RANGE, LIDAR_NUM)
        self.action_filter = ActionMask()

        self.action_space = spaces.Box(
            np.array([VALID_STEER[0], VALID_SPEED[0]]).astype(np.float32),
            np.array([VALID_STEER[1], VALID_SPEED[1]]).astype(np.float32),
        ) # steer, speed

    def _get_lidar_observation(self, clusters:List[LineString]):
        # obs_list = [(cluster.x,cluster.y) for cluster in clusters]  #需要转换一下数据格式  
        lidar_view = self.lidar.get_observation(self.vehicle.state, clusters)
        return lidar_view
    
    def _get_targt_repr(self, parking_goal):
        # target position representation
        dest_pos = parking_goal
        ego_pos = (self.vehicle.state.loc.x, self.vehicle.state.loc.y, self.vehicle.state.heading)
        rel_distance = math.sqrt((dest_pos[0]-ego_pos[0])**2 + (dest_pos[1]-ego_pos[1])**2)
        rel_angle = math.atan2(dest_pos[1]-ego_pos[1], dest_pos[0]-ego_pos[0]) - ego_pos[2]
        rel_dest_heading = dest_pos[2] - ego_pos[2]
        tgt_repr = np.array([rel_distance, math.cos(rel_angle), math.sin(rel_angle),\
            math.cos(rel_dest_heading), math.cos(rel_dest_heading)])
        return tgt_repr 

    def calc_obervation_feature(self, initial_state, parking_goal, clusters:List[LineString]):
        assert self.vehicle is not None
        self.vehicle.reset(initial_state)
        observation = {'lidar':None, 'target':None, 'action_mask':None}
        if self.use_lidar_observation:
            observation['lidar'] = self._get_lidar_observation(clusters)
        if self.use_action_mask:
            observation['action_mask'] = self.action_filter.get_steps(observation['lidar'])
        observation['target'] = self._get_targt_repr(parking_goal) #获得[rel_distance, math.cos(rel_angle), math.sin(rel_angle), math.cos(rel_dest_heading), math.cos(rel_dest_heading)]      
        return observation


