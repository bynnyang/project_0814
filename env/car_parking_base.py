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
from heapdict import heapdict
try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
except ImportError:
    raise DependencyNotInstalled(
        "pygame is not installed, run `pip install pygame`"
    )

from env.vehicle import *
from env.map_base import *
from env.lidar_simulator import LidarSimlator
from env.action_mask import ActionMask
from vehicle_config import *
from shapely.geometry import LineString
from env.parking_map_normal import ParkingMapNormal
from env.parking_map_dlp import ParkingMapDLP
import env.reeds_shepp as rsCurve
from env.observation_processor import Obs_Processor
from utils.vec2d import Vec2d
from utils.box2d import Box2d
from utils.pose_utils import CustomizePose

class CarParking(gym.Env):

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
        use_lidar_observation: bool =USE_LIDAR,
        use_img_observation: bool=USE_IMG,
        use_action_mask: bool=USE_ACTION_MASK,
    ):
        super().__init__()

        self.verbose = verbose
        self.use_lidar_observation = use_lidar_observation
        self.use_img_observation = use_img_observation
        self.use_action_mask = use_action_mask
        self.render_mode = "rgb_array" if render_mode is None else render_mode
        self.fps = fps
        self.screen: Optional[pygame.Surface] = None
        self.clock = None
        self.is_open = True
        self.t = 0.0
        self.k = None
        self.level = MAP_LEVEL
        self.tgt_repr_size = 5 # relative_distance, cos(theta), sin(theta), cos(phi), sin(phi)
        self.dest_pose_vcs = None
        self.clusters_frame_info_vcs = None
        self.obervattion_clusters_info_vcs = None
        self.history_traj_vcs =None
        self.history_traj_len =None

        if self.level in ['Normal', 'Complex', 'Extrem']:
            self.map = ParkingMapNormal(self.level)
        elif self.level == 'dlp':
            self.map = ParkingMapDLP()
        self.vehicle = Vehicle(n_step=NUM_STEP, step_len=STEP_LENGTH)
        self.lidar = LidarSimlator(LIDAR_RANGE, LIDAR_NUM)
        self.reward = 0.0
        self.prev_reward = 0.0
        self.accum_arrive_reward = 0.0
        self.meter_per_pixel = 0.04

        self.action_space = spaces.Box(
            np.array([VALID_STEER[0], VALID_SPEED[0]]).astype(np.float32),
            np.array([VALID_STEER[1], VALID_SPEED[1]]).astype(np.float32),
        ) # steer, speed
       
        self.observation_space = {}
        if self.use_action_mask:
            self.action_filter = ActionMask()
            self.observation_space['action_mask'] = spaces.Box(low=0, high=1, 
                shape=(N_DISCRETE_ACTION,), dtype=np.float32
            )
        if self.use_img_observation:
            self.img_processor = Obs_Processor()
            self.observation_space['image'] = spaces.Box(low=0, high=255, 
                shape=(OBS_W//self.img_processor.downsample_rate, OBS_H//self.img_processor.downsample_rate, 
                self.img_processor.n_channels), dtype=np.uint8
            )             # 数据类型不匹配，dtype是float类型
            self.raw_img_shape = (OBS_W, OBS_H, 3)
        if self.use_lidar_observation:
            # the observation is composed of lidar points and target representation
            # the target representation is (relative_distance, cos(theta), sin(theta), cos(phi), sin(phi))
            # where the theta indicates the relative angle of parking lot, and phi means the heading of 
            # parking lot in the polar coordinate of the ego car's view
            low_bound, high_bound = np.zeros((LIDAR_NUM)), np.ones((LIDAR_NUM))*LIDAR_RANGE
            self.observation_space['lidar'] = spaces.Box(
                low=low_bound, high=high_bound, shape=(LIDAR_NUM,), dtype=np.float32
            )
        low_bound = np.array([0,-1,-1,-1,-1])
        high_bound = np.array([MAX_DIST_TO_DEST,1,1,1,1])
        self.observation_space['target'] = spaces.Box(
            low=low_bound, high=high_bound, shape=(self.tgt_repr_size,), dtype=np.float32
        )
        low_bound = np.array([-10.0, -10.0, -1, -1])
        high_bound = np.array([10.0, 10.0, 1, 1])
        self.observation_space['park_target_point'] = spaces.Box(
            low=low_bound, high=high_bound, shape=(4,), dtype=np.float32
        )
    
    def set_level(self, level:str=None):
        if level is None:
            self.map = ParkingMapNormal()
            return
        self.level = level
        if self.level in ['Normal', 'Complex', 'Extrem',]:
            self.map = ParkingMapNormal(self.level)
        elif self.level == 'dlp':
            self.map = ParkingMapDLP()

    def reset(self, case_id: int = None, data_dir: str = None, level: str = None,) -> np.ndarray:
        self.reward = 0.0
        self.prev_reward = 0.0
        self.accum_arrive_reward = 0.0
        self.t = 0.0

        if level is not None:
            self.set_level(level)
        initial_state = self.map.reset(case_id, data_dir)
        self.vehicle.reset(initial_state)
        return self.step()[0]


    def _detect_collision(self):
        # return False
        for obstacle in self.map.obstacles:
            if self.vehicle.box.intersects(obstacle.shape):
                return True
        return False
    
    def _detect_outbound(self):
        x, y = self.vehicle.state.loc.x, self.vehicle.state.loc.y
        return x>self.map.xmax or x<self.map.xmin or y>self.map.ymax or y<self.map.ymin

    def _check_arrived(self):
        vehicle_box = Polygon(self.vehicle.box)
        dest_box = Polygon(self.map.dest_box)
        union_area = vehicle_box.intersection(dest_box).area
        if union_area / dest_box.area > 0.95:
            return True
        return False
    
    def _check_time_exceeded(self):
        return self.t > TOLERANT_TIME

    def _check_status(self):
        if self._detect_collision():
            return Status.COLLIDED
        if self._detect_outbound():
            return Status.OUTBOUND
        if self._check_arrived():
            return Status.ARRIVED
        if self._check_time_exceeded():
            return Status.OUTTIME
        return Status.CONTINUE

    def _get_reward(self, prev_state: State, curr_state: State):

        # time penalty
        time_cost = - np.tanh(self.t / (10*TOLERANT_TIME))

        # RS distance reward
        if REWARD_WEIGHT['rs_dist_reward'] != 0:
            radius = math.tan(VALID_STEER[-1])/WHEEL_BASE
            curr_rs_dist = rsCurve.calc_optimal_path(*curr_state.get_pos(), *self.map.dest.get_pos(), radius , 0.1).L
            prev_rs_dist = rsCurve.calc_optimal_path(*prev_state.get_pos(), *self.map.dest.get_pos(), radius, 0.1).L
            rs_dist_norm_ratio = rsCurve.calc_optimal_path(*self.map.start.get_pos(), *self.map.dest.get_pos(), radius, 0.1).L
            rs_dist_reward = math.exp(-curr_rs_dist/rs_dist_norm_ratio) - \
                math.exp(-prev_rs_dist/rs_dist_norm_ratio)
        else:
            rs_dist_reward = 0

        # Euclidean distance reward & angle reward
        def get_angle_diff(angle1, angle2):
            # norm to 0 ~ pi/2
            angle_dif = math.acos(math.cos(angle1 - angle2)) # 0~pi
            # return angle_dif if angle_dif<math.pi/2 else math.pi-angle_dif
            return angle_dif
        dist_diff = curr_state.loc.distance(self.map.dest.loc)
        angle_diff = get_angle_diff(curr_state.heading, self.map.dest.heading)
        prev_dist_diff = prev_state.loc.distance(self.map.dest.loc)
        prev_angle_diff = get_angle_diff(prev_state.heading, self.map.dest.heading)
        dist_norm_ratio = max(self.map.dest.loc.distance(self.map.start.loc),10)
        angle_norm_ratio = math.pi
        dist_reward = prev_dist_diff/dist_norm_ratio - dist_diff/dist_norm_ratio
        angle_reward = prev_angle_diff/angle_norm_ratio - angle_diff/angle_norm_ratio
        
        # Box union reward
        vehicle_box = Polygon(self.vehicle.box)
        dest_box = Polygon(self.map.dest_box)
        union_area = vehicle_box.intersection(dest_box).area
        box_union_reward = union_area/(2*dest_box.area - union_area)
        if box_union_reward < self.accum_arrive_reward:
            box_union_reward = 0 
        else:
            prev_arrive_reward = self.accum_arrive_reward
            self.accum_arrive_reward = box_union_reward
            box_union_reward -= prev_arrive_reward
        # Gear shift reward (D<->R). First shift is free, subsequent shifts: -2 each.
        def gear_sign(v: float) -> int:
            # treat v==0 as "no gear" (ignored)
            return 1 if v > 0 else (-1 if v < 0 else 0)

        prev_gear = gear_sign(prev_state.speed)
        curr_gear = gear_sign(curr_state.speed)

        # lazy init counter/flag
        if not hasattr(self, "_gear_shift_seen"):
            self._gear_shift_seen = False  # whether we have already counted the first shift

        gear_reward = 0
        # count shift only when both gears are valid and sign changes
        if prev_gear != 0 and curr_gear != 0 and prev_gear != curr_gear:
            if self._gear_shift_seen:
                gear_reward = -0.1
            else:
                # first shift is free
                self._gear_shift_seen = True
        return [time_cost ,rs_dist_reward ,dist_reward ,angle_reward ,box_union_reward, gear_reward]
        
    def get_reward(self, status, prev_state):
        reward_info = [0,0,0,0,0,0]
        if status == Status.CONTINUE:
            reward_info = self._get_reward(prev_state, self.vehicle.state)
        return reward_info

    def step(self, action:np.ndarray = None):
        '''
        Parameters:
        ----------
        `action`: `np.ndarray`

        Returns:
        ----------
        ``obsercation`` (Dict): 
            the observation of image based surroundings, lidar view and target representation.
            If `use_lidar_observation` is `True`, then `obsercation['image'] = None`.
            If `use_lidar_observation` is `False`, then `obsercation['lidar'] = None`. 

        ``reward_info`` (OrderedDict): different types of reward information, including:
                time_cost ,rs_dist_reward ,dist_reward ,angle_reward ,box_union_reward
        `status` (`Status`): represent the state of vehicle, including:
                `CONTINUE`, `ARRIVED`, `COLLIDED`, `OUTBOUND`, `OUTTIME`
        `info` (`OrderedDict`): other information.
        '''
        assert self.vehicle is not None
        prev_state = self.vehicle.state
        collide = False
        arrive = False
        if action is not None:
            for simu_step_num in range(NUM_STEP):   #在这里控制的仿真时间是10 * 0.05 = 0.5s
                prev_info = self.vehicle.step(action,step_time=1)
                if self._check_arrived():
                    arrive = True
                    break
                if self._detect_collision():
                    if simu_step_num == 0:
                        collide = ENV_COLLIDE
                        self.vehicle.retreat(prev_info)
                    else:
                        collide = ENV_COLLIDE
                        self.vehicle.retreat(prev_info)
                    simu_step_num -= 1
                    break
            simu_step_num += 1
            # remove redundant trajectory
            if simu_step_num > 1:
                del self.vehicle.trajectory[-simu_step_num:-1]

        self.t += 1
        observation = self.render(self.render_mode)
        if arrive:
            status = Status.ARRIVED
        else:
            status = Status.COLLIDED if collide else self._check_status()

        reward_list = self.get_reward(status, prev_state)
        reward_info = OrderedDict({'time_cost':reward_list[0],\
            'rs_dist_reward':reward_list[1],\
            'dist_reward':reward_list[2],\
            'angle_reward':reward_list[3],\
            'box_union_reward':reward_list[4],
            'gear_shift_reward':reward_list[5]})

        info = OrderedDict({'reward_info':reward_info,
            'path_to_dest':None})
        if self.t > 1 and status==Status.CONTINUE\
            and self.vehicle.state.loc.distance(self.map.dest.loc)<RS_MAX_DIST:
            rs_path_to_dest = self.find_rs_path(status)
            if rs_path_to_dest is not None:
                info['path_to_dest'] = rs_path_to_dest

        return observation, reward_info, status, info

    def _process_img_observation(self, img):
        '''
        Process the img into channels of different information.

        Parameters
        ------
        img (np.ndarray): RGB image of shape (OBS_W, OBS_H, 3)

        Returns
        ------
        processed img (np.ndarray): shape (OBS_W//downsample_rate, OBS_H//downsample_rate, n_channels )
        '''
        processed_img = self.img_processor.process_img(img)
        return processed_img

    def _get_lidar_observation(self,):
        init_state = State([0.0,0.0,0.0])
        lidar_view = self.lidar.get_observation(init_state, self.obervattion_clusters_info_vcs)
        return lidar_view
    
    def _get_targt_repr(self,):
        # target position representation
        init_state = State([0.0,0.0,0.0])
        dest_pos = (self.dest_pose_vcs.loc.x, self.dest_pose_vcs.loc.y, self.dest_pose_vcs.heading)
        ego_pos = (init_state.loc.x, init_state.loc.y, init_state.heading)
        rel_distance = math.sqrt((dest_pos[0]-ego_pos[0])**2 + (dest_pos[1]-ego_pos[1])**2)
        rel_angle = math.atan2(dest_pos[1]-ego_pos[1], dest_pos[0]-ego_pos[0]) - ego_pos[2]
        rel_dest_heading = dest_pos[2] - ego_pos[2]
        tgt_repr = np.array([rel_distance, math.cos(rel_angle), math.sin(rel_angle),\
            math.cos(rel_dest_heading), math.sin(rel_dest_heading)])
        return tgt_repr   #获得[rel_distance, math.cos(rel_angle), math.sin(rel_angle), math.cos(rel_dest_heading), math.sin(rel_dest_heading)]

    def _get_park_targrt_point(self,):
        dest_pos = (self.dest_pose_vcs.loc.x, self.dest_pose_vcs.loc.y, self.dest_pose_vcs.heading)
        tgt_point = np.array([self.dest_pose_vcs.loc.x, self.dest_pose_vcs.loc.y, math.cos(
            self.dest_pose_vcs.heading), math.sin(self.dest_pose_vcs.heading)])
        return tgt_point  
    def render(self, mode: str = "human"):
        assert mode in self.metadata["render_mode"]
        assert self.vehicle is not None

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
        self.coord_transform_to_vcs(self.vehicle.state, self.map.obstacles, self.map.dest)
        observation = {'image':None, 'lidar':None, 'target':None, 'action_mask':None, 'park_target_point':None}
        if self.use_img_observation:
            raw_observation = self._get_img_observation()
            observation['image'] = self._process_img_observation(raw_observation)
        if self.use_lidar_observation:
            observation['lidar'] = self._get_lidar_observation()
        if self.use_action_mask:
            observation['action_mask'] = self.action_filter.get_steps(observation['lidar'])
        observation['target'] = self._get_targt_repr() #获得[rel_distance, math.cos(rel_angle), math.sin(rel_angle), math.cos(rel_dest_heading), math.sin(rel_dest_heading)]
        observation['park_target_point'] = self._get_park_targrt_point()
        pygame.display.update()
        self.clock.tick(self.fps)

        observation['lidar'] = observation['lidar'] / LIDARRANGE
        observation['target'][0] = np.clip(observation['target'][0] / np.sqrt(TRAJXRANGE**2 + TRAJYRANGE**2), -1.0, 1.0)
        observation['park_target_point'][0] = np.clip(observation['target'][0] / TRAJXRANGE, -1.0, 1.0)
        observation['park_target_point'][1] = np.clip(observation['park_target_point'][1] / TRAJYRANGE, -1.0, 1.0)

        return observation

    def find_rs_path(self,status):
        '''
        Find collision-free RS path. 

        Returns:
            path (PATH): the related PATH object which is collision-free.
        '''
        startX, startY, startYaw = self.vehicle.state.loc.x, self.vehicle.state.loc.y, self.vehicle.state.heading
        goalX, goalY, goalYaw = self.map.dest.loc.x, self.map.dest.loc.y, self.map.dest.heading
        radius = math.tan(VALID_STEER[-1])/WHEEL_BASE
        #  Find all possible reeds-shepp paths between current and goal node
        reedsSheppPaths = rsCurve.calc_all_paths(startX, startY, startYaw, goalX, goalY, goalYaw, radius, 0.1)

        # Check if reedsSheppPaths is empty
        if not reedsSheppPaths:
            return None

        # Find path with lowest cost considering non-holonomic constraints
        costQueue = heapdict()
        for path in reedsSheppPaths:
            costQueue[path] = path.L

        # Find first path in priority queue that is collision free
        min_path_len = -1
        idx = 0
        while len(costQueue)!=0:
            idx += 1
            path = costQueue.popitem()[0]
            if min_path_len < 0:
                min_path_len = path.L
            if path.L > 1.6*min_path_len and idx > 2:
                break
            traj=[]
            traj = [[path.x[k],path.y[k],path.yaw[k]] for k in range(len(path.x))]
            traj_valid = self.is_traj_valid(traj)
            if traj_valid:
                return path
        return None
    
    def is_traj_valid(self, traj):
        car_coords1 = np.array(VehicleBox.coords)[:4] # (4,2)
        car_coords2 = np.array(VehicleBox.coords)[1:] # (4,2)
        car_coords_x1 = car_coords1[:,0].reshape(1,-1)
        car_coords_y1 = car_coords1[:,1].reshape(1,-1) # (1,4)
        car_coords_x2 = car_coords2[:,0].reshape(1,-1)
        car_coords_y2 = car_coords2[:,1].reshape(1,-1) # (1,4)
        vxs = np.array([t[0] for t in traj])
        vys = np.array([t[1] for t in traj])
        # check outbound
        if np.min(vxs) < self.map.xmin or np.max(vxs) > self.map.xmax \
        or np.min(vys) < self.map.ymin or np.max(vys) > self.map.ymax:
            return False
        vthetas = np.array([t[2] for t in traj])
        cos_theta = np.cos(vthetas).reshape(-1,1) # (T,1)
        sin_theta = np.sin(vthetas).reshape(-1,1)
        vehicle_coords_x1 = cos_theta*car_coords_x1 - sin_theta*car_coords_y1 + vxs.reshape(-1,1) # (T,4)
        vehicle_coords_y1 = sin_theta*car_coords_x1 + cos_theta*car_coords_y1 + vys.reshape(-1,1)
        vehicle_coords_x2 = cos_theta*car_coords_x2 - sin_theta*car_coords_y2 + vxs.reshape(-1,1) # (T,4)
        vehicle_coords_y2 = sin_theta*car_coords_x2 + cos_theta*car_coords_y2 + vys.reshape(-1,1)
        vx1s = vehicle_coords_x1.reshape(-1,1)
        vx2s = vehicle_coords_x2.reshape(-1,1)
        vy1s = vehicle_coords_y1.reshape(-1,1)
        vy2s = vehicle_coords_y2.reshape(-1,1)
        # Line 1: the edges of vehicle box, ax + by + c = 0
        a = (vy2s - vy1s).reshape(-1,1) # (4*t,1)
        b = (vx1s - vx2s).reshape(-1,1)
        c = (vy1s*vx2s - vx1s*vy2s).reshape(-1,1)
        
        # convert obstacles(LinerRing) to edges ((x1,y1), (x2,y2))
        x_max = np.max(vx1s) + 5
        x_min = np.min(vx1s) - 5
        y_max = np.max(vy1s) + 5
        y_min = np.min(vy1s) - 5

        x1s, x2s, y1s, y2s = [], [], [], []
        for obst in self.map.obstacles:
            if isinstance(obst, Area):
                obst = obst.shape
            obst_coords = np.array(obst.coords) # (n+1,2)
            if (obst_coords[:,0] > x_max).all() or (obst_coords[:,0] < x_min).all()\
                or (obst_coords[:,1] > y_max).all() or (obst_coords[:,1] < y_min).all():
                continue
            x1s.extend(list(obst_coords[:-1, 0]))
            x2s.extend(list(obst_coords[1:, 0]))
            y1s.extend(list(obst_coords[:-1, 1]))
            y2s.extend(list(obst_coords[1:, 1]))
        if len(x1s) == 0: # no obstacle around
            return True
        x1s, x2s, y1s, y2s  = np.array(x1s).reshape(1,-1), np.array(x2s).reshape(1,-1),\
            np.array(y1s).reshape(1,-1), np.array(y2s).reshape(1,-1), 
        # Line 2: the edges of obstacles, dx + ey + f = 0
        d = (y2s - y1s).reshape(1,-1) # (1,E)
        e = (x1s - x2s).reshape(1,-1)
        f = (y1s*x2s - x1s*y2s).reshape(1,-1)

        # calculate the intersections
        det = a*e - b*d # (4, E)
        parallel_line_pos = (det==0) # (4, E)
        det[parallel_line_pos] = 1 # temporarily set "1" to avoid "divided by zero"
        raw_x = (b*f - c*e)/det # (4, E)
        raw_y = (c*d - a*f)/det

        collide_map_x = np.ones_like(raw_x, dtype=np.uint8)
        collide_map_y = np.ones_like(raw_x, dtype=np.uint8)
        # the false positive intersections on line L2(not on edge L2)
        collide_map_x[raw_x>np.maximum(x1s, x2s)] = 0
        collide_map_x[raw_x<np.minimum(x1s, x2s)] = 0
        collide_map_y[raw_y>np.maximum(y1s, y2s)] = 0
        collide_map_y[raw_y<np.minimum(y1s, y2s)] = 0
        # the false positive intersections on line L1(not on edge L1)
        collide_map_x[raw_x>np.maximum(vx1s, vx2s)] = 0
        collide_map_x[raw_x<np.minimum(vx1s, vx2s)] = 0
        collide_map_y[raw_y>np.maximum(vy1s, vy2s)] = 0
        collide_map_y[raw_y<np.minimum(vy1s, vy2s)] = 0

        collide_map = collide_map_x*collide_map_y
        collide_map[parallel_line_pos] = 0
        collide = np.sum(collide_map) > 0

        if collide:
            return False
        return True

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.is_open = False
            pygame.quit()

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
    
    def _compute_vehicle_pixel_corners(self, pose_vcs, yaw, scale = 1.0):
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
        vehicle_width = WIDTH * scale
        vehicle_length = LENGTH * scale
        vehicle_rear_overhang = REAR_HANG * scale
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

            history_pixel_corners = self._compute_vehicle_pixel_corners(
                history_pose, history_yaw,
                scale=HISTORY_BOX_SCALE
            )

            pygame.draw.polygon(
                self.screen,
                TRAJ_COLORS[idx],
                history_pixel_corners,
                width=1   # 薄线就够了
            )

    def _get_img_observation(self,):
       
        # 3. 渲染自车位置
        vehicle_angle = 0.0
        vehicle_position = Vec2d(0.0, 0.0)
        vehicle_pixel_corners = self._compute_vehicle_pixel_corners(vehicle_position, vehicle_angle)
        pygame.draw.polygon(self.screen, COLOR_POOL[0], vehicle_pixel_corners)

        ego_u, ego_v = self._vcs_to_pixel(vehicle_position.x_, vehicle_position.y_)
        pygame.draw.circle(self.screen, EGO_CENTER_COLOR, (ego_u, ego_v), 3)

        # 4. 渲染起点位置

        '''
        # start_pose = Vec2d(start_pose_vsc.x, start_pose_vsc.y)
        # start_pose_yaw = start_pose_vsc.yaw

        # start_pose_pixel_corners = self._compute_vehicle_pixel_corners(start_pose, start_pose_yaw)

        # pygame.draw.polygon(self.screen, START_COLOR, start_pose_pixel_corners, width=1)
        '''

        #5、渲染目标位置
 
        target_pose = Vec2d(self.dest_pose_vcs.loc.x, self.dest_pose_vcs.loc.y)
        target_pose_yaw = self.dest_pose_vcs.heading

        target_pose_pixel_corners = self._compute_vehicle_pixel_corners(target_pose, target_pose_yaw)

        pygame.draw.polygon(self.screen, DEST_COLOR, target_pose_pixel_corners, width=2)

        tx, ty = self.dest_pose_vcs.loc.x, self.dest_pose_vcs.loc.y
        tu, tv = self._vcs_to_pixel(tx, ty)
        pygame.draw.circle(self.screen, TARGET_CENTER_COLOR, (tu, tv), 3)

    

        # 6. 画障碍物 / cluster（假设是多条折线）
        # cluster_info_vcs: List[List[(x, y)]]
        for cluster in self.clusters_frame_info_vcs:
            p0 = cluster["p0"]
            p1 = cluster["p1"]

            # 自车坐标系 (x 前, y 左) → 像素坐标 (u, v)
            u0, v0 = self._vcs_to_pixel(p0.x, p0.y)
            u1, v1 = self._vcs_to_pixel(p1.x, p1.y)
            # 折线
            pygame.draw.line(self.screen, OBSTACLE_COLOR, (u0, v0), (u1, v1), width= 2)

        self.render_history_traj(self.history_traj_vcs, self.history_traj_len)

        pygame.display.update()
        self.clock.tick(self.fps)

        obs_str = pygame.image.tostring(self.screen, "RGB")
        img = np.frombuffer(obs_str, dtype=np.uint8)
        img = img.reshape((OBS_H, OBS_W, 3))

        # measurements_path = os.path.join(filename, str(ego_index))
        # os.makedirs(measurements_path, exist_ok=True)
        # measurements_path_final = os.path.join(measurements_path, "cnn.png")

        # image = Image.fromarray(img)      # numpy → PIL
        # image.save(measurements_path_final)          # 保存为 png/jpg 都可以

        # observation = {"img": img}
        
        return img
    
    def coord_transform_to_vcs(self, curr_state: State, obstacles:list, Dest_state: State):
        ego_x, ego_y, ego_heading = curr_state.get_pos()
        ego_pose = CustomizePose(ego_x, ego_y, 0.0, 0.0, ego_heading / 3.14 * 180, 0.0)
        world2ego_mat = ego_pose.get_homogeneous_transformation().get_inverse_matrix()
        dest_x, dest_y, dest_heading = Dest_state.get_pos()
        dest_pose = CustomizePose(dest_x, dest_y, 0.0, 0.0, dest_heading / 3.14 * 180, 0.0)
        dest_pose_in_ego = dest_pose.get_pose_in_ego(world2ego_mat)
        dest_yaw_refine = self.get_safe_yaw(dest_pose_in_ego.yaw) 
        self.dest_pose_vcs = State([dest_pose_in_ego.x, dest_pose_in_ego.y, dest_yaw_refine, 0.0, 0.0])
        cluster_frame_in_vcs =[]
        cluster_dict_template = {
                "id": None,
                "p0": {},
                "p1": {}
            }
        start_id = 0
        for obs in obstacles:
            obs = obs.shape
            coords = list(obs.coords)
            if isinstance(obs, LinearRing):
                coords = coords[:-1]
            for i in range(len(coords)):
                p0_coord = coords[i]
                p1_coord = coords[(i + 1) % len(coords)]  # 对 LinearRing 循环连接
                
                # 跳过零长度线段
                if np.allclose(p0_coord, p1_coord):
                    continue
                p0 = CustomizePose(p0_coord[0], p0_coord[1], 0.0, 0.0, 0.0, 0.0)
                p1 = CustomizePose(p1_coord[0], p1_coord[1], 0.0, 0.0, 0.0, 0.0)
                
                # 创建 cluster 字典
                each_cluster_vcs = copy.deepcopy(cluster_dict_template)
                each_cluster_vcs["id"] = start_id + i
                each_cluster_vcs["p0"] = p0.get_pose_in_ego(world2ego_mat)
                each_cluster_vcs["p1"] = p1.get_pose_in_ego(world2ego_mat)
                cluster_frame_in_vcs.append(each_cluster_vcs)
        self.clusters_frame_info_vcs = cluster_frame_in_vcs
            
        clusters: List[LineString] = []

        for cl in cluster_frame_in_vcs:
            p0 = cl["p0"]
            p1 = cl["p1"]

            # Shapely LineString
            line = LineString([(p0.x, p0.y), (p1.x, p1.y)])
            clusters.append(line)

        self.obervattion_clusters_info_vcs = clusters
        self.history_traj_vcs, self.history_traj_len = self.create_history_point(world2ego_mat)
    def get_safe_yaw(slef, yaw):
        if yaw <= -180:
            yaw += 360
        if yaw > 180:
            yaw -= 360
        yaw = yaw / 180.0 * 3.14
        return yaw
    
    def create_history_point(self, world2ego_mat: np.array):
        history_trajector_vcs = []
        history_traj_len = 0
        for i in range(1, 13):  # predict iteration
            ds = -0.5 * i + self.vehicle.trajectory[-1].s
            if(ds < 0):
                return history_trajector_vcs, history_traj_len
            history_state_in_world = self.vehicle.interpolate_by_s(ds)
            history_pose_in_world = CustomizePose(history_state_in_world.loc.x, history_state_in_world.loc.y, 0.0,
                                                  0.0, history_state_in_world.heading / 3.14 * 180.0, 0.0, history_state_in_world.s)
            history_pose_in_ego = history_pose_in_world.get_pose_in_ego(world2ego_mat)
            history_pose_in_ego.yaw = self.get_safe_yaw(history_pose_in_ego.yaw)
            history_trajector_vcs.append(history_pose_in_ego)
            history_trajector_vcs = history_trajector_vcs[::-1]
            history_traj_len = len(history_trajector_vcs)

        return history_trajector_vcs, history_traj_len






