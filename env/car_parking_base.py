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
        self._risk_prev_d = None
        self._risk_in_zone = None
        self._stuck_steps = 0
        for k in ["_gear_shift_count","_gear_heavy_tail","_gear_shift_seen", "_last_shift_xy", "_last_shift_step"]:
            if hasattr(self, k):
                delattr(self, k)

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
        if union_area / dest_box.area > 0.90:
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
    
    def lateral_longitudinal_error(
        self,
        car_xy,        # (x_c, y_c)
        slot_xy,       # (x_s, y_s)
        slot_yaw       # theta_s (rad)
    ):
        dx = car_xy[0] - slot_xy[0]
        dy = car_xy[1] - slot_xy[1]

        # 车位坐标系
        e_parallel = np.array([np.cos(slot_yaw), np.sin(slot_yaw)])
        e_perp     = np.array([-np.sin(slot_yaw), np.cos(slot_yaw)])

        d_lon = dx * e_parallel[0] + dy * e_parallel[1]
        d_lat = dx * e_perp[0]     + dy * e_perp[1]

        return d_lat, d_lon
    
    def _soft_hinge(self, x: float, sharpness: float = 10.0) -> float:
        """
        平滑版 max(0, x)，sharpness 越大越接近 ReLU。
        """
        # softplus(x) = log(1+exp(x))
        return np.log1p(np.exp(sharpness * x)) / sharpness

    def _get_reward(self, prev_state: State, curr_state: State, lidar_dist: List):

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

        abs_dist_pen = -0.05 * (dist_diff / dist_norm_ratio)
        abs_ang_pen  = -0.02 * (angle_diff  / angle_norm_ratio)
        d_lat, _ = self.lateral_longitudinal_error((curr_state.loc.x, curr_state.loc.y),
                                                (self.map.dest.loc.x, self.map.dest.loc.y), self.map.dest.heading)

        near_bonus = 0.0
        if dist_diff < 5.0 and angle_diff < (40.0 * math.pi / 180.0) and abs(d_lat)<1.0:
            near_bonus += 0.04
        if dist_diff < 3.0 and angle_diff < (20.0 * math.pi / 180.0) and abs(d_lat)<1.0:
            near_bonus += 0.06
        
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

        effective_prev_gear = prev_gear
        effective_curr_gear = curr_gear if curr_gear != 0 else prev_gear

        shifted = (effective_prev_gear != 0 and effective_curr_gear != 0 and effective_prev_gear != effective_curr_gear)

        
        if not hasattr(self, "_gear_shift_count"):
            self._gear_shift_count = 0
        # "heavy tail" penalty counter: once triggered, penalize next N steps
        if not hasattr(self, "_gear_heavy_tail"):
            self._gear_heavy_tail = 0
        # lazy init counter/flag
        if not hasattr(self, "_gear_shift_seen"):
            self._gear_shift_seen = False  # whether we have already counted the first shift

        if not hasattr(self, "_last_shift_xy"):
            self._last_shift_xy = None  # (x, y)

        # base penalty (first shift can be free if you want)
        BASE_SHIFT_PEN = 0.1   # 每次换挡基础惩罚：0.2~0.6 之间调
        SHIFT_PEN_HEAVY = 0.2 
        FIRST_FREE = True

        # "near last shift point" heavy penalty
        NEAR_SHIFT_DIST = 1.0   # 两次换挡点距离阈值（米）：0.25~0.6 之间调
        HEAVY_PEN = 0.5          # 重罚强度：0.5~1.5 之间调
        MIN_STEP_GAP = 3         # 防止同一步/极短步误触：>=2~5
        HEAVY_TAIL_STEPS = 10           # 之后的路径（后续步数）也要罚
        HEAVY_TAIL_PEN = 0.02           # 每步持续罚（后续路径重罚）
        FREE_SHIFTS = 6


        gear_reward = 0
        # count shift only when both gears are valid and sign changes

        if self._gear_heavy_tail > 0:
            gear_reward += -HEAVY_TAIL_PEN
            self._gear_heavy_tail -= 1
        if shifted:
              # 1) base penalty
            if FIRST_FREE and (not self._gear_shift_seen):
                self._gear_shift_seen = True
                gear_reward += 0.0
            elif self._gear_shift_count <= FREE_SHIFTS:
                gear_reward += -BASE_SHIFT_PEN
            else:
                 # over 4 shifts: heavy instant penalty
                gear_reward += -SHIFT_PEN_HEAVY
                # and heavy tail: penalize following steps (path)
                self._gear_heavy_tail = max(self._gear_heavy_tail, HEAVY_TAIL_STEPS)


            # 2) heavy penalty if shift happens near last shift point
            curr_xy = (curr_state.loc.x, curr_state.loc.y)
            if self._last_shift_xy is not None:
                dx = curr_xy[0] - self._last_shift_xy[0]
                dy = curr_xy[1] - self._last_shift_xy[1]
                dist_from_last_shift = math.hypot(dx, dy)

                # only apply heavy penalty if not too close in time (avoid double count jitter)
                if dist_from_last_shift < NEAR_SHIFT_DIST:
                    # stronger when closer: (near -> heavier)
                    # ratio in [0,1], 1 means exactly same point
                    ratio = 1.0 - (dist_from_last_shift / NEAR_SHIFT_DIST)
                    gear_reward += -HEAVY_PEN * ratio

            # update last shift record
            self._last_shift_xy = curr_xy
        '''
        增加靠近障碍物的惩罚和远离障碍物的奖励
        1、首先是没有进入车位， 没有发生面积的overlap， union_area = vehicle_box.intersection(dest_box).area = 0
        2、当D档时，判断前方距离小于0.5时，开始惩罚，且越靠近越惩罚；
           R档时，判断后方距离小于0.5时，开始惩罚，且越靠近越惩罚
           进入惩罚后，如果下一步会远离前方障碍物，或者后方障碍物，则可以获得奖励，奖励为定值
           前向距离信号为min_front_distance，后向距离信号为min_rear_distance

        '''
         # ==============================
        # Encourage gear shift near wall (optional, set to 0 to disable)
        # ==============================

        front_distance = np.concatenate([lidar_dist[0:5], lidar_dist[114:120]])
        rear_distance = np.array(lidar_dist[42:77])
        # distance signals (m)
        min_front_distance = float(front_distance.min())
        min_rear_distance  = float(rear_distance.min())

        if effective_curr_gear == 1:
            danger_d = min_front_distance
        elif effective_curr_gear == -1:
            danger_d = min_rear_distance
        else:
            danger_d = None

        in_shift_zone = (danger_d is not None and danger_d < 0.5)

        if in_shift_zone and shifted:
            # cancel any shift penalty and optionally add bonus
            gear_reward = 0.0
            gear_reward += BASE_SHIFT_PEN


        # ==============================
        # Risk reward (obstacle distance shaping)
        # includes:
        #   1) barrier penalty (closer -> much larger penalty)
        #   2) away bonus (if moving away when already in risk zone)
        #   3) approach penalty (if moving closer when already in risk zone)
        # ==============================
        risk_reward = 0.0

        # ---- thresholds you asked ----
        SHIFT_START = 0.3   # you want start shifting around 0.3m
        RISK_START  = 0.5   # mild risk starts here
        eps = 0.05          # anti-div0 / smooth

        # ---- strengths (set to 0 to disable) ----
        # state-based barrier penalties
        K_MILD   = 0.08     # mild barrier in (0.5m ~ 0m)
        K_STRONG = 0.25     # extra strong barrier inside 0.3m

        # trend bonus/penalty (set to 0 to disable)
        AWAY_BONUS     = 0.05   # +reward if distance increases while in risk zone
        AWAY_EPS       = 0.02   # require at least +2cm to count as moving away
        APPROACH_PEN_K = 0.08   # -penalty if distance decreases while in risk zone
        APPROACH_EPS   = 0.01   # require at least -1cm to count as moving closer

        # Optional: extra "pushing closer" penalty inside SHIFT_START (set to 0 to disable)
        K_PUSH   = 0.20
        PUSH_EPS = 0.01

        # effective gear: avoid risk disabled when speed ~ 0
        effective_gear = curr_gear if curr_gear != 0 else prev_gear

        # Only before entering slot (no overlap)
        if union_area <= 1e-9 and (angle_diff  / angle_norm_ratio) < 0.5:

            # select risk distance direction
            if effective_gear == 1:      # D -> front
                d = min_front_distance
            elif effective_gear == -1:   # R -> rear
                d = min_rear_distance
            else:
                d = None

            # init memory
            if not hasattr(self, "_risk_prev_d"):
                self._risk_prev_d = None
            if not hasattr(self, "_risk_in_zone"):
                self._risk_in_zone = False

            if d is not None and d >= 0.0:
                prev_d = self._risk_prev_d
                in_zone_now = (d < RISK_START)

                # 1) state-based barrier penalties (only in risk zone)
                if in_zone_now:
                    # mild barrier
                    term_m = (1.0/(d + eps) - 1.0/(RISK_START + eps))
                    risk_reward += -K_MILD * (min((term_m ** 2), 10.0))

                    # strong barrier inside 0.3m
                    if d < SHIFT_START:
                        term_s = (1.0/(d + eps) - 1.0/(SHIFT_START + eps))
                        risk_reward += -K_STRONG * (min((term_s ** 2), 4.0))

                # 2) trend bonus/penalty (only when we were already in zone)
                if self._risk_in_zone and (prev_d is not None) and in_zone_now:
                    # away bonus
                    if d > prev_d + AWAY_EPS:
                        risk_reward += AWAY_BONUS

                    # approach penalty
                    if d < prev_d - APPROACH_EPS:
                        approach_delta = min(prev_d - d, 0.20)  # cap 20cm
                        approach_ratio = max(0.0, min(1.0, approach_delta / RISK_START))
                        risk_reward += -APPROACH_PEN_K * approach_ratio

                # 3) extra push penalty only inside 0.3m (optional)
                if self._risk_in_zone and (prev_d is not None) and (d < SHIFT_START):
                    if d < prev_d - PUSH_EPS:
                        push_delta = min(prev_d - d, 0.10)  # cap 10cm
                        push_ratio = max(0.0, min(1.0, push_delta / SHIFT_START))
                        risk_reward += -K_PUSH * push_ratio

                # update memory
                self._risk_prev_d = d
                self._risk_in_zone = in_zone_now

        else:
            # reset when entered slot
            if hasattr(self, "_risk_prev_d"):
                self._risk_prev_d = None
            if hasattr(self, "_risk_in_zone"):
                self._risk_in_zone = False
        
        # ==============================
        # Stuck penalty: punish "no movement / no progress" loops
        # ==============================
        stuck_pen = 0.0

        # thresholds (tune)
        MIN_MOVE = 0.05             # 2cm per step considered "moved"
        MIN_YAW  = 1.0 * math.pi/180.0  # 1 deg considered "turned"
        STUCK_START = 4             # allow a few steps for fine control
        STUCK_K = 0.05              # penalty per extra stuck step (set 0 to disable)

        # compute movement
        delta_pos = curr_state.loc.distance(prev_state.loc)
        delta_yaw = abs(get_angle_diff(curr_state.heading, prev_state.heading))

        # also consider progress to goal (optional, helps)
        progress = prev_dist_diff - dist_diff  # >0 means closer to goal

        if not hasattr(self, "_stuck_steps"):
            self._stuck_steps = 0

        # define "no effective change"
        # no_change = (delta_pos < MIN_MOVE) and (delta_yaw < MIN_YAW) and (progress < 0.01)
        no_change = delta_pos < MIN_MOVE

        if no_change:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0

        if self._stuck_steps >= STUCK_START:
            # linearly increasing penalty to strongly break loops
            stuck_pen = -STUCK_K * (self._stuck_steps - STUCK_START + 1)

        high_speed = 0.0
        v = curr_state.speed
        v_th = 1.0
        w_v = 5.0
        v_excess = abs(v) - v_th
        v_pen = self._soft_hinge(v_excess, sharpness=10.0)      # >=0
        high_speed -= w_v * (v_pen ** 2)                       # 二次惩罚：越大惩罚增长更快
        high_speed = max(high_speed, -5.0)


        big_steer = 0.0
        steer = curr_state.steering
        steer_th = 0.55    # 例：弧度 ~20deg；如果你的单位是度就改成 20
        w_steer  = 5.0

        steer_excess = abs(steer) - steer_th
        steer_pen = self._soft_hinge(steer_excess, sharpness=10.0)
        big_steer -= w_steer * steer_pen


        return [time_cost, rs_dist_reward, dist_reward, angle_reward, box_union_reward, gear_reward, abs_dist_pen + abs_ang_pen, near_bonus, stuck_pen, risk_reward, high_speed, big_steer]
        
    def get_reward(self, status, prev_state, observation):
        reward_info = [0,0,0,0,0,0,0,0,0,0,0,0]
        lidar_dist = observation['lidar'] * LIDARRANGE
        if status == Status.CONTINUE:
            reward_info = self._get_reward(prev_state, self.vehicle.state, lidar_dist)
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

        reward_list = self.get_reward(status, prev_state, observation)
        reward_info = OrderedDict({'time_cost':reward_list[0],\
            'rs_dist_reward':reward_list[1],\
            'dist_reward':reward_list[2],\
            'angle_reward':reward_list[3],\
            'box_union_reward':reward_list[4],
            'gear_shift_reward':reward_list[5],
            'abs_shape':reward_list[6],
            'near_bonus':reward_list[7],
            'low_speed':reward_list[8],
            'risk_reward':reward_list[9],
            'high_speed':reward_list[10],
            'big_steer':reward_list[11]})

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
            math.cos(rel_dest_heading), math.sin(rel_dest_heading)], dtype=np.float32)
        return tgt_repr   #获得[rel_distance, math.cos(rel_angle), math.sin(rel_angle), math.cos(rel_dest_heading), math.sin(rel_dest_heading)]

    def _get_park_targrt_point(self,):
        dest_pos = (self.dest_pose_vcs.loc.x, self.dest_pose_vcs.loc.y, self.dest_pose_vcs.heading)
        tgt_point = np.array([self.dest_pose_vcs.loc.x, self.dest_pose_vcs.loc.y, math.cos(
            self.dest_pose_vcs.heading), math.sin(self.dest_pose_vcs.heading)], dtype=np.float32)
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
        # radius = math.tan(VALID_STEER[-1])/WHEEL_BASE
        radius = math.tan(0.496)/WHEEL_BASE
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






