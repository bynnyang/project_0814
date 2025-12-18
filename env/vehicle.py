from typing import Callable, List
from enum import Enum
import copy

import numpy as np
from shapely.geometry import Point, LinearRing
from shapely.affinity import affine_transform
from vehicle_config import *


class Status(Enum):
    CONTINUE = 1
    ARRIVED = 2
    COLLIDED = 3
    OUTBOUND = 4
    OUTTIME = 5


class State:
    def __init__(self, raw_state: list, s =0.0):
        self.loc: Point = Point(raw_state[:2])
        self.heading: float = raw_state[2]
        if len(raw_state) == 3:
            self.speed: float = 0
            self.steering: float = 0
        else:
            self.speed: float = raw_state[3]
            self.steering: float = raw_state[4]
        self.s = s

    def create_box(self) -> LinearRing:
        cos_theta = np.cos(self.heading)
        sin_theta = np.sin(self.heading)
        mat = [cos_theta, -sin_theta, sin_theta, cos_theta, self.loc.x, self.loc.y]
        return affine_transform(VehicleBox, mat)

    def get_pos(self,):
        return (self.loc.x, self.loc.y, self.heading)


class KSModel(object):
    """Update the state of a vehicle by the Kinematic Single-Track Model.

    Kinematic Single-Track Model use the vehicle's current speed, heading, location, 
    acceleration, and velocity of steering angle as input. Then it returns the estimation of 
    speed, heading, steering angle and location after a small time step.

    Use the center of vehicle's rear wheels as the origin of local coordinate system.

    Assume the vehicle is front-wheel-only drive.
    """
    def __init__(
        self, 
        wheel_base: float,
        step_len: float,
        n_step: int,
        speed_range: list,
        angle_range: list
    ):
        self.wheel_base = wheel_base
        self.step_len = step_len
        self.n_step = n_step
        self.speed_range = speed_range
        self.angle_range = angle_range
        self.mini_iter = 1


    def step(self, state: State, action: list, step_time:int=NUM_STEP) -> State:
        """Update the state of a vehicle with the Kinematic Single-Track Model.

        Args:
            state (list): [x, y, car_angle, speed, steering]
            action (list): [steer, speed].
            step (float, optional): the step length for each simulation.
            n_step (int): number of step of updating the physical state. This value is decide by
                (physics simulation step length : rendering step length).

        """
        new_state = copy.deepcopy(state)
        x, y = new_state.loc.x, new_state.loc.y
        steer, speed = action
        new_state.steering = steer
        new_state.speed = speed
        new_state.speed = np.clip(new_state.speed, *self.speed_range)
        new_state.steering = np.clip(new_state.steering, *self.angle_range)

        for _ in range(step_time): #1步
            for _ in range(self.mini_iter): # self.mini_iter =20
                step_dist = abs(new_state.speed) * self.step_len / self.mini_iter
                new_state.s += step_dist
                x += new_state.speed * np.cos(new_state.heading) * self.step_len/self.mini_iter #self.step_len = 0.05
                y += new_state.speed * np.sin(new_state.heading) * self.step_len/self.mini_iter
                new_state.heading += \
                    new_state.speed * np.tan(new_state.steering) / self.wheel_base * self.step_len/self.mini_iter 
                new_state.heading = self.get_safe_yaw(new_state.heading)

        new_state.loc = Point(x, y)
        return new_state
    
    def get_safe_yaw(slef, yaw):
        if yaw <= -np.pi:
            yaw += 2 * np.pi
        if yaw > np.pi:
            yaw -= 2 * np.pi
        return yaw


class Vehicle:
    """_summary_
    """
    def __init__(
        self,
        wheel_base: float = WHEEL_BASE,
        step_len: float = STEP_LENGTH,
        n_step: int = NUM_STEP,
        speed_range: list = VALID_SPEED, 
        angle_range: list = VALID_STEER
    ) -> None:

        self.initial_state: list = None
        self.state: State = None
        self.box: LinearRing = None
        self.trajectory: List[State] = []
        self.kinetic_model: Callable = \
            KSModel(wheel_base, step_len, n_step, speed_range, angle_range)
        self.v_max = None
        self.v_min = None

    def reset(self, initial_state: State):
        """
        Args:
            init_pos (list): [x0, y0, theta0]
        """
        self.initial_state = initial_state
        self.state = self.initial_state
        self.v_max = self.initial_state.speed
        self.v_min = self.initial_state.speed
        self.box = self.state.create_box()
        self.trajectory.clear()
        self.trajectory.append(self.state)
        self.tmp_trajectory = self.trajectory.copy()

    def step(self, action: np.ndarray, step_time: int=NUM_STEP):
        """
        Args:
            action (list): [steer, speed]
        """
        prev_info = copy.deepcopy((self.state, self.box, self.v_max, self.v_min))
        self.state = self.kinetic_model.step(self.state, action, step_time)  # 1s 
        self.box = self.state.create_box()
        self.trajectory.append(self.state)
        self.tmp_trajectory.append(self.state)
        self.v_max = self.state.speed if self.state.speed > self.v_max else self.v_max
        self.v_min = self.state.speed if self.state.speed < self.v_min else self.v_min
        return prev_info

    def retreat(self, prev_info):
        '''
        Retreat the vehicle state from previous one.

        Args:
            prev_info (tuple): (state, box, v_max, v_min)
        '''
        self.state, self.box, self.v_max, self.v_min = prev_info
        self.trajectory.pop(-1)

    def interpolate_by_s(self, target_s: float) -> State:
        """
        按路径长度 s 在轨迹中插值查找状态
        
        Args:
            target_s: 目标路径长度（>=0）
        
        Returns:
            插值后的 State 对象（包含插值后的位置、朝向、速度、转向角、路径长度）
        """
        if not self.trajectory:
            raise ValueError("轨迹为空，无法插值")
        
        # 计算累积路径长度
        cumulative_s = [state.s for state in self.trajectory]
        total_s = cumulative_s[-1]
        
        # 边界处理
        if target_s <= 0:
            return self.trajectory[0]
        if target_s >= total_s:
            return self.trajectory[-1]
        
        # 找到所在区间
        for i in range(1, len(cumulative_s)):
            if target_s <= cumulative_s[i]:
                prev_state = self.trajectory[i-1]
                curr_state = self.trajectory[i]
                prev_s = cumulative_s[i-1]
                curr_s = cumulative_s[i]
                
                # 插值比例
                t = (target_s - prev_s) / (curr_s - prev_s) if curr_s > prev_s else 0.0
                
                # 位置和朝向插值
                x = prev_state.loc.x + t * (curr_state.loc.x - prev_state.loc.x)
                y = prev_state.loc.y + t * (curr_state.loc.y - prev_state.loc.y)
                
                # 角度插值（处理 -π 到 π 的跳变）
                h0, h1 = prev_state.heading, curr_state.heading
                heading_diff = (h1 - h0 + np.pi) % (2 * np.pi) - np.pi
                heading = h0 + t * heading_diff
                heading = (heading + np.pi) % (2 * np.pi) - np.pi
                
                # 速度和转向角线性插值
                speed = prev_state.speed + t * (curr_state.speed - prev_state.speed)
                steering = prev_state.steering + t * (curr_state.steering - prev_state.steering)
                
                # 构造新状态（s 直接设为 target_s）
                return State([x, y, heading, speed, steering, target_s])
        
        # 兜底返回最后一个点
        return self.trajectory[-1]
    