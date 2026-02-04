from vehicle_config import *
import numpy as np
class RsPlanner(object):
    def __init__(self, step_ratio:float) -> None:
        self.route = None
        self.actions = []
        self.step_ratio = step_ratio

    def reset(self,):
        self.route = None
        self.actions.clear()
    
    def set_rs_path(self, rs_path):
        action_type = {'L':1, 'S':0, 'R':-1}
        self.route = rs_path
        step_ratio = self.step_ratio
        action_list = []
        for i in range(len(rs_path.ctypes)):
            steer = action_type[rs_path.ctypes[i]]
            step_len = rs_path.lengths[i]/step_ratio
            action_list.append([steer, step_len])

        # divide the action
        filtered_actions = []
        speed_scale = VALID_SPEED[1]
        for action in action_list:
            action[0] = action[0] * (0.496 / VALID_STEER[1])
            if abs(action[1])<1 and abs(action[1])>1e-3:
                filtered_actions.append(action)
            elif action[1]>1:
                while action[1]>1:
                    filtered_actions.append([action[0], 1 / speed_scale])
                    action[1] -= 1
                if abs(action[1])>1e-3:
                    filtered_actions.append([action[0], action[1] / speed_scale])
            elif action[1]<-1:
                while action[1]<-1:
                    filtered_actions.append([action[0], -1 / speed_scale])
                    action[1] += 1
                if abs(action[1])>1e-3:
                    filtered_actions.append([action[0], action[1] / speed_scale])
        
        self.actions = filtered_actions

    def get_action(self, ):
        action = self.actions.pop(0)
        if len(self.actions) == 0 and self.route is not None:
            self.reset()
        return action

class ParkingAgent(object):
    def __init__(
        self, rl_agent, planner=None,
    ) -> None:
        self.agent = rl_agent
        self.planner = planner
        self._last_use_rs = None
        self._state_start_frame = None

    def __getattr__(self, name: str):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.agent, name)
    
    def reset(self,):
        if self.planner is not None:
            self.planner.reset()

    def set_planner_path(self, path=None, forced=False):
        if self.planner is None:
            return
        if path is not None and (forced or self.planner.route is None):
            self.planner.set_rs_path(path)

    @property
    def executing_rs(self,):
        return not (self.planner is None or self.planner.route is None)
    
    def get_log_prob(self, obs, action, predict_pose_list):
        return self.agent.get_log_prob(obs, action, predict_pose_list)

    def choose_action(self, obs, predict_pose_list):
        '''
        Get the fused decision from the planner and the agent.
        The action is clipped to the range of the safe action space using action mask.

        Params:
            obs(dict): the observation of the environment

        Return:
            action(np.array): the fused decision
            other: the other information, such as the log_prob of the action in case of PPO
        '''
        if not self.executing_rs:
            return self.agent.choose_action(obs, predict_pose_list)
        else:
            action = self.planner.get_action()
            log_prob = self.agent.get_log_prob(obs, action, predict_pose_list)
            return action, log_prob
        

    def sample_action_with_mask(self, obs):
    
            return self.agent.sample_action_with_mask(obs)
        
    def choose_action_eval(self, obs, predict_pose_list):
        '''
        Get the fused decision from the planner and the agent.
        The action is clipped to the range of the safe action space using action mask.

        Params:
            obs(dict): the observation of the environment

        Return:
            action(np.array): the fused decision
            other: the other information, such as the log_prob of the action in case of PPO
        '''
        if not self.executing_rs:
            return self.agent.choose_action_eval(obs, predict_pose_list)
        else:
            action = self.planner.get_action()
            log_prob = self.agent.get_log_prob(obs, action, predict_pose_list)
            return action, log_prob
        
    def get_action(self, obs, predict_pose_list):
        '''
        Get the fused decision from the planner and the agent.

        Params:
            obs(dict): the observation of the environment

        Return:
            action(np.array): the fused decision
            other: the other information, such as the log_prob of the action in case of PPO
        '''
        if not self.executing_rs:
            return self.agent.get_action(obs, predict_pose_list)
        else:
            action = self.planner.get_action()
            log_prob = 0.0
            return action, log_prob
        
    def vcs_action_step(self, prev_point, action):
        """
        prev_point: [B, 4] = (x_norm, y_norm, cos_yaw, sin_yaw)
        action:     [B, 2] = (delta_norm, v_norm)  (tanh 输出)
        return:     [B, 4] = 下一步的 (x_norm, y_norm, cos_yaw, sin_yaw)
        """

        # 1. 反归一化上一步的状态
        x = prev_point[0] * TRAJXRANGE       # [B]
        y = prev_point[1] * TRAJYRANGE       # [B]
        cos_yaw = prev_point[2]
        sin_yaw = prev_point[3]
        yaw = np.arctan2(sin_yaw, cos_yaw)           # [-pi, pi]

        # 2. 反归一化动作（根据你自己的映射方式调整）
        # 假设 last_step_pred_action ∈ [-1,1]（tanh 输出
        delta_norm  = action[0]
        v_norm = action[1]

        v     = v_norm * VALID_SPEED[1]      # 映射到 [-v_max, v_max]，你也可以直接 v = v_norm * cfg.v_max
        delta = delta_norm * VALID_STEER[1]  

        # 3. 单轨运动学模型离散更新
        dt = STEP_TIME_AND_LENGHT
        L  = WHEEL_BASE

        x_next   = x + v * np.cos(yaw) * dt
        y_next   = y + v * np.sin(yaw) * dt
        yaw_next = yaw + v / L * np.tan(delta) * dt

        # 4. wrap yaw 到 [-pi, pi]（可微写法）
        yaw_next = np.arctan2(np.sin(yaw_next), np.cos(yaw_next))

        # 5. 再归一化 x,y，并重新编码 yaw 为 cos/sin
        x_next_norm = x_next / TRAJXRANGE
        y_next_norm = y_next / TRAJYRANGE

        cos_next = np.cos(yaw_next)
        sin_next = np.sin(yaw_next)

        pred_point = np.stack([x_next_norm, y_next_norm, cos_next, sin_next], axis=-1).reshape(4,)  # [B,4]
        return pred_point
            
    def push_memory(self, experience):
        self.agent.push_memory(experience)

    def update(self, step, i):
        return self.agent.update(step, i)
    
    def save(self, *args, **kwargs ):
        self.agent.save(*args, **kwargs )

    def load(self, *args, **kwargs ):
        self.agent.load(*args, **kwargs)