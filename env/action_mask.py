import numpy as np
import torch
from shapely.geometry import LineString, Point
from scipy.ndimage.filters import minimum_filter1d

from vehicle_config import *

class ActionMask():
    def __init__(self, VehicleBox=VehicleBox, n_iter=10) -> None:
        print('initializing action mask')
        self.vehicle_box_base = VehicleBox
        self.n_iter = n_iter
        self.action_space = discrete_actions
        self.vehicle_boxes = self.init_vehicle_box() # (42, n_iter, 4, 2)
        self.lidar_num = LIDAR_NUM
        self.lidar_range = LIDAR_RANGE
        self.vehicle_lidar_base = self.get_vehicle_lidar_base()
        self.up_sample_rate = 10
        self.dist_star = self.precompute()
        self.act_rng  = np.random.default_rng()

    def get_vehicle_lidar_base(self, ):
        lidar_base = []
        ORIGIN = Point((0,0))
        for l in range(self.lidar_num):
            lidar_line = LineString(((0,0), (np.cos(l*np.pi/self.lidar_num*2)*self.lidar_range,\
                 np.sin(l*np.pi/self.lidar_num*2)*self.lidar_range)))
            distance = lidar_line.intersection(VehicleBox).distance(ORIGIN)
            lidar_base.append(distance)
        return np.array(lidar_base)

    def _intersect(self, e1:np.ndarray, e2:np.ndarray):
        """
        Calcualte the intersection of 2 group of edges.

        Params:
            e1 (array): coordinates of m edges in shape (m*2*2)
            e2 (array): coordinates of n edges in shape (n*2*2)

        Return:
            intersections (array): coordinates of m*n intersections in shape (m*n*2).
            If 2 edges do not intersects, the corresponding value is set as +np.inf.
        """
        # calculate the params for edges 1
        x1s1, x2s1, y1s1, y2s1 = e1[:, 0, 0].reshape(-1,1), e1[:, 1, 0].reshape(-1,1), e1[:, 0, 1].reshape(-1,1), e1[:, 1, 1].reshape(-1,1)
        a = (y2s1 - y1s1).reshape(-1,1) # (m,1)
        b = (x1s1 - x2s1).reshape(-1,1)
        c = (y1s1*x2s1 - x1s1*y2s1).reshape(-1,1)

        # calculate the params for edges 2
        x1s2, x2s2, y1s2, y2s2 = e2[:, 0, 0].reshape(1,-1), e2[:, 1, 0].reshape(1,-1), e2[:, 0, 1].reshape(1,-1), e2[:, 1, 1].reshape(1,-1)
        d = (y2s2 - y1s2).reshape(1,-1) # (1,n)
        e = (x1s2 - x2s2).reshape(1,-1)
        f = (y1s2*x2s2 - x1s2*y2s2).reshape(1,-1)

        # calculate the intersections by line(instead of edge) 1 & 2
        det = a*e - b*d # (m, n)
        parallel_line_pos = (det==0) # (m, n)
        det[parallel_line_pos] = 1 # temporarily set "1" to avoid "divided by zero"
        raw_x = (b*f - c*e)/det # (m, n)
        raw_y = (c*d - a*f)/det

        # select the true intersections, set the false positive interesections to inf
        tmp_inf = np.inf
        tolerrance = 1e-8
        # the false positive intersections on line L1(not on ray L1)
        raw_x[raw_x>np.maximum(x1s1, x2s1)+tolerrance] = tmp_inf
        raw_x[raw_x<np.minimum(x1s1, x2s1)-tolerrance] = tmp_inf
        raw_y[raw_y>np.maximum(y1s1, y2s1)+tolerrance] = tmp_inf
        raw_y[raw_y<np.minimum(y1s1, y2s1)-tolerrance] = tmp_inf
        # the false positive intersections on line L2(not on edge L2)
        raw_x[raw_x>np.maximum(x1s2, x2s2)+tolerrance] = tmp_inf
        raw_x[raw_x<np.minimum(x1s2, x2s2)-tolerrance] = tmp_inf
        raw_y[raw_y>np.maximum(y1s2, y2s2)+tolerrance] = tmp_inf
        raw_y[raw_y<np.minimum(y1s2, y2s2)-tolerrance] = tmp_inf
        # the (L1, L2) which are parallel
        raw_x[parallel_line_pos] = tmp_inf

        intersections = np.stack([raw_x, raw_y], axis=2) # (m, n, 2)
        # print(e1.shape, e2.shape, intersections.shape)
        assert intersections.shape ==  (e1.shape[0], e2.shape[0], 2)

        return intersections
    
    def init_vehicle_box(self,):
        VehicleBox = self.vehicle_box_base
        vehicle_boxes = []
        x,y,theta = 0,0,0
        actions = np.array(self.action_space)
        radius = 1/(np.tan(actions[:,0])/WHEEL_BASE)
        car_coords = np.array(VehicleBox.coords)[:4] # (4,2)
        car_coords_x = car_coords[:,0].reshape(1,-1)
        car_coords_y = car_coords[:,1].reshape(1,-1) # (1,4)
        Ox = x-radius*np.sin(theta)
        Oy = y+radius*np.cos(theta)
        delta_phi = MASK_STEP_TIME * actions[:,1]/10/radius # (42)  #0.5s仿真时间
        ptheta = theta
        px, py = x,y

        for _ in range(self.n_iter):
            ptheta = ptheta + delta_phi # (42)
            px = Ox + radius*np.sin(ptheta) # (42)
            py = Oy - radius*np.cos(ptheta) # (42)

            # coords transform
            cos_theta = np.cos(ptheta).reshape(-1,1) # (42)
            sin_theta = np.sin(ptheta).reshape(-1,1)
            vehicle_coords_x = cos_theta*car_coords_x - sin_theta*car_coords_y + px.reshape(-1,1) # (42,4)
            vehicle_coords_y = sin_theta*car_coords_x + cos_theta*car_coords_y + py.reshape(-1,1) # (42,4)
            vehicle_coords = np.concatenate((np.expand_dims(vehicle_coords_x, axis=-1), np.expand_dims(vehicle_coords_y, axis=-1)), axis=-1) # (42,4,2)
            vehicle_boxes.append(vehicle_coords)
        
        return np.array(vehicle_boxes).transpose(1,0,2,3)

    def precompute(self,):
        """
        Precomputation of dist_star, which can accelerate the calculatio of action mask.

        Return:
            dist_star(array): shape (lidar_num, n_action, n_iter). 
            dist_star[i,j,k] stands for the minimal distance of obstacle 
            detected by the i-th lidar line for the j-th action to step k iterations without collision.
        """
        lidar_line_idx = np.arange(self.lidar_num)
        max_distance = LIDAR_RANGE*10
        lidar_line_end_x = np.cos(lidar_line_idx/self.lidar_num*2*np.pi)*max_distance
        lidar_line_end_y = np.sin(lidar_line_idx/self.lidar_num*2*np.pi)*max_distance
        lidar_line_end_coords = np.stack([lidar_line_end_x, lidar_line_end_y], axis=1) # (lidar_num, 2)
        origin = np.zeros_like(lidar_line_end_coords)
        lidar_edges = np.stack([origin, lidar_line_end_coords], axis=1) # (lidar_num, 2, 2)

        vehicle_boxes_shifed = np.zeros_like(self.vehicle_boxes)
        vehicle_boxes_shifed[:, :, :-1, :] = self.vehicle_boxes[:, :, 1:, :]
        vehicle_boxes_shifed[:, :, -1, :] = self.vehicle_boxes[:, :, 0, :]
        vehicle_edges = np.stack([vehicle_boxes_shifed, self.vehicle_boxes],axis=3) # (n_action, n_iter, 4, 2, 2)
        vehicle_edges = vehicle_edges.reshape(-1, 2, 2) # (n_action*n_iter*4, 2, 2)，把所有的维度去掉，只留下这一行把所有动作 × 所有时间步 × 车身 4 条边，整理成一个统一的“线段列表”，每个线段是形状 (2,2)，分别是起点和终点坐标。
        intersections = self._intersect(lidar_edges, vehicle_edges) # (lidar_num, n_action*n_iter*4, 2) #120条雷达线，和 42x10x4=1680车身线段求交点
        # intersections[intersections==np.inf] = 0
        intersections = intersections.reshape(self.lidar_num, len(self.action_space), self.n_iter, 4, 2)  # 120个雷达线和 420个车框，每个车框4条线，都应该有个交点，最后的2表示点的维度（x,y）
        intersections = np.linalg.norm(intersections, axis=-1) # (lidar_num, n_action, n_iter, 4条边)
        intersections[intersections==np.inf] = 0 
        dist_star = np.max(intersections, axis=-1) # (lidar_num, n_action, n_iter)  得到的是，单线雷达上，42个动作，一个动作10步，420个车框都可能与单线雷达相交，每一个单线雷达上与每一个车框四条边相交的最大距离
        dist_star = self._linear_interpolate(dist_star, self.up_sample_rate) #这个是在雷达的维度上插值，把雷达从120条，插到1200条，相当于每0.3度检测一次碰撞
        return dist_star
    
    def _linear_interpolate(self, x:np.ndarray, upsample_rate:int=None):
        '''
        Interpolate the input array by linear interpolation on the first axis.

        Parameters:
            `x`: the input array
            `upsample_rate`: the upsample rate
        Returns:
            `y`: y[j,...] = x[j//upsample_rate,...]*((j)%upsample_rate)/upsample_rate + x[(j)//upsample_rate+1,...]*(1-(j%upsample_rate)/upsample_rate)
        '''
        if upsample_rate is None:
            upsample_rate = self.up_sample_rate
        y = np.zeros((x.shape[0]*upsample_rate,) + x.shape[1:])
        x = np.concatenate([x, x[0:1]], axis=0) # assume the input is circular
        j = np.arange(y.shape[0])
        tmp_shape = (y.shape[0],) + (1,)*(len(x.shape)-1)
        y = x[j//upsample_rate,...]*(1-(j%upsample_rate)/upsample_rate).reshape(tmp_shape) +\
            x[(j)//upsample_rate+1,...]*(((j)%upsample_rate)/upsample_rate).reshape(tmp_shape)
        return y


    def get_steps(self, raw_lidar_obs:np.ndarray):
        '''
        raw_lidar_obs: the raw lidar obs which already substract the vehicle base.
        '''
        lidar_obs = np.clip(raw_lidar_obs, 0, 10) + self.vehicle_lidar_base
        dist_obs = self._linear_interpolate(lidar_obs.reshape(-1), self.up_sample_rate).reshape(-1,1,1) # (lidar_num*upsample_rate, 1, 1)
        step_save = np.zeros_like(self.dist_star) # (lidar_num, n_action, n_iter)
        step_save[self.dist_star<=dist_obs] = 1
        step_save[self.dist_star>dist_obs] = 0
        # find the first "0" on the last axis
        max_step = np.argmin(step_save, axis=-1) # (lidar_num, n_action)   返回的是每个动作对应的最小步数，物理理解就是在一条雷达下（单一方向碰撞检测），扫到的每个动作的10步对应的最小值
        max_step[np.sum(step_save, axis=-1) == self.n_iter] = self.n_iter  #对全部无碰撞，max_step返回0的位置做修正，改成满的10步
        
        step_len = np.min(max_step, axis=0) # (n_action)   #取所有动作中最小的允许步数，目的是计算42个动作中，每个动作所允许的最大前进距离。物理理解就是在所有雷达下（所有方向碰撞检测），允许的不碰撞距离。

        step_len = self.post_process(step_len)
        if np.sum(step_len) == 0:
            return np.clip(step_len, 0.01, 1)
        return step_len
    
    def post_process(self, step_len:np.ndarray):
        kernel = 5
        forward_step_len = step_len[:len(step_len)//2]
        backward_step_len = step_len[len(step_len)//2:]
        forward_step_len[0] -= 1
        forward_step_len[-1] -= 1
        backward_step_len[0] -= 1
        backward_step_len[-1] -= 1
        forward_step_len_ = minimum_filter1d(forward_step_len, kernel)
        backward_step_len_ = minimum_filter1d(backward_step_len, kernel)
        out = np.clip(np.concatenate((forward_step_len_, backward_step_len_)), 0, self.n_iter)/self.n_iter
        return out.astype(np.float32, copy=False)
    
    def atanh(self,x):
        return 0.5 * (np.log1p(x) - np.log1p(-x))
    
    
    def choose_action(self, action_mean, action_std, action_mask):

        if isinstance(action_mean, torch.Tensor):
            action_mean = action_mean.cpu().numpy()
            action_std = action_std.cpu().numpy()
        if isinstance(action_mask, torch.Tensor):
            action_mask = action_mask.cpu().numpy()
        if len(action_mean.shape) == 2:
            action_mean = action_mean.squeeze(0)
            action_std = action_std.squeeze(0)
        if len(action_mask.shape) == 2:
            action_mask = action_mask.squeeze(0)

        def calculate_probability(mean, std, values):
            z_scores = (values - mean) / std
            log_probabilities = -0.5 * z_scores ** 2 - np.log((np.sqrt(2 * np.pi) * std))
            return np.sum(np.clip(log_probabilities, -10, 10), axis=1)

        possible_actions = np.array(self.action_space)
        # deal the scaling
        scale_steer = VALID_STEER[1]
        # scale_speed = 1
        scale_speed = VALID_SPEED[1] 
        possible_actions = possible_actions/np.array([scale_steer, scale_speed])
        a = possible_actions
        eps = 0.001
        a = np.clip(a, -1.0 + eps, 1.0 - eps)
        u = self.atanh(a)
        prob = calculate_probability(action_mean, action_std, u)
        # prob = prob - np.sum(np.log(1.0 - a**2 + 1e-6), axis=1)
        logits = prob + np.log(action_mask + 1e-12)   # 软权重进log域
        logits = logits - np.max(logits)              # 防止exp溢出
        exp_logits = np.exp(logits)
        prob_softmax = exp_logits / (np.sum(exp_logits) + 1e-12)
        actions = np.arange(len(possible_actions))
        action_chosen = np.random.choice(actions, p=prob_softmax)
        return possible_actions[action_chosen]
    
    def sample_action_with_mask(self, action_mask):

        possible_actions = np.array(self.action_space)
        # deal the scaling
        scale_steer = VALID_STEER[1]
        # scale_speed = 1
        scale_speed = VALID_SPEED[1] 
        possible_actions = possible_actions/np.array([scale_steer, scale_speed])
        logits =np.log(action_mask + 1e-12)   # 软权重进log域
        logits = logits - np.max(logits)              # 防止exp溢出
        exp_logits = np.exp(logits)
        prob_softmax = exp_logits / (np.sum(exp_logits) + 1e-12)
        actions = np.arange(len(possible_actions))
        action_chosen = self.act_rng.choice(actions, p=prob_softmax)
        return possible_actions[action_chosen]
    


# class ActionFilter:
#     def __init__(self, action_space, scale_steer, scale_speed, eps=1e-12):
#         """
#         action_space: list of [steer_phys, speed_phys] 或者你存的任何二维动作（物理量）
#         scale_*: 用于映射到 norm 空间（和预训练一致）
#         """
#         self.action_space = np.array(action_space, dtype=np.float32)  # (K, 2) 物理动作
#         self.scale = np.array([scale_steer, scale_speed], dtype=np.float32)
#         self.eps = eps

#         # 预先把候选动作映射到 norm 空间（K,2）
#         self.actions_norm = self.action_space / self.scale

#     @staticmethod
#     def _gaussian_logpdf(actions, mean, std):
#         """
#         actions: (K, D)
#         mean,std: (D,)
#         return: (K,)  log N(actions | mean, std) assuming independent dims
#         """
#         z = (actions - mean) / std
#         logp_each_dim = -0.5 * (z ** 2) - np.log(np.sqrt(2.0 * np.pi) * std)
#         logp = np.sum(logp_each_dim, axis=-1)
#         return logp

#     def sample_exec(self, mean, std, action_mask):
#         """
#         mean,std: torch.Tensor or np.ndarray, shape (D,) or (1,D)
#         action_mask: torch.Tensor or np.ndarray, shape (K,) or (1,K)
#         Return:
#           action_norm (np.ndarray, shape (D,))
#           action_index (int)
#           log_prob_exec (float)  # log π_exec(k|s)
#           probs (np.ndarray, shape (K,))  # 可选 debug
#         """
#         # to numpy & squeeze
#         if hasattr(mean, "detach"):
#             mean = mean.detach().cpu().numpy()
#         if hasattr(std, "detach"):
#             std = std.detach().cpu().numpy()
#         if hasattr(action_mask, "detach"):
#             action_mask = action_mask.detach().cpu().numpy()

#         mean = np.squeeze(mean)       # (D,)
#         std  = np.squeeze(std)        # (D,)
#         mask = np.squeeze(action_mask).astype(np.float32)  # (K,)

#         # 防止 std 过小
#         std = np.maximum(std, 1e-6)

#         # 1) log N(a_k | mean, std)
#         logp = self._gaussian_logpdf(self.actions_norm, mean, std)  # (K,)

#         # 2) mask：不可行动作置为 -inf（严格数学意义）
#         logp_masked = np.where(mask > 0.5, logp, -np.inf)

#         # 3) 归一化：log-sum-exp
#         max_logp = np.max(logp_masked)
#         if not np.isfinite(max_logp):
#             # 全不可行，给一个兜底：退化为均匀选（或你自定义）
#             valid_idx = np.where(mask > 0.5)[0]
#             if len(valid_idx) == 0:
#                 # 真的一个都没有：硬兜底选 0
#                 k = 0
#                 return self.actions_norm[k], int(k), float(-np.inf), None
#             k = int(np.random.choice(valid_idx))
#             # 均匀分布 log_prob
#             log_prob_exec = -np.log(len(valid_idx))
#             return self.actions_norm[k], k, float(log_prob_exec), None

#         # logsumexp over valid
#         exp_shifted = np.exp(logp_masked - max_logp)  # invalid -> exp(-inf)=0
#         Z = np.sum(exp_shifted) + self.eps
#         probs = exp_shifted / Z

#         # 4) 采样
#         k = int(np.random.choice(np.arange(len(probs)), p=probs))

#         # 5) 严格一致 log_prob_exec
#         log_prob_exec = np.log(probs[k] + self.eps)

#         return self.actions_norm[k], k, float(log_prob_exec), probs

#     def log_prob_exec(self, mean, std, action_mask, action_index):
#         """
#         给 PPO update 用：在当前 mean,std 下，计算同一个 action_index 的 log π_exec
#         """
#         if hasattr(mean, "detach"):
#             mean = mean.detach().cpu().numpy()
#         if hasattr(std, "detach"):
#             std = std.detach().cpu().numpy()
#         if hasattr(action_mask, "detach"):
#             action_mask = action_mask.detach().cpu().numpy()

#         mean = np.squeeze(mean)
#         std  = np.maximum(np.squeeze(std), 1e-6)
#         mask = np.squeeze(action_mask).astype(np.float32)

#         logp = self._gaussian_logpdf(self.actions_norm, mean, std)
#         logp_masked = np.where(mask > 0.5, logp, -np.inf)

#         max_logp = np.max(logp_masked)
#         if not np.isfinite(max_logp):
#             return float(-np.inf)

#         exp_shifted = np.exp(logp_masked - max_logp)
#         Z = np.sum(exp_shifted) + self.eps
#         probs = exp_shifted / Z

#         return float(np.log(probs[int(action_index)] + self.eps))