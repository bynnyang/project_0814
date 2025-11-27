from torch import nn

from utils.config import Configuration
from utils.metrics import CustomizedMetric
from utils.metrics import CustomizedMetricBev

import torch
from vehicle_config import *
import torch.nn.functional as F

class TokenTrajPointLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super(TokenTrajPointLoss, self).__init__()
        self.cfg = cfg
        self.PAD_token = self.cfg.token_nums + self.cfg.append_token - 1
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.PAD_token)

    def forward(self, pred, data, global_step):
        pre_raw = pred
        pred = pred[:, :-1,:]
        pred_traj_point = pred.reshape(-1, pred.shape[-1])
        gt_traj_point_token = data['gt_traj_point_token'][:, 1:-1].reshape(-1).to(self.cfg.device)
        traj_point_loss = self.ce_loss(pred_traj_point, gt_traj_point_token)
        if (global_step + 1) % 20 == 0:
            val_loss_dict = {}
            val_loss_dict.update({"val_loss": traj_point_loss})
            customized_metric = CustomizedMetric(self.cfg, pre_raw, data)
            val_loss_dict.update(customized_metric.calculate_distance(pre_raw, data))
            print(val_loss_dict)
        return traj_point_loss


# class TrajPointLoss(nn.Module):
#     def __init__(self, cfg: Configuration):
#         super(TrajPointLoss, self).__init__()
#         self.cfg = cfg
#         self.mse_loss = nn.MSELoss()

#     def forward(self, pred, data, global_step):
#         gt = data['gt_traj_point'].view(-1, self.cfg.autoregressive_points, 2)
#         traj_point_loss = self.mse_loss(pred, gt)
#         return traj_point_loss
    

class TrajPointLoss(nn.Module):
    def __init__(self, cfg, lambda_yaw: float = 0.05):
        super().__init__()
        self.cfg = cfg
        self.lambda_yaw = lambda_yaw
        # 初始值随便给一个，真正使用时用 update_weights 覆盖
        self.w_local = 1.0
        self.w_global = 0.0

    def update_weights(self, ss_ratio: float):
        """
        根据当前 scheduled_sampling_ratio 更新 local/global loss 的权重
        ss_ratio: 你 decoder 里的 self.scheduled_sampling_ratio
        """
        # 限制一下范围，防止极端值
        ss_ratio = float(ss_ratio)
        ss_ratio = max(0.0, min(1.0, ss_ratio))

        # 例如：local 跟随 ss_ratio，global 跟随 (1 - ss_ratio)
        self.w_local  = ss_ratio          # 前期 1.0，后期接近 0
        self.w_global = 1.0 - ss_ratio    # 前期 0，后期接近 1

        # 也可以稍微偏向 global 一点，比如：
        # self.w_local  = 0.5 + 0.5 * ss_ratio
        # self.w_global = 0.5 + 0.5 * (1.0 - ss_ratio)


    def kinematic_step(self, prev_point, action):
        """
        prev_point: [B, 4] = (x_norm, y_norm, cos_yaw, sin_yaw)
        action:     [B, 2] = (delta_norm, v_norm)  (tanh 输出 in [-1,1])
        return:     [B, 4] = 下一步的 (x_norm, y_norm, cos_yaw, sin_yaw)
        """
       # 1. 反归一化上一步的状态
        x = prev_point[:, 0] * self.cfg.traj_x_range       # [B]
        y = prev_point[:, 1] * self.cfg.traj_y_range       # [B]
        cos_yaw = prev_point[:, 2]
        sin_yaw = prev_point[:, 3]
        yaw = torch.atan2(sin_yaw, cos_yaw)           # [-pi, pi]

        # 2. 反归一化动作（根据你自己的映射方式调整）
        # 假设 last_step_pred_action ∈ [-1,1]（tanh 输出）
        delta_norm    = action[:, 0]
        v_norm = action[:, 1]

        v     = v_norm * VALID_SPEED[1]      # 映射到 [0, v_max]，你也可以直接 v = v_norm * cfg.v_max
        delta = delta_norm * VALID_STEER[1]            # [-steer_max, steer_max]

        # 3. 单轨运动学模型离散更新
        dt = 0.5
        L  = WHEEL_BASE

        x_next   = x + v * torch.cos(yaw) * dt
        y_next   = y + v * torch.sin(yaw) * dt
        yaw_next = yaw + v / L * torch.tan(delta) * dt

        # 4. wrap yaw 到 [-pi, pi]（可微写法）
        yaw_next = torch.atan2(torch.sin(yaw_next), torch.cos(yaw_next))

        # 5. 再归一化 x,y，并重新编码 yaw 为 cos/sin
        x_next_norm = x_next / self.cfg.traj_x_range
        y_next_norm = y_next / self.cfg.traj_y_range

        cos_next = torch.cos(yaw_next)
        sin_next = torch.sin(yaw_next)

        pred_point = torch.stack([x_next_norm, y_next_norm, cos_next, sin_next], dim=-1)  # [B,4]
        return pred_point

    def rollout_traj(self, init_point, pred_actions):
        """
        从 init_point 开始，用动作序列 rollout 整条预测轨迹。

        init_point:   [B, 4]，一般用 gt_traj[:,0,:]
        pred_actions: [B, T-1, 2]
        return:       [B, T, 4]，包含起点 + T-1 个 rollout 点
        """
        B, Tm1, _ = pred_actions.shape
        traj = []
        curr = init_point              # [B,4]
        traj.append(curr)

        for t in range(Tm1):
            action_t = pred_actions[:, t, :]             # [B,2]
            next_point = self.kinematic_step(curr, action_t)  # [B,4]
            traj.append(next_point)
            curr = next_point

        traj = torch.stack(traj, dim=1)  # [B, T, 4]
        return traj

    def forward(self, pred_actions, gt_traj, global_step):
        """
        pred_actions: [B, T-1, 2]，来自 TrajectoryDecoder.forward 的输出
        gt_traj:      [B, T, 4] = (x_norm, y_norm, cos_yaw, sin_yaw)

        返回总 loss 和一个 dict。
        
        """
        B, T, D = gt_traj.shape
        assert D == 4, "gt_traj 应为 [B,T,4] (x_norm,y_norm,cos,sin)"
        B2, Tm1, A = pred_actions.shape
        assert B2 == B and A == 2, "pred_actions 应为 [B,T-1,2]"
        assert Tm1 == T - 1, "动作长度应为 T-1"

        # ========================
        # 1) local 一步 loss
        # ========================
        # 每个时间步：用 gt 状态 p_t + 动作 a_t 推 p_{t+1}^hat_local
        gt_curr = gt_traj[:, :-1, :]      # [B, T-1, 4] -> p_t^gt
        gt_next = gt_traj[:, 1:, :]       # [B, T-1, 4] -> p_{t+1}^gt

        gt_curr_flat   = gt_curr.reshape(-1, 4)          # [B*(T-1),4]
        actions_flat   = pred_actions.reshape(-1, 2)     # [B*(T-1),2]
        local_next_flat = self.kinematic_step(gt_curr_flat, actions_flat)  # [B*(T-1),4]
        local_next = local_next_flat.view(B, T-1, 4)     # [B,T-1,4]

        pos_loss_local = F.mse_loss(local_next[..., :2], gt_next[..., :2])
        yaw_loss_local = F.mse_loss(local_next[..., 2:], gt_next[..., 2:])
        loss_local = pos_loss_local + self.lambda_yaw * yaw_loss_local

        # ========================
        # 2) global 多步 rollout loss
        # ========================
        init_point = gt_traj[:, 0, :]             # [B,4]
        global_traj = self.rollout_traj(init_point, pred_actions)  # [B,T,4]

        global_next = global_traj[:, 1:, :]       # [B,T-1,4]

        pos_loss_global = F.mse_loss(global_next[..., :2], gt_next[..., :2])
        yaw_loss_global = F.mse_loss(global_next[..., 2:], gt_next[..., 2:])
        loss_global = pos_loss_global + self.lambda_yaw * yaw_loss_global

        # ========================
        # 3) 组合 loss
        # ========================
        total_loss = self.w_local * loss_local + self.w_global * loss_global

        log_dict = {
            "loss": total_loss.detach(),
            "loss_local": loss_local.detach(),
            "loss_global": loss_global.detach(),
            "pos_loss_local": pos_loss_local.detach(),
            "yaw_loss_local": yaw_loss_local.detach(),
            "pos_loss_global": pos_loss_global.detach(),
            "yaw_loss_global": yaw_loss_global.detach(),
            "w_local": self.w_local,
            "w_global": self.w_global
        }

        if (global_step + 1) % 100 == 0:
            print(f"[step {global_step}] loss_dict: {{")
            for k, v in log_dict.items():
                if isinstance(v, torch.Tensor):
                    v_val = v.item()
                else:
                    v_val = float(v)  # numpy.float64 / float 都可以转
                print(f"  {k}: {v_val:.6f}")
            print("}")


        if global_step == 19:
            customized_metric = CustomizedMetricBev(self.cfg, global_next, gt_traj[:,1:,:])
            val_loss_dict = {}
            val_loss_dict.update(customized_metric.calculate_distance(global_next, gt_traj[:,1:,:]))
            print(val_loss_dict)

        return total_loss, log_dict