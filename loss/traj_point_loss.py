from torch import nn

from utils.config import Configuration
from utils.metrics import CustomizedMetric


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


class TrajPointLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super(TrajPointLoss, self).__init__()
        self.cfg = cfg
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, data, global_step):
        gt = data['gt_traj_point'].view(-1, self.cfg.autoregressive_points, 2)
        traj_point_loss = self.mse_loss(pred, gt)
        return traj_point_loss