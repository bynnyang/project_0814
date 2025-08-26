#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-06-18 22:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com
import torch
from utils.metrics import CustomizedMetric
from pprint import pprint
from typing import List
from loss.traj_point_loss import TokenTrajPointLoss
from loss.traj_point_loss import TrajPointLoss


def get_eval_metric_results(config_obj, model, data_loader, device, global_step):
 
    model.eval()
    traj_point_loss_func = None
    if config_obj.decoder_method == "transformer":
        traj_point_loss_func = TokenTrajPointLoss(config_obj)
    elif config_obj.decoder_method == "gru":
         traj_point_loss_func = TrajPointLoss(config_obj)
    with torch.no_grad():
        loss = 0
        for data in data_loader:
            for key, val in data.items():
                if isinstance(val, torch.Tensor):
                    data[key] = val.to(device)
            out = model(data)
            loss+=traj_point_loss_func(out, data, global_step)
        loss = loss / len(data_loader)
        return loss

def eval_loss():
    raise NotImplementedError("not finished yet")
    model.eval()
    from utils.viz_utils import show_pred_and_gt
    with torch.no_grad():
        accum_loss = .0
        for sample_id, data in enumerate(train_loader):
            data = data.to(device)
            gt = data.y.view(-1, out_channels).to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.mse_loss(out, gt)
            accum_loss += batch_size * loss.item()
            print(f"loss for sample {sample_id}: {loss.item():.3f}")

            for i in range(gt.size(0)):
                pred_y = out[i].numpy().reshape((-1, 2)).cumsum(axis=0)
                y = gt[i].numpy().reshape((-1, 2)).cumsum(axis=0)
                show_pred_and_gt(pred_y, y)
                plt.show()
        print(f"eval overall loss: {accum_loss / len(ds):.3f}")
