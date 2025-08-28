import os
from PIL import Image
from typing import List

import numpy as np
import torch.utils.data
import tqdm

from utils.config import Configuration
from utils.trajectory_utils import TrajectoryInfoParser, tokenize_traj_point
from dataset import GraphData
from dataset import GraphDataset
import json
from ruamel.yaml import YAML

class ParkingDataModuleReal(torch.utils.data.Dataset):
    def __init__(self, config: Configuration, is_train):
        super(ParkingDataModuleReal, self).__init__()
        self.cfg = config

        self.BOS_token = self.cfg.token_nums
        self.EOS_token = self.cfg.token_nums + self.cfg.append_token - 2
        self.PAD_token = self.cfg.token_nums + self.cfg.append_token - 1

        self.root_dir = self.cfg.data_dir
        self.is_train = is_train

        self.task_index_list = []

        self.fuzzy_target_point = []
        self.traj_point = []
        self.traj_point_token = []
        self.target_point = []
        self.create_gt_data()
        self.gnndir = "./interm_data"
        if is_train == 1:
            self.folder = "train"
        elif is_train == 0:
            self.folder = "val"
        else:
            self.folder = "test"
        self.dataptpath = os.path.join(self.gnndir, f"{self.folder}_intermediate")

        # self.graph_dataset = torch.load(self.dataptpath, weights_only=False)
        self.graph_dataset = GraphDataset(self.dataptpath)
        # self.peace = self.graph_dataset[0][1]['x']
        if is_train == 1:
            all_x = self.traj_point[:, 0::2]          # 已 flatten，每样本 (30*2,)
            all_y = self.traj_point[:, 1::2]
            self.traj_x_min, self.traj_x_max = all_x.min(), all_x.max()
            self.traj_y_min, self.traj_y_max = all_y.min(), all_y.max()
            # 可选：把统计量写进 config，让验证/测试集直接复用
            config.traj_norm_x_min, config.traj_norm_x_max = self.traj_x_min, self.traj_x_max
            config.traj_norm_y_min, config.traj_norm_y_max = self.traj_y_min, self.traj_y_max

            all_target_point_x = self.target_point[:,0]
            all_target_point_y = self.target_point[:,1]
            all_target_point_theta = self.target_point[:,2]
            self.target_point_x_min, self.target_point_x_max = all_target_point_x.min(), all_target_point_x.max()
            self.target_point_y_min, self.target_point_y_max = all_target_point_y.min(), all_target_point_y.max()
            self.target_point_theta_min, self.target_point_theta_max = all_target_point_theta.min(), all_target_point_theta.max()

            config.target_point_x_min, config.target_point_x_max = self.target_point_x_min, self.target_point_x_max
            config.target_point_y_min, config.target_point_y_max = self.target_point_y_min, self.target_point_y_max
            config.target_point_theta_min, config.target_point_theta_max = self.target_point_theta_min, self.target_point_theta_max




            all_nodes = []
            for g in self.graph_dataset:
                all_nodes.append(g.x[:, :3])      # 取前3列：x,y,heading
            all_nodes = torch.cat(all_nodes, dim=0)

            self.graph_norm_x_min = all_nodes[:, 0].min().item()
            self.graph_norm_x_max = all_nodes[:, 0].max().item()
            self.graph_norm_y_min = all_nodes[:, 1].min().item()
            self.graph_norm_y_max = all_nodes[:, 1].max().item()
            self.graph_norm_theta_min = all_nodes[:, 2].min().item()
            self.graph_norm_theta_max = all_nodes[:, 2].max().item()

            config.graph_norm_x_min, config.graph_norm_x_max = self.graph_norm_x_min, self.graph_norm_x_max
            config.graph_norm_y_min, config.graph_norm_y_max = self.graph_norm_y_min, self.graph_norm_y_max
            config.graph_norm_theta_min, config.graph_norm_theta_max = self.graph_norm_theta_min, self.graph_norm_theta_max
            yaml = YAML()
            yaml.preserve_quotes = True    
            yaml.width = 4096               

            with open("./config/training_real.yaml", "r", encoding="utf-8") as f:
                cfg_dict = yaml.load(f)
            cfg_dict["traj_norm_x_min"] = float(config.traj_norm_x_min)
            cfg_dict["traj_norm_x_max"] = float(config.traj_norm_x_max)
            cfg_dict["traj_norm_y_min"] = float(config.traj_norm_y_min)
            cfg_dict["traj_norm_y_max"] = float(config.traj_norm_y_max)
            cfg_dict["graph_norm_x_min"] = float(config.graph_norm_x_min)
            cfg_dict["graph_norm_x_max"] = float(config.graph_norm_x_max)
            cfg_dict["graph_norm_y_min"] = float(config.graph_norm_y_min)
            cfg_dict["graph_norm_y_max"] = float(config.graph_norm_y_max)
            cfg_dict["graph_norm_theta_min"] = float(config.graph_norm_theta_min)
            cfg_dict["graph_norm_theta_max"] = float(config.graph_norm_theta_max)
            cfg_dict["target_point_x_min"] = float(config.target_point_x_min)
            cfg_dict["target_point_x_max"] = float(config.target_point_x_max)
            cfg_dict["target_point_y_min"] = float(config.target_point_y_min)
            cfg_dict["target_point_y_max"] = float(config.target_point_y_max)
            cfg_dict["target_point_theta_min"] = float(config.target_point_theta_min)
            cfg_dict["target_point_theta_max"] = float(config.target_point_theta_max)

            with open("./config/training_real.yaml", "w", encoding="utf-8") as f:
                yaml.dump(cfg_dict, f)   

        else:
            self.traj_x_min, self.traj_x_max = config.traj_norm_x_min, config.traj_norm_x_max
            self.traj_y_min, self.traj_y_max = config.traj_norm_y_min, config.traj_norm_y_max

            self.graph_norm_x_min, self.graph_norm_x_max = config.graph_norm_x_min, config.graph_norm_x_max
            self.graph_norm_y_min, self.graph_norm_y_max = config.graph_norm_y_min, config.graph_norm_y_max
            self.graph_norm_theta_min, self.graph_norm_theta_max = config.graph_norm_theta_min, config.graph_norm_theta_max   



            self.target_point_x_min, self.target_point_x_max = config.target_point_x_min, config.target_point_x_max
            self.target_point_y_min, self.target_point_y_max = config.target_point_y_min, config.target_point_y_max
            self.target_point_theta_min, self.target_point_theta_max = config.target_point_theta_min, config.target_point_theta_max


    def __len__(self):
        return len(self.traj_point)

    def __getitem__(self, index):
        g: GraphData = self.graph_dataset[index].clone()  # 这是 GraphData 实例

        x = g.x.clone()

        x[:, 0] = (x[:, 0] - self.graph_norm_x_min) / (self.graph_norm_x_max - self.graph_norm_x_min)
        x[:, 1] = (x[:, 1] - self.graph_norm_y_min) / (self.graph_norm_y_max - self.graph_norm_y_min)
        x[:, 2] = (x[:, 2] - self.graph_norm_theta_min) / (self.graph_norm_theta_max - self.graph_norm_theta_min)
        g.x = x
        # 把轨迹/目标等张量挂到图上成为额外属性
        traj = self.traj_point[index].copy()
        traj[0::2] = (traj[0::2] - self.traj_x_min) / (self.traj_x_max - self.traj_x_min)
        traj[1::2] = (traj[1::2] - self.traj_y_min) / (self.traj_y_max - self.traj_y_min)
        
        target_point = self.target_point[index].copy()
        target_point[0] = (target_point[0] - self.target_point_x_min) / (self.target_point_x_max - self.target_point_x_min)
        target_point[1] = (target_point[1] - self.target_point_y_min) / (self.target_point_y_max - self.target_point_y_min)
        target_point[2] = (target_point[2] - self.target_point_theta_min) / (self.target_point_theta_max - self.target_point_theta_min)
        # g.gt_traj_point        = torch.from_numpy(np.array(self.traj_point[index]))
        g.gt_traj_point        = torch.from_numpy(traj.astype(np.float32))
        g.gt_traj_point_token  = torch.from_numpy(np.array(self.traj_point_token[index]))
        g.target_point         = torch.from_numpy(target_point.astype(np.float32))
        # g.fuzzy_target_point   = torch.from_numpy(self.fuzzy_target_point[index])

        return g  

    # def __getitem__(self, index):
    #     data = self.graph_dataset[index]  # Get the graph data for the given index
    #     data_dict = {
    #         'x': data.x,
    #         'y': data.y,
    #         'cluster': data.cluster,
    #         'edge_index': data.edge_index,
    #         'valid_len': data.valid_len,
    #         'time_step_len': data.time_step_len,
    #         'gt_traj_point': torch.from_numpy(np.array(self.traj_point[index])),
    #         'gt_traj_point_token': torch.from_numpy(np.array(self.traj_point_token[index])),
    #         'target_point': torch.from_numpy(self.target_point[index]),
    #         'fuzzy_target_point': torch.from_numpy(self.fuzzy_target_point[index])
    #     }
    #     return data_dict
    def save_measurements(self, measurements, ego_index, filename, cnt, measurement_tag="measurements"):
        measurements_path = os.path.join(filename, str(ego_index))
        os.makedirs(measurements_path, exist_ok=True)
        measurements_path_final = os.path.join(measurements_path, measurement_tag)
        os.makedirs(measurements_path_final, exist_ok=True)
        measurements_filename = os.path.join(measurements_path_final, "{:04d}.json".format(cnt))
        if measurements == None:
            return
        with open(measurements_filename, 'w') as json_file:
            json.dump(measurements, json_file, indent=4)

    def parser_measurements_pred(self, pred_point,switch):
        pose_ret = {
            'x':pred_point[0],
            'y':pred_point[1],
            'dir': switch,
        }

        return pose_ret

    def create_gt_data(self):
        all_tasks = self.get_all_tasks()

        for task_index, task_path in tqdm.tqdm(enumerate(all_tasks)):  # task iteration
            traje_info_obj = TrajectoryInfoParser(task_index, task_path)
            judge_ego_pose = traje_info_obj.get_trajectory_point(0)
            judge_world2ego_mat = judge_ego_pose.get_homogeneous_transformation().get_inverse_matrix()
            finally_pose_in_ego = traje_info_obj.trajectory_list[-1].get_pose_in_ego(judge_world2ego_mat)
            judge_pose_in_ego = traje_info_obj.trajectory_list[-50].get_pose_in_ego(judge_world2ego_mat)
            switch_side = -1.0 if finally_pose_in_ego.y > judge_pose_in_ego.y else 1.0
            for ego_index in range(0, traje_info_obj.total_frames):  # ego iteration
                ego_pose = traje_info_obj.get_trajectory_point(ego_index)
                world2ego_mat = ego_pose.get_homogeneous_transformation().get_inverse_matrix()
                # create predict point
                predict_point_token_gt, predict_point_gt = self.create_predict_point_gt(traje_info_obj, ego_index, world2ego_mat, switch_side, task_path)
                # create parking goal
                fuzzy_parking_goal, parking_goal = self.create_parking_goal_gt(traje_info_obj, world2ego_mat, switch_side)

                self.traj_point.append(predict_point_gt)

                self.traj_point_token.append(predict_point_token_gt)
                self.target_point.append(parking_goal)
                self.fuzzy_target_point.append(fuzzy_parking_goal)
                self.task_index_list.append(task_index)

        self.format_transform()

    def create_predict_point_gt(self, traje_info_obj: TrajectoryInfoParser, ego_index: int, world2ego_mat: np.array, switch_side: float, filename: str) -> List[int]:
        predict_point, predict_point_token = [], []
        for predict_index in range(self.cfg.autoregressive_points):  # predict iteration
            ds = 0.1 * predict_index + traje_info_obj.get_trajectory_point(ego_index).s
            predict_stride_index = self.get_clip_stride_index(predict_index = predict_index, 
                                                                start_index=ego_index, 
                                                                max_index=traje_info_obj.total_frames - 1, 
                                                                stride=self.cfg.traj_downsample_stride)
            predict_pose_in_world = traje_info_obj.get_trajectory_point_by_s(ego_index, ds)
            predict_pose_in_ego = predict_pose_in_world.get_pose_in_ego(world2ego_mat)
            predict_pose_in_ego.y = predict_pose_in_ego.y * switch_side
            progress = traje_info_obj.get_progress(predict_stride_index)
            predict_point.append([predict_pose_in_ego.x, predict_pose_in_ego.y])
            tokenize_ret = tokenize_traj_point(predict_pose_in_ego.x, predict_pose_in_ego.y, 
                                                progress, self.cfg.token_nums, self.cfg.xy_max)
            tokenize_ret_process = tokenize_ret[:2] if self.cfg.item_number == 2 else tokenize_ret
            predict_point_token.append(tokenize_ret_process)

            if predict_pose_in_world.s == traje_info_obj.get_trajectory_point(traje_info_obj.total_frames - 1).s or predict_index == self.cfg.autoregressive_points - 1:
                break

        predict_point_gt = [item for sublist in predict_point for item in sublist]
        # for index, point in enumerate(predict_point):
        #     point_record = self.parser_measurements_pred(point, switch_side)
        #     self.save_measurements(point_record, ego_index, filename,index,"pred")
        append_pad_num = self.cfg.autoregressive_points * self.cfg.item_number - len(predict_point_gt)
        assert append_pad_num >= 0
        predict_point_gt = predict_point_gt + (append_pad_num // 2) * [predict_point_gt[-2], predict_point_gt[-1]]

        predict_point_token_gt = [item for sublist in predict_point_token for item in sublist]
        predict_point_token_gt.insert(0, self.BOS_token)
        predict_point_token_gt.append(self.EOS_token)
        predict_point_token_gt.append(self.PAD_token)
        append_pad_num = self.cfg.autoregressive_points * self.cfg.item_number + self.cfg.append_token - len(predict_point_token_gt)
        assert append_pad_num >= 0
        predict_point_token_gt = predict_point_token_gt + append_pad_num * [self.PAD_token]
        return predict_point_token_gt, predict_point_gt
    
    def get_safe_yaw(slef, yaw):
        if yaw <= -180:
            yaw += 360
        if yaw > 180:
            yaw -= 360
        yaw = yaw / 180.0 * 3.14
        return yaw

    def create_parking_goal_gt(self, traje_info_obj: TrajectoryInfoParser, world2ego_mat: np.array, switch_side: float):
        candidate_target_pose_in_world = traje_info_obj.get_random_candidate_target_pose()
        candidate_target_pose_in_ego = candidate_target_pose_in_world.get_pose_in_ego(world2ego_mat)
        candidate_target_pose_in_ego.y = candidate_target_pose_in_ego.y * switch_side
        fuzzy_parking_goal = [candidate_target_pose_in_ego.x, candidate_target_pose_in_ego.y]

        target_pose_in_world = traje_info_obj.get_precise_target_pose()
        target_pose_in_ego = target_pose_in_world.get_pose_in_ego(world2ego_mat)
        target_pose_in_ego.y = target_pose_in_ego.y * switch_side
        yaw_refine = self.get_safe_yaw(target_pose_in_ego.yaw) * switch_side
        parking_goal = [target_pose_in_ego.x, target_pose_in_ego.y, yaw_refine]

        return fuzzy_parking_goal, parking_goal
    def get_all_tasks(self):
        all_tasks = []
        train_data_dir = os.path.join(self.root_dir, self.cfg.training_dir)
        val_data_dir = os.path.join(self.root_dir, self.cfg.validation_dir)
        test_data_dir = os.path.join(self.root_dir, self.cfg.test_dir)
        if self.is_train == 1:
            data_dir = train_data_dir
        elif self.is_train == 0:
            data_dir = val_data_dir
        else:
            data_dir = test_data_dir
        for scene_item in os.listdir(data_dir):
            scene_path = os.path.join(data_dir, scene_item)
            all_tasks.append(scene_path)
        return all_tasks

    def format_transform(self):
        self.traj_point = np.array(self.traj_point).astype(np.float32)
        self.traj_point_token = np.array(self.traj_point_token).astype(np.int64)
        self.target_point = np.array(self.target_point).astype(np.float32)
        self.fuzzy_target_point = np.array(self.fuzzy_target_point).astype(np.float32)
        self.task_index_list = np.array(self.task_index_list).astype(np.int64)

    def get_clip_stride_index(self, predict_index, start_index, max_index, stride):
        return int(np.clip(start_index + stride * (0 + predict_index), 0, max_index))
