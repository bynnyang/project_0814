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
        else:
            self.folder = "val"
        self.dataptpath = os.path.join(self.gnndir, f"{self.folder}_intermediate")

        # self.graph_dataset = torch.load(self.dataptpath, weights_only=False)
        self.graph_dataset = GraphDataset(self.dataptpath)
        # self.peace = self.graph_dataset[0][1]['x']
        self.acb = 1

    def __len__(self):
        return len(self.traj_point)

    # def __getitem__(self, index):
    #     data = {}
    #     keys = ['x', 'y', 'cluster', 'edge_index', 'valid_len', 'time_step_len',
    #             'target_point', 'gt_traj_point', 'gt_traj_point_token', 'fuzzy_target_point']
    #     for key in keys: 
    #         data[key] = []
    #     data['x'] = self.gcndata['x'][index]
    #     data['y'] = self.gcndata['y'][index]
    #     data['cluster'] = self.gcndata['cluster'][index]
    #     data['edge_index'] = self.gcndata['edge_index'][index]
    #     data['valid_len'] = self.gcndata['valid_len'][index]
    #     data['time_step_len'] = self.gcndata['time_step_len'][index]
    #     data["gt_traj_point"] = torch.from_numpy(np.array(self.traj_point[index]))
    #     data['gt_traj_point_token'] = torch.from_numpy(np.array(self.traj_point_token[index]))
    #     data['target_point'] = torch.from_numpy(self.target_point[index])
    #     data["fuzzy_target_point"] = torch.from_numpy(self.fuzzy_target_point[index])

    #     return data

    def __getitem__(self, index):
        data = self.graph_dataset[index]  # Get the graph data for the given index
        data_dict = {
            'x': data.x,
            'y': data.y,
            'cluster': data.cluster,
            'edge_index': data.edge_index,
            'valid_len': data.valid_len,
            'time_step_len': data.time_step_len,
            'gt_traj_point': torch.from_numpy(np.array(self.traj_point[index])),
            'gt_traj_point_token': torch.from_numpy(np.array(self.traj_point_token[index])),
            'target_point': torch.from_numpy(self.target_point[index]),
            'fuzzy_target_point': torch.from_numpy(self.fuzzy_target_point[index])
        }
        return data_dict

    def create_gt_data(self):
        all_tasks = self.get_all_tasks()

        for task_index, task_path in tqdm.tqdm(enumerate(all_tasks)):  # task iteration
            traje_info_obj = TrajectoryInfoParser(task_index, task_path)

            for ego_index in range(0, traje_info_obj.total_frames):  # ego iteration
                ego_pose = traje_info_obj.get_trajectory_point(ego_index)
                world2ego_mat = ego_pose.get_homogeneous_transformation().get_inverse_matrix()
                # create predict point
                predict_point_token_gt, predict_point_gt = self.create_predict_point_gt(traje_info_obj, ego_index, world2ego_mat)
                # create parking goal
                fuzzy_parking_goal, parking_goal = self.create_parking_goal_gt(traje_info_obj, world2ego_mat)

                self.traj_point.append(predict_point_gt)

                self.traj_point_token.append(predict_point_token_gt)
                self.target_point.append(parking_goal)
                self.fuzzy_target_point.append(fuzzy_parking_goal)
                self.task_index_list.append(task_index)

        self.format_transform()

    def create_predict_point_gt(self, traje_info_obj: TrajectoryInfoParser, ego_index: int, world2ego_mat: np.array) -> List[int]:
        predict_point, predict_point_token = [], []
        for predict_index in range(self.cfg.autoregressive_points):  # predict iteration
            predict_stride_index = self.get_clip_stride_index(predict_index = predict_index, 
                                                                start_index=ego_index, 
                                                                max_index=traje_info_obj.total_frames - 1, 
                                                                stride=self.cfg.traj_downsample_stride)
            predict_pose_in_world = traje_info_obj.get_trajectory_point(predict_stride_index)
            predict_pose_in_ego = predict_pose_in_world.get_pose_in_ego(world2ego_mat)
            progress = traje_info_obj.get_progress(predict_stride_index)
            predict_point.append([predict_pose_in_ego.x, predict_pose_in_ego.y])
            tokenize_ret = tokenize_traj_point(predict_pose_in_ego.x, predict_pose_in_ego.y, 
                                                progress, self.cfg.token_nums, self.cfg.xy_max)
            tokenize_ret_process = tokenize_ret[:2] if self.cfg.item_number == 2 else tokenize_ret
            predict_point_token.append(tokenize_ret_process)

            if predict_stride_index == traje_info_obj.total_frames - 1 or predict_index == self.cfg.autoregressive_points - 1:
                break

        predict_point_gt = [item for sublist in predict_point for item in sublist]
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

    def create_parking_goal_gt(self, traje_info_obj: TrajectoryInfoParser, world2ego_mat: np.array):
        candidate_target_pose_in_world = traje_info_obj.get_random_candidate_target_pose()
        candidate_target_pose_in_ego = candidate_target_pose_in_world.get_pose_in_ego(world2ego_mat)
        fuzzy_parking_goal = [candidate_target_pose_in_ego.x, candidate_target_pose_in_ego.y]

        target_pose_in_world = traje_info_obj.get_precise_target_pose()
        target_pose_in_ego = target_pose_in_world.get_pose_in_ego(world2ego_mat)
        parking_goal = [target_pose_in_ego.x, target_pose_in_ego.y]

        return fuzzy_parking_goal, parking_goal
    def get_all_tasks(self):
        all_tasks = []
        train_data_dir = os.path.join(self.root_dir, self.cfg.training_dir)
        val_data_dir = os.path.join(self.root_dir, self.cfg.validation_dir)
        data_dir = train_data_dir if self.is_train == 1 else val_data_dir
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
        return int(np.clip(start_index + stride * (1 + predict_index), 0, max_index))
