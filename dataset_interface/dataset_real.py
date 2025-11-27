import os
from PIL import Image
from typing import List

import numpy as np
import torch.utils.data
import tqdm

from utils.config import Configuration
from utils.trajectory_utils import TrajectoryInfoParser, tokenize_traj_point
from utils.cluster_utils import ClusterInfoParser
from dataset import GraphData
from dataset import GraphDataset
from dataset_interface.bev_render import BevRender
import copy
# from dataset_interface.bev_render import BevRender
import json
from ruamel.yaml import YAML
from utils.pose_utils import CustomizePose
from utils.vec2d import Vec2d
from utils.box2d import Box2d
from utils.box2d import LineSegment
import cv2
from env.car_parking_base import CarParking
from shapely.geometry import LineString
from env.vehicle import State
import shutil

class Obs_Processor():
    def __init__(self) -> None:
        self.downsample_rate = 1
        self.n_channels = 3

    def process_img(self, img):
        processed_img = self.change_bg_color(img)
        H, W = img.shape[:2]
        processed_img = cv2.resize(processed_img, (W//self.downsample_rate, H//self.downsample_rate))
        # plt.imshow(processed_img)  # 直接显示
        # plt.savefig('processed_img.png')  # 保存到当前目录
        processed_img = processed_img.transpose(2,0,1)
        processed_img = processed_img/255.0

        return processed_img

    def change_bg_color(self, img):
        processed_img = img.copy()
        # bg_pos = img==BG_COLOR[:3]
        # bg_pos = (np.sum(bg_pos,axis=-1) == 3)
        # processed_img[bg_pos] = (0,0,0)
        return processed_img


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
        self.history_trajector_vcs = []
        self.render_cnn = BevRender()
        self.car_parking_date = CarParking()
        self.img_processor = Obs_Processor()
        self.img_cnn_path_list = []
        self.lidar = []
        self.action_mask = []
        self.target = []
       
        self.gnndir = "./interm_data"
        if is_train == 1:
            self.folder = "train"
            self.e2e_dataset = os.path.join("./e2e_dataset", "train", "e2e_dataset.pt")
        elif is_train == 0:
            self.folder = "val"
            self.e2e_dataset = os.path.join("./e2e_dataset", "val", "e2e_dataset.pt")
        else:
            self.folder = "test"
            self.e2e_dataset = os.path.join("./e2e_dataset", "test", "e2e_dataset.pt")
        self.dataptpath = os.path.join(self.gnndir, f"{self.folder}_intermediate")

        if os.path.exists(self.e2e_dataset):
            # Load the dataset.pt file
            self.load_dataset()
        else:
            # Run create_gt_data to generate the data and save it
            self.create_gt_data()
            self.save_dataset()

        # self.graph_dataset = GraphDataset(self.dataptpath)
        if is_train == 1:
            if self.cfg.item_number == 2:
                all_x = self.traj_point[:, 0::2]          # 已 flatten，每样本 (30*2,)
                all_y = self.traj_point[:, 1::2]
            else:
                all_x = self.traj_point[:, 0::3]          # 已 flatten，每样本 (30*2,)
                all_y = self.traj_point[:, 1::3]
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


            # all_nodes = []
            # for g in self.graph_dataset:
            #     all_nodes.append(g.x[:, :3])      # 取前3列：x,y,heading
            # all_nodes = torch.cat(all_nodes, dim=0)

            # self.graph_norm_x_min = all_nodes[:, 0].min().item()
            # self.graph_norm_x_max = all_nodes[:, 0].max().item()
            # self.graph_norm_y_min = all_nodes[:, 1].min().item()
            # self.graph_norm_y_max = all_nodes[:, 1].max().item()
            # self.graph_norm_theta_min = all_nodes[:, 2].min().item()
            # self.graph_norm_theta_max = all_nodes[:, 2].max().item()

            # config.graph_norm_x_min, config.graph_norm_x_max = self.graph_norm_x_min, self.graph_norm_x_max
            # config.graph_norm_y_min, config.graph_norm_y_max = self.graph_norm_y_min, self.graph_norm_y_max
            # config.graph_norm_theta_min, config.graph_norm_theta_max = self.graph_norm_theta_min, self.graph_norm_theta_max
            yaml = YAML()
            yaml.preserve_quotes = True    
            yaml.width = 4096               

            with open("./config/training_real.yaml", "r", encoding="utf-8") as f:
                cfg_dict = yaml.load(f)
            cfg_dict["traj_norm_x_min"] = float(config.traj_norm_x_min)
            cfg_dict["traj_norm_x_max"] = float(config.traj_norm_x_max)
            cfg_dict["traj_norm_y_min"] = float(config.traj_norm_y_min)
            cfg_dict["traj_norm_y_max"] = float(config.traj_norm_y_max)
            # cfg_dict["graph_norm_x_min"] = float(config.graph_norm_x_min)
            # cfg_dict["graph_norm_x_max"] = float(config.graph_norm_x_max)
            # cfg_dict["graph_norm_y_min"] = float(config.graph_norm_y_min)
            # cfg_dict["graph_norm_y_max"] = float(config.graph_norm_y_max)
            # cfg_dict["graph_norm_theta_min"] = float(config.graph_norm_theta_min)
            # cfg_dict["graph_norm_theta_max"] = float(config.graph_norm_theta_max)
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

            # self.graph_norm_x_min, self.graph_norm_x_max = config.graph_norm_x_min, config.graph_norm_x_max
            # self.graph_norm_y_min, self.graph_norm_y_max = config.graph_norm_y_min, config.graph_norm_y_max
            # self.graph_norm_theta_min, self.graph_norm_theta_max = config.graph_norm_theta_min, config.graph_norm_theta_max   



            self.target_point_x_min, self.target_point_x_max = config.target_point_x_min, config.target_point_x_max
            self.target_point_y_min, self.target_point_y_max = config.target_point_y_min, config.target_point_y_max
            self.target_point_theta_min, self.target_point_theta_max = config.target_point_theta_min, config.target_point_theta_max


    def load_dataset(self):
        # Load the dataset.pt file
        data = torch.load(self.e2e_dataset, weights_only=False)
        self.task_index_list = data['task_index_list']
        self.fuzzy_target_point = data['fuzzy_target_point']
        self.traj_point = data['traj_point']
        self.traj_point_token = data['traj_point_token']
        self.target_point = data['target_point']
        self.img_cnn_path_list = data['img_cnn_path_list']
        self.lidar = data['lidar']
        self.action_mask = data['action_mask']
        self.target = data['target']

    def save_dataset(self):
        # Save the dataset to dataset.pt file
        data = {
            'task_index_list': self.task_index_list,
            'fuzzy_target_point': self.fuzzy_target_point,
            'traj_point': self.traj_point,
            'traj_point_token': self.traj_point_token,
            'target_point': self.target_point,
            'img_cnn_path_list':self.img_cnn_path_list,
            'lidar':self.lidar,
            'action_mask':self.action_mask,
            'target':self.target
        }
        torch.save(data, self.e2e_dataset)

    def __len__(self):
        return len(self.traj_point)

    # def __getitem__(self, index):
    #     g: GraphData = self.graph_dataset[index].clone()  # 这是 GraphData 实例

    #     x = g.x.clone()

    #     x[:, 0] = (x[:, 0] - self.graph_norm_x_min) / (self.graph_norm_x_max - self.graph_norm_x_min)
    #     x[:, 1] = (x[:, 1] - self.graph_norm_y_min) / (self.graph_norm_y_max - self.graph_norm_y_min)
    #     x[:, 2] = (x[:, 2] - self.graph_norm_theta_min) / (self.graph_norm_theta_max - self.graph_norm_theta_min)
    #     g.x = x
    #     # 把轨迹/目标等张量挂到图上成为额外属性
    #     traj = self.traj_point[index].copy()
    #     traj[0::2] = (traj[0::2] - self.traj_x_min) / (self.traj_x_max - self.traj_x_min)
    #     traj[1::2] = (traj[1::2] - self.traj_y_min) / (self.traj_y_max - self.traj_y_min)
        
    #     target_point = self.target_point[index].copy()
    #     target_point[0] = (target_point[0] - self.target_point_x_min) / (self.target_point_x_max - self.target_point_x_min)
    #     target_point[1] = (target_point[1] - self.target_point_y_min) / (self.target_point_y_max - self.target_point_y_min)
    #     target_point[2] = (target_point[2] - self.target_point_theta_min) / (self.target_point_theta_max - self.target_point_theta_min)
    #     # g.gt_traj_point        = torch.from_numpy(np.array(self.traj_point[index]))
    #     # g.gt_traj_point        = torch.from_numpy(traj.astype(np.float32))   增加了heading的预测后，没有进行修正，先不要用
    #     g.gt_traj_point_token  = torch.from_numpy(np.array(self.traj_point_token[index]))
    #     g.target_point         = torch.from_numpy(target_point.astype(np.float32))
    #     # g.fuzzy_target_point   = torch.from_numpy(self.fuzzy_target_point[index])

    #     return g  

    def __getitem__(self, index):
 
    # ---- 1. 轨迹点归一化 + cos/sin 编码 ----
        traj = self.traj_point[index].copy().astype(np.float32)   # [T,3] = (x,y,yaw)

        # x,y 归一化到 [-1,1]
        traj_x = np.clip(traj[:, 0] / self.cfg.traj_x_range, -1.0, 1.0)
        traj_y = np.clip(traj[:, 1] / self.cfg.traj_y_range, -1.0, 1.0)
        traj_yaw = traj[:, 2]   # 假设是弧度

        # [T,4] = (x_norm, y_norm, cos(yaw), sin(yaw))
        gt_traj_point_np = np.stack(
            [traj_x, traj_y, np.cos(traj_yaw), np.sin(traj_yaw)],
            axis=-1
        )   # [T,4]
        gt_traj_point = torch.from_numpy(gt_traj_point_np.astype(np.float32))

        target_feature = self.target[index].copy()
        target_feature[0] = target_feature[0] / np.sqrt(self.cfg.traj_x_range**2 + self.cfg.traj_y_range**2)
        target_feature[0] = np.clip(target_feature[0], -1.0, 1.0)
        target_feature        = torch.from_numpy(target_feature.astype(np.float32))
        
        target_point_raw = self.target_point[index].copy().astype(np.float32)  # [N_tp,3] 通常 N_tp=1
        tp_x = np.clip(target_point_raw[0] / self.cfg.traj_x_range, -1.0, 1.0)
        tp_y = np.clip(target_point_raw[1] / self.cfg.traj_y_range, -1.0, 1.0)
        tp_yaw = target_point_raw[2]
        target_point_np = np.stack(
            [tp_x, tp_y, np.cos(tp_yaw), np.sin(tp_yaw)],
            axis=-1
        )   # [N_tp,4] 一般是 [1,4]
        target_point = torch.from_numpy(target_point_np.astype(np.float32))

        gt_traj_point_token  = torch.from_numpy(np.array(self.traj_point_token[index]))

        img_path = self.img_cnn_path_list[index]
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        processed_img = self.img_processor.process_img(img)
        processed_img         = torch.from_numpy(processed_img.astype(np.float32))

        action_mask = self.action_mask[index].copy()
        action_mask = torch.from_numpy(action_mask.astype(np.float32))

        lidar = self.lidar[index].copy()
        lidar = lidar / self.cfg.traj_x_range
        lidar = torch.from_numpy(lidar.astype(np.float32))


        data = {
            "image": processed_img,               # Tensor [3,H,W]
            "gt_traj_point": gt_traj_point,                # [T,4] = (x_norm,y_norm,cos,sin)
            "target_point": target_point,         # [N_tp,4] 一般 [1,4]
            "gt_traj_point_token": gt_traj_point_token,
            "lidar": lidar,
            "action_mask": action_mask,
            "target": target_feature
        }
        return data
    def save_measurements(self, measurements, ego_index, filename, cnt, measurement_tag="measurements"):
        measurements_path = os.path.join(filename, str(ego_index))
        os.makedirs(measurements_path, exist_ok=True)
        measurements_path_final = os.path.join(measurements_path, measurement_tag)
        if cnt == 0 and measurement_tag == "pred":
            if os.path.exists(measurements_path_final):
                shutil.rmtree(measurements_path_final)   # 删除整个 pred 文件夹
            os.makedirs(measurements_path_final, exist_ok=True)  # 重建空文件夹
        else:
            os.makedirs(measurements_path_final, exist_ok=True)
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
            'yaw':pred_point[2],
            'dir': switch,
        }

        return pose_ret
    
    def parser_measurements_target(self, target_point, switch):
        pose_ret = {
            'x':target_point[0],
            'y':target_point[1],
            'theta':target_point[2],
            'dir': switch,
        }

        return pose_ret
    
    def parser_clusters_pred(self, cluster_frame_in_vcs):
        clusters_list =[]
        for clusters in cluster_frame_in_vcs:
            my_dict = {
                "id": clusters["id"],
                "p0": {
                    "x": clusters["p0"].x,
                    "y": clusters["p0"].y
                },
                "p1": {
                    "x": clusters["p1"].x,
                    "y": clusters["p1"].y
                }
            }
            clusters_list.append(my_dict)
        return clusters_list
    
    def create_clusters_info_vcs(self, cluster_info_obj: ClusterInfoParser, world2ego_mat: np.array, ego_index, switch_side: float, filename):
        cluster_frame_in_world = cluster_info_obj.get_clusters(ego_index)
        cluster_frame_in_vcs =[]  
        cluster_dict_template = {
                "id": None,
                "p0": {},
                "p1": {}
            }
        for index, each_cluster in enumerate(cluster_frame_in_world):
            if each_cluster["p0"].x == each_cluster["p1"].x and each_cluster["p0"].y == each_cluster["p1"].y:
                continue
            each_cluster_vcs = copy.deepcopy(cluster_dict_template)
            each_cluster_vcs["id"] = each_cluster["id"] * switch_side
            each_cluster_vcs["p0"] = each_cluster["p0"].get_pose_in_ego(world2ego_mat)
            each_cluster_vcs["p0"].y = each_cluster_vcs["p0"].y * switch_side
            each_cluster_vcs["p1"] = each_cluster["p1"].get_pose_in_ego(world2ego_mat)
            each_cluster_vcs["p1"].y = each_cluster_vcs["p1"].y * switch_side
            cluster_frame_in_vcs.append(each_cluster_vcs)
        cluster_frame_in_switch = self.parser_clusters_pred(cluster_frame_in_vcs)
        self.save_measurements(cluster_frame_in_switch, ego_index, filename, 0, "pre_cluster")

        return cluster_frame_in_vcs
    
    def judge_clusters_info_vcs(self, cluster_info_obj: ClusterInfoParser, ego_index, ego_pose: CustomizePose, filename):
        vehicle_width = 1.781 
        vehicle_length = 3.99 
        vehicle_rear_overhang = 0.704
        center_offset = vehicle_length / 2.0 - vehicle_rear_overhang
        center_pose = Vec2d(ego_pose.x, ego_pose.y)
        vehicle_center = center_pose + Vec2d.create_unit_vec2d(ego_pose.yaw/180*3.14) * center_offset

        # 2. 创建 Box2d
        vehicle_box = Box2d(vehicle_center, ego_pose.yaw/180*3.14, vehicle_length, vehicle_width)
        for i in range(0, cluster_info_obj.total_frames):
            cluster_frame_in_world = cluster_info_obj.get_clusters(i)
            for index, each_cluster in enumerate(cluster_frame_in_world):
                x_0 = each_cluster["p0"].x
                y_0 = each_cluster["p0"].y
                x_1 = each_cluster["p1"].x
                y_1 = each_cluster["p1"].y
                cluster_line = LineSegment(Vec2d(x_0,y_0), Vec2d(x_1, y_1))
                overlap = vehicle_box.overlap_with_segment(cluster_line)
                if(overlap):
                    print("cluster",i)
                    print("filename", filename)
                    print("ego_index", ego_index)
    def create_history_point(self, traje_info_obj: TrajectoryInfoParser, ego_index: int, world2ego_mat: np.array, switch_side: float):
        history_trajector_vcs = []
        history_traj_len = 0
        if ego_index == 0:
            return history_trajector_vcs, history_traj_len
        for i in range(1, 13):  # predict iteration
            ds = -0.5 * i + traje_info_obj.get_trajectory_point(ego_index).s
            if(ds < 0):
                return history_trajector_vcs, history_traj_len
            history_pose_in_world = traje_info_obj.get_trajectory_point_by_s_dec(ego_index, ds)
            history_pose_in_ego = history_pose_in_world.get_pose_in_ego(world2ego_mat)
            history_pose_in_ego.y = history_pose_in_ego.y * switch_side
            history_pose_in_ego.yaw = self.get_safe_yaw(history_pose_in_ego.yaw) * switch_side
            history_trajector_vcs.append(history_pose_in_ego)
            history_trajector_vcs = history_trajector_vcs[::-1]
            history_traj_len = len(history_trajector_vcs)

        return history_trajector_vcs, history_traj_len
    
    def convert_clusters_to_geometry(sefl, cluster_frame_in_vcs):
        clusters: List[LineString] = []

        for cl in cluster_frame_in_vcs:
            p0 = cl["p0"]
            p1 = cl["p1"]

            # Shapely LineString
            line = LineString([(p0.x, p0.y), (p1.x, p1.y)])
            clusters.append(line)

        return clusters
#########

#自车坐标系下，车头朝向为x轴，右手坐标系，左侧为y轴

    """
    CNN版本新增 cluster的自车坐标系转换, 泊车目标的转换，历史轨迹点收集，渲染图生成


    """
######
    def create_gt_data(self):
        all_tasks = self.get_all_tasks()

        for task_index, task_path in tqdm.tqdm(enumerate(all_tasks)):  # task iteration
            name = os.path.splitext(os.path.basename(task_path))[0]
            if name.startswith("e2e_"):
                continue
            traje_info_obj = TrajectoryInfoParser(task_index, task_path)
            cluster_info_obj = ClusterInfoParser(task_index, task_path)
            judge_ego_pose = traje_info_obj.get_trajectory_point(0)
            judge_world2ego_mat = judge_ego_pose.get_homogeneous_transformation().get_inverse_matrix()
            finally_pose_in_ego = traje_info_obj.trajectory_list[-1].get_pose_in_ego(judge_world2ego_mat)
            judge_pose_in_ego = traje_info_obj.trajectory_list[-50].get_pose_in_ego(judge_world2ego_mat)
            switch_side = -1.0 if finally_pose_in_ego.y > judge_pose_in_ego.y else 1.0
            for ego_index in range(0, traje_info_obj.total_frames):  # ego iteration
                ego_pose = traje_info_obj.get_trajectory_point(ego_index)
                world2ego_mat = ego_pose.get_homogeneous_transformation().get_inverse_matrix()
                # create predict point
                # predict_point_token_gt, predict_point_gt = self.create_predict_point_gt(traje_info_obj, ego_index, world2ego_mat, switch_side, task_path, 0)
                # create parking goal
                fuzzy_parking_goal, parking_goal = self.create_parking_goal_gt(traje_info_obj, world2ego_mat, switch_side)

                # target_pose = self.parser_measurements_target(parking_goal, switch_side)
                # self.save_measurements(target_pose, ego_index, task_path, 0, "pre_target")

                cluster_frame_info_vcs = self.create_clusters_info_vcs(cluster_info_obj, world2ego_mat, ego_index, switch_side, task_path)

                # self.judge_clusters_info_vcs(cluster_info_obj, ego_index, ego_pose, str(task_path))  # 脏数据筛查 判断自车的所有轨迹有没有与cluster碰撞的

                history_trajector_vcs, history_traj_len = self.create_history_point(traje_info_obj, ego_index, world2ego_mat, switch_side)

                obervattion_clusters = self.convert_clusters_to_geometry(cluster_frame_info_vcs)

                if ego_index == 0:
                    i = 0
                    while i < self.cfg.autoregressive_points - 1:
                        predict_point_token_gt, predict_point_gt = self.create_predict_point_gt(traje_info_obj, ego_index, world2ego_mat, switch_side, task_path, i)
                        init_state = State([0.0,0.0,0.0])
                
                        observation = self.car_parking_date.calc_obervation_feature(init_state, parking_goal, obervattion_clusters)
                        target_pose = self.parser_measurements_target(parking_goal, switch_side)
                        self.save_measurements(target_pose, ego_index - i, task_path, 0, "pre_target")
                        cluster_frame_in_switch = self.parser_clusters_pred(cluster_frame_info_vcs)
                        self.save_measurements(cluster_frame_in_switch, ego_index - i, task_path, 0, "pre_cluster")

                        start_pose = traje_info_obj.get_trajectory_point(0)
                        start_pose_vcs = start_pose.get_pose_in_ego(world2ego_mat)
                        start_pose_vcs.y = start_pose_vcs.y * switch_side
                        start_pose_vcs.yaw = self.get_safe_yaw(start_pose_vcs.yaw) * switch_side
                        # imge_cnn = self.render_cnn.render(start_pose_vcs, history_trajector_vcs, history_traj_len, parking_goal, cluster_frame_info_vcs, ego_index - i, task_path)
                        measurements_path = os.path.join(task_path, str(ego_index - i))
                        measurements_path_final = os.path.join(measurements_path, "cnn.png")

                        self.traj_point.append(predict_point_gt)

                        self.traj_point_token.append(predict_point_token_gt)
                        self.target_point.append(parking_goal)
                        self.fuzzy_target_point.append(fuzzy_parking_goal)
                        self.task_index_list.append(task_index)
                        self.img_cnn_path_list.append(measurements_path_final)
                        self.lidar.append(observation['lidar'])
                        self.action_mask.append(observation['action_mask'])
                        self.target.append(observation['target'])
                        i = i + 1
                else:
                    predict_point_token_gt, predict_point_gt = self.create_predict_point_gt(traje_info_obj, ego_index, world2ego_mat, switch_side, task_path, 0)

                    init_state = State([0.0,0.0,0.0])
                    
                    observation = self.car_parking_date.calc_obervation_feature(init_state, parking_goal, obervattion_clusters)

                    start_pose = traje_info_obj.get_trajectory_point(0)
                    start_pose_vcs = start_pose.get_pose_in_ego(world2ego_mat)
                    start_pose_vcs.y = start_pose_vcs.y * switch_side
                    start_pose_vcs.yaw = self.get_safe_yaw(start_pose_vcs.yaw) * switch_side
                    # imge_cnn = self.render_cnn.render(start_pose_vcs, history_trajector_vcs, history_traj_len, parking_goal, cluster_frame_info_vcs, ego_index, task_path)
                    measurements_path = os.path.join(task_path, str(ego_index))
                    measurements_path_final = os.path.join(measurements_path, "cnn.png")

                    self.traj_point.append(predict_point_gt)

                    self.traj_point_token.append(predict_point_token_gt)
                    self.target_point.append(parking_goal)
                    self.fuzzy_target_point.append(fuzzy_parking_goal)
                    self.task_index_list.append(task_index)
                    self.img_cnn_path_list.append(measurements_path_final)
                    self.lidar.append(observation['lidar'])
                    self.action_mask.append(observation['action_mask'])
                    self.target.append(observation['target'])

        self.format_transform()

    def create_predict_point_gt(self, traje_info_obj: TrajectoryInfoParser, ego_index: int, world2ego_mat: np.array, switch_side: float, filename: str, index_i) -> List[int]:
        predict_point, predict_point_token = [], []
        for predict_index in range(self.cfg.autoregressive_points - index_i):  # predict iteration
            ds = 0.5 * predict_index + traje_info_obj.get_trajectory_point(ego_index).s
            predict_stride_index = self.get_clip_stride_index(predict_index = predict_index, 
                                                                start_index=ego_index, 
                                                                max_index=traje_info_obj.total_frames - 1, 
                                                                stride=self.cfg.traj_downsample_stride)
            predict_pose_in_world = traje_info_obj.get_trajectory_point_by_s(ego_index, ds)
            predict_pose_in_ego = predict_pose_in_world.get_pose_in_ego(world2ego_mat)
            predict_pose_in_ego.y = predict_pose_in_ego.y * switch_side
            predict_pose_in_ego.yaw = self.get_safe_yaw(predict_pose_in_ego.yaw) * switch_side
            progress = traje_info_obj.get_progress(predict_stride_index)
            if self.cfg.item_number == 2:
                predict_point.append([predict_pose_in_ego.x, predict_pose_in_ego.y])
            else:
                predict_point.append([predict_pose_in_ego.x, predict_pose_in_ego.y, predict_pose_in_ego.yaw])
            tokenize_ret = tokenize_traj_point(predict_pose_in_ego.x, predict_pose_in_ego.y, 
                                                predict_pose_in_ego.yaw, self.cfg.token_nums, self.cfg.xy_max)
            tokenize_ret_process = tokenize_ret[:2] if self.cfg.item_number == 2 else tokenize_ret
            predict_point_token.append(tokenize_ret_process)

            if predict_pose_in_world.s == traje_info_obj.get_trajectory_point(traje_info_obj.total_frames - 1).s or predict_index == self.cfg.autoregressive_points - 1:
                break

        predict_point_gt = [item for sublist in predict_point for item in sublist]
        for index, point in enumerate(predict_point):
            point_record = self.parser_measurements_pred(point, switch_side)
            self.save_measurements(point_record, ego_index - index_i, filename, index, "pred")
        append_pad_num = self.cfg.autoregressive_points * self.cfg.item_number - len(predict_point_gt)
        assert append_pad_num >= 0
        if index_i == 0:
            if self.cfg.item_number == 2:
                predict_point_gt = predict_point_gt + (append_pad_num // 2) * [predict_point_gt[-2], predict_point_gt[-1]]
                predict_point_gt = np.array(predict_point_gt, dtype=np.float32).reshape(-1, 2)
            else:
                predict_point_gt = predict_point_gt + (append_pad_num // 3) * [predict_point_gt[-3], predict_point_gt[-2], predict_point_gt[-1]]
                predict_point_gt = np.array(predict_point_gt, dtype=np.float32).reshape(-1, 3)
        else:
            if self.cfg.item_number == 2:
                predict_point_gt = (append_pad_num // 2) * [predict_point_gt[0], predict_point_gt[1]] + predict_point_gt
                predict_point_gt = np.array(predict_point_gt, dtype=np.float32).reshape(-1, 2)
            else:
                predict_point_gt = (append_pad_num // 3) * [predict_point_gt[0], predict_point_gt[1], predict_point_gt[2]] + predict_point_gt
                predict_point_gt = np.array(predict_point_gt, dtype=np.float32).reshape(-1, 3)
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
        self.lidar = np.array(self.lidar).astype(np.float32)
        self.action_mask = np.array(self.action_mask).astype(np.float32)
        self.target = np.array(self.target).astype(np.float32)

    def get_clip_stride_index(self, predict_index, start_index, max_index, stride):
        return int(np.clip(start_index + stride * (0 + predict_index), 0, max_index))
