from collections import OrderedDict
import numpy as np
import threading
import time

import torch
import torchvision
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion

from model_interface.model.parking_model_real import ParkingModelReal
from utils.config import InferenceConfiguration
from utils.pose_utils import PoseFlow, pose2customize_pose
from utils.traj_post_process import calculate_tangent, fitting_curve
from utils.trajectory_utils import detokenize_traj_point
import os
import matplotlib.pyplot as plt
from utils.pose_utils import CustomizePose
from torch_geometric.data import Data, Batch
from dataset import GraphData
from dataset_interface.dataset_real import ParkingDataModuleReal
from torch.utils.data import DataLoader
from torch.fx import symbolic_trace

# ========== 1. Encoder部分 ==========
class EncoderONNXWrapper(torch.nn.Module):
    def __init__(self, model, cfg, device):
        super().__init__()
        self.subgraph = model.subgraph
        self.self_atten_layer = model.self_atten_layer
        self.cfg = cfg
        self.device = device

    def forward(self, x, cluster, edge_index, valid_len, time_step_len):
        x, cluster, edge_index, valid_len, time_step_len = [
            t.to(torch.int64) if t.dtype == torch.int32 else t
            for t in [x, cluster, edge_index, valid_len, time_step_len]
        ]
        x_new = x.squeeze(0)
        x_new = x_new.squeeze(0)
        edge_index_new = edge_index.squeeze(0)
        edge_index_new = edge_index_new.squeeze(0)
        cluster_new = cluster.reshape(-1)
        valid_len_new = valid_len.reshape(-1)
        time_step_new = time_step_len.reshape(-1)
        dummy_input = Data(
            x = x_new,
            cluster = cluster_new,
            edge_index = edge_index_new,
            valid_len= valid_len_new,
            time_step_len = time_step_new
        )
        # dummy_input = Data(
        #     x = x,
        #     cluster = cluster,
        #     edge_index = edge_index,
        #     valid_len= valid_len,
        #     time_step_len = time_step_len
        # )
        dummy_input.to(self.device)
        time_step_len =dummy_input["time_step_len"][0]
        valid_lens = dummy_input["valid_len"]
        sub_graph_out = self.subgraph(dummy_input)
        x = sub_graph_out.view(-1, time_step_len, self.cfg.subgraph_width)
        out = self.self_atten_layer(x, valid_lens)
        return out  # encoder输出 global_feat
    

# ========== 2. TrajInputEmbedding部分 ==========
class TrajInputEmbeddingONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.target_point_encoder = model.target_point_encoder

    def forward(self, target_point):
        return self.target_point_encoder(target_point)


# ========== 3. Decoder部分 ==========
class DecoderONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.trajectory_decoder = model.trajectory_decoder

    def forward(self, encoder_out, point_out, gt_traj_point_token):
        if gt_traj_point_token.dtype == torch.int64:
            gt_traj_point_token = gt_traj_point_token.to(torch.int32)
        return self.trajectory_decoder(encoder_out, point_out, gt_traj_point_token)

class ParkingModelONNXWrapper(torch.nn.Module):
    def __init__(self, model, cfg, device):
        super().__init__()
        self.subgraph = model.subgraph
        self.self_atten_layer = model.self_atten_layer
        self.target_point_encoder = model.target_point_encoder
        self.trajectory_decoder = model.trajectory_decoder
        self.cfg =cfg
        self.device = device

    def forward(self, x, cluster, edge_index, valid_len, time_step_len, target_point, gt_traj_point_token):
        x, cluster, edge_index, valid_len, time_step_len = [
            t.to(torch.int64) if t.dtype == torch.int32 else t
            for t in [x, cluster, edge_index, valid_len, time_step_len]
        ]
        x_new = x.squeeze(0)
        x_new = x_new.squeeze(0)
        edge_index_new = edge_index.squeeze(0)
        edge_index_new = edge_index_new.squeeze(0)
        cluster_new = cluster.reshape(-1)
        valid_len_new = valid_len.reshape(-1)
        time_step_new = time_step_len.reshape(-1)
        dummy_input = Data(
            x = x_new,
            cluster = cluster_new,
            edge_index = edge_index_new,
            valid_len= valid_len_new,
            time_step_len = time_step_new
        )
        target_point_new = target_point.squeeze(0)
        target_point_new = target_point_new.squeeze(0)
        gt_traj_point_token_new = gt_traj_point_token.squeeze(0)
        gt_traj_point_token_new = gt_traj_point_token_new.squeeze(0)
        dummy_input.target_point = target_point_new
        dummy_input.gt_traj_point_token = gt_traj_point_token_new
        dummy_input.to(self.device)
        time_step_len =dummy_input["time_step_len"][0]
        valid_lens = dummy_input["valid_len"]
        sub_graph_out = self.subgraph(dummy_input)
        x = sub_graph_out.view(-1, time_step_len, self.cfg.subgraph_width)
        out = self.self_atten_layer(x, valid_lens)
        point_out = self.target_point_encoder(dummy_input["target_point"].to(self.cfg.device))
        # Auto Regressive Decoder
        autoregressive_point = dummy_input['gt_traj_point_token'].to(self.cfg.device) # During inference, we regard BOS as gt_traj_point_token.
        predict_token_num=self.cfg.item_number*self.cfg.autoregressive_points
        for _ in range(predict_token_num):
            pred_traj_point = self.trajectory_decoder.predict(out, point_out, autoregressive_point)
            autoregressive_point = torch.cat([autoregressive_point, pred_traj_point], dim=1)
        pred_traj_point_update = autoregressive_point[0][1:]
        pred_traj_point_update = self.remove_invalid_content(pred_traj_point_update)

        delta_predicts = detokenize_traj_point(pred_traj_point_update, self.cfg.token_nums, 
                                            self.cfg.item_number, 
                                            self.cfg.xy_max)
        return delta_predicts
    
    def remove_invalid_content(self, pred_traj_point_update):
        finish_index = -1
        index_tensor = torch.where(pred_traj_point_update == self.cfg.token_nums + self.cfg.append_token - 2)[0]
        if len(index_tensor):
            finish_index = torch.where(pred_traj_point_update == self.EOS_token)[0][0].item()
            finish_index = finish_index - finish_index % self.cfg.item_number
        if finish_index != -1:
            pred_traj_point_update = pred_traj_point_update[: finish_index]
        return pred_traj_point_update


class ParkingInferenceModuleReal:
    def __init__(self, inference_cfg: InferenceConfiguration):
        self.cfg = inference_cfg
        self.model = None
        self.device = None

        self.load_model(self.cfg.model_ckpt_path)

        # self.export_onnx(self.cfg.model_ckpt_path, self.cfg)
        
        self.BOS_token = self.cfg.train_meta_config.token_nums

        self.traj_start_point_info = Pose()
        self.traj_start_point_lock = threading.Lock()
        
        self.EOS_token = self.cfg.train_meta_config.token_nums + self.cfg.train_meta_config.append_token - 2

        self.predict_points_record = None
        
        self.traj_yaw_path_record = None

        self.cnt = 17
        self.pre = None
        self.cur = None

    def predict(self, test_data, cnt, judge_ego2world_mat = None, mode="service"):
        if mode == "topic":
            self.pub_path(test_data, cnt)
        elif mode == "simulation":
            delta_predicts_map, traj_yaw_path_map= self.pub_simulation(test_data, judge_ego2world_mat)
            return delta_predicts_map, traj_yaw_path_map
        else:
            assert print("Can't support %s mode!".format(mode))

    def pub_path(self, test_data, cnt):
 
       
        filename = "./e2e_dataset/test/20250622T101821"
        start_token = [self.BOS_token]
        test_data["gt_traj_point_token"][0,:] = torch.tensor([start_token], dtype=torch.int64).to(self.device)
        test_data["gt_traj_point_token"] = test_data["gt_traj_point_token"][:,0:1]
        self.model = self.model.to(device=self.device)
        self.model.eval()
        self.export_onnx(self.cfg.model_ckpt_path, self.cfg, test_data)
        delta_predicts = self.inference(test_data)
        delta_predicts = np.array(delta_predicts, dtype=np.float32)
        # delta_predicts[:,0::2] = delta_predicts[:,0::2] * (self.cfg.train_meta_config.traj_norm_x_max - self.cfg.train_meta_config.traj_norm_x_min) + self.cfg.train_meta_config.traj_norm_x_min
        # delta_predicts[:,1::2] = delta_predicts[:,1::2] * (self.cfg.train_meta_config.traj_norm_y_max - self.cfg.train_meta_config.traj_norm_y_min) + self.cfg.train_meta_config.traj_norm_y_min 
        # delta_predicts = fitting_curve(delta_predicts, num_points=self.cfg.train_meta_config.autoregressive_points, item_number=self.cfg.train_meta_config.item_number)
        # traj_yaw_path = calculate_tangent(np.array(delta_predicts)[:, :2], mode="five_point")

        x_coords = []
        y_coords = []
        theta_coords = []
        for point_item in delta_predicts:
            if self.cfg.train_meta_config.item_number == 2:
                x, y= point_item
                x_coords.append(x)
                y_coords.append(y)
            elif self.cfg.train_meta_config.item_number == 3:
                # x, y, progress_bar = point_item
                # if abs(progress_bar) < 1 - self.cfg.progress_threshold:
                #     break
                x, y, theta= point_item
                x_coords.append(x)
                y_coords.append(y)
                theta_coords.append(theta / 180 * 3.14)
        save_folder = os.path.join(filename,str(cnt),"test")
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, "test")
        plt.ioff()
        plt.figure(figsize=(6, 6))
        plt.scatter(x_coords, y_coords, color='blue', s = 2, label='Coordinates')
        # L = 0.08   # 箭头长度，按你的坐标系调
        # dx = L * np.cos(theta_coords)
        # dy = L * np.sin(theta_coords)
        # plt.quiver(x_coords, y_coords, dx, dy,
        #             color='red', width=0.003, scale=1, scale_units='xy', angles='xy')
        plt.title(f'Scene {save_path}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()    

    def pub_simulation(self, test_data, judge_ego2world_mat):
 
       
        start_token = [self.BOS_token]
        test_data["gt_traj_point_token"] = torch.tensor([start_token], dtype=torch.int64).to(self.device)
        test_data["gt_traj_point_token"] = test_data["gt_traj_point_token"][:,0:1]

        self.model.eval()
        delta_predicts = self.inference(test_data)
        delta_predicts = np.array(delta_predicts, dtype=np.float32)
        delta_predicts = fitting_curve(delta_predicts, num_points=delta_predicts.shape[0], item_number=self.cfg.train_meta_config.item_number)
        traj_yaw_path = calculate_tangent(np.array(delta_predicts)[:, :2], mode="five_point")
        points = np.array(delta_predicts)[:, :2]
        max_index = np.argmax(points[:, 0])
        max_points = points.shape[0]
        delta_predicts_map = []
        traj_yaw_path_map = []
        for index in range(len(delta_predicts)):
            vcs_point = CustomizePose(x=delta_predicts[index][0], y=delta_predicts[index][1], z=0.0, roll=0.0, yaw=traj_yaw_path[index], pitch=0.0)
            map_point = vcs_point.get_pose_in_world(judge_ego2world_mat)
            delta_predicts_map.append([map_point.x, map_point.y])
            traj_yaw_path_map.append(map_point.yaw / 180 * 3.14)
        if (max_index == 0 or max_index == (max_points -1) or  max_index > 5) and self.cnt > 15 and max_points > 5:
            self.predict_points_record = delta_predicts_map
            self.traj_yaw_path_record = traj_yaw_path_map
            self.cur = True
        else:
            self.predict_points_record = self.predict_points_record[1:]
            self.traj_yaw_path_record = self.traj_yaw_path_record[1:]
            self.cur = False
        if self.pre == True and self.cur == False:
            self.cnt = 0
        self.pre = self.cur
        self.cnt = self.cnt + 1
        if self.cnt > 17:
            self.cnt = 17
        return self.predict_points_record, self.traj_yaw_path_record

    def inference(self, data):
        delta_predicts = []
        with torch.no_grad():
            if self.cfg.train_meta_config.decoder_method == "transformer":
                delta_predicts = self.inference_transformer(data)
            elif self.cfg.train_meta_config.decoder_method == "gru":
                delta_predicts = self.inference_gru(data)
            else:
                raise ValueError(f"Don't support decoder_method '{self.cfg.decoder_method}'!")
        delta_predicts = delta_predicts.tolist()
        return delta_predicts

    def inference_transformer(self, data):
        pred_traj_point= self.model.predict_transformer(data, predict_token_num=self.cfg.train_meta_config.item_number*self.cfg.train_meta_config.autoregressive_points)
        pred_traj_point_update = pred_traj_point[0][1:]
        pred_traj_point_update = self.remove_invalid_content(pred_traj_point_update)

        delta_predicts = detokenize_traj_point(pred_traj_point_update, self.cfg.train_meta_config.token_nums, 
                                            self.cfg.train_meta_config.item_number, 
                                            self.cfg.train_meta_config.xy_max)

        return delta_predicts

    def inference_gru(self, data):
        delta_predicts = self.model.predict_gru(data)

        return delta_predicts

    def remove_invalid_content(self, pred_traj_point_update):
        finish_index = -1
        index_tensor = torch.where(pred_traj_point_update == self.cfg.train_meta_config.token_nums + self.cfg.train_meta_config.append_token - 2)[0]
        if len(index_tensor):
            finish_index = torch.where(pred_traj_point_update == self.EOS_token)[0][0].item()
            finish_index = finish_index - finish_index % self.cfg.train_meta_config.item_number
        if finish_index != -1:
            pred_traj_point_update = pred_traj_point_update[: finish_index]
        return pred_traj_point_update

    def get_posestamp_info(self, x, y, yaw):
        predict_pose = PoseStamped()
        pose_flow_obj = PoseFlow(att_input=[yaw, 0, 0], type="euler", deg_or_rad="deg")
        quad = pose_flow_obj.get_quad()
        predict_pose.pose.position = Point(x=x, y=y, z=0.0)
        predict_pose.pose.orientation = Quaternion(x=quad.x, y=quad.y,z=quad.z, w=quad.w)
        return predict_pose


    def load_model(self, parking_pth_path):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = ParkingModelReal(self.cfg.train_meta_config)

        ckpt = torch.load(parking_pth_path, map_location='cpu')
        # state_dict = OrderedDict([(k.replace('parking_model.', ''), v) for k, v in ckpt['state_dict'].items()])
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def export_onnx(self, checkpoint_path, cfg, data_set, export_dir="./onnx_exports", date="1013"):
        """
        将 PyTorch 模型导出为 ONNX 格式。
        """
        os.makedirs(export_dir, exist_ok=True)

        self.model_onnx = ParkingModelReal(self.cfg.train_meta_config)
        
        # 加载权重  
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model_onnx.load_state_dict(checkpoint['state_dict'])
        self.model_onnx = self.model_onnx.to(device=self.device)
        self.model_onnx.eval()

        # 构造一个示例输入（必须与模型 forward 输入匹配）
        # 假设模型输入是一个 GraphData 或 Batch 图结构：
        # 可根据 ParkingModelReal 的 forward 接口修改

        # export_onnx_model = ParkingModelONNXWrapper(self.model_onnx, self.cfg, self.device)
        # export_onnx_model = export_onnx_model.to(device=self.device)
        # export_path = os.path.join(export_dir, f"model_export_{date}.onnx")


        encoder_wrapper = EncoderONNXWrapper(self.model_onnx, self.cfg.train_meta_config, self.device).eval()
        encoder_wrapper = encoder_wrapper.to(device=self.device)
        
        export_path_EncoderONNXWrapper = os.path.join(export_dir, f"EncoderONNXWrapper_{date}.onnx")

        allcoder_wrapper = ParkingModelONNXWrapper(self.model_onnx, self.cfg.train_meta_config, self.device).eval()
        allcoder_wrapper = allcoder_wrapper.to(device=self.device)
        
        export_path_AllcoderONNXWrapper = os.path.join(export_dir, f"AllcoderONNXWrapper_{date}.onnx")

        num_nodes = 82         # 假设总节点数
        num_edges = 4
        batch_size= 1
        # xyz = torch.rand(num_nodes, 3) * 2 + 1  # [-1, 1] 范围随机值
        # polyline_id = torch.arange(num_nodes).unsqueeze(1).float()  # [0, 1, 2, ..., num_nodes-1]
        # x = torch.cat([xyz, polyline_id], dim=1)
        # y = torch.randn(batch_size)
        x = torch.tensor([[0.5103,  0.5121,  0.5000,  0.0000],
                          [0.4554,  0.5122,  0.5000,  0.0000],
                          [0.4553,  0.3777,  0.5000,  0.0000],
                          [0.5102,  0.3776,  0.5000,  0.0000],
                          [0.4702,  0.4856,  0.4995,  1.0000],
                          [0.7858,  0.2795,  0.2488,  2.0000],
                          [0.7855,  0.2357,  0.2488,  2.0000],
                          [0.7812,  0.5448,  0.5777,  3.0000],
                          [0.7903,  0.5502,  0.5777,  3.0000],
                          [0.7903,  0.5502,  0.5417,  4.0000],
                          [0.8028,  0.5539,  0.5417,  4.0000],
                          [0.7820,  0.5535,  0.4456,  5.0000],
                          [0.7903,  0.5502,  0.4456,  5.0000],
                          [0.8028,  0.5539,  0.5548,  6.0000],
                          [0.8142,  0.5584,  0.5548,  6.0000],
                          [0.5847,  0.5876,  0.7494,  7.0000],
                          [0.5847,  0.5941,  0.7494,  7.0000],
                          [0.5766,  0.5625,  0.6929,  8.0000],
                          [0.5842,  0.5850,  0.6929,  8.0000],
                          [0.5961,  0.3259,  0.4015,  9.0000],
                          [0.6011,  0.3220,  0.4015,  9.0000],
                          [0.5621,  0.5270,  0.4944, 10.0000],
                          [0.5648,  0.5269,  0.4944, 10.0000],
                          [0.5376,  0.5269,  0.5008, 11.0000],
                          [0.5590,  0.5270,  0.5008, 11.0000],
                          [0.4744,  0.5663,  0.4990, 12.0000],
                          [0.5240,  0.5659,  0.4990, 12.0000],
                          [0.5622,  0.2818,  0.2476, 13.0000],
                          [0.5617,  0.2452,  0.2476, 13.0000],
                          [0.5795,  0.3522,  0.3585, 14.0000],
                          [0.5872,  0.3417,  0.3585, 14.0000],
                          [0.5882,  0.3117,  0.1861, 15.0000],
                          [0.5853,  0.3043,  0.1861, 15.0000],
                          [0.5897,  0.3148,  0.7666, 16.0000],
                          [0.5875,  0.3387,  0.7666, 16.0000],
                          [0.7806,  0.5323,  0.2465, 17.0000],
                          [0.7804,  0.5217,  0.2465, 17.0000],
                          [0.5840,  0.3031,  0.1484, 18.0000],
                          [0.5761,  0.2914,  0.1484, 18.0000],
                          [0.5016,  0.3506,  0.5070, 19.0000],
                          [0.5157,  0.3513,  0.5070, 19.0000],
                          [0.8940,  0.5448,  0.4987, 20.0000],
                          [0.9473,  0.5443,  0.4987, 20.0000],
                          [0.5357,  0.2725,  0.4975, 21.0000],
                          [0.5636,  0.2720,  0.4975, 21.0000],
                          [0.5636,  0.2720,  0.2475, 22.0000],
                          [0.5632,  0.2461,  0.2475, 22.0000],
                          [0.5632,  0.2461,  0.9975, 23.0000],
                          [0.5353,  0.2466,  0.9975, 23.0000],
                          [0.5353,  0.2466,  0.7475, 24.0000],
                          [0.5357,  0.2725,  0.7475, 24.0000],
                          [0.8114,  0.2425,  0.9965, 25.0000],
                          [0.7861,  0.2431,  0.9965, 25.0000],
                          [0.7861,  0.2431,  0.7465, 26.0000],
                          [0.7867,  0.2690,  0.7465, 26.0000],
                          [0.7867,  0.2690,  0.4965, 27.0000],
                          [0.8119,  0.2684,  0.4965, 27.0000],
                          [0.8119,  0.2684,  0.2465, 28.0000],
                          [0.8114,  0.2425,  0.2465, 28.0000],
                          [0.4343,  0.5193,  0.2459, 29.0000],
                          [0.4328,  0.4519,  0.2459, 29.0000],
                          [0.4828,  0.4855,  0.5000, 30.0000],
                          [0.4828,  0.4855,  0.5000, 31.0000],
                          [0.4828,  0.4855,  0.5000, 32.0000],
                          [0.4828,  0.4855,  0.5000, 33.0000],
                          [0.4828,  0.4855,  0.5000, 34.0000],
                          [0.4828,  0.4855,  0.5000, 35.0000],
                          [0.4828,  0.4855,  0.5000, 36.0000],
                          [0.4828,  0.4855,  0.5000, 37.0000],
                          [0.4828,  0.4855,  0.5000, 38.0000],
                          [0.4828,  0.4855,  0.5000, 39.0000],
                          [0.4828,  0.4855,  0.5000, 40.0000],
                          [0.4828,  0.4855,  0.5000, 41.0000],
                          [0.4828,  0.4855,  0.5000, 42.0000],
                          [0.4828,  0.4855,  0.5000, 43.0000],
                          [0.4828,  0.4855,  0.5000, 44.0000],
                          [0.4828,  0.4855,  0.5000, 45.0000],
                          [0.4828,  0.4855,  0.5000, 46.0000],
                          [0.4828,  0.4855,  0.5000, 47.0000],
                          [0.4828,  0.4855,  0.5000, 48.0000],
                          [0.4828,  0.4855,  0.5000, 49.0000],
                          [0.4828,  0.4855,  0.5000, 50.0000]], dtype=torch.float32)
        polyline_id = torch.tensor([ 0,  0,  0,  0,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,
         8,  9,  9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17,
        17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26,
        26, 27, 27, 28, 28, 29, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50])
        # x = torch.cat([xyz, polyline_id.unsqueeze(1).float()], dim=1)
        x = x[None, None, :, :]
        # cluster = torch.arange(num_nodes)
        # cluster = torch.cat([
        # torch.full((n,), i, dtype=torch.long, device=self.device)   # ✅ fill_value 是数字 i
        #     for i, n in enumerate(num_nodes)])
        cluster = polyline_id.to(torch.int32)
        cluster = cluster[None, None, None, :]
        # cluster = polyline_id.to(torch.int32).squeeze(1)      # 聚类信息
        # edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.int32)  # 边索引
        edge_index = torch.tensor([[0,  1,  2,  3,  1,  2,  3,  0,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,
                                    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                                    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
                                   [1,  2,  3,  0,  0,  1,  2,  3,  4,  6,  5,  8,  7, 10,  9, 12, 11, 14,
                                    13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25, 28, 27, 30, 29, 32,
                                    31, 34, 33, 36, 35, 38, 37, 40, 39, 42, 41, 44, 43, 46, 45, 48, 47, 50,
                                    49, 52, 51, 54, 53, 56, 55, 58, 57, 60, 59]], dtype=torch.int32)
        # edge_index = torch.arange(num_edges, dtype=torch.int32).unsqueeze(0).repeat(2, 1)
        edge_index = edge_index[None, None, :, :]
        valid_len = torch.tensor([29], dtype=torch.int32) 
        valid_len = valid_len[None, None, None, :]      
        time_step_len = torch.tensor([51], dtype=torch.int32)
        time_step_len = time_step_len[None, None, None, :]       
        start_token = [self.BOS_token]
        target_point = torch.randn(1, 3)
        target_point = target_point[None, None, :, :]
        gt_traj_point_token = torch.tensor([start_token]).to(torch.int32)
        gt_traj_point_token = gt_traj_point_token[None, None, :, :]
        x.numpy().tofile("x_input.bin")
        cluster.numpy().tofile("cluster_input.bin")
        edge_index.numpy().tofile("edge_index_input.bin")
        valid_len.numpy().tofile("valid_len_input.bin")
        time_step_len.numpy().tofile("time_step_len_input.bin")
        target_point.numpy().tofile("target_point_input.bin")
        gt_traj_point_token.numpy().tofile("gt_traj_point_token_input.bin")



        # 导出模型
        # torch.onnx.export(
        #     export_onnx_model,                     # 模型
        #     (x, y, cluster, edge_index, valid_len, time_step_len, target_point, gt_traj_point_token),                      # 示例输入
        #     export_path,                         # 导出路径
        #     export_params=True,                  # 保存权重参数
        #     opset_version=11,                    # ONNX opset版本
        #     do_constant_folding=True,            # 常量折叠优化
        #     input_names=["x", "y", "cluster", "edge_index", "valid_len", "time_step_len", "target_point","gt_traj_point_token"],
        #     output_names=["pred_traj_point_token"],
        #     dynamic_axes={
        #         "x": {0: "num_nodes"},
        #         "y": {0: "batch_size"},
        #         "cluster": {0: "num_nodes"},
        #         "edge_index": {1: "num_edges"},
        #         "valid_len": {0: "batch_size"},
        #         "time_step_len": {0: "batch_size"},
        #         "target_point":{0: "batch_size"},
        #         "gt_traj_point_token":{0: "batch_size"},
        #         "pred_traj_point_token": {0: "batch_size"}
        #     },
        #     verbose=True
        # )

        torch.onnx.export(
            allcoder_wrapper,                     # 模型
            (x, cluster, edge_index, valid_len, time_step_len, target_point, gt_traj_point_token),                      # 示例输入
            export_path_AllcoderONNXWrapper,                    # 导出路径
            export_params=True,                  # 保存权重参数
            opset_version=11,                    # ONNX opset版本
            do_constant_folding=True,            # 常量折叠优化
            input_names=["x", "cluster", "edge_index", "valid_len", "time_step_len", "target_point", "gt_traj_point_token"],
            output_names=["pred_traj_point"],
            dynamic_axes=None
        )
        print(f"✅ ONNX 模型已导出到: {export_path_EncoderONNXWrapper}")

        torch.onnx.export(
            encoder_wrapper,                     # 模型
            (x, cluster, edge_index, valid_len, time_step_len),                      # 示例输入
            export_path_EncoderONNXWrapper,                    # 导出路径
            export_params=True,                  # 保存权重参数
            opset_version=11,                    # ONNX opset版本
            do_constant_folding=True,            # 常量折叠优化
            input_names=["x", "cluster", "edge_index", "valid_len", "time_step_len"],
            output_names=["global_feat"],
            dynamic_axes=None
        )
        print(f"✅ ONNX 模型已导出到: {export_path_EncoderONNXWrapper}")

        traj_input_wrapper = TrajInputEmbeddingONNXWrapper(self.model_onnx).eval()
        traj_input_wrapper = traj_input_wrapper.to(device=self.device)
        export_path_TrajInputEmbeddingONNXWrapper = os.path.join(export_dir, f"TrajInputEmbeddingONNXWrapper_{date}.onnx")

        dummy_target_point = torch.randn(1, 3).to(device=self.device)

        torch.onnx.export(
            traj_input_wrapper,
            (dummy_target_point,),
            export_path_TrajInputEmbeddingONNXWrapper,
            opset_version=11,
            do_constant_folding=True,
            input_names=["target_point"],
            output_names=["point_out"],
            dynamic_axes={"target_point": {0: "batch_size"}, "point_out": {0: "batch_size"}}
        )
        print(f"✅ ONNX 模型已导出到: {export_path_TrajInputEmbeddingONNXWrapper}")



        decoder_wrapper = DecoderONNXWrapper(self.model_onnx).eval()
        decoder_wrapper = decoder_wrapper.to(device=self.device)
        export_path_DecoderONNXWrapper = os.path.join(export_dir, f"DecoderONNXWrapper_{date}.onnx")

        dummy_encoder_out = torch.randn(batch_size, 1, self.cfg.train_meta_config.global_graph_width).to(device=self.device)
        dummy_point_out = torch.randn(1, self.cfg.train_meta_config.global_graph_width).to(device=self.device)
        dummy_gt_token = torch.randint(0, self.cfg.train_meta_config.token_nums, (1, 60)).to(device=self.device).to(torch.int32)

        torch.onnx.export(
            decoder_wrapper,
            (dummy_encoder_out, dummy_point_out, dummy_gt_token),
            export_path_DecoderONNXWrapper,
            opset_version=11,
            do_constant_folding=True,
            input_names=["encoder_out", "point_out", "gt_traj_point_token"],
            output_names=["pred_traj_point"],
            dynamic_axes={
            }
        )
        print(f"✅ ONNX 模型已导出到: {export_path_DecoderONNXWrapper}")




