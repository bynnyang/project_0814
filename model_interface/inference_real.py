from collections import OrderedDict
import numpy as np
import threading
import time

import torch
import torchvision
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion

from model_interface.model.parking_model_real import ParkingModelReal
from model_interface.model.parking_model_real import ParkingModelInference
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

class ParkingModelONNXWrapper(torch.nn.Module):
    def __init__(self, model, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.multi_encoder = model.multi_encoder
             
        self.target_point_encoder = model.target_point_encoder

        # Trajectory Decoder
        self.trajectory_decoder = model.trajectory_decoder

    def forward(self, data, predict_token_num):
        # Encoder
    
        encoder_out = self.multi_encoder(data)
        point_out = self.target_point_encoder(data["target_point"].to(self.cfg.device))
        # Auto Regressive Decoder
        autoregressive_point = data['gt_traj_point'].to(self.cfg.device) # During inference, we regard BOS as gt_traj_point_token.
        for _ in range(predict_token_num):
            pred_traj_point = self.trajectory_decoder.predict(encoder_out, point_out, autoregressive_point)
            pred_traj_point = pred_traj_point.unsqueeze(1)
            autoregressive_point = torch.cat([autoregressive_point, pred_traj_point], dim=1)

        return autoregressive_point


class ParkingInferenceModuleReal:
    def __init__(self, inference_cfg: InferenceConfiguration):
        self.cfg = inference_cfg
        self.model = None
        self.device = None

        self.load_model(self.cfg.model_ckpt_path)

        # self.export_onnx(self.cfg.model_ckpt_path, self.cfg)
        
        self.BOS_token = self.cfg.train_meta_config.token_nums
      
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
        test_data["gt_traj_point"] = test_data["gt_traj_point"][:,0:1].to(self.device)
        self.model = self.model.to(device=self.device)
        self.model.eval()
        self.export_onnx(self.cfg.model_ckpt_path, test_data)
        delta_predicts = self.inference(test_data)
        delta_predicts = np.array(delta_predicts, dtype=np.float32)
        delta_predicts = delta_predicts.squeeze(0)
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
                x, y, cos_theta, sin_theta = point_item
                x_coords.append(x)
                y_coords.append(y)
                yaw = np.arctan2(sin_theta, cos_theta) 
                theta_coords.append(yaw)
        save_folder = os.path.join(filename,str(cnt),"test")
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, "test")
        plt.ioff()
        plt.figure(figsize=(12, 8),dpi= 300)
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
 
        test_data["gt_traj_point"] = test_data["gt_traj_point"][:,0:1]

        self.model.eval()
        delta_predicts = self.inference(test_data)
        delta_predicts = np.array(delta_predicts, dtype=np.float32)
        delta_predicts = delta_predicts.squeeze(0)
        delta_predicts = fitting_curve(delta_predicts, num_points=delta_predicts.shape[0], item_number = 4)
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
        # if (max_index <= 1 or max_index == (max_points -1) or  max_index > 5) and self.cnt > 4 and max_points > 5:
        #     self.predict_points_record = delta_predicts_map
        #     self.traj_yaw_path_record = traj_yaw_path_map
        #     self.cur = True
        # else:
        #     self.predict_points_record = self.predict_points_record[1:]
        #     self.traj_yaw_path_record = self.traj_yaw_path_record[1:]
        #     self.cur = False
        # if self.pre == True and self.cur == False:
        #     self.cnt = 0
        # self.pre = self.cur
        # self.cnt = self.cnt + 1
        # if self.cnt > 10:
        #     self.cnt = 10
        self.predict_points_record = delta_predicts_map
        self.traj_yaw_path_record = traj_yaw_path_map
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
        pred_traj_point= self.model.predict_transformer(data, predict_token_num = self.cfg.train_meta_config.autoregressive_points - 1)
        # pred_traj_point_update = pred_traj_point[0][1:]
        # pred_traj_point_update = self.remove_invalid_content(pred_traj_point_update)

        # delta_predicts = detokenize_traj_point(pred_traj_point_update, self.cfg.train_meta_config.token_nums, 
        #                                     self.cfg.train_meta_config.item_number, 
        #                                     self.cfg.train_meta_config.xy_max)

        pred_traj_point = np.array(pred_traj_point.cpu().numpy())
        pred_traj_point[..., 0] *= self.cfg.train_meta_config.traj_x_range
        pred_traj_point[..., 1] *= self.cfg.train_meta_config.traj_y_range

        return pred_traj_point

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
        self.model = ParkingModelInference(self.cfg.train_meta_config)

        ckpt = torch.load(parking_pth_path, map_location = self.device)
        # state_dict = OrderedDict([(k.replace('parking_model.', ''), v) for k, v in ckpt['state_dict'].items()])
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def export_onnx(self, checkpoint_path, data_set, export_dir="./onnx_exports", date="1013"):
        """
        将 PyTorch 模型导出为 ONNX 格式。
        """
        os.makedirs(export_dir, exist_ok=True)

        self.model_onnx = ParkingModelInference(self.cfg.train_meta_config)
        
        # 加载权重  
        checkpoint = torch.load(checkpoint_path, map_location = self.device)
        self.model_onnx.load_state_dict(checkpoint['state_dict'])
        self.model_onnx = self.model_onnx.to(device=self.device)
        self.model_onnx.eval()
        number_points = self.cfg.train_meta_config.autoregressive_points - 1



        allcoder_wrapper = ParkingModelONNXWrapper(self.model_onnx, self.cfg.train_meta_config, self.device).eval()
        allcoder_wrapper = allcoder_wrapper.to(device=self.device)
        
        export_path_AllcoderONNXWrapper = os.path.join(export_dir, f"CNNONNXWrapper_{date}.onnx")


        with torch.no_grad():
            torch.onnx.export(
                allcoder_wrapper,                     # 模型
                (data_set, number_points),                      # 示例输入
                export_path_AllcoderONNXWrapper,                    # 导出路径
                export_params=True,                  # 保存权重参数
                opset_version=11,                    # ONNX opset版本
                do_constant_folding=True,            # 常量折叠优化
                input_names=["data_set", "number_points"],
                output_names=["pred_traj_point"],
                dynamic_axes=None
            )
            print(f"✅ ONNX 模型已导出到: {export_path_AllcoderONNXWrapper}")






