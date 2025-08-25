from collections import OrderedDict
import numpy as np
import threading
import time

import rospy
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


class ParkingInferenceModuleReal:
    def __init__(self, inference_cfg: InferenceConfiguration):
        self.cfg = inference_cfg
        self.model = None
        self.device = None

        self.load_model(self.cfg.model_ckpt_path)
        
        self.BOS_token = self.cfg.train_meta_config.token_nums

        self.traj_start_point_info = Pose()
        self.traj_start_point_lock = threading.Lock()
        
        self.EOS_token = self.cfg.train_meta_config.token_nums + self.cfg.train_meta_config.append_token - 2

    def predict(self, test_data, cnt, mode="service"):
        if mode == "topic":
            self.pub_path(test_data, cnt)
        else:
            assert print("Can't support %s mode!".format(mode))

    def pub_path(self, test_data, cnt):
 
       
        filename = "./e2e_dataset/test/20250622T101821"
        start_token = [self.BOS_token]
        test_data["gt_traj_point_token"][0,:] = torch.tensor([start_token], dtype=torch.int64).to(self.device)
        test_data["gt_traj_point_token"] = test_data["gt_traj_point_token"][:,0:1]

        self.model.eval()
        delta_predicts = self.inference(test_data)
        delta_predicts = np.array(delta_predicts, dtype=np.float32)
        delta_predicts[:,0::2] = delta_predicts[:,0::2] * (self.cfg.train_meta_config.traj_norm_x_max - self.cfg.train_meta_config.traj_norm_x_min) + self.cfg.train_meta_config.traj_norm_x_min
        delta_predicts[:,1::2] = delta_predicts[:,1::2] * (self.cfg.train_meta_config.traj_norm_y_max - self.cfg.train_meta_config.traj_norm_y_min) + self.cfg.train_meta_config.traj_norm_y_min 
        delta_predicts = fitting_curve(delta_predicts, num_points=self.cfg.train_meta_config.autoregressive_points, item_number=self.cfg.train_meta_config.item_number)
        traj_yaw_path = calculate_tangent(np.array(delta_predicts)[:, :2], mode="five_point")

        x_coords = []
        y_coords = []
        for point_item in delta_predicts:
            if self.cfg.train_meta_config.item_number == 2:
                x, y = point_item
                x_coords.append(x)
                y_coords.append(y)
            elif self.cfg.train_meta_config.item_number == 3:
                x, y, progress_bar = point_item
                if abs(progress_bar) < 1 - self.cfg.progress_threshold:
                    break
        save_folder = os.path.join(filename,str(cnt),"test")
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, "test")
        plt.ioff()
        plt.figure(figsize=(6, 6))
        plt.scatter(x_coords, y_coords, color='blue', s = 2, label='Coordinates')
        plt.title(f'Scene {save_path}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()    

        
        

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


