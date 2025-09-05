import argparse
from dataset_interface.dataloader import ParkingDataloaderModule
from dataset_interface.dataset_real import ParkingDataModuleReal
from utils.config import get_inference_config_obj
from dataset import GraphData
from dataset import GraphDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from loss.traj_point_loss import TokenTrajPointLoss
from torch_geometric.data import Batch
from torch.utils.data._utils.collate import default_collate
from utils.config import InferenceConfiguration
from model_interface.model_interface import get_parking_model
from tqdm import tqdm
import time
from niodds_py3 import niodds
from function.parking.par_fusion_pb2 import ParkingFusion
from function.parking.par_planning_pb2 import ParkingTrajectory, SlotTrajectory
from function.parking.parking_zongmu_20ms_pb2 import ZMData20ms
from function.parking.par_pnc_point_pb2 import PathPoint, TrajectoryPoint
from function.parking.par_trajectory_pb2 import ADCTrajectory
import threading
from model_interface.inference_real import ParkingInferenceModuleReal
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from utils.trajectory_utils import TrajectoryInfoParser
from utils.cluster_utils import ClusterInfoParser
from utils.vec2d import Vec2d
from utils.box2d import Box2d
from utils.pose_utils import CustomizePose
import copy

NODE_NAME = "inference_node"
SUB_TOPIC_FUSION = "function/parking/par_fusion"       # 接收 sensor 数据
SUB_TOPIC_DR = "function/parking/parking_zongmu_20ms"       # 接收 sensor 数据
PUB_TOPIC = "function/parking/par_planning"   # 发布轨迹
PERIOD_S = 0.1              # 100 ms

g_latest_sensor_fusion = None      # 最新收到的 sensor 数据
g_latest_sensor_dr = None      # 最新收到的 sensor 数据
g_sensor_fusion_lock = threading.Lock()
g_sensor_dr_lock = threading.Lock()
slot_data = None

    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)
    return checkpoint_path['end_epoch']

def get_safe_yaw(yaw):
    if yaw <= -180:
        yaw += 360
    if yaw > 180:
        yaw -= 360
    yaw = yaw / 180.0 * 3.14
    return yaw


def parser_dr_msg(msg) -> CustomizePose: 
    pose_ret = CustomizePose(x=msg.ego_pose.Pose.x, y=msg.ego_pose.Pose.y, z=0.0, roll=0.0, yaw=msg.ego_pose.Pose.theta / 3.14 * 180, pitch=0.0)
    return pose_ret

def parser_slot_msg(msg):
    if len(msg.fusion_slots) > 0:
        slot_ret = {
            'v0': CustomizePose(x=msg.fusion_slots[0].polygon.points[0].x, y=msg.fusion_slots[0].polygon.points[0].y, z=0.0, roll=0.0, yaw=0.0, pitch=0.0),
            'v1': CustomizePose(x=msg.fusion_slots[0].polygon.points[1].x, y=msg.fusion_slots[0].polygon.points[1].y, z=0.0, roll=0.0, yaw=0.0, pitch=0.0),
            'v2': CustomizePose(x=msg.fusion_slots[0].polygon.points[2].x, y=msg.fusion_slots[0].polygon.points[2].y, z=0.0, roll=0.0, yaw=0.0, pitch=0.0),
            'v3': CustomizePose(x=msg.fusion_slots[0].polygon.points[3].x, y=msg.fusion_slots[0].polygon.points[3].y, z=0.0, roll=0.0, yaw=0.0, pitch=0.0),
        }
    else:
        raise RuntimeError("slot ret failed")
    return slot_ret


def parser_clusters_msg(msg):
    clusters_list =[]
    clusters_msg = msg.zm_data_50ms
    for clusters in clusters_msg.ups_cluster_info.clusters:
        for index, line in enumerate(clusters.lines):
            if (line.height != 2):
                my_dict = {
                    "id": clusters.id,
                    "p0": CustomizePose(x=line.p0.x, y=line.p0.y, z=0.0, roll=0.0, yaw=0.0, pitch=0.0),
                    "p1": CustomizePose(x=line.p1.x, y=line.p1.y, z=0.0, roll=0.0, yaw=0.0, pitch=0.0),
                }
            clusters_list.append(my_dict)
    return clusters_list

def create_parking_goal_vcs(target_pose_in_ego: CustomizePose, switch_side: float):
    target_pose_in_ego.y = target_pose_in_ego.y * switch_side
    yaw_refine = get_safe_yaw(target_pose_in_ego.yaw) * switch_side
    parking_goal = [target_pose_in_ego.x, target_pose_in_ego.y, yaw_refine]
    return parking_goal

def create_clusters_info_vcs(cluster_info_obj, world2ego_mat: np.array, switch_side: float):
    cluster_frame_in_vcs =[]  
    cluster_dict_template = {
            "id": None,
            "p0": {},
            "p1": {}
        }
    for index, each_cluster in enumerate(cluster_info_obj):
        each_cluster_vcs = copy.deepcopy(cluster_dict_template)
        each_cluster_vcs["id"] = each_cluster["id"] * switch_side
        each_cluster_vcs["p0"] = each_cluster["p0"].get_pose_in_ego(world2ego_mat)
        each_cluster_vcs["p0"].y = each_cluster_vcs["p0"].y * switch_side
        each_cluster_vcs["p1"] = each_cluster["p1"].get_pose_in_ego(world2ego_mat)
        each_cluster_vcs["p1"].y = each_cluster_vcs["p1"].y * switch_side
        cluster_frame_in_vcs.append(each_cluster_vcs)

    return cluster_frame_in_vcs

def get_agent_feature_ls():
    vehicle_width = 1.8
    vehicle_length = 3.99
    vehicle_rear_overhang = 3.2
    vehicle_angle = 3.14 / 2.0
    
    vehicle_position = Vec2d(0.0, 0.0)

    vehicle_center = vehicle_position + Vec2d.create_unit_vec2d(vehicle_angle) * (vehicle_length / 2.0 - vehicle_rear_overhang)

    vehicle_box = Box2d(vehicle_center, vehicle_angle, vehicle_length, vehicle_width)

    vehicle_corners = vehicle_box.GetAllCorners()

    return [vehicle_corners, 0]

def  get_target_point_vcs_feature_ls(target_point_vcs):

    return [target_point_vcs, 0]

def get_clusters_feature_ls(clusters_info_vcs):
    clusters_feature_ls = []

    for index in range(0, len(clusters_info_vcs)):
        start_pose = np.array([clusters_info_vcs[index]["p0"].x, clusters_info_vcs[index]["p0"].y])
        end_pose = np.array([clusters_info_vcs[index]["p1"].x, clusters_info_vcs[index]["p1"].y])
        clusters_feature_ls.append([start_pose, end_pose, clusters_info_vcs[index]["id"], index])
    return clusters_feature_ls


def compute_feature_for_one_seq(data_fusion, data_dr) -> List[List]:
    """
    return lane & track features
    args:
        mode: 'rect' or 'nearby'
    returns:
        agent_feature_ls:
            list of (doubeld_track, object_type, timetamp, track_id, not_doubled_groudtruth_feature_trajectory)
        obj_feature_ls:
            list of list of (doubled_track, object_type, timestamp, track_id)
        lane_feature_ls:
            list of list of lane a segment feature, formatted in [left_lane, right_lane, is_traffic_control, is_intersection, lane_id]
        norm_center np.ndarray: (2, )
    """
    # normalize timestamps
    cluster_info_data = parser_clusters_msg(data_fusion)
    judge_ego_pose = parser_dr_msg(data_dr)
    global slot_data
    if slot_data == None:
        slot_data = parser_slot_msg(data_fusion)
    judge_world2ego_mat = judge_ego_pose.get_homogeneous_transformation().get_inverse_matrix()
    vcs_slot_point_0 = slot_data["v0"].get_pose_in_ego(judge_world2ego_mat)
    vcs_slot_point_1 = slot_data["v1"].get_pose_in_ego(judge_world2ego_mat)
    vcs_slot_point_2 = slot_data["v2"].get_pose_in_ego(judge_world2ego_mat)
    vcs_slot_point_3 = slot_data["v3"].get_pose_in_ego(judge_world2ego_mat)
    map_point = CustomizePose(x=0.0, y=0.0, z=0.0, roll=0.0, yaw=0.0, pitch=0.0)
    vcs_map_point = map_point.get_pose_in_ego(judge_world2ego_mat)
    judge_ego2world_mat = judge_ego_pose.get_homogeneous_transformation().get_matrix()

    middlw_position = Vec2d((vcs_slot_point_0.x + vcs_slot_point_1.x)/2, (vcs_slot_point_0.y + vcs_slot_point_1.y)/2)

    delta_x = vcs_slot_point_1.x - vcs_slot_point_2.x
    delta_y = vcs_slot_point_1.y - vcs_slot_point_2.y
    angle_rad = np.arctan2(delta_y, delta_x)
    park_slot_angle_rad = get_safe_yaw(np.degrees(angle_rad))


    vec_park_slot = middlw_position + Vec2d.create_unit_vec2d(park_slot_angle_rad + 3.14) * 4.0

    switch_side = -1.0 if vcs_slot_point_2.y > vcs_slot_point_1.y else 1.0

    park_slot_vcs = CustomizePose(x=vec_park_slot.x_, y=vec_park_slot.y_, z=0.0, roll=0.0, yaw=(park_slot_angle_rad / 3.14 * 180), pitch=0.0)

    target_point_vcs = create_parking_goal_vcs(park_slot_vcs, switch_side)
    clusters_info_vcs = create_clusters_info_vcs(cluster_info_data, judge_world2ego_mat, switch_side)

    agent_feature = get_agent_feature_ls()

    park_slot_feature_ls = get_target_point_vcs_feature_ls(target_point_vcs)
    # pdb.set_trace()

    # search nearby moving objects from the last observed point of agent
    clusters_feature_ls = get_clusters_feature_ls(clusters_info_vcs)
    # get agent features

    return [agent_feature, clusters_feature_ls, park_slot_feature_ls, judge_ego2world_mat]


def encoding_features(agent_feature, clusters_feature, park_slot_feature):

    """"
    polyline_features: vstack[
                (xs, ys, xe, ye, theta, polyline_id), 车身轮廓四条边等于四个节点构成一个polyline
                (xs, ys, xe, ye, theta,  polyline_id), cluster一条边等同于一个节点,构成一个polyline
                (xs, ys, xs, ys, theta,  polyline_id), 目标位置一个点,等同于一个节点，成一个polyline
                ]


    """

    """
    20250821重构特征

    polyline_features: vstack[
                (x, y, theta, polyline_id), 车身轮廓的每个顶点一共四个节点构成一个polyline
                (x, y, theta,  polyline_id), cluster的每个顶点,一共两个节点构成一个polyline
                (x, y, theta,  polyline_id), 目标位置一个点,等同于一个节点,自连接构成一个polyline
                ]
    
    """
    polyline_id = 0
    agent_id2mask, cluster_id2mask, park_slot_id2mask= {}, {},{}
    agent_nd, cluster_nd, park_slot_nd = np.empty((0, 4)), np.empty((0, 4)), np.empty((0, 4))
    agent_feature_points_list = []
    pre_agent_len = agent_nd.shape[0]
    for index in range(0,len(agent_feature[0])):
        x_array = agent_feature[0][index].x_
        y_array = agent_feature[0][index].y_
        agent_feature_points_list.append([x_array,y_array])
    agent_feature_points_array = np.array(agent_feature_points_list)
    # first_row = agent_feature_points_array[0,:]
    # rest_of_rows = agent_feature_points_array[1:,:]
    # new_array = np.vstack((rest_of_rows, first_row))
    # agent_feature_points_array = np.hstack((agent_feature_points_array, new_array))
    agent_len = agent_feature_points_array.shape[0]
    agent_nd = np.hstack((agent_feature_points_array, np.ones((agent_len, 1)) * 0.0, np.ones((agent_len, 1)) * polyline_id))

    assert agent_nd.shape[1] == 4

    agent_id2mask[polyline_id] = (pre_agent_len, agent_nd.shape[0])
    pre_agent_len = agent_nd.shape[0]
    polyline_id += 1


    pre_park_slot_len = park_slot_nd.shape[0]
    park_slot_feature_2d = np.array([park_slot_feature[0][0],park_slot_feature[0][1]]).reshape(1, -1)
    park_slot_points_array = park_slot_feature_2d
    park_len = park_slot_points_array.shape[0]
    park_slot_nd = np.hstack((park_slot_points_array, np.ones((park_len, 1)) * park_slot_feature[0][2], np.ones((park_len, 1)) * polyline_id))

    assert park_slot_nd.shape[1] == 4

    park_slot_id2mask[polyline_id] = (pre_park_slot_len, park_slot_nd.shape[0])
    pre_park_slot_len = park_slot_nd.shape[0]
    polyline_id += 1


    pre_cluster_len = cluster_nd.shape[0]
    for line in clusters_feature:
        p0_x = line[0][0]
        p0_y = line[0][1]
        p0_feature_2d = np.array([p0_x,p0_y]).reshape(1, -1)
        p1_x = line[1][0]
        p1_y = line[1][1]
        p1_feature_2d = np.array([p1_x,p1_y]).reshape(1, -1)
        delta_x = p1_x - p0_x
        delta_y = p1_y - p0_y
        angle_rad = np.arctan2(delta_y, delta_x)
        angle_rad = get_safe_yaw(np.degrees(angle_rad))
        # line_points_array = np.hstack((p0_feature_2d, p1_feature_2d))
        # line_len = line_points_array.shape[0]
        # one_cluster_nd = np.hstack((line_points_array, np.ones((line_len, 1)) * angle_rad, np.ones((line_len, 1)) * polyline_id))
        p0_points_array = p0_feature_2d
        p0_len = p0_points_array.shape[0]
        p0_cluster_nd = np.hstack((p0_points_array, np.ones((p0_len, 1)) * angle_rad, np.ones((p0_len, 1)) * polyline_id))
        p1_points_array = p1_feature_2d
        p1_len = p1_points_array.shape[0]
        p1_cluster_nd = np.hstack((p1_points_array, np.ones((p1_len, 1)) * angle_rad, np.ones((p1_len, 1)) * polyline_id))
        cluster_nd = np.vstack((cluster_nd, p0_cluster_nd, p1_cluster_nd))
        cluster_id2mask[polyline_id] = (pre_cluster_len, cluster_nd.shape[0])
        pre_cluster_len = cluster_nd.shape[0]
        polyline_id += 1

    # don't ignore the id
    polyline_features = np.vstack((agent_nd, park_slot_nd, cluster_nd))
    data = [[polyline_features.astype(
        np.float32), agent_id2mask, park_slot_id2mask, cluster_id2mask, agent_nd.shape[0], park_slot_nd.shape[0], cluster_nd.shape[0]]]

    return pd.DataFrame(
        data,
        columns=["POLYLINE_FEATURES", "AGENT_ID_TO_MASK",
                 "PARK_SLOT_ID_TO_MASK", "CLUSTER_ID_TO_MASK", "AGENT_LEN", "PARK_SLOT_LEN", "CLUSTER_LEN"]
    )

def get_agent_edge_index(num_nodes, start=0):
    """
    return a tensor(2, edges), indicing edge_index
    """
    to_ = np.arange(num_nodes, dtype=np.int64)
    edge_index = np.empty((2, 0))
    for i in range(num_nodes):
        from_ = np.ones(1, dtype=np.int64) * i
        to_ = from_ + 1
        if to_ == num_nodes:
            to_ = 0
        # FIX BUG: no self loop in ful connected nodes graphs
        edge_index = np.hstack((edge_index, np.vstack((from_, to_))))
    first_row = edge_index[0, :]
    second_row = edge_index[1, :]
    edge_index = np.hstack((edge_index, np.vstack((second_row, first_row))))
    edge_index = edge_index + start

    return edge_index.astype(np.int64), num_nodes + start



def get_cluster_edge_index(num_nodes, start=0):
    """
    return a tensor(2, edges), indicing edge_index
    """
    edge_index = np.empty((2, 0))
    for i in range(num_nodes):
        # FIX BUG: no self loop in ful connected nodes graphs
        from_ = np.ones(1, dtype=np.int64) * i
        to_ = from_ + 1
        if to_ == num_nodes:
            to_ = 0
        # FIX BUG: no self loop in ful connected nodes graphs
        edge_index = np.hstack((edge_index, np.vstack((from_, to_))))
    edge_index = edge_index + start

    return edge_index.astype(np.int64), num_nodes + start


def get_park_slot_edge_index(num_nodes, start=0):
    """
    return a tensor(2, edges), indicing edge_index
    """
    edge_index = np.empty((2, 0))
    for i in range(num_nodes):
        # FIX BUG: no self loop in ful connected nodes graphs
        edge_index = np.hstack((edge_index, np.vstack((i, i))))
    edge_index = edge_index + start

    return edge_index.astype(np.int64), num_nodes + start


def inference(inference_cfg: InferenceConfiguration, parking_inference_model:ParkingInferenceModuleReal, data_fusion, data_dr):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    agent_feature, clusters_feature, park_slot_feature, judge_ego2world_mat = compute_feature_for_one_seq(data_fusion, data_dr)
    feature_pd = encoding_features(agent_feature, clusters_feature, park_slot_feature)
    valid_len_ls = []
    data_ls = []
    x_ls = []
    y = np.array([1])
    cluster = None
    edge_index_ls = []
    all_in_features = feature_pd['POLYLINE_FEATURES'].values[0]
    agen_len = feature_pd['AGENT_LEN'].values[0]
    park_slot_len = feature_pd['PARK_SLOT_LEN'].values[0]
    cluster_len = feature_pd['CLUSTER_LEN'].values[0]
    cluster = all_in_features[:, -1].reshape(-1).astype(np.int32)
    valid_len_ls.append(cluster.max())
    agent_mask, park_slot_mask, clusert_mask = feature_pd["AGENT_ID_TO_MASK"].values[0], feature_pd['PARK_SLOT_ID_TO_MASK'].values[0], feature_pd['CLUSTER_ID_TO_MASK'].values[0]
    agent_id = 0
    edge_index_start = 0
    assert all_in_features[agent_id][-1] == 0, f"agent id is wrong. id {agent_id}: type {all_in_features[agent_id][4]}"

    for id_, mask_ in agent_mask.items():
        data_ = all_in_features[mask_[0]:mask_[1]]
        edge_index_, edge_index_start = get_agent_edge_index(data_.shape[0], start=edge_index_start)
        x_ls.append(data_)
        edge_index_ls.append(edge_index_)

    for id_, mask_ in park_slot_mask.items():
        data_ = all_in_features[mask_[0] + agen_len: mask_[1] + agen_len]
        edge_index_, edge_index_start = get_park_slot_edge_index(data_.shape[0], edge_index_start)
        x_ls.append(data_)
        edge_index_ls.append(edge_index_)
                
    for id_, mask_ in clusert_mask.items():
        data_ = all_in_features[mask_[0] + agen_len + park_slot_len: mask_[1] + agen_len + park_slot_len]
        edge_index_, edge_index_start = get_cluster_edge_index(data_.shape[0], edge_index_start)
        x_ls.append(data_)
        edge_index_ls.append(edge_index_)
    edge_index = np.hstack(edge_index_ls)
    x = np.vstack(x_ls)
    data_ls.append([x, y, cluster, edge_index])

        # [x, y, cluster, edge_index, valid_len]
    g_ls = []
    padd_to_index = np.max(valid_len_ls)
    feature_len = data_ls[0][0].shape[1]
    for ind, tup in enumerate(data_ls):
        pad_matrix = np.zeros((padd_to_index - tup[-2].max(), feature_len), dtype=tup[0].dtype)
        pad_matrix[:, -1] = np.arange(tup[-2].max() + 1, padd_to_index + 1)
        tup[0] = np.vstack([tup[0], pad_matrix])
        tup[-2] = np.hstack([tup[2], np.arange(tup[-2].max()+1, padd_to_index+1)])
        g_data = GraphData(
            x=torch.from_numpy(tup[0]),
            y=torch.from_numpy(tup[1]),
            cluster=torch.from_numpy(tup[2]),
            edge_index=torch.from_numpy(tup[3]),
            valid_len=torch.tensor([valid_len_ls[ind]]),
            time_step_len=torch.tensor([padd_to_index + 1])
            # time_step_len=torch.tensor([valid_len_ls[ind] + 1])
        )
        g_ls.append(g_data)

    g: GraphData = g_ls[0].clone()  # 这是 GraphData 实例

    x = g.x.clone()

    x[:, 0] = (x[:, 0] - inference_cfg.train_meta_config.graph_norm_x_min) / (inference_cfg.train_meta_config.graph_norm_x_max - inference_cfg.train_meta_config.graph_norm_x_min)
    x[:, 1] = (x[:, 1] - inference_cfg.train_meta_config.graph_norm_y_min) / (inference_cfg.train_meta_config.graph_norm_y_max - inference_cfg.train_meta_config.graph_norm_y_min)
    x[:, 2] = (x[:, 2] - inference_cfg.train_meta_config.graph_norm_theta_min) / (inference_cfg.train_meta_config.graph_norm_theta_max - inference_cfg.train_meta_config.graph_norm_theta_min)
    g.x = x
    # 把轨迹/目标等张量挂到图上成为额外属性

    target_point = park_slot_feature[0]
    target_point[0] = (target_point[0] - inference_cfg.train_meta_config.target_point_x_min) / (inference_cfg.train_meta_config.target_point_x_max - inference_cfg.train_meta_config.target_point_x_min)
    target_point[1] = (target_point[1] - inference_cfg.train_meta_config.target_point_y_min) / (inference_cfg.train_meta_config.target_point_y_max - inference_cfg.train_meta_config.target_point_y_min)
    target_point[2] = (target_point[2] - inference_cfg.train_meta_config.target_point_theta_min) / (inference_cfg.train_meta_config.target_point_theta_max - inference_cfg.train_meta_config.target_point_theta_min)
    # g.gt_traj_point        = torch.from_numpy(np.array(self.traj_point[index]))
    start_token = [inference_cfg.train_meta_config.token_nums]
    traj_point_token = start_token
    g.gt_traj_point_token = torch.from_numpy(np.array(traj_point_token))
    g.target_point = torch.from_numpy(np.array(target_point).astype(np.float32))
    g.to(device)
    t1 = time.time()
    delta_predicts, traj_yaw_path = parking_inference_model.predict(g, 0, "simulation")
    t2 = time.time()
    print(t2 - t1)
    delta_predicts_map = []
    traj_yaw_path_map = []
    for index in range(len(delta_predicts)):
        vcs_point = CustomizePose(x=delta_predicts[index][0], y=delta_predicts[index][1], z=0.0, roll=0.0, yaw=traj_yaw_path[index], pitch=0.0)
        map_point = vcs_point.get_pose_in_world(judge_ego2world_mat)
        delta_predicts_map.append([map_point.x, map_point.y])
        traj_yaw_path_map.append(map_point.yaw / 180 * 3.14)
    return delta_predicts_map, traj_yaw_path_map

def sensor_fusion_callback(msg):
    global g_latest_sensor_fusion
    with g_sensor_fusion_lock:
        g_latest_sensor_fusion = msg

def sensor_dr_callback(msg):
    global g_latest_sensor_dr
    with g_sensor_dr_lock:
        g_latest_sensor_dr = msg
    
def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--inference_config_path', default='./config/inference_real.yaml', type=str)
    args = arg_parser.parse_args()
    inference_cfg = get_inference_config_obj(args.inference_config_path)
    ParkingInferenceModelModule = get_parking_model(data_mode=inference_cfg.train_meta_config.data_mode, run_mode="inference")
    parking_inference_model = ParkingInferenceModelModule(inference_cfg)
    if not niodds.init(NODE_NAME):
        raise RuntimeError("niodds init failed")

    node = niodds.Node(NODE_NAME)
    qos = niodds.qos()
    sub_fusion = node.create_subscriber(
        name=SUB_TOPIC_FUSION,
        data_type=ParkingFusion,
        qos=qos,
        callback=sensor_fusion_callback
    )
    if sub_fusion is None:
        raise RuntimeError("create subscriber failed")
    
    sub_dr = node.create_subscriber(
        name=SUB_TOPIC_DR,
        data_type=ZMData20ms,
        qos=qos,
        callback=sensor_dr_callback
    )
    if sub_dr is None:
        raise RuntimeError("create subscriber failed")
    
    pub = node.create_publisher(
        name=PUB_TOPIC,
        data_type=ParkingTrajectory,
        qos=qos
    )
    while not niodds.is_shutdown():
            t0 = time.time()

            # 取最新 sensor 数据
            with g_sensor_fusion_lock:
                global g_latest_sensor_fusion
                data_fusion = g_latest_sensor_fusion
                g_latest_sensor_fusion = None  # 用后即弃，避免重复处理

            with g_sensor_dr_lock:
                global g_latest_sensor_dr
                data_dr = g_latest_sensor_dr
                g_latest_sensor_dr = None  # 用后即弃，避免重复处理

            if data_fusion is not None and data_dr is not None:
                # 推理
                # t1 = time.time()
                delta_predicts, traj_yaw_path = inference(inference_cfg, parking_inference_model, data_fusion, data_dr)
                # t2 = time.time()
                # print(t2 - t1)
                msg = ParkingTrajectory()
                traj = ADCTrajectory()
                slottrajectory = SlotTrajectory()
                for (point_item, traj_yaw) in zip(delta_predicts, traj_yaw_path):
                    x, y = point_item
                    pathpoint = PathPoint()
                    traj_point = TrajectoryPoint()
                    pathpoint.x = x
                    pathpoint.y = y
                    pathpoint.theta = traj_yaw
                    traj_point.path_point.CopyFrom(pathpoint)
                    traj.trajectory_point.append(traj_point)
                slottrajectory.trajectory.CopyFrom(traj)
                msg.slot_trajectory.append(slottrajectory)
                pub.publish(msg)

            # 睡眠补足 100 ms
            elapsed = time.time() - t0
            time.sleep(max(0, PERIOD_S - elapsed))



if __name__ == '__main__':
    main()