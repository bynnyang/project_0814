import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os
from tqdm import tqdm
import re
import pickle
from utils.trajectory_utils import TrajectoryInfoParser
from utils.cluster_utils import ClusterInfoParser
from utils.vec2d import Vec2d
from utils.box2d import Box2d
# %matplotlib inline


def get_safe_yaw(yaw):
    if yaw <= -180:
        yaw += 360
    if yaw > 180:
        yaw -= 360
    yaw = yaw / 180.0 * 3.14
    return yaw

def create_parking_goal_vcs(traje_info_obj: TrajectoryInfoParser, world2ego_mat: np.array):
    target_pose_in_world = traje_info_obj.get_precise_target_pose()
    target_pose_in_ego = target_pose_in_world.get_pose_in_ego(world2ego_mat)
    yaw_refine = get_safe_yaw(target_pose_in_ego.yaw)
    parking_goal = [target_pose_in_ego.x, target_pose_in_ego.y, yaw_refine]
    return parking_goal

def create_clusters_info_vcs(cluster_info_obj: ClusterInfoParser, world2ego_mat: np.array, ego_index):
    cluster_frame_in_world = cluster_info_obj.get_clusters(ego_index)
    cluster_frame_in_vcs =[]  
    cluster_dict_template = {
            "id": None,
            "p0": {},
            "p1": {}
        }
    for each_cluster in cluster_frame_in_world:
        each_cluster_vcs = cluster_dict_template.copy()
        each_cluster_vcs["id"] = each_cluster["id"]
        each_cluster_vcs["p0"] = each_cluster["p0"].get_pose_in_ego(world2ego_mat)
        each_cluster_vcs["p1"] = each_cluster["p1"].get_pose_in_ego(world2ego_mat)
        cluster_frame_in_vcs.append(each_cluster_vcs)
    return cluster_frame_in_vcs

def get_agent_feature_ls(total_frames):
    vehicle_width = 1.8
    vehicle_length = 3.99
    vehicle_rear_overhang = 3.2
    vehicle_angle = 3.14 / 2.0
    
    vehicle_position = Vec2d(0.0, 0.0)

    vehicle_center = vehicle_position + Vec2d.create_unit_vec2d(vehicle_angle) * (vehicle_length / 2.0 - vehicle_rear_overhang)

    vehicle_box = Box2d(vehicle_center, vehicle_angle, vehicle_length, vehicle_width)

    vehicle_corners = vehicle_box.GetAllCorners()

    vehicle_corners_list = []


    for index in range(0, total_frames):
        vehicle_corners_list.append([vehicle_corners, index])

    return vehicle_corners_list

def  get_target_point_vcs_feature_ls(target_point_vcs):
    park_slot_feature_ls = []

    for index in range(len(target_point_vcs)):
        park_slot_feature_ls.append([target_point_vcs[index], index])

    return park_slot_feature_ls

def get_clusters_feature_ls(clusters_info_vcs):
    clusters_feature_ls = []

    for index in range(0, len(clusters_info_vcs)):
        cluster_feature_ls = []
        for line in clusters_info_vcs[index]:
            start_pose = np.array([line["p0"].x, line["p0"].y])
            end_pose = np.array([line["p1"].x, line["p1"].y])
            cluster_feature_ls.append([start_pose, end_pose, line["id"], index])
        clusters_feature_ls.append(cluster_feature_ls)
    return clusters_feature_ls





def encoding_features(agent_feature, clusters_feature, park_slot_feature):

    """"
    polyline_features: vstack[
                (xs, ys, xe, ye, theta, polyline_id), 车身轮廓四条边等于四个节点构成一个polyline
                (xs, ys, xe, ye, theta,  polyline_id), cluster一条边等同于一个节点,构成一个polyline
                (xs, ys, xs, ys, theta,  polyline_id), 目标位置一个点,等同于一个节点，成一个polyline
                ]


    """
    polyline_id = 0
    agent_id2mask, cluster_id2mask, park_slot_id2mask= {}, {},{}
    agent_nd, cluster_nd, park_slot_nd = np.empty((0, 6)), np.empty((0, 6)), np.empty((0, 6))
    agent_feature_points_list = []
    pre_agent_len = agent_nd.shape[0]
    for index in range(0,len(agent_feature[0])):
        x_array = agent_feature[0][index].x_
        y_array = agent_feature[0][index].y_
        agent_feature_points_list.append([x_array,y_array])
    agent_feature_points_array = np.array(agent_feature_points_list)
    first_row = agent_feature_points_array[0,:]
    rest_of_rows = agent_feature_points_array[1:,:]
    new_array = np.vstack((rest_of_rows, first_row))
    agent_feature_points_array = np.hstack((agent_feature_points_array, new_array))
    agent_len = agent_feature_points_array.shape[0]
    agent_nd = np.hstack((agent_feature_points_array, np.ones((agent_len, 1)) * 0.0, np.ones((agent_len, 1)) * polyline_id))

    assert agent_nd.shape[1] == 6

    agent_id2mask[polyline_id] = (pre_agent_len, agent_nd.shape[0])
    pre_agent_len = agent_nd.shape[0]
    polyline_id += 1


    pre_park_slot_len = park_slot_nd.shape[0]
    park_slot_feature_2d = np.array([park_slot_feature[0][0],park_slot_feature[0][1]]).reshape(1, -1)
    park_slot_points_array = np.hstack((park_slot_feature_2d, park_slot_feature_2d))
    park_len = park_slot_points_array.shape[0]
    park_slot_nd = np.hstack((park_slot_points_array, np.ones((park_len, 1)) * park_slot_feature[0][2], np.ones((park_len, 1)) * polyline_id))

    assert park_slot_nd.shape[1] == 6

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
        line_points_array = np.hstack((p0_feature_2d, p1_feature_2d))
        line_len = line_points_array.shape[0]
        one_cluster_nd = np.hstack((line_points_array, np.ones((line_len, 1)) * angle_rad, np.ones((line_len, 1)) * polyline_id))
        cluster_nd = np.vstack((cluster_nd, one_cluster_nd))
        cluster_id2mask[polyline_id]=(pre_cluster_len,cluster_nd.shape[0])
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

def compute_feature_for_one_seq(filename) -> List[List]:
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
    traje_info_obj = TrajectoryInfoParser(1.0, filename)
    cluster_info_obj = ClusterInfoParser(1.0, filename)
    assert (len(traje_info_obj.trajectory_list) == len(cluster_info_obj.clusters_list))

    target_point_vcs = []
    clusters_info_vcs = []

    for ego_index in range(0, traje_info_obj.total_frames): # ego iteration
        ego_pose = traje_info_obj.get_trajectory_point(ego_index)
        world2ego_mat = ego_pose.get_homogeneous_transformation().get_inverse_matrix()
        parking_goal = create_parking_goal_vcs(traje_info_obj, world2ego_mat)
        target_point_vcs.append(parking_goal)
        cluster_frame_info_vcs = create_clusters_info_vcs(cluster_info_obj, world2ego_mat, ego_index)
        clusters_info_vcs.append(cluster_frame_info_vcs)

    assert (len(target_point_vcs) == len(clusters_info_vcs))


    agent_feature = get_agent_feature_ls(traje_info_obj.total_frames)

    park_slot_feature_ls = get_target_point_vcs_feature_ls(target_point_vcs)
    # pdb.set_trace()

    # search nearby moving objects from the last observed point of agent
    clusters_feature_ls = get_clusters_feature_ls(clusters_info_vcs)
    # get agent features

    return [agent_feature, clusters_feature_ls, park_slot_feature_ls]

def save_features(df, name, cnt, dir_=None):
    if dir_ is None:
        dir_ = './input_data'
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    name = f"features_{name}"
    feature_path= os.path.join(dir_, name)
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
    df.to_pickle(
        os.path.join(feature_path, "{:04d}.pkl".format(cnt))
    )
    # df.to_csv(
    #     os.path.join(feature_path, "{:04d}.csv".format(cnt)),index =False
    # )


if __name__ == "__main__":

    DATA_DIR = "./e2e_dataset"
    INTERMEDIATE_DATA_DIR = './interm_data'
    for folder in os.listdir(DATA_DIR):
        #if not re.search(r'val', folder):
        # FIXME: modify the target folder by hand ('val|train|sample|test')
        # if not re.search(r'test', folder):
        #    continue
        print(f"folder: {folder}")
        each_mcap_floder = os.path.join(DATA_DIR, folder)
        for name in tqdm(os.listdir(each_mcap_floder)):
            filepath = os.path.join(each_mcap_floder, name)

            agent_feature, clusters_feature, park_slot_feature = compute_feature_for_one_seq(filepath)
            for index in range(0, len(agent_feature)):

                df = encoding_features(agent_feature[index], clusters_feature[index], park_slot_feature[index])
                # df = df = pd.DataFrame(
                #     {"column1": [1, 2, 3],
                #      "column2": [4, 5, 6]
                #      })
                save_features(df, name, index, os.path.join(INTERMEDIATE_DATA_DIR, f"{folder}_intermediate"))


# %%


# %%
