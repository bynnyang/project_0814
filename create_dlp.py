
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os
import tqdm
import re
import pickle
from utils.trajectory_utils import TrajectoryInfoParser
from utils.cluster_utils import ClusterInfoParser
from utils.vec2d import Vec2d
from utils.box2d import Box2d
import copy
import json
from utils.common import get_json_content
from shapely.geometry import LineString


def get_safe_yaw(yaw):
    if yaw <= -180:
        yaw += 360
    if yaw > 180:
        yaw -= 360
    yaw = yaw / 180.0 * 3.14
    return yaw

def create_parking_goal_gt(traje_info_obj: TrajectoryInfoParser, world2ego_mat: np.array, switch_side: float):

    target_pose_in_world = traje_info_obj.get_precise_target_pose()
    target_pose_in_ego = target_pose_in_world.get_pose_in_ego(world2ego_mat)
    target_pose_in_ego.y = target_pose_in_ego.y * switch_side
    yaw_refine = get_safe_yaw(target_pose_in_ego.yaw) * switch_side
    parking_goal = (target_pose_in_ego.x, target_pose_in_ego.y, yaw_refine)
    return parking_goal

def get_all_tasks(data_dir):
    all_tasks = []
    for scene_item in os.listdir(data_dir):
        scene_path = os.path.join(data_dir, scene_item)
        all_tasks.append(scene_path)
    return all_tasks

def create_clusters_info_vcs(cluster_info_obj: ClusterInfoParser, world2ego_mat: np.array, ego_index, switch_side: float):
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
    clusters: List[LineString] = []

    for cl in cluster_frame_in_vcs:
        p0 = cl["p0"]
        p1 = cl["p1"]

        # Shapely LineString
        line = LineString([(p0.x, p0.y), (p1.x, p1.y)])
        clusters.append(line)
    return clusters
    
def create_map_data(data_dir, map_data):
    all_tasks = get_all_tasks(data_dir)

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
        slot_type_path = os.path.join(task_path, "slot_type", "{}.json".format(str(0).zfill(4)))
        slot_type_json = get_json_content(slot_type_path)
        slot_type = slot_type_json["slot_type"]
        if slot_type != 40002:
            print(task_path)
        parking_goal = create_parking_goal_gt(traje_info_obj, judge_world2ego_mat, switch_side)
        ego_index = np.random.randint(0, traje_info_obj.total_frames -1)
        obstacles = create_clusters_info_vcs(cluster_info_obj, judge_world2ego_mat, ego_index, switch_side)
        start_pose = (0.0, 0.0, 0.0)
        map_data.append((start_pose, parking_goal, obstacles, slot_type))
        


if __name__ == "__main__":

    DATA_DIR = "./e2e_dataset"
    map_data =[]
    for folder in os.listdir(DATA_DIR):
        print(f"folder: {folder}")
        each_mcap_floder = os.path.join(DATA_DIR, folder)
        create_map_data(each_mcap_floder, map_data)
    output_path = "./data/map_data.pkl"  # 指定保存路径
    with open(output_path, 'wb') as f:
        pickle.dump(map_data, f)
    print("finish save map data")