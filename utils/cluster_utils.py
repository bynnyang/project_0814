import os
from typing import List

import numpy as np
import torch
from shapely.geometry import LineString
from shapely.measurement import hausdorff_distance

from utils.common import get_json_content
from utils.pose_utils import CustomizePose
import copy


class ClusterInfoParser:
    def __init__(self, task_index, task_path):
        self.task_index = task_index
        self.task_path = task_path
        self.total_frames = self._get_clusters_num()
        self.clusters_list = self.make_clusters()

    def _get_clusters_num(self) -> int:
        return len(os.listdir(os.path.join(self.task_path, "clusters")))
    
    def get_clusters(self, clusters_index) -> CustomizePose:
        return self.clusters_list[clusters_index]

    def get_clusters_path(self, clusters_index) -> str:
        return os.path.join(self.task_path, "clusters", "{}.json".format(str(clusters_index).zfill(4)))

    def make_clusters(self) -> List[CustomizePose]:
        clusters_list = []
        cluster_dict_template = {
            "id": None,
            "p0": {},
            "p1": {}
        }
        for frame in range(0, self.total_frames):
            cluster_list = []
            data = get_json_content(self.get_clusters_path(frame))
            for item in data:
                mycluster_dict = copy.deepcopy(cluster_dict_template)
                mycluster_dict["id"] = item["id"]
                p0 = CustomizePose(x=item["p0"]["x"], y=item["p0"]["y"], z=0.0, roll=0.0, yaw=0.0, pitch=0.0)
                p1 = CustomizePose(x=item["p1"]["x"], y=item["p1"]["y"], z=0.0, roll=0.0, yaw=0.0, pitch=0.0)
                mycluster_dict["p0"] = p0
                mycluster_dict["p1"] = p1
                cluster_list.append(mycluster_dict)
            clusters_list.append(cluster_list)
        return clusters_list
    
