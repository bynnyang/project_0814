# %%

import os
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data, DataLoader
import torch
from tqdm import tqdm


# %%
def get_fc_edge_index(num_nodes, start=0):
    """
    return a tensor(2, edges), indicing edge_index
    """
    to_ = np.arange(num_nodes, dtype=np.int64)
    edge_index = np.empty((2, 0))
    for i in range(num_nodes):
        from_ = np.ones(num_nodes, dtype=np.int64) * i
        # FIX BUG: no self loop in ful connected nodes graphs
        edge_index = np.hstack((edge_index, np.vstack((np.hstack([from_[:i], from_[i+1:]]), np.hstack([to_[:i], to_[i+1:]])))))
    edge_index = edge_index + start

    return edge_index.astype(np.int64), num_nodes + start


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
        edge_index = np.hstack((edge_index, np.vstack((i, i))))
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


# %%




class GraphData(Data):
    """
    override key `cluster` indicating which polyline_id is for the vector
    """

    def __inc__(self, key, value,index):
        if key == 'edge_index':
            return self.x.size(0)
        elif key == 'cluster':
            return int(self.cluster.max().item()) + 1
        else:
            return 0

# %%


class GraphDataset(InMemoryDataset):
    """
    dataset object similar to `torchvision` 
    """

    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0],weights_only = False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['dataset.pt']

    def download(self):
        pass

    def process(self):

        def get_data_path_ls(dir_):
            return [os.path.join(dir_, data_path) for data_path in os.listdir(dir_)]
        
        # make sure deterministic results
        mcap_floder = sorted(get_data_path_ls(self.root))

        valid_len_ls = []
        data_ls = []
        for each_floder in tqdm(mcap_floder):
            name = os.path.splitext(os.path.basename(each_floder))[0]
            if not name.startswith("features"):
                continue
            data_path_ls = sorted(get_data_path_ls(each_floder)) 
            for data_p in tqdm(data_path_ls):
                if not data_p.endswith('pkl'):
                    continue
                x_ls = []
                y = np.array([1])
                cluster = None
                edge_index_ls = []
                data = pd.read_pickle(data_p)
                all_in_features = data['POLYLINE_FEATURES'].values[0]
                agen_len = data['AGENT_LEN'].values[0]
                park_slot_len = data['PARK_SLOT_LEN'].values[0]
                cluster_len = data['CLUSTER_LEN'].values[0]
                cluster = all_in_features[:, -1].reshape(-1).astype(np.int32)
                valid_len_ls.append(cluster.max())

                agent_mask, park_slot_mask, clusert_mask = data["AGENT_ID_TO_MASK"].values[0], data['PARK_SLOT_ID_TO_MASK'].values[0], data['CLUSTER_ID_TO_MASK'].values[0]
                agent_id = 0
                edge_index_start = 0
                assert all_in_features[agent_id][
                    -1] == 0, f"agent id is wrong. id {agent_id}: type {all_in_features[agent_id][4]}"

                for id_, mask_ in agent_mask.items():
                    data_ = all_in_features[mask_[0]:mask_[1]]
                    edge_index_, edge_index_start = get_agent_edge_index(
                        data_.shape[0], start=edge_index_start)
                    x_ls.append(data_)
                    edge_index_ls.append(edge_index_)

                for id_, mask_ in park_slot_mask.items():
                    data_ = all_in_features[mask_[0]+agen_len: mask_[1]+agen_len]
                    edge_index_, edge_index_start = get_park_slot_edge_index(
                        data_.shape[0], edge_index_start)
                    x_ls.append(data_)
                    edge_index_ls.append(edge_index_)
                
                for id_, mask_ in clusert_mask.items():
                    data_ = all_in_features[mask_[0] + agen_len + park_slot_len: mask_[1] + agen_len + park_slot_len]
                    edge_index_, edge_index_start = get_park_slot_edge_index(
                        data_.shape[0], edge_index_start)
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
            tup[0] = np.vstack(
                [tup[0], np.zeros((padd_to_index - tup[-2].max(), feature_len), dtype=tup[0].dtype)])
            tup[-2] = np.hstack(
                [tup[2], np.arange(tup[-2].max()+1, padd_to_index+1)])
            g_data = GraphData(
                x=torch.from_numpy(tup[0]),
                y=torch.from_numpy(tup[1]),
                cluster=torch.from_numpy(tup[2]),
                edge_index=torch.from_numpy(tup[3]),
                valid_len=torch.tensor([valid_len_ls[ind]]),
                time_step_len=torch.tensor([padd_to_index + 1])
            )
            g_ls.append(g_data)
        data, slices = self.collate(g_ls)
        torch.save((data, slices), self.processed_paths[0])


# %%
if __name__ == "__main__":

    DATA_DIR = "./e2e_dataset"
    INTERMEDIATE_DATA_DIR = './interm_data'
    for folder in os.listdir(DATA_DIR):
        dataset_input_path = os.path.join(
            INTERMEDIATE_DATA_DIR, f"{folder}_intermediate")
        dataset = GraphDataset(dataset_input_path)
        dataset_x = dataset[0]
        batch_iter = DataLoader(dataset, batch_size=256)
        batch = next(iter(batch_iter))



# %%
