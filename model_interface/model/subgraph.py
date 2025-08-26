from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import MessagePassing, max_pool
from torch_geometric.nn import avg_pool_x
from torch_geometric.nn import GCNConv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pdb
import os


class ResBottleneck(nn.Module):
    def __init__(self, in_channels, hidden_unit, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_unit // 4),
            # nn.LayerNorm(hidden_unit // 4),
            nn.GELU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_unit // 4, hidden_unit // 4),
            # nn.LayerNorm(hidden_unit // 4),
            nn.GELU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_unit // 4, hidden_unit),
            # nn.Dropout(dropout)
        )
        self.skip = nn.Linear(in_channels, hidden_unit) if in_channels != hidden_unit else nn.Identity()

    def forward(self, x):
        return F.gelu(self.skip(x) + self.net(x))

class SubGraph(nn.Module):
    """
    Subgraph that computes all vectors in a polyline, and get a polyline-level feature
    """

    def __init__(self, in_channels, num_subgraph_layres=9, hidden_unit=256, max_id = 64, dropout=0.0001, use_residual=True, use_norm=True):
        super(SubGraph, self).__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if use_norm else None
        self.dropout = dropout
        self.use_residual = use_residual
   
        id_dim  = 8                 # 嵌入后的维度

        self.id_emb = nn.Embedding(max_id + 1, id_dim)

        self.feature_encoder = ResBottleneck(in_channels, hidden_unit, dropout=0.0002)
        
        # 输入层
        self.convs.append(GCNConv(hidden_unit, hidden_unit))
        if use_norm:
            self.norms.append(nn.LayerNorm(hidden_unit))
        
        # 隐藏层
        for _ in range(num_subgraph_layres - 2):
            self.convs.append(GCNConv(hidden_unit, hidden_unit))
            if use_norm:
                self.norms.append(nn.LayerNorm(hidden_unit))
        
        # 输出层
        if num_subgraph_layres > 1:
            self.convs.append(GCNConv(hidden_unit, hidden_unit))
            if use_norm:
                self.norms.append(nn.LayerNorm(hidden_unit))

    def forward(self, sub_data):
        """
        polyline vector set in torch_geometric.data.Data format
        args:
            sub_data (Data): [x, y, cluster, edge_index, valid_len]
        """

        geo_feat = sub_data.x[:, :3]                     # 几何特征 (N,3)
        id_index = sub_data.x[:, 3].long()               # id 列 (N,)
        id_feat  = self.id_emb(id_index)             # (N, 8)

        # 拼接
        node_feat = torch.cat([geo_feat, id_feat], dim=-1)  # (N, 11)


        encoder_x = self.feature_encoder(node_feat)

        data = sub_data
        x, edge_index = encoder_x, data.edge_index

        if self.use_residual:
            original_x = x

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            # 应用归一化
            if self.norms is not None:
                x = self.norms[i](x)
            
            # 应用激活函数
            x = F.relu(x)
            
            # 应用dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # 应用残差连接（如果维度匹配）
            if self.use_residual and i == 0 and x.shape[1] == original_x.shape[1]:
                x = x + original_x

        data.x = x
        out_data = avg_pool_x(data.cluster, data.x, data.batch)
        # try:
        # assert out_data[0].shape[0] % int(data["time_step_len"][0]) == 0
        # except:
            # from pdb import set_trace; set_trace()
        norm_x = F.normalize(out_data[0], p=2, dim=0, eps=1e-6)
        return norm_x
        # node_feature, _ = torch.max(x, dim=0)
        # # l2 noramlize node_feature before feed it to global graph
        # node_feature = node_feature / node_feature.norm(dim=0)
        # return node_feature

# %%


class GraphLayerProp(MessagePassing):
    """
    Message Passing mechanism for infomation aggregation
    """

    def __init__(self, in_channels, hidden_unit=64, verbose=False):
        super(GraphLayerProp, self).__init__(
            aggr='max')  # MaxPooling aggragation
        self.verbose = verbose
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU(),
            nn.Linear(hidden_unit, in_channels)
        )

    def forward(self, x, edge_index):
        if self.verbose:
            print(f'x before mlp: {x}')
        x = self.mlp(x)
        if self.verbose:
            print(f"x after mlp: {x}")
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, x):
        if self.verbose:
            print(f"x after mlp: {x}")
            print(f"aggr_out: {aggr_out}")
        return torch.cat([x, aggr_out], dim=1)


if __name__ == "__main__":
    data = Data(x=torch.tensor([[1.0], [7.0]]), edge_index=torch.tensor([[0, 1], [1, 0]]))
    print(data)
    layer = GraphLayerProp(1, 1, True)
    for k, v in layer.state_dict().items():
        if k.endswith('weight'):
            v[:] = torch.tensor([[1.0]])
        elif k.endswith('bias'):
            v[:] = torch.tensor([1.0])
    y = layer(data.x, data.edge_index)