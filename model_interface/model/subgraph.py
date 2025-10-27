from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import MessagePassing, max_pool
from torch_geometric.nn import avg_pool_x
# from torch_geometric.nn import GCNConv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pdb
import os

import torch
import torch.nn as nn


class MyGCNConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = False,
        normalize: bool = True,
        bias: bool = True,
    ):
        super(MyGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        # 参数
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        # 缓存
        self.cached_edge_index = None
        self.cached_norm = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.cached_edge_index = None
        self.cached_norm = None

    def forward(self, x, edge_index, edge_weight=None):
        device = x.device
        N = x.size(0)

        # === 归一化边权 ===
        if self.normalize:
            if self.cached and self.cached_edge_index is not None:
                edge_index, norm = self.cached_edge_index, self.cached_norm
            else:
                edge_index, norm = self.gcn_norm(
                    edge_index, edge_weight, N, self.improved, self.add_self_loops, device
                )
                if self.cached:
                    self.cached_edge_index = edge_index
                    self.cached_norm = norm
        else:
            norm = edge_weight if edge_weight is not None else torch.ones(edge_index.size(1), device=device)

        # === 线性变换 ===
        x = x @ self.weight

        # === 消息传递 ===
        # row, col = edge_index
        edge_index_split = edge_index.to(torch.float32)
        row, col = edge_index_split
        row = row.long()
        col = col.long()
        out = torch.zeros_like(x)
        # out.index_add_(0, row, norm.unsqueeze(1) * x[col])
        def index_add_manual(out, row, updates):
            """
            手动实现 out.index_add_(0, row, updates)

            Args:
                out: (N, F) 张量，存放累加结果
                row: (E,) 长度索引张量，指明 updates 写入 out 的行
                updates: (E, F) 张量，沿行累加到 out
            Returns:
                out: 累加后的张量
            """
            for i in range(row.size(0)):
                out[row[i]] += updates[i]
            return out
        updates = norm.unsqueeze(1) * x[col]
        out = index_add_manual(out, row, updates)

        # === 加偏置 ===
        if self.bias is not None:
            out += self.bias

        return out

    @staticmethod
    def gcn_norm(edge_index, edge_weight, num_nodes, improved=False, add_self_loops=True, device=None):
        """
        安全版 gcn_norm()，解决：
        1. 度为0导致 inf
        2. 自环重复
        3. improved 只影响自环
        """
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=device)

        # === 移除已有自环 ===
        # if add_self_loops:
        #     mask = edge_index[0] != edge_index[1]
        #     edge_index = edge_index[:, mask]
        #     edge_weight = edge_weight[mask]

        #     # 添加自环
        #     loop_index = torch.arange(num_nodes, device=device)
        #     loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        #     loop_val = torch.full((num_nodes,), 2.0 if improved else 1.0, device=device)
        #     edge_index = torch.cat([edge_index, loop_index], dim=1)
        #     edge_weight = torch.cat([edge_weight, loop_val])

        # === 计算度矩阵 ===
        # row, col = edge_index
        edge_index_split = edge_index.to(torch.float32)
        row, col = edge_index_split
        row = row.long()
        col = col.long()
        # deg = torch.zeros(num_nodes, device=device)
        # for i in range(row.size(0)):
        #     deg[row[i]] += edge_weight[i]
        # deg = torch.zeros(num_nodes, device=device).scatter_add_(0, row.long(), edge_weight)
        deg = torch.zeros(num_nodes, device=device)

        for i in range(row.size(0)):
            deg[row[i]] += edge_weight[i]

        # 防止度为0导致 inf
        # deg_inv_sqrt = deg.pow(-0.5)
        # deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        deg_inv_sqrt = deg.pow(-0.5) * (deg > 0).float()

        # === 归一化边权 ===
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return edge_index, norm

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
        )
        self.skip = nn.Linear(in_channels, hidden_unit) if in_channels != hidden_unit else nn.Identity()

    def forward(self, x):
        return F.gelu(self.skip(x) + self.net(x))

class SubGraph(nn.Module):
    """
    Subgraph that computes all vectors in a polyline, and get a polyline-level feature
    """

    def __init__(self, in_channels, num_subgraph_layres=9, hidden_unit=256, max_id = 64, dropout=0.001, use_residual=False, use_norm=False):
        super(SubGraph, self).__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if use_norm else None
        self.dropout = dropout
        self.use_residual = use_residual
   
        id_dim  = 8                 # 嵌入后的维度

        self.id_emb = nn.Embedding(np.int32(max_id + 1), np.int32(id_dim))

        self.feature_encoder = ResBottleneck(np.int32(in_channels), np.int32(hidden_unit), dropout=0.001)
        
        # 输入层
        self.convs.append(MyGCNConv(np.int32(hidden_unit), np.int32(hidden_unit)))
        if use_norm:
            self.norms.append(nn.LayerNorm(np.int32(hidden_unit)))
        
        # 隐藏层
        for _ in range(num_subgraph_layres - 2):
            self.convs.append(MyGCNConv(np.int32(hidden_unit), np.int32(hidden_unit)))
            if use_norm:
                self.norms.append(nn.LayerNorm(np.int32(hidden_unit)))
        
        # 输出层
        if num_subgraph_layres > 1:
            self.convs.append(MyGCNConv(np.int32(hidden_unit), np.int32(hidden_unit)))
            if use_norm:
                self.norms.append(nn.LayerNorm(np.int32(hidden_unit)))

    def forward(self, sub_data):
        """
        polyline vector set in torch_geometric.data.Data format
        args:
            sub_data (Data): [x, y, cluster, edge_index, valid_len]
        """
        sub_data.cluster = sub_data.cluster.to(torch.int64)
        sub_data.edge_index = sub_data.edge_index.to(torch.int64)
        sub_data.valid_len = sub_data.valid_len.to(torch.int64)
        sub_data.time_step_len = sub_data.time_step_len.to(torch.int64)
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
            
            if self.norms is not None:
                x = self.norms[i](x)
            
            x = F.relu(x)

            if self.use_residual and i == 0 and x.shape[1] == original_x.shape[1]:
                x = x + original_x

        data.x = x

        # num_clusters = data.cluster.max() + 1
        # out = torch.zeros((num_clusters, data.x.size(1)), device=data.x.device)
        # for i in range(num_clusters):
        #     mask = (data.cluster == i)
        #     out[i] = data.x[mask].max(dim=0)[0]
#         num_nodes = data.x.size(0)
        num_clusters = data.cluster.max() + 1
        

# 生成 one-hot: [N, num_clusters]
        cluster_onehot = torch.nn.functional.one_hot(data.cluster.long(), num_clusters).float()

# 由于 max 不能直接用 one-hot 乘法表示最大值，只能模拟 min/max 通过mask：
        masked = data.x.unsqueeze(1) * cluster_onehot.unsqueeze(2)  # [N, num_clusters, F]
        out = masked.max(dim=0)[0]  # [num_clusters, F]
        # norm_x = F.normalize(out_data.x, p=2, dim=0, eps=1e-6)
        # return norm_x
        return out
        # node_feature, _ = torch.max(x, dim=0)
        # # l2 noramlize node_feature before feed it to global graph
        # node_feature = node_feature / node_feature.norm(dim=0)
        # return node_feature

