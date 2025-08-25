from torch_geometric.data import Data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import MessagePassing, max_pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pdb
import os


# def masked_softmax(X, valid_len):
#     """
#     masked softmax for attention scores
#     args:
#         X: 3-D tensor, valid_len: 1-D or 2-D tensor
#     """
#     if valid_len is None:
#         return nn.functional.softmax(X, dim=-1)
#     else:
#         shape = X.shape
#         if valid_len.dim() == 1:
#             valid_len = torch.repeat_interleave(
#                 valid_len, repeats=shape[1], dim=0)
#         else:
#             valid_len = valid_len.reshape(-1)
#         # Fill masked elements with a large negative, whose exp is 0
#         X = X.reshape(-1, shape[-1])
#         for count, row in enumerate(X):
#             try:
#                 X[count][int(valid_len[count]):] = -1e6
#             except:
#                 break
#         return nn.functional.softmax(X.reshape(shape), dim=-1)


# class SelfAttentionLayer(nn.Module):
#     """
#     Self-attention layer. no scale_factor d_k
#     """

#     def __init__(self, in_channels, global_graph_width, need_scale=False):
#         super(SelfAttentionLayer, self).__init__()
#         self.in_channels = in_channels
#         self.q_lin = nn.Linear(in_channels, global_graph_width)
#         self.k_lin = nn.Linear(in_channels, global_graph_width)
#         self.v_lin = nn.Linear(in_channels, global_graph_width)
#         self.scale_factor_d = 1 + \
#             int(np.sqrt(self.in_channels)) if need_scale else 1

#     def forward(self, x, valid_len):
#         # print(x.shape)
#         # print(self.q_lin)
#         query = self.q_lin(x)
#         key = self.k_lin(x)
#         value = self.v_lin(x)
#         scores = torch.bmm(query, key.transpose(1, 2))
#         attention_weights = masked_softmax(scores, valid_len + 1)
#         return torch.bmm(attention_weights, value)


class SelfAttentionLayer(nn.Module):
    def __init__(self, in_channels, global_graph_width, num_heads=8):
        super().__init__()
        assert global_graph_width % num_heads == 0
        self.mha = nn.MultiheadAttention(
            embed_dim=global_graph_width,
            num_heads=num_heads,
            batch_first=True
        )
        self.lin = nn.Linear(in_channels, global_graph_width)
        self.norm = nn.LayerNorm(global_graph_width)

    def forward(self, x, valid_len):
        x = self.lin(x)
        mask = torch.arange(x.size(1), device=x.device)[None, :] > valid_len[:, None]
        out, _ = self.mha(x, x, x, key_padding_mask=mask)
        return self.norm(out + x)   # 残差 + LayerNorm
