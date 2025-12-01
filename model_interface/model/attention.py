

import torch
from torch import nn
from einops import rearrange
from tqdm import trange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # x: [b, n, f]
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # [b, h, n, n]

        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v) # v: [b, h, n, d]   attn: [b, h, n, n]
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out) # [b, n, h*d] -> [b, n, f]
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(), # nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    
class AttentionNetwork(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, n_features, hidden_dim, output_dim):
        super().__init__()
        self.encoder = Transformer(dim, depth, heads, dim_head, mlp_dim,)
        # self.output = nn.Sequential(
        #     nn.Linear(n_features*dim, hidden_dim),
        #     nn.Tanh(),
        #     nn.Linear(hidden_dim, output_dim),
        #     # nn.Tanh(),
        # )
        # self.view_embed = nn.Parameter(torch.zeros(1, n_features, dim))

    def forward(self, x):
        # x = x + self.view_embed
        x = self.encoder(x)
        # x = rearrange(x, 'b n d -> b (n d)')
        return x   #self.output(x)
    
class AttentionNetworkBuiltin(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim,
                 n_features, hidden_dim, output_dim, dropout=0.0):
        super().__init__()

        # 可选：确保 dim 可以被 heads 整除（否则官方 MultiheadAttention 直接报错）
        assert dim % heads == 0, "PyTorch MultiheadAttention 要求 dim % heads == 0"

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",      # 你原来用 Tanh，这里也可以改成 "tanh"
            batch_first=True,       # 输入输出都是 [batch, seq, dim]
            norm_first=True,        # 和你自己的 PreNorm 行为一致
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

    def forward(self, x):
        # x: [b, n_features, dim]
        # 如果你想用 view_embed，就解开这行：
        # x = x + self.view_embed      # [1, n, dim] 会自动 broadcast 到 batch 维
        x = self.encoder(x)           # [b, n, dim]
        return x


    