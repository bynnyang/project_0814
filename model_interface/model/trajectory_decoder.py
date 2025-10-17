import torch
from torch import nn
from timm.models.layers import trunc_normal_

from utils.config import Configuration


class TrajectoryDecoder(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        self.PAD_token = self.cfg.token_nums + self.cfg.append_token - 1

        self.scheduled_sampling_ratio = 1.0  # 初始完全使用真实值
        self.scheduled_sampling_decay_step = 1000  # 每多少步降低一次采样率
        self.scheduled_sampling_decay_rate = 0.98  # 衰减率

        self.embedding = nn.Embedding(self.cfg.token_nums + self.cfg.append_token, self.cfg.tf_de_dim)
        self.pos_drop = nn.Dropout(self.cfg.tf_de_dropout)

        item_cnt = self.cfg.autoregressive_points

        self.pos_embed = nn.Parameter(torch.randn(1, self.cfg.item_number*item_cnt + 2, self.cfg.tf_de_dim) * .02)

        tf_layer = nn.TransformerDecoderLayer(d_model=self.cfg.tf_de_dim, nhead=self.cfg.tf_de_heads)
        self.tf_decoder = nn.TransformerDecoder(tf_layer, num_layers=self.cfg.tf_de_layers)
        self.output = nn.Linear(self.cfg.tf_de_dim, self.cfg.token_nums + self.cfg.append_token)
        self.out_drop = nn.Dropout(self.cfg.tf_de_dropout)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if 'pos_embed' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        trunc_normal_(self.pos_embed, std=.02)

    def update_scheduled_sampling_ratio(self, global_step):
        """更新计划采样比率"""
        if global_step % self.scheduled_sampling_decay_step == 0:
            self.scheduled_sampling_ratio *= self.scheduled_sampling_decay_rate
            # 确保比率不低于最小值
            self.scheduled_sampling_ratio = max(self.scheduled_sampling_ratio, 0.01)

    def create_mask(self, tgt):
        tgt_mask = (torch.triu(torch.ones((tgt.shape[1], tgt.shape[1]), device=self.cfg.device)) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        tgt_padding_mask = (tgt == self.PAD_token)

        return tgt_mask, tgt_padding_mask

    def decoder(self, encoder_out, tgt_embedding, tgt_mask, tgt_padding_mask):
        encoder_out = encoder_out.transpose(0, 1)
        tgt_embedding = tgt_embedding.transpose(0, 1)
        pred_traj_points = self.tf_decoder(tgt=tgt_embedding,
                                        memory=encoder_out,
                                        tgt_mask=tgt_mask,
                                        tgt_key_padding_mask=tgt_padding_mask)
        pred_traj_points = pred_traj_points.transpose(0, 1)
        return pred_traj_points

    def forward(self, encoder_out, point_out, tgt, global_step = None):
        if global_step is not None:
            self.update_scheduled_sampling_ratio(global_step)

        global_context = point_out
        
        # 保存原始目标序列
        original_tgt = tgt.clone()
        tgt = tgt[:, :-1]
        batch_size, seq_len = tgt.size()
        output_sequence = torch.zeros_like(tgt)
        output_sequence[:, 0] = tgt[:, 0]

        for t in range(1, seq_len):
            # 创建当前输入序列
            current_input = output_sequence.clone()[:, :t]
            
            # 创建掩码
            tgt_mask, tgt_padding_mask = self.create_mask(current_input)
            
            # 嵌入
            tgt_embedding = self.embedding(current_input)
            step_global_context = global_context.unsqueeze(1).repeat(1, t, 1)
            tgt_embedding = tgt_embedding + step_global_context
            tgt_embedding = self.pos_drop(tgt_embedding + self.pos_embed[:, :t, :])
            
            # 解码
            pred_traj_points = self.decoder(encoder_out[:,[0]], tgt_embedding, tgt_mask, tgt_padding_mask)
            
            # 获取最后一步的预测
            last_step_pred = pred_traj_points[:, -1, :]
            last_step_pred = self.output(last_step_pred)
            last_step_pred = self.out_drop(last_step_pred)
            
            # 应用softmax并选择最可能的token
            pred_token = torch.softmax(last_step_pred, dim=-1).argmax(dim=-1)
            
            # 计划采样：决定是使用真实值还是预测值
            use_ground_truth = torch.rand(batch_size, device=self.cfg.device) < self.scheduled_sampling_ratio
            next_token = torch.where(use_ground_truth, tgt[:, t], pred_token)
            
            # 更新输出序列
            if t < seq_len:
                output_sequence[:, t] = next_token
        
        final_global_context = global_context.unsqueeze(1).repeat(1, tgt.size(1), 1)



        tgt_mask, tgt_padding_mask = self.create_mask(output_sequence)

        tgt_embedding = self.embedding(output_sequence)
        tgt_embedding = tgt_embedding + final_global_context
        tgt_embedding = self.pos_drop(tgt_embedding + self.pos_embed[:, :seq_len, :])

        pred_traj_points = self.decoder(encoder_out[:,[0]], tgt_embedding, tgt_mask, tgt_padding_mask)
        pred_traj_points = self.output(pred_traj_points)
        pred_traj_points = self.out_drop(pred_traj_points)
        return pred_traj_points
    
    def predict(self, encoder_out, point_out, tgt):
        length = tgt.size(1)
        padding_num = self.cfg.item_number * self.cfg.autoregressive_points + 2 - length

        global_context = point_out.reshape(-1, self.cfg.tf_de_dim)
        
        offset = 1
        if padding_num > 0:
            padding = torch.ones(tgt.size(0), padding_num).fill_(self.PAD_token).long().to(self.cfg.device)
            tgt = torch.cat([tgt, padding], dim=1)

        tgt_mask, tgt_padding_mask = self.create_mask(tgt)
        final_global_context = global_context.unsqueeze(1).repeat(1, tgt.size(1), 1)

        tgt_embedding = self.embedding(tgt)
        tgt_embedding = tgt_embedding + final_global_context
        tgt_embedding = tgt_embedding + self.pos_embed[:, :tgt.size(1), :]

        pred_traj_points = self.decoder(encoder_out[:,[0]], tgt_embedding, tgt_mask, tgt_padding_mask)
        pred_traj_points = self.output(pred_traj_points)[:, length - offset, :]

        pred_traj_points = torch.softmax(pred_traj_points, dim=-1)
        pred_traj_points = pred_traj_points.argmax(dim=-1).view(-1, 1)
        return pred_traj_points
    


# ----------------------------
# ONNX-friendly MultiheadAttention
# ----------------------------
class ONNXMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, key_padding_mask=None):
        # x: B, T, C
        B, T, C = x.size()
        H = self.num_heads
        D = self.head_dim

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 手动 reshape绕过 aten::unflatten
        Q = Q.view(B, T, H, D).transpose(1, 2)  # B, H, T, D
        K = K.view(B, T, H, D).transpose(1, 2)
        V = V.view(B, T, H, D).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]  # B,1,1,T
            scores = scores.masked_fill(mask, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # B, H, T, D

        # 合并头
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        return out


# ----------------------------
# ONNX-friendly Transformer Decoder Layer
# ----------------------------
class ONNXTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = ONNXMultiheadAttention(d_model, num_heads)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, tgt, memory=None, tgt_mask=None, tgt_key_padding_mask=None):
        # Self-attention
        tgt2 = self.self_attn(tgt, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Feedforward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt


# ----------------------------
# ONNX-friendly Transformer Decoder
# ----------------------------
class ONNXTransformerDecoder(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])

    def forward(self, tgt, memory=None, tgt_mask=None, tgt_key_padding_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, tgt_key_padding_mask)
        return tgt


# ----------------------------
# ONNX-friendly TrajectoryDecoder
# ----------------------------
class TrajectoryDecoderONNX(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        self.PAD_token = self.cfg.token_nums + self.cfg.append_token - 1

        self.scheduled_sampling_ratio = 1.0
        self.scheduled_sampling_decay_step = 1000
        self.scheduled_sampling_decay_rate = 0.98

        self.embedding = nn.Embedding(self.cfg.token_nums + self.cfg.append_token, self.cfg.tf_de_dim)
        self.pos_drop = nn.Dropout(self.cfg.tf_de_dropout)

        item_cnt = self.cfg.autoregressive_points
        self.pos_embed = nn.Parameter(torch.randn(1, self.cfg.item_number * item_cnt + 2, self.cfg.tf_de_dim) * .02)

        # 使用 ONNX-friendly Transformer
        tf_layer = ONNXTransformerDecoderLayer(d_model=self.cfg.tf_de_dim, num_heads=self.cfg.tf_de_heads,
                                               dim_feedforward=self.cfg.tf_de_dim*4, dropout=self.cfg.tf_de_dropout)
        self.tf_decoder = ONNXTransformerDecoder(tf_layer, num_layers=self.cfg.tf_de_layers)

        self.output = nn.Linear(self.cfg.tf_de_dim, self.cfg.token_nums + self.cfg.append_token)
        self.out_drop = nn.Dropout(self.cfg.tf_de_dropout)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if 'pos_embed' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        trunc_normal_(self.pos_embed, std=.02)

    def update_scheduled_sampling_ratio(self, global_step):
        if global_step % self.scheduled_sampling_decay_step == 0:
            self.scheduled_sampling_ratio *= self.scheduled_sampling_decay_rate
            self.scheduled_sampling_ratio = max(self.scheduled_sampling_ratio, 0.01)

    def create_mask(self, tgt):
        tgt_mask = (torch.triu(torch.ones((tgt.shape[1], tgt.shape[1]), device=self.cfg.device)) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        tgt_padding_mask = (tgt == self.PAD_token)
        return tgt_mask, tgt_padding_mask

    def decoder(self, encoder_out, tgt_embedding, tgt_mask, tgt_padding_mask):
        encoder_out = encoder_out  # memory 可以保留原形状
        pred_traj_points = self.tf_decoder(tgt=tgt_embedding,
                                           memory=encoder_out,
                                           tgt_mask=tgt_mask,
                                           tgt_key_padding_mask=tgt_padding_mask)
        return pred_traj_points

    def forward(self, encoder_out, point_out, tgt, global_step=None):
        if global_step is not None:
            self.update_scheduled_sampling_ratio(global_step)

        batch_size, seq_len = tgt.size()
        output_sequence = torch.zeros_like(tgt)
        output_sequence[:, 0] = tgt[:, 0]

        global_context = point_out

        for t in range(1, seq_len):
            current_input = output_sequence.clone()[:, :t]
            tgt_mask, tgt_padding_mask = self.create_mask(current_input)

            tgt_embedding = self.embedding(current_input)
            step_global_context = global_context.unsqueeze(1).repeat(1, t, 1)
            tgt_embedding = tgt_embedding + step_global_context
            tgt_embedding = self.pos_drop(tgt_embedding + self.pos_embed[:, :t, :])

            pred_traj_points = self.decoder(encoder_out[:, [0]], tgt_embedding, tgt_mask, tgt_padding_mask)
            last_step_pred = self.output(pred_traj_points[:, -1, :])
            last_step_pred = self.out_drop(last_step_pred)

            pred_token = torch.softmax(last_step_pred, dim=-1).argmax(dim=-1)
            use_ground_truth = torch.rand(batch_size, device=self.cfg.device) < self.scheduled_sampling_ratio
            next_token = torch.where(use_ground_truth, tgt[:, t], pred_token)
            output_sequence[:, t] = next_token

        final_global_context = global_context.unsqueeze(1).repeat(1, tgt.size(1), 1)
        tgt_mask, tgt_padding_mask = self.create_mask(output_sequence)
        tgt_embedding = self.embedding(output_sequence)
        tgt_embedding = tgt_embedding + final_global_context
        tgt_embedding = self.pos_drop(tgt_embedding + self.pos_embed[:, :seq_len, :])

        pred_traj_points = self.decoder(encoder_out[:, [0]], tgt_embedding, tgt_mask, tgt_padding_mask)
        pred_traj_points = self.output(pred_traj_points)
        pred_traj_points = self.out_drop(pred_traj_points)
        return pred_traj_points
    

    def predict(self, encoder_out, point_out, tgt):
        length = tgt.size(1)
        padding_num = self.cfg.item_number * self.cfg.autoregressive_points + 2 - length

        global_context = point_out.reshape(-1, self.cfg.tf_de_dim)
        
        offset = 1
        if padding_num > 0:
            padding = torch.ones(tgt.size(0), padding_num).fill_(self.PAD_token).long().to(self.cfg.device)
            tgt = torch.cat([tgt, padding], dim=1)

        tgt_mask, tgt_padding_mask = self.create_mask(tgt)
        final_global_context = global_context.unsqueeze(1).repeat(1, tgt.size(1), 1)

        tgt_embedding = self.embedding(tgt)
        tgt_embedding = tgt_embedding + final_global_context
        tgt_embedding = tgt_embedding + self.pos_embed[:, :tgt.size(1), :]

        pred_traj_points = self.decoder(encoder_out[:,[0]], tgt_embedding, tgt_mask, tgt_padding_mask)
        pred_traj_points = self.output(pred_traj_points)[:, length - offset, :]

        pred_traj_points = torch.softmax(pred_traj_points, dim=-1)
        pred_traj_points = pred_traj_points.argmax(dim=-1).view(-1, 1)
        return pred_traj_points

    


