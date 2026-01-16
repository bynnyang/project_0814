import torch
from torch import nn
from timm.models.layers import trunc_normal_

from utils.config import Configuration
import numpy as np
from vehicle_config import * 
import copy


class TrajectoryDecoder(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        self.PAD_token = self.cfg.token_nums + self.cfg.append_token - 1

        self.scheduled_sampling_ratio = 1.0  # 初始完全使用真实值
        self.scheduled_sampling_decay_step = 100  # 每多少步降低一次采样率
        self.scheduled_sampling_decay_rate = 0.97  # 衰减率

        self.traj_embedding = nn.Linear(4, self.cfg.tf_de_dim)
        self.pos_drop = nn.Dropout(self.cfg.tf_de_dropout)

        item_cnt = self.cfg.autoregressive_points - 1

        self.pos_embed = nn.Parameter(torch.randn(1, item_cnt, self.cfg.tf_de_dim) * .02)

        tf_layer = nn.TransformerDecoderLayer(d_model=self.cfg.tf_de_dim, nhead=self.cfg.tf_de_heads, dim_feedforward = 512, dropout= 0.05, activation="gelu")
        self.tf_decoder = nn.TransformerDecoder(tf_layer, num_layers=self.cfg.tf_de_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(self.cfg.tf_de_dim, 2),
            nn.Tanh()
        )

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
            print(f"scheduled_sampling_ratio , {self.scheduled_sampling_ratio:6f}")

    def create_mask(self, tgt):
        tgt_mask = (torch.triu(torch.ones((tgt.shape[1], tgt.shape[1]), device=self.cfg.device)) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        tgt_padding_mask = (tgt == self.PAD_token).all(dim=-1)

        return tgt_mask, tgt_padding_mask
    
    def kinematic_step(self, prev_point, action):
        """
        prev_point: [B, 4] = (x_norm, y_norm, cos_yaw, sin_yaw)
        action:     [B, 2] = (v_norm, delta_norm)  (tanh 输出)
        return:     [B, 4] = 下一步的 (x_norm, y_norm, cos_yaw, sin_yaw)
        """

        # 1. 反归一化上一步的状态
        x = prev_point[:, 0] * self.cfg.traj_x_range       # [B]
        y = prev_point[:, 1] * self.cfg.traj_y_range       # [B]
        cos_yaw = prev_point[:, 2]
        sin_yaw = prev_point[:, 3]
        yaw = torch.atan2(sin_yaw, cos_yaw)           # [-pi, pi]

        # 2. 反归一化动作（根据你自己的映射方式调整）
        # 假设 last_step_pred_action ∈ [-1,1]（tanh 输出）
        delta_norm    = action[:, 0]
        v_norm = action[:, 1]

        v     = v_norm * VALID_SPEED[1]      # 映射到 [0, v_max]，你也可以直接 v = v_norm * cfg.v_max
        delta = delta_norm * VALID_STEER[1]            # [-steer_max, steer_max]

        # 3. 单轨运动学模型离散更新
        dt = STEP_TIME_AND_LENGHT
        L  = WHEEL_BASE

        x_next   = x + v * torch.cos(yaw) * dt
        y_next   = y + v * torch.sin(yaw) * dt
        yaw_next = yaw + v / L * torch.tan(delta) * dt

        # 4. wrap yaw 到 [-pi, pi]（可微写法）
        yaw_next = torch.atan2(torch.sin(yaw_next), torch.cos(yaw_next))

        # 5. 再归一化 x,y，并重新编码 yaw 为 cos/sin
        x_next_norm = x_next / self.cfg.traj_x_range
        y_next_norm = y_next / self.cfg.traj_y_range

        cos_next = torch.cos(yaw_next)
        sin_next = torch.sin(yaw_next)

        pred_point = torch.stack([x_next_norm, y_next_norm, cos_next, sin_next], dim=-1)  # [B,4]
        return pred_point

    def decoder(self, encoder_out, tgt_embedding, tgt_mask, tgt_key_padding_mask = None):
        encoder_out = encoder_out.transpose(0, 1)
        tgt_embedding = tgt_embedding.transpose(0, 1)
        pred_traj_points = self.tf_decoder(tgt=tgt_embedding,
                                        memory=encoder_out,
                                        tgt_mask=tgt_mask,
                                        tgt_key_padding_mask = tgt_key_padding_mask)
        pred_traj_points = pred_traj_points.transpose(0, 1)
        return pred_traj_points

    def forward(self, encoder_out, point_out, tgt, global_step = None):
        if global_step is not None:
            self.update_scheduled_sampling_ratio(global_step)

        global_context = point_out
        
        # 保存原始目标序列
        original_tgt = tgt.clone()
        tgt = tgt[:, :-1, :]
        batch_size, seq_len, feat_dim = tgt.size()
        assert feat_dim == 4
        output_sequence = torch.zeros_like(tgt)
        output_sequence[:, 0,:] = tgt[:, 0, :]

        for t in range(1, seq_len):
            # 创建当前输入序列
            current_input = output_sequence[:, :t, :].detach()
            
            # 创建掩码
            tgt_mask, tgt_padding_mask = self.create_mask(current_input)
            
            # 嵌入
            tgt_embedding = self.traj_embedding(current_input)
            step_global_context = global_context.unsqueeze(1).repeat(1, t, 1)
            tgt_embedding = tgt_embedding + step_global_context
            tgt_embedding = self.pos_drop(tgt_embedding + self.pos_embed[:, :t, :])
            
            # 解码
            pred_actions_logtis = self.decoder(encoder_out, tgt_embedding, tgt_mask)
            
            # 获取最后一步的预测
            last_step_pred_action_logti = pred_actions_logtis[:, -1, :]
            last_step_pred_action = self.output_layer(last_step_pred_action_logti)
            
            prev_point = output_sequence[:, t-1, :].detach()          # [B,4]
            pred_point = self.kinematic_step(prev_point, last_step_pred_action)  # [B,4]
            
            # 计划采样：决定是使用真实值还是预测值
            use_ground_truth = torch.rand(batch_size, 1, device=self.cfg.device) < self.scheduled_sampling_ratio
            next_token = torch.where(use_ground_truth, tgt[:, t, :], pred_point)
            
            # 更新输出序列
            if t < seq_len:
                output_sequence[:, t, :] = next_token
        
        final_global_context = global_context.unsqueeze(1).repeat(1, tgt.size(1), 1)

        output_seq_detached = output_sequence.detach()

        tgt_mask, tgt_padding_mask = self.create_mask(output_seq_detached)

        tgt_embedding = self.traj_embedding(output_seq_detached)
        tgt_embedding = tgt_embedding + final_global_context
        tgt_embedding = self.pos_drop(tgt_embedding + self.pos_embed[:, :seq_len, :])

        pred_actions_logtis = self.decoder(encoder_out, tgt_embedding, tgt_mask)
        pred_actions = self.output_layer(pred_actions_logtis)
        return pred_actions
    
    def predict(self, encoder_out, point_out, tgt):
        batch_size, length, feat_dim = tgt.size()
        assert feat_dim == 4

        padding_num = self.cfg.autoregressive_points -1 - length

        global_context = point_out.reshape(-1, self.cfg.tf_de_dim)
        
        offset = 1
        if padding_num > 0:
            padding = torch.ones(batch_size, padding_num, 4, device=self.cfg.device) * self.PAD_token
            tgt = torch.cat([tgt, padding], dim=1)

        tgt_mask, tgt_padding_mask = self.create_mask(tgt)
        final_global_context = global_context.unsqueeze(1).repeat(1, tgt.size(1), 1)

        tgt_embedding = self.traj_embedding(tgt)
        tgt_embedding = tgt_embedding + final_global_context
        tgt_embedding = tgt_embedding + self.pos_embed[:, :tgt.size(1), :]

        pred_actions_logtis = self.decoder(encoder_out, tgt_embedding, tgt_mask, tgt_padding_mask)

        last_step_pred_action_logti = pred_actions_logtis[:, length - offset, :]
        
        pred_actions = self.output_layer(last_step_pred_action_logti)

        prev_point = tgt[:, length - 1, :]         # [B,4]
        pred_point = self.kinematic_step(prev_point, pred_actions)  # [B,4]

        return pred_point
    


# ----------------------------
# ONNX-friendly MultiheadAttention
# ----------------------------
    
class ONNXMultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
 
        assert (
             self.d_k * n_heads == d_model
         ), f"d_model {d_model} not divisible by n_heads {n_heads}"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, tgt_mask=None, tgt_key_padding_mask=None):
         # Q: (batch_size, n_heads, seq_len, d_k)
         # K: (batch_size, n_heads, seq_len, d_k)
         # V: (batch_size, n_heads, seq_len, d_k)

        # scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (batch_size, n_heads, seq_len, seq_len)
        # if tgt_mask is not None:
        #     # 如果是 2D mask（L, L），需要 broadcast 到 batch/n_heads
        #     if tgt_mask.dim() == 2:
        #         tgt_mask = tgt_mask[None, None,  :, :] 
        #     scores = scores + tgt_mask.to(scores.device)
        # if tgt_key_padding_mask is not None:
        #     mask = tgt_key_padding_mask[:, None, None, :]  # B,1,1,T
        #     scores = scores.masked_fill(mask, float('-inf'))
         
        # attn_weights = torch.softmax(scores, dim=-1)    # (batch_size, n_heads, seq_len, seq_len)
        # attn_weights = self.dropout(attn_weights)    # apply dropout to attention weights
        # output = torch.matmul(attn_weights, V)    # (batch_size, n_heads, seq_len, d_k)
        # return output
        LARGE_NEG = 1e9
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        if tgt_mask is not None:
            # ensure additive numeric mask
            if tgt_mask.dim() == 2:
                additive = tgt_mask[None, None, :, :].to(dtype=scores.dtype, device=scores.device)
                scores = scores + additive
            else:
                scores = scores + tgt_mask.to(dtype=scores.dtype, device=scores.device)

        if tgt_key_padding_mask is not None:
            mask = tgt_key_padding_mask[:, None, None, :].to(dtype=scores.dtype, device=scores.device)
            scores = scores - mask * LARGE_NEG

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)
        return output
     
    def forward(self, Q, K, V, tgt_mask=None, tgt_key_padding_mask=None):
         # Q: (batch_size, seq_len, d_model)
         # K: (batch_size, seq_len, d_model)
         # V: (batch_size, seq_len, d_model)

        batch_size = Q.size(0)

         # (batch_size, seq_len, d_model) -> (batch_size, n_heads, seq_len, d_k)
        Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)    # (batch_size, n_heads, seq_len, d_k)
        K = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)    # (batch_size, n_heads, seq_len, d_k)
        V = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)    # (batch_size, n_heads, seq_len, d_k)

         # scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)    # (batch_size, n_heads, seq_len, d_k)

         # (batch_size, n_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)    # (batch_size, seq_len, d_model)
        output = self.W_o(attn_output)    # (batch_size, seq_len, d_model)
        return output    # (batch_size, seq_len, d_model)
    

class ONNXFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
         # x: (batch_size, seq_len, d_model)
        x = self.linear1(x)    # (batch_size, seq_len, d_ff)
        x = self.activation(x)    # (batch_size, seq_len, d_ff)
        x = self.dropout(x)    # (batch_size, seq_len, d_ff)
        x = self.linear2(x)    # (batch_size, seq_len, d_model)
        return x    # (batch_size, seq_len, d_model)


# ----------------------------
# ONNX-friendly Transformer Decoder Layer
# ----------------------------

class ONNXTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = ONNXMultiheadAttention(d_model, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = ONNXMultiheadAttention(d_model, n_heads, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = ONNXFeedForward(d_model, d_ff, dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, src, tgt_mask=None, tgt_key_padding_mask=None):
        if src is None:
            src = torch.zeros(tgt.size(0), 1, tgt.size(2), device=tgt.device, dtype=tgt.dtype)
        elif src.size(1) == 0:
            src = torch.zeros(src.size(0), 1, src.size(2), device=src.device, dtype=src.dtype)

        # x = tgt
        # output = self.self_attn(x, x, x, tgt_mask, tgt_key_padding_mask)
        # x = self.norm1(x + self.dropout1(output))

        # output = self.cross_attn(x, src, src)
        # x = self.norm2(x + self.dropout2(output))

        # output = self.ffn(x)
        # x = self.norm3(x + self.dropout3(output))
        x = tgt
        sa = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x),
                        tgt_mask, tgt_key_padding_mask)
        x = x + self.dropout1(sa)

        # cross-attn
        ca = self.cross_attn(self.norm2(x), src, src)
        x = x + self.dropout2(ca)

        # ffn
        ff = self.ffn(self.norm3(x))
        x = x + self.dropout3(ff)
        '''
        等价的norm_first=True的写法，强化学习PPO重新使用该方法

        x = tgt
        sa = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x),
                        tgt_mask, tgt_key_padding_mask)
        x = x + self.dropout1(sa)

        # cross-attn
        ca = self.cross_attn(self.norm2(x), src, src)
        x = x + self.dropout2(ca)

        # ffn
        ff = self.ffn(self.norm3(x))
        x = x + self.dropout3(ff)
        
        x = self.final_norm(x) #经过实际测试，预训练阶段可以取得更好的效果

        '''
        return x


# ----------------------------
# ONNX-friendly Transformer Decoder
# ----------------------------
class ONNXTransformerDecoder(nn.Module):
    def __init__(self, layer, num_layers, d_model, use_final_norm: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model) if use_final_norm else None

    def forward(self, tgt, memory=None, tgt_mask=None, tgt_key_padding_mask=None):
        x = tgt
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, tgt_key_padding_mask)
        if self.final_norm is not None:
            x = self.final_norm(x)
        return x


# ----------------------------
# ONNX-friendly TrajectoryDecoder
# ----------------------------
class TrajectoryDecoderONNX(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        self.PAD_token = self.cfg.token_nums + self.cfg.append_token - 1

        self.scheduled_sampling_ratio = 1.0
        self.scheduled_sampling_decay_step = 100
        self.scheduled_sampling_decay_rate = 0.97

        self.traj_embedding = nn.Linear(4, self.cfg.tf_de_dim)
        self.pos_drop = nn.Dropout(self.cfg.tf_de_dropout)

        item_cnt = self.cfg.autoregressive_points - 1

        self.pos_embed = nn.Parameter(torch.randn(1, item_cnt, self.cfg.tf_de_dim) * .02)
        # 使用 ONNX-friendly Transformer
        tf_layer = ONNXTransformerDecoderLayer(d_model=self.cfg.tf_de_dim, n_heads=self.cfg.tf_de_heads, d_ff=512, dropout=0.0)
        self.tf_decoder = ONNXTransformerDecoder(tf_layer, num_layers=self.cfg.tf_de_layers, d_model=self.cfg.tf_de_dim, use_final_norm=True)
        self.output_layer = nn.Sequential(
            nn.Linear(self.cfg.tf_de_dim, 2),
            nn.Tanh()
        )

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
        # mask = (torch.arange(tgt.shape[1]).unsqueeze(1) >= torch.arange(tgt.shape[1]).unsqueeze(0)).float()
        # tgt_mask = mask.to(self.cfg.device)
        # # tgt_mask = (torch.triu(torch.ones((tgt.shape[1], tgt.shape[1]), device=self.cfg.device)) == 1).transpose(0, 1)
        # tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        # tgt_padding_mask = (tgt == self.PAD_token)
        # return tgt_mask, tgt_padding_mask
        # ONNX-safe numeric additive mask
        L = tgt.shape[1]
        device = tgt.device
        causal = (torch.arange(L, device=device).unsqueeze(1) >= torch.arange(L, device=device).unsqueeze(0)).float()
        additive = causal.clone().to(dtype=torch.float32)
        additive = additive.masked_fill(additive == 0, -1e9).masked_fill(additive == 1, 0.0)
        tgt_mask = additive.unsqueeze(0).unsqueeze(0)  # (1,1,L,L)
        tgt_padding_mask = (tgt == self.PAD_token).all(dim=-1)
        return tgt_mask, tgt_padding_mask
    
    @staticmethod
    def _ensure_nonempty_memory(mem):
        if mem is None:
            return None
        if mem.size(1) == 0:
            return torch.zeros(mem.size(0), 1, mem.size(2), device=mem.device, dtype=mem.dtype)
        return mem
    
    def kinematic_step(self, prev_point, action):
        """
        prev_point: [B, 4] = (x_norm, y_norm, cos_yaw, sin_yaw)
        action:     [B, 2] = (v_norm, delta_norm)  (tanh 输出)
        return:     [B, 4] = 下一步的 (x_norm, y_norm, cos_yaw, sin_yaw)
        """

        # 1. 反归一化上一步的状态
        x = prev_point[:, 0] * self.cfg.traj_x_range       # [B]
        y = prev_point[:, 1] * self.cfg.traj_y_range       # [B]
        cos_yaw = prev_point[:, 2]
        sin_yaw = prev_point[:, 3]
        yaw = torch.atan2(sin_yaw, cos_yaw)           # [-pi, pi]

        # 2. 反归一化动作（根据你自己的映射方式调整）
        # 假设 last_step_pred_action ∈ [-1,1]（tanh 输出）
        delta_norm    = action[:, 0]
        v_norm = action[:, 1]

        v     = v_norm * VALID_SPEED[1]      # 映射到 [-v_max, v_max]，你也可以直接 v = v_norm * cfg.v_max
        delta = delta_norm * VALID_STEER[1]            # [-steer_max, steer_max]

        # 3. 单轨运动学模型离散更新
        dt = STEP_TIME_AND_LENGHT
        L  = WHEEL_BASE

        x_next   = x + v * torch.cos(yaw) * dt
        y_next   = y + v * torch.sin(yaw) * dt
        yaw_next = yaw + v / L * torch.tan(delta) * dt

        # 4. wrap yaw 到 [-pi, pi]（可微写法）
        yaw_next = torch.atan2(torch.sin(yaw_next), torch.cos(yaw_next))

        # 5. 再归一化 x,y，并重新编码 yaw 为 cos/sin
        x_next_norm = x_next / self.cfg.traj_x_range
        y_next_norm = y_next / self.cfg.traj_y_range

        cos_next = torch.cos(yaw_next)
        sin_next = torch.sin(yaw_next)

        pred_point = torch.stack([x_next_norm, y_next_norm, cos_next, sin_next], dim=-1)  # [B,4]
        return pred_point

    def decoder(self, encoder_out, tgt_embedding, tgt_mask, tgt_padding_mask = None):
        # encoder_out = encoder_out  # memory 可以保留原形状
        encoder_out_safe = self._ensure_nonempty_memory(encoder_out)
        pred_traj_points = self.tf_decoder(tgt=tgt_embedding,
                                           memory=encoder_out_safe,
                                           tgt_mask=tgt_mask,
                                           tgt_key_padding_mask=tgt_padding_mask)
        return pred_traj_points

    def forward_sigle_step(self, encoder_out, point_out, tgt, global_step=None):
        if global_step is not None:
            self.update_scheduled_sampling_ratio(global_step)

        global_context = point_out
        
        # 保存原始目标序列
        original_tgt = tgt.clone()
        # tgt = tgt[:, :-1, :]
        batch_size, seq_len, feat_dim = tgt.size()
        assert feat_dim == 4
        max_possible_len = self.cfg.autoregressive_points -1
        output_sequence = torch.zeros(batch_size, max_possible_len, feat_dim).to(self.cfg.device)
        output_sequence[:, 0,:] = tgt[:, 0, :]
        pred_actions_list = []
        step_pred_action_logti_list = []

        for t in range(1, self.cfg.autoregressive_points - 1):
            # 创建当前输入序列
            current_input = output_sequence[:, :t, :].detach().clone()
            
            # 创建掩码
            tgt_mask, tgt_padding_mask = self.create_mask(current_input)
            
            # 嵌入
            tgt_embedding = self.traj_embedding(current_input)
            step_global_context = global_context.unsqueeze(1).repeat(1, t, 1)
            tgt_embedding = tgt_embedding + step_global_context
            tgt_embedding = self.pos_drop(tgt_embedding + self.pos_embed[:, :t, :])
            
            # 解码
            pred_actions_logtis = self.decoder(encoder_out, tgt_embedding, tgt_mask)
            
            # 获取最后一步的预测
            last_step_pred_action_logti = pred_actions_logtis[:, -1, :]
            step_pred_action_logti_list.append(last_step_pred_action_logti)
            last_step_pred_action = self.output_layer(last_step_pred_action_logti)
            pred_actions_list.append(last_step_pred_action)
            
            prev_point = output_sequence[:, t-1, :].detach().clone()          # [B,4]
            pred_point = self.kinematic_step(prev_point, last_step_pred_action)  # [B,4]
            
            # 计划采样：决定是使用真实值还是预测值
            next_token = pred_point
        
            # 更新输出序列
            if t < self.cfg.autoregressive_points -1:
                output_sequence[:, t, :] = next_token
        
        final_global_context = global_context.unsqueeze(1).repeat(1, max_possible_len, 1)

        output_seq_detached = output_sequence.detach().clone()

        tgt_mask, tgt_padding_mask = self.create_mask(output_seq_detached)

        tgt_embedding = self.traj_embedding(output_seq_detached)
        tgt_embedding = tgt_embedding + final_global_context
        tgt_embedding = self.pos_drop(tgt_embedding + self.pos_embed[:, :(self.cfg.autoregressive_points-1), :])

        pred_actions_logtis = self.decoder(encoder_out, tgt_embedding, tgt_mask)
        pred_actions = self.output_layer(pred_actions_logtis)
        pred_actions_list.append(pred_actions[:,-1,:])
        return pred_actions, pred_actions_list[0], step_pred_action_logti_list[0]
        # return pred_actions, pred_actions_list[0]
    

    def forward(self, encoder_out, point_out, tgt, lengths):
        batch_size, length, feat_dim = tgt.size()
        assert feat_dim == 4

        padding_num = self.cfg.autoregressive_points -1 - length

        global_context = point_out.reshape(-1, self.cfg.tf_de_dim)
        
        offset = 1
        if padding_num > 0:
            padding = torch.ones(batch_size, padding_num, 4, device=self.cfg.device) * self.PAD_token
            tgt = torch.cat([tgt, padding], dim=1)

        tgt_mask, tgt_padding_mask = self.create_mask(tgt)
        final_global_context = global_context.unsqueeze(1).repeat(1, tgt.size(1), 1)

        tgt_embedding = self.traj_embedding(tgt)
        tgt_embedding = tgt_embedding + final_global_context
        tgt_embedding = tgt_embedding + self.pos_embed[:, :tgt.size(1), :]

        pred_actions_logtis = self.decoder(encoder_out, tgt_embedding, tgt_mask, tgt_padding_mask)
        idx = (lengths - 1).clamp(min=0)

        last_step_pred_action_logti = pred_actions_logtis[torch.arange(batch_size, device=tgt.device), idx, :]
        
        pred_actions = self.output_layer(last_step_pred_action_logti)
        mean_u = self.output_layer[0](last_step_pred_action_logti)

        prev_point = tgt[:, idx, :]         # [B,4]
        # pred_point = self.kinematic_step(prev_point, pred_actions)  # [B,4]

        return prev_point, pred_actions, mean_u
    

class TrajectoryValueDecoderONNX(TrajectoryDecoderONNX):
    """
    复用 TrajectoryDecoderONNX 的:
      - traj_embedding / pos_embed / tf_decoder / create_mask / decoder(...)
    只把 output_layer 改为输出 1 维 value。
    """
    def __init__(self, cfg: Configuration):
        super().__init__(cfg)
        self.output_layer = nn.Sequential(
            nn.Linear(self.cfg.tf_de_dim, 1)
        )
        self.output_layer = nn.Sequential(
            nn.LayerNorm(self.cfg.tf_de_dim),
            nn.Linear(self.cfg.tf_de_dim, 2 * self.cfg.tf_de_dim),
            nn.GELU(),
            nn.Linear(2 * self.cfg.tf_de_dim, 1),
        )
        self._init_value_head()

    def _init_value_head(self):
        for m in self.output_layer.modules():
            if isinstance(m, nn.Linear):
                # 最后一层 out_features==1 用小 gain
                gain = 0.05 if m.out_features == 1 else 1.0
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, encoder_out, point_out, tgt, lengths):
        # 直接复用你 TrajectoryDecoderONNX.forward 的流程 :contentReference[oaicite:3]{index=3}
        batch_size, length, feat_dim = tgt.size()
        assert feat_dim == 4

        padding_num = self.cfg.autoregressive_points - 1 - length
        global_context = point_out.reshape(-1, self.cfg.tf_de_dim)

        if padding_num > 0:
            padding = torch.ones(batch_size, padding_num, 4, device=tgt.device) * self.PAD_token
            tgt = torch.cat([tgt, padding], dim=1)

        tgt_mask, tgt_padding_mask = self.create_mask(tgt)

        # 用 expand 更省显存（repeat 也可以）
        final_global_context = global_context.unsqueeze(1).expand(-1, tgt.size(1), -1)

        tgt_embedding = self.traj_embedding(tgt) + final_global_context + self.pos_embed[:, :tgt.size(1), :]
        pred_h = self.decoder(encoder_out, tgt_embedding, tgt_mask, tgt_padding_mask)
        idx = (lengths - 1).clamp(min=0)

        h_t = pred_h[torch.arange(batch_size, device=tgt.device), idx, :]     # 和 actor 一样取当前步 token
        value = self.output_layer(h_t)     # [B,1]
        return value

    


