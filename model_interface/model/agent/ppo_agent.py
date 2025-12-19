from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Beta
import numpy as np

from model_interface.model.agent_base import ConfigBase, AgentBase
from model_interface.model.network import *
from model_interface.model.replay_memory import ReplayMemory
from model_interface.model.state_norm import StateNorm
from env.action_mask import ActionMask
from typing import Optional
from model_interface.model.parking_model_real import TrajInputEmbedding
from model_interface.model.trajectory_decoder import TrajectoryDecoderONNX
import itertools
from utils.config import Configuration
from vehicle_config import *

def _extract_sub_state_dict(state_dict, prefix: str):
    out = {}
    plen = len(prefix)
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[plen:]] = v
    return out
class PPOConfig(ConfigBase):
    def __init__(self, configs):
        super().__init__()

        # hyperparameters
        self.lr_actor = self.lr
        self.lr_critic = self.lr*5
        self.lr_backbone = 2e-5
        self.lr_log_std = 1e-3
        self.adam_epsilon = 1e-8
        self.dist_type = "gaussian"
        self.hidden_size = 256
        self.mini_epoch = 10
        self.mini_batch = 256

        self.clip_epsilon = 0.2
        self.lambda_ = 0.95
        self.var_max = 1

        # tricks
        self.adv_norm = True
        self.state_norm = False
        self.reward_norm = False
        self.use_gae = True
        self.reward_scaling = False
        self.gradient_clip = False
        self.policy_entropy = False
        self.entropy_coef = 0.01

        self.merge_configs(configs)

class CriticNetwork(nn.Module):
    """
    Input:
      encoder_out: (B, N, D)  # 来自 shared multi_encoder
      point_out:   (B, D) optional  # 如果你actor也用这个global context，critic建议也用
    Output:
      value: (B, 1)
    """
    def __init__(
        self,
        embed_dim: int,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_cls: bool = True,
        use_point_token: bool = True,
        value_hidden: int = 256,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_cls = use_cls
        self.use_point_token = use_point_token

        ff_dim = ff_dim or 4 * embed_dim

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)

        if use_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls, std=0.02)

        if use_point_token:
            self.point_proj = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
            )

        self.value_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, value_hidden),
            nn.GELU(),
            nn.Linear(value_hidden, 1),
        )

        self.orthogonal_init()

    def orthogonal_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 最后一层（value head）gain 要小
                if m.out_features == 1:
                    nn.init.orthogonal_(m.weight, gain=0.01)
                else:
                    nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        encoder_out: torch.Tensor,             # (B,N,D)
        point_out: Optional[torch.Tensor]=None,# (B,D)
        key_padding_mask: Optional[torch.Tensor]=None, # (B, N_total) True=mask
    ) -> torch.Tensor:
        B, N, D = encoder_out.shape
        assert D == self.embed_dim

        tokens = []

        if self.use_cls:
            tokens.append(self.cls.expand(B, 1, D))      # (B,1,D)

        tokens.append(encoder_out)                       # (B,N,D)

        if self.use_point_token and point_out is not None:
            pt = self.point_proj(point_out).unsqueeze(1) # (B,1,D)
            tokens.append(pt)

        x = torch.cat(tokens, dim=1)                     # (B, N_total, D)

        h = self.tr(x, src_key_padding_mask=key_padding_mask)

        if self.use_cls:
            pooled = h[:, 0, :]                          # CLS pooling
        else:
            pooled = h.mean(dim=1)                       # mean pooling baseline

        return self.value_head(pooled)                   # (B,1)


class PPOAgent(AgentBase):
    def __init__(
        self, config_obj: Configuration, configs: dict, discrete: bool = False, verbose: bool = False,
        save_params: bool = False, load_params: bool = False
    ) -> None:

        super().__init__(PPOConfig, configs, verbose, save_params, load_params)
        self.discrete = discrete
        self.action_filter = ActionMask()
        self.cfg = config_obj

        # debug
        self.actor_loss_list = []
        self.critic_loss_list = []

        # the networks
        retrain_model_path = self.cfg.pretrain_model_path
        self._init_network(retrain_model_path)

        # As a on-policy RL algorithm, PPO does not have memory, the self.memory represents
        # the buffer
        self.memory = ReplayMemory(self.configs.batch_size, ["log_prob","next_obs"])

        # tricks
        if self.configs.state_norm:
            self.state_normalize = StateNorm(self.configs.observation_shape)

        traj = [[0.0,0.0,0.0]]
        traj = np.array(traj, dtype=np.float32)   # [T,3] = (x,y,yaw)
            # x,y 归一化到 [-1,1]
        traj_x = np.clip(traj[:,0] / TRAJXRANGE, -1.0, 1.0)
        traj_y = np.clip(traj[:,1] / TRAJYRANGE, -1.0, 1.0)
        traj_yaw = traj[:,2]   # 假设是弧度

            # [T,4] = (x_norm, y_norm, cos(yaw), sin(yaw))
        gt_traj_point_np = np.stack(
                [traj_x, traj_y, np.cos(traj_yaw), np.sin(traj_yaw)],
                axis=-1
            )   # [T,4]
        self.gt_traj_point = torch.from_numpy(gt_traj_point_np.astype(np.float32))
        
    
    def load_pretrained_from_parkingmodelreal_ckpt(self, ckpt_path: str, strict: bool = False):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt["state_dict"]   # 你的保存函数就是这个结构

        # 1) multi_encoder
        me_sd = _extract_sub_state_dict(state_dict, "multi_encoder.")
        missing, unexpected = self.multi_encoder.load_state_dict(me_sd, strict=strict)
        print(f"[load] multi_encoder: loaded={len(me_sd)}, missing={len(missing)}, unexpected={len(unexpected)}")

        # 2) target_point_encoder
        tp_sd = _extract_sub_state_dict(state_dict, "target_point_encoder.")
        missing, unexpected = self.target_point_encoder.load_state_dict(tp_sd, strict=strict)
        print(f"[load] target_point_encoder: loaded={len(tp_sd)}, missing={len(missing)}, unexpected={len(unexpected)}")

        # 3) actor / trajectory decoder
        dec_sd = _extract_sub_state_dict(state_dict, "trajectory_decoder.")
        missing, unexpected = self.actor_net.load_state_dict(dec_sd, strict=strict)
        print(f"[load] actor_net(trajectory_decoder): loaded={len(dec_sd)}, missing={len(missing)}, unexpected={len(unexpected)}")

    def freeze_multi_encoder_embed_img(self):
        if hasattr(self.multi_encoder, "embed_img"):
            # 1. 关闭梯度
            for p in self.multi_encoder.embed_img.parameters():
                p.requires_grad = False

            # 2. 固定 BN / Dropout 行为（如果 ConvBlock 里有）
            self.multi_encoder.embed_img.eval()

            print("[freeze] multi_encoder.embed_img frozen.")

    def build_optimizer(self):
        param_groups = [
            {
                "params": [p for p in self.multi_encoder.parameters() if p.requires_grad],
                "lr": self.configs.lr_backbone,
            },
            {
                "params": [p for p in self.target_point_encoder.parameters() if p.requires_grad],
                "lr": self.configs.lr_actor,
            },
            {
                "params": [p for p in self.actor_net.parameters() if p.requires_grad],
                "lr": self.configs.lr_actor,
            },
            {
                "params": [self.log_std],
                "lr": self.configs.lr_log_std,   # 可单独控制
            },
            {
                "params": [p for p in self.critic_net.parameters() if p.requires_grad],
                "lr": self.configs.lr_critic,
            },
        ]

        self.actor_critic_optimizer = torch.optim.Adam(param_groups)

    def _init_network(self, pretrain_ckpt_path=None):
        '''
        Initialize 1.the network, 2.the optimizer, 3.the checklist.
        '''

        self.multi_encoder = MultiObsEmbedding(self.configs.actor_layers).to(self.device)
        
         
        self.target_point_encoder = TrajInputEmbedding(self.cfg.global_graph_width).to(self.device)

        self.actor_net = TrajectoryDecoderONNX(self.cfg).to(self.device)

        self.critic_net = CriticNetwork(self.configs.actor_layers["embed_size"]).to(self.device)

        self.critic_target = deepcopy(self.critic_net).to(self.device)

        if pretrain_ckpt_path is not None:
            self.load_pretrained_from_parkingmodelreal_ckpt(pretrain_ckpt_path, strict=False)
            self.freeze_multi_encoder_embed_img()

        self.log_std = \
            nn.Parameter(
                torch.full((1, self.configs.action_dim), -1.0, device=self.device), requires_grad=True
            )
        self.build_optimizer()
   
            
        for n, p in self.multi_encoder.named_parameters():
            if n.startswith("embed_img."):
                assert p.requires_grad is False
            if n.startswith("re_embed_img."):
                assert p.requires_grad is True
        
        # save and load
        self.check_list = [  # (name, item, save_state_dict)
            ("configs", self.configs, 0),

            # 共享/actor侧
            ("multi_encoder", self.multi_encoder, 1),
            ("target_point_encoder", self.target_point_encoder, 1),
            ("actor_net", self.actor_net, 1),

            # critic
            ("critic_net", self.critic_net, 1),
            ("critic_target", self.critic_target, 1),

            # optimizer
            ("actor_critic_optimizer", self.actor_critic_optimizer, 1),

            # gaussian policy 的可学习方差
            ("log_std", self.log_std, 1),
        ]
    def _actor_forward(self, obs) -> torch.distributions.Distribution: # to be replaced
        observation = deepcopy(obs)
        if self.configs.state_norm:
            observation = self.state_normalize.state_norm(observation)
        observation = self.obs2tensor(observation)
        self.multi_encoder.eval()
        self.target_point_encoder.eval()
        self.actor_net.eval()
            
        with torch.no_grad():
            traj_point_start = self.gt_traj_point.to(self.device)
            traj_point_start = traj_point_start.unsqueeze(1)
            encoder_out = self.multi_encoder(observation)
            point_out = self.target_point_encoder(observation['park_target_point'])
            _, policy_out = self.actor_net(encoder_out, point_out, traj_point_start)
            # if policy_out.dim() == 2 and policy_out.size(0) > 1:
            #     raise NotImplementedError("Actor forward expects single-sample inference, got batch > 1")
            # if policy_out.dim() == 2 and policy_out.size(0) == 1:
            #     policy_out = policy_out.squeeze(0)  # -> (act_dim,)
            if self.discrete:
                dist = Categorical(F.softmax(policy_out, dim=1))
            elif self.configs.dist_type == "beta":
                alpha, beta = torch.chunk(policy_out, 2, dim=-1)
                alpha = F.softplus(alpha) + 1.0
                beta = F.softplus(beta) + 1.0
                dist = Beta(alpha, beta)
            elif self.configs.dist_type == "gaussian":
                mean =  torch.clamp(policy_out,-1,1)  
                log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
                log_std = torch.clamp(log_std, min=-2.0, max=-0.5)
                std = torch.exp(log_std)
                dist = Normal(mean, std)
            else:
                raise NotImplementedError
            
        return dist
    
    def _post_process_action(self, action_dist:torch.distributions.Distribution , action_mask=None): # to be replaced
        if False and action_mask is not None:
            mean, std = action_dist.mean, action_dist.stddev
            action = self.action_filter.choose_action(mean, std, action_mask)
            action = torch.FloatTensor(action).to(self.device)
        else:
            action = action_dist.sample()

        if not self.discrete and self.configs.dist_type == "gaussian":
                action = torch.clamp(action, -1, 1)
        log_prob = action_dist.log_prob(action)
        action = action.detach().cpu().numpy().flatten()
        log_prob = log_prob.detach().cpu().numpy().flatten()
        return action, log_prob


    def choose_action(self, obs):

        dist = self._actor_forward(obs)
        action_mask = obs['action_mask']
        action, other_info = self._post_process_action(dist, action_mask)
                
        return action, other_info
    
    # def _post_process_action(self, action_dist, action_mask=None):
    #     if action_mask is not None:
    #         mean, std = action_dist.mean, action_dist.stddev

    #         # 1) 用离散执行策略采样（严格一致）
    #         action_norm_np, action_index, log_prob_exec, _ = \
    #             self.action_filter.sample_exec(mean, std, action_mask)

    #         action = torch.as_tensor(action_norm_np, dtype=torch.float32, device=self.device)

    #         # 注意：这里 log_prob 已经是 π_exec 的 log_prob（不是 Normal 的 log_prob）
    #         log_prob = torch.tensor([log_prob_exec], dtype=torch.float32, device=self.device)

    #         # 你最好把 action_index 也返回并存进 memory（强烈建议）
    #         other_info = {
    #             "log_prob": log_prob,
    #             "action_index": action_index,
    #         }

    #     else:
    #         # 无 mask：就是真正从 Normal 采样，此时 log_prob 用 Normal 的即可（一致）
    #         action = action_dist.sample()
    #         if (not self.discrete) and (self.configs.dist_type == "gaussian"):
    #             action = torch.clamp(action, -1, 1)

    #         log_prob = action_dist.log_prob(action)
    #         if log_prob.dim() > 0:
    #             log_prob = log_prob.sum(dim=-1, keepdim=True)  # 多维动作求和

    #         other_info = {"log_prob": log_prob, "action_index": None}

    #     action_np = action.detach().cpu().numpy().flatten()
    #     log_prob_np = other_info["log_prob"].detach().cpu().numpy().flatten()
    #     return action_np, other_info  # 这里我建议把 other_info 返回，包含 action_index
    
    # def choose_action(self, obs):
    #     dist = self._actor_forward(obs)
    #     action_mask = obs.get("action_mask", None)

    #     action, info = self._post_process_action(dist, action_mask)

    #     # info: {"log_prob": tensor([..]), "action_index": int or None}
    #     # 你把它存进 memory：action、log_prob、action_index
    #     return action, info


    def get_action(self, obs: np.ndarray):
        '''Take action based on one observation. 

        Args:
            observation(np.ndarray): np.ndarray with the same shape of self.state_dim.

        Returns:
            action: If self.discrete, the action is an (int) index. 
                If the action space is continuous, the action is an (np.ndarray).
            log_prob(np.ndarray): the log probability of taken action.
        '''
        dist = self._actor_forward(obs)
        action, log_prob = self._post_process_action(dist)
                
        return action, log_prob

    def get_log_prob(self, obs: np.ndarray, action: np.ndarray):
        '''get the log probability for given action based on current policy

        Args:
            observation(np.ndarray): np.ndarray with the same shape of self.state_dim.

        Returns:
            log_prob(np.ndarray): the log probability of taken action.
        '''
        dist = self._actor_forward(obs)
        
        action = torch.FloatTensor(action).to(self.device)
        log_prob = dist.log_prob(action)
        log_prob = log_prob.detach().cpu().numpy().flatten()
        return log_prob
    
    '''
    连续动作空间
        # 策略网络输出
        mean, std = policy_network(state)
        action_dist = torch.distributions.Normal(mean, std)

        # 采样动作
        action = action_dist.sample()

        # 计算对数概率（用于策略梯度）
        log_prob = action_dist.log_prob(action)  # 这里要使用高斯公式计算一下概率密度
        # 形状: [batch_size, action_dim]

        # 通常需要压缩为标量
        log_prob = log_prob.sum(dim=-1)  # [batch_size]

    离散动作区间
        # 策略网络输出 logits
        logits = policy_network(state)
        action_dist = torch.distributions.Categorical(logits=logits)

        # 采样动作
        action = action_dist.sample()

        # 计算对数概率
        log_prob = action_dist.log_prob(action)  # 这里计算的就是这个离散的action被选中的概率
        # 形状: [batch_size]

    '''

    def push_memory(self, observations):
        '''
        Args:
            observations(tuple): (obs, action, reward, done, log_prob, next_obs)
        '''
        obs, action, reward, done, log_prob, next_obs = deepcopy(observations)
        if self.configs.state_norm:
            obs = self.state_normalize.state_norm(obs)
            next_obs = self.state_normalize.state_norm(next_obs,update=True)
        observations = (obs, action, reward, done, log_prob, next_obs)
        self.memory.push(observations)

    def _reward_norm(self, reward):
        return (reward - reward.mean()) / (reward.std() + 1e-8)

    def obs2tensor(self, obs):
        if isinstance(obs, list):
            merged_obs = {}
            for obs_type in self.configs.observation_shape.keys():
                merged_obs[obs_type] = []
                for o in obs:
                    merged_obs[obs_type].append(o[obs_type])
                merged_obs[obs_type] = torch.FloatTensor(np.array(merged_obs[obs_type])).to(self.device)
            obs = merged_obs 
        elif isinstance(obs, dict):
            for obs_type in self.configs.observation_shape.keys():
                obs[obs_type] = torch.FloatTensor(obs[obs_type]).to(self.device).unsqueeze(0)
        else:
            raise NotImplementedError()
        return obs
    
    def get_obs(self, obs, ids):
        return {k:obs[k][ids] for k in obs }

    def update(self, step): # to be replaced
        # convert batches to tensors

        # GAE computation cannot use shuffled data
        # batches = self.memory.shuffle()
        batches = self.memory.get_items(np.arange(len(self.memory)))
        state_batch = self.obs2tensor(batches["state"])
        next_state_batch = self.obs2tensor(batches["next_obs"])

        if self.discrete:
            action_np = np.asarray(batches["action"], dtype=np.int32)
            action_batch = torch.from_numpy(action_np).to(self.device)
        else:
            action_np = np.asarray(batches["action"], dtype=np.float32)
            action_batch = torch.from_numpy(action_np).to(self.device)
        # 3) reward：一次性 numpy 化，并确保 shape=[N,1]
        reward_np = np.asarray(batches["reward"], dtype=np.float32).reshape(-1, 1)
        rewards = torch.from_numpy(reward_np).to(self.device)
        reward_batch = self._reward_norm(rewards) if self.configs.reward_norm else rewards

        # 4) done：很多 buffer 里 done 是 bool；转 float32 并 reshape=[N,1]
        done_np = np.asarray(batches["done"], dtype=np.float32).reshape(-1, 1)
        done_batch = torch.from_numpy(done_np).to(self.device)

        # 5) old log prob：同理一次性 numpy 化
        logp_np = np.asarray(batches["log_prob"], dtype=np.float32)
        old_log_prob_batch = torch.from_numpy(logp_np).to(self.device)
        self.memory.clear()

        def encode_obs(obs_dict):
            """
            obs_dict: dict[str, Tensor], shapes follow your obs2tensor
            """
            encoder_out = self.multi_encoder(obs_dict)  # 通常输出 (B, N, D) 或 (B, D)

            # ⚠️ 如果你的 target_point_encoder 只吃某个字段，比如 obs_dict["target_point"]
            # 请改成：point_out = self.target_point_encoder(obs_dict["target_point"])
            point_out = self.target_point_encoder(obs_dict['park_target_point'])

            return encoder_out, point_out

        # GAE
        gae = 0
        adv_list = []

        self.multi_encoder.eval()
        self.target_point_encoder.eval()
        self.actor_net.eval()
        self.critic_net.eval()
        # critic_target 通常保持 eval 即可（本来就是 target）
        self.critic_target.eval()

        with torch.no_grad():
            enc, pt = encode_obs(state_batch)
            next_enc, next_pt = encode_obs(next_state_batch)
            value = self.critic_net(enc, pt)
            next_value = self.critic_net(next_enc, next_pt)
            deltas = reward_batch + self.configs.gamma * (1 - done_batch) * next_value - value
            if self.configs.use_gae:
                for delta, done in zip(reversed(deltas.cpu().flatten().numpy()), reversed(done_batch.cpu().flatten().numpy())):
                    gae = delta + self.configs.gamma * self.configs.lambda_ * gae * (1.0 - done)
                    adv_list.append(gae)
                adv_list.reverse()
                # adv = torch.FloatTensor(adv_list).view(-1, 1).to(self.device)
                adv = torch.as_tensor(adv_list, dtype=torch.float32, device=self.device).view(-1, 1)
            else:
                adv = deltas
            v_target = (adv + self.critic_target(enc, pt)).detach() #Vtarget​(st​)=AtGAE​+V(st​)≈Q(st​,at​)  实际上是在拟合构造 “Q-like target”  所以要加上AtGAE，就是构造TD target，所以要计算一下
            
            if self.configs.adv_norm: # advantage normalization
                adv = (adv - adv.mean()) / (adv.std() + 1e-5)


        self.multi_encoder.train()
        self.target_point_encoder.train()
        self.actor_net.train()
        self.critic_net.train()
        # critic_target 通常保持 eval 即可（本来就是 target）
        self.critic_target.eval()

        # 但你冻结的 embed_img 希望永远 eval，就再强制一下：
        if hasattr(self.multi_encoder, "embed_img"):
            self.multi_encoder.embed_img.eval()
        
        # apply multi update epoch
        mini_batch = self.configs.mini_batch
        batchsize = self.configs.batch_size
        train_times = batchsize//mini_batch if batchsize%mini_batch==0 else batchsize//mini_batch+1
        traj_point_start = self.gt_traj_point.to(self.device)
        for _ in range(self.configs.mini_epoch):
            # use mini batch and shuffle data
            random_idx = np.arange(batchsize)
            np.random.shuffle(random_idx)
            for i in range(train_times):
                if i == batchsize//mini_batch:
                    ri = random_idx[i*mini_batch:]
                else:
                    ri = random_idx[i*mini_batch:(i+1)*mini_batch]
                # state = state_batch[ri]
                obs = self.get_obs(state_batch, ri)  # 仍然返回 dict[str, Tensor]
                enc_train, pt_train = encode_obs(obs)
                B = enc_train.size(0)
                traj_point_start_mb = (
                    traj_point_start
                    .to(enc_train.device)
                    .unsqueeze(1)          # (1, 4) -> (1, 1, 4)
                    .expand(B, -1, -1)     # (B, 1, 4)
                    .clone()               # 防止 inplace 梯度错误
                )
                if self.discrete:
                    _, policy_dist = self.actor_net(enc_train, pt_train, traj_point_start_mb)
                    dist = Categorical(F.softmax(policy_dist,dim=-1))
                    dist_entropy = dist.entropy().view(-1, 1)
                    log_prob= dist.log_prob(action_batch[ri].squeeze()).view(-1, 1)
                    old_log_prob = old_log_prob_batch[ri].view(-1,1)

                elif self.configs.dist_type == "beta":
                    _, policy_dist = self.actor_net(enc_train, pt_train, traj_point_start_mb)
                    alpha, beta = torch.chunk(policy_dist, 2, dim=-1)
                    alpha = F.softplus(alpha) + 1.0
                    beta = F.softplus(beta) + 1.0
                    dist = Beta(alpha, beta)
                    dist_entropy = dist.entropy().sum(1, keepdim=True)
                    log_prob = dist.log_prob(action_batch[ri])
                    log_prob =torch.sum(log_prob,dim=1, keepdim=True)
                    old_log_prob =torch.sum(old_log_prob_batch[ri],dim=1, keepdim=True)

                elif self.configs.dist_type == "gaussian":
                    _, policy_dist = self.actor_net(enc_train, pt_train, traj_point_start_mb)
                    mean = torch.clamp(policy_dist, -1, 1)
                    log_std = self.log_std.expand_as(mean)
                    std = torch.exp(log_std)
                    dist = Normal(mean, std)
                    dist_entropy = dist.entropy().sum(1, keepdim=True)
                    log_prob = dist.log_prob(action_batch[ri])
                    log_prob =torch.sum(log_prob,dim=1, keepdim=True)
                    old_log_prob =torch.sum(old_log_prob_batch[ri],dim=1, keepdim=True)
                prob_ratio = (log_prob - old_log_prob).exp()
                ''''
                # 取该样本当时存的 action_index
                    k = batches["action_index"][ri]   # 需要你在 memory 里存下来

                    # 用 action_filter 计算 new_log_prob_exec（逐样本）
                    new_log_probs = []
                    for b in range(len(ri)):
                        new_lp = self.action_filter.log_prob_exec(
                            mean[b], std[b], obs["action_mask"][b], int(k[b])
                        )
                        new_log_probs.append(new_lp) 

                    new_log_prob = torch.tensor(new_log_probs, device=self.device).view(-1,1) #维度是(b,1)
                    old_log_prob = old_log_prob_batch[ri].view(-1,1)

                prob_ratio = (new_log_prob - old_log_prob).exp()
                '''
                loss1 = prob_ratio * adv[ri]
                loss2 = torch.clamp(prob_ratio, 1 - self.configs.clip_epsilon, 1 + self.configs.clip_epsilon) * adv[ri]

                actor_loss = - torch.min(loss1, loss2)
                if self.configs.policy_entropy:
                    actor_loss = actor_loss - self.configs.entropy_coef * dist_entropy

                v_pred = self.critic_net(enc_train, pt_train)
                critic_loss = F.mse_loss(v_target[ri], v_pred, reduction="none")

                self.actor_critic_optimizer.zero_grad(set_to_none=True)
                total_loss = actor_loss.mean() + critic_loss.mean()
                total_loss.backward()
                
                self.actor_loss_list.append(actor_loss.mean().item())
                self.critic_loss_list.append(critic_loss.mean().item())
                if self.configs.gradient_clip:
                # 对所有可训练参数一起 clip（包含 log_std）
                    trainable_params = [p for p in itertools.chain(
                        self.multi_encoder.parameters(),
                        self.target_point_encoder.parameters(),
                        self.actor_net.parameters(),
                        self.critic_net.parameters(),
                        [self.log_std],
                    ) if p.requires_grad]
                    nn.utils.clip_grad_norm_(trainable_params, 0.5)
            self.actor_critic_optimizer.step()

            self._soft_update(self.critic_target, self.critic_net)

        if self.configs.lr_decay: # learning rate decay
            self.actor_critic_optimizer.param_groups[0]["lr"] = self.lr_decay(self.configs.lr_backbone, step)
            self.actor_critic_optimizer.param_groups[1]["lr"] = self.lr_decay(self.configs.lr_actor, step)
            self.actor_critic_optimizer.param_groups[2]["lr"] = self.lr_decay(self.configs.lr_actor, step)
            self.actor_critic_optimizer.param_groups[3]["lr"] = self.lr_decay(self.configs.lr_log_std, step)
            self.actor_critic_optimizer.param_groups[4]["lr"] = self.lr_decay(self.configs.lr_critic, step)

        # for debug
        a = actor_loss.detach().cpu().numpy()[0][0]
        b = critic_loss.mean().item()
        return a, b

    def save(self, path: str = None, params_only: bool = None) -> None: # to be replaced
        """Store the model structure and corresponding parameters to a file.
        """
        if params_only is not None:
            self.save_params = params_only
        if self.save_params and len(self.check_list) > 0:
            checkpoint = {}
            for name, item, save_state_dict in self.check_list:
                if save_state_dict:
                    if isinstance(item, nn.Module):
                        checkpoint[name] = item.state_dict()
                    elif isinstance(item, torch.optim.Optimizer):
                        checkpoint[name] = item.state_dict()
                    elif isinstance(item, nn.Parameter):
                        checkpoint[name] = item.detach().cpu()
                    elif torch.is_tensor(item):
                        checkpoint[name] = item.detach().cpu()
                    else:
                        checkpoint[name] = item  # 例如 configs 这种
                else:
                    checkpoint[name] = item
            if self.configs.state_norm:
                checkpoint['state_norm'] = self.state_normalize # (self.state_mean, self.state_std, self.S, self.n_state)
            torch.save(checkpoint, path)
        else:
            torch.save(self, path)
        
        if self.verbose:
            print("Save current model to %s" % path)

    def load(self, path: str = None, params_only: bool = None) -> None: # to be replaced
        """Load the model structure and corresponding parameters from a file.
        """
        if params_only is not None:
            self.load_params = params_only
        if self.load_params and len(self.check_list) > 0:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            # 1) 按 check_list 加载
            for name, item, save_state_dict in self.check_list:
                if name not in checkpoint:
                    if getattr(self, "verbose", False):
                        print(f"[load][WARN] key '{name}' not in checkpoint, skip.")
                    continue

                ckpt_val = checkpoint[name]

                if save_state_dict:
                    # (a) nn.Module / Optimizer：用 load_state_dict
                    if isinstance(item, nn.Module):
                        missing, unexpected = item.load_state_dict(ckpt_val, strict=False)
                        if getattr(self, "verbose", False) and (missing or unexpected):
                            print(f"[load][WARN] {name}: missing={len(missing)} unexpected={len(unexpected)}")

                    elif isinstance(item, torch.optim.Optimizer):
                        item.load_state_dict(ckpt_val)

                    # (b) nn.Parameter：用 data.copy_ 恢复
                    elif isinstance(item, nn.Parameter):
                        # ckpt_val 可能是 cpu tensor
                        t = ckpt_val.to(self.device)
                        if item.data.shape != t.shape:
                            raise RuntimeError(f"[load][ERR] {name} shape mismatch: "
                                            f"param {tuple(item.data.shape)} vs ckpt {tuple(t.shape)}")
                        item.data.copy_(t)

                    # (c) Tensor：copy_
                    elif torch.is_tensor(item):
                        t = ckpt_val.to(self.device)
                        if item.shape != t.shape:
                            raise RuntimeError(f"[load][ERR] {name} shape mismatch: "
                                            f"tensor {tuple(item.shape)} vs ckpt {tuple(t.shape)}")
                        item.copy_(t)

                    else:
                        # 兜底：如果 item 不是上面几类，就直接 setattr（但一般不建议走到这）
                        setattr(self, name, ckpt_val)

                else:
                    # 直接赋值到 self 上（configs 这类）
                    setattr(self, name, ckpt_val)

            # 2) 恢复 state normalize
            if "state_norm" in checkpoint:
                self.state_normalize = checkpoint["state_norm"]

            if self.configs.dist_type == "gaussian":
                found = any(
                    self.log_std is p
                    for g in self.actor_critic_optimizer.param_groups
                    for p in g["params"]
                )
                if not found and getattr(self, "verbose", False):
                    print("[load][WARN] log_std not found in optimizer param_groups. "
                        "Make sure build_optimizer() adds it before load().")

        else:
            obj = torch.load(path, map_location=self.device)
            raise RuntimeError("Loading full object is not supported safely here. Use params_only=True.")

        if getattr(self, "verbose", False):
            print(f"Load the model from {path}")
