from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from model_interface.model.agent_base import ConfigBase, AgentBase
from model_interface.model.network import *
from model_interface.model.replay_memory import ReplayMemory
from model_interface.model.state_norm import StateNorm
from env.action_mask import ActionMask
from utils.config import Configuration
from model_interface.model.parking_model_real import TrajInputEmbedding
from model_interface.model.trajectory_decoder import TrajectoryDecoderONNX, TrajectoryValueDecoderONNX
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from typing import Dict, List, Tuple, Optional
import torch.distributed as dist_gpu
from torch.nn.utils.rnn import pad_sequence
from contextlib import contextmanager


@contextmanager
def temporary_freeze_modules(modules):
    """
    modules: Iterable[nn.Module]
    进入时：把这些模块所有参数 requires_grad=False
    退出时：恢复进入前每个参数原本的 requires_grad（不会误打开你永久冻结的层）
    """
    params = []
    old_flags = []
    for m in modules:
        if m is None:
            continue
        for p in m.parameters(recurse=True):
            params.append(p)
            old_flags.append(p.requires_grad)
            p.requires_grad = False
    try:
        yield
    finally:
        for p, flag in zip(params, old_flags):
            p.requires_grad = flag

@contextmanager
def temporary_eval(*modules):
    old = []
    ms = []
    for m in modules:
        if m is None:
            continue
        ms.append(m)
        old.append(m.training)   # True/False
        m.eval()
    try:
        yield
    finally:
        for m, was_train in zip(ms, old):
            m.train(was_train)


@torch.no_grad()
def load_common_state_dict(
    dst_module: torch.nn.Module,
    src_state_dict: Dict[str, torch.Tensor],
    *,
    dst_submodule_attr: Optional[str] = None,     # 例如 critic adapter 里用 "net"
    include_prefixes: Optional[Tuple[str, ...]] = None,  # 只加载这些前缀开头的key
    exclude_prefixes: Tuple[str, ...] = (),
    strict: bool = False,
    verbose: bool = True,
    tag: str = "[partial-load]",
    print_missing_topk: int = 10,
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """
    从 src_state_dict 迁移到 dst_module（或 dst_module.<dst_submodule_attr>）：
    - key 同名
    - shape 一致
    - 可通过 include_prefixes 限制只加载某些子模块（强烈推荐用于 actor->critic encoder）
    - 可通过 exclude_prefixes 排除某些前缀

    返回 (match_dict, missing_keys) 供你继续打印/统计。
    """
    if dst_submodule_attr is not None:
        dst_module = getattr(dst_module, dst_submodule_attr)

    dst_sd = dst_module.state_dict()

    def allowed(k: str) -> bool:
        if include_prefixes is not None:
            if not any(k.startswith(p) for p in include_prefixes):
                return False
        if exclude_prefixes and any(k.startswith(p) for p in exclude_prefixes):
            return False
        return True

    match: Dict[str, torch.Tensor] = {}
    for k_dst, v_dst in dst_sd.items():
        if not allowed(k_dst):
            continue
        v_src = src_state_dict.get(k_dst, None)
        if v_src is None or v_src.shape != v_dst.shape:
            continue
        match[k_dst] = v_src

    # missing：dst里允许加载但没匹配到的key（诊断用）
    missing = [k for k in dst_sd.keys() if allowed(k) and (k not in match)]

    if verbose:
        total_allowed = sum(1 for k in dst_sd.keys() if allowed(k))
        print(f"{tag} matched keys: {len(match)}/{total_allowed} (allowed)")
        if missing:
            print(f"{tag} missing keys (first {min(print_missing_topk, len(missing))}):")
            for k in missing[:print_missing_topk]:
                print("  -", k)

    dst_module.load_state_dict(match, strict=strict)
    return match, missing
def _extract_sub_state_dict(state_dict, prefix: str):
    out = {}
    plen = len(prefix)
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[plen:]] = v
    return out
class SACCriticEncoderAdapter(nn.Module):
    def __init__(self, configs: dict, action_dim:int=2):
        super().__init__()
        self.configs = deepcopy(configs)
        self.configs['input_action_dim'] = action_dim
        self.configs['n_modal'] += 1
        self.net = MultiObsEmbedding(self.configs)

    def forward(self, state: dict, action: torch.Tensor) -> torch.Tensor:
        state_action = dict(state)        # 浅拷贝，避免污染外部
        state_action['action'] = action
        return self.net(state_action)

class SACBundle(nn.Module):
    def __init__(self, cfg, configs, device):
        super().__init__()
        self.cfg = cfg
        self.configs = configs
        A = configs.action_dim

        # --- actor ---
        self.actor_encoder = MultiObsEmbedding(configs.actor_layers)
        self.actor_point_encoder = TrajInputEmbedding(cfg.global_graph_width)
        self.actor_net = TrajectoryDecoderONNX(cfg)

        self.log_std = nn.Parameter(
            torch.tensor([[-0.5, -0.1]], device=device),
            requires_grad=True
        )

        # --- Q1 ---
        self.q1_encoder = SACCriticEncoderAdapter(configs.actor_layers, action_dim=A)
        self.q1_point_encoder = TrajInputEmbedding(cfg.global_graph_width)
        self.q1_net = TrajectoryValueDecoderONNX(cfg)

        # --- Q2 ---
        self.q2_encoder = SACCriticEncoderAdapter(configs.actor_layers, action_dim=A)
        self.q2_point_encoder = TrajInputEmbedding(cfg.global_graph_width)
        self.q2_net = TrajectoryValueDecoderONNX(cfg)

        # --- target Q ---
        self.q1_target_encoder = deepcopy(self.q1_encoder)
        self.q1_target_point_encoder = deepcopy(self.q1_point_encoder)
        self.q1_target_net = deepcopy(self.q1_net)

        self.q2_target_encoder = deepcopy(self.q2_encoder)
        self.q2_target_point_encoder = deepcopy(self.q2_point_encoder)
        self.q2_target_net = deepcopy(self.q2_net)

    def encode_actor_obs(self, obs_dict):
        enc = self.actor_encoder(obs_dict)
        pt = self.actor_point_encoder(obs_dict["park_target_point"])
        return enc, pt

    def encode_q1_obs(self, obs_dict, action):
        enc = self.q1_encoder(obs_dict, action)
        pt = self.q1_point_encoder(obs_dict["park_target_point"])
        return enc, pt

    def encode_q2_obs(self, obs_dict, action):
        enc = self.q2_encoder(obs_dict, action)
        pt = self.q2_point_encoder(obs_dict["park_target_point"])
        return enc, pt
    def encode_q1_target_obs(self, obs_dict, action):
        enc = self.q1_target_encoder(obs_dict, action)
        pt = self.q1_target_point_encoder(obs_dict["park_target_point"])
        return enc, pt

    def encode_q2_target_obs(self, obs_dict, action):
        enc = self.q2_target_encoder(obs_dict, action)
        pt = self.q2_target_point_encoder(obs_dict["park_target_point"])
        return enc, pt


class SACConfig(ConfigBase):
    def __init__(self, configs):
        super().__init__()

        # hyperparameters
        self.lr_actor = 1e-4
        self.lr_critic = 1e-4
        self.lr_alpha = 1e-4
        self.lr_backbone = 5e-5
        self.lr_log_std = 1e-4
        self.tau = 0.005
        self.adam_epsilon = 1e-8
        self.dist_type = "gaussian"
        self.hidden_size = 256
        self.memory_size = 10240
        self.batch_size = 128
        # self.mini_batch_size = 32
        self.mini_epoch = 1
        self.initial_temperature = 0.01
        self.action_dim = 2
        self.target_entropy = -self.action_dim

        # tricks
        self.learn_temperature = True
        self.state_norm = False
        self.reward_norm = False
        self.reward_scaling = False

        self.merge_configs(configs)


class SACAgent(AgentBase):
    def __init__(
        self, config_obj: Configuration, configs: dict, discrete: bool = False, verbose: bool = False,
        save_params: bool = False, load_params: bool = False
    ) -> None:

        super().__init__(SACConfig, configs, verbose, save_params, load_params)
        self.discrete = discrete
        self.action_filter = ActionMask()
        self.cfg = config_obj
        self.cfg.device = self.device

        # debug
        self.actor_loss_list = []
        self.critic_loss_list = []

        # the networks
        retrain_model_path = self.cfg.pretrain_model_path
        self._init_network(retrain_model_path)

        # As a on-policy RL algorithm, PPO does not have memory, the self.memory represents
        # the buffer
        self.memory = ReplayMemory(self.configs.memory_size, ["log_prob","next_obs", "predict_pose_list", "seg_done"])

        # tricks
        if self.configs.state_norm:
            self.state_normalize = StateNorm(self.configs.observation_shape)


    def _unwrap(self, m):
        return m.module if hasattr(m, "module") else m
    

    def load_pretrained_from_parkingmodelreal_ckpt(self, ckpt_path: str, strict: bool = False):
        """
        复用你 PPO 的加载思路：multi_encoder / target_point_encoder / trajectory_decoder
        但这里要加载到：actor_* 以及 q1/q2 的 image encoder / point encoder / decoder。
        """
        b = self._unwrap(self.bundle)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt["state_dict"]

        me_sd = _extract_sub_state_dict(state_dict, "multi_encoder.")
        tp_sd = _extract_sub_state_dict(state_dict, "target_point_encoder.")
        dec_sd = _extract_sub_state_dict(state_dict, "trajectory_decoder.")

        # actor
        b.actor_encoder.load_state_dict(me_sd, strict=strict)
        b.actor_point_encoder.load_state_dict(tp_sd, strict=strict)
        b.actor_net.load_state_dict(dec_sd, strict=strict)

        # Q 网络：encoder 结构不同（多了 action 模态），所以：
        # --- Q encoder：只迁移公共 embed 子模块（不会碰 net 第一层，也不会动 critic 新增的 embed_action） ---
        common_embed_prefixes = (
            "embed_img.", "re_embed_img.",
            "embed_lidar.", "embed_tgt.", "embed_am.",
            # 注意：不要包含 "embed_action."，因为 actor 没有这个分支
        )

        load_common_state_dict(
            b.q1_encoder,
            me_sd,
            dst_submodule_attr="net",                 # adapter 里 .net 是 MultiObsEmbedding
            include_prefixes=common_embed_prefixes,
            strict=False,
            verbose=True,
            tag="[Q1 encoder init]"
        )

        load_common_state_dict(
            b.q2_encoder,
            me_sd,
            dst_submodule_attr="net",
            include_prefixes=common_embed_prefixes,
            strict=False,
            verbose=True,
            tag="[Q2 encoder init]"
        )

        b.q1_target_encoder = deepcopy(b.q1_encoder).to(self.device)
        b.q2_target_encoder = deepcopy(b.q2_encoder).to(self.device)

        # point encoder：可以直接加载
        b.q1_point_encoder.load_state_dict(tp_sd, strict=strict)
        b.q2_point_encoder.load_state_dict(tp_sd, strict=strict)
        b.q1_target_point_encoder.load_state_dict(tp_sd, strict=strict)
        b.q2_target_point_encoder.load_state_dict(tp_sd, strict=strict)

        # Q net：用 decoder 权重，但你 PPO 里对 critic 去掉 output_layer.* :contentReference[oaicite:10]{index=10}
        # 对 SAC 的 Q：也是输出 1 维，所以同样可以 strict=False 或剔除某些 key
        sd = b.q1_net.state_dict()
        match = {k: v for k, v in dec_sd.items()
                if k in sd and sd[k].shape == v.shape and not k.startswith("output_layer.")}

        missing = [k for k in sd.keys() if k not in match]
        print(f"[Q init] matched keys: {len(match)}/{len(sd)}")
        if len(match) < len(sd):
            print("[Q init] missing keys (first 10):")
            for k in missing[:10]:
                print("  -", k)
        b.q1_net.load_state_dict(match, strict=False)
        b.q2_net.load_state_dict(match, strict=False)
        b.q1_target_net = deepcopy(b.q1_net).to(self.device)
        b.q2_target_net = deepcopy(b.q2_net).to(self.device)

    def freeze_multi_encoder_embed_img(self):
        """
        对齐 PPO：冻结 actor encoder 的 embed_img；SAC 的 q1/q2 encoder 也冻结。
        """
        b = self._unwrap(self.bundle)
        def freeze_embed_img(m):
            # m 可能是 MultiObsEmbedding，也可能是 Adapter
            if hasattr(m, "net"):
                m = m.net
            if hasattr(m, "embed_img"):
                for p in m.embed_img.parameters():
                    p.requires_grad = False
                m.embed_img.eval()
        for enc in [b.actor_encoder, b.q1_encoder, b.q2_encoder, b.q1_target_encoder, b.q2_target_encoder]:
            freeze_embed_img(enc)
        print("[freeze] embed_img frozen for actor/q1/q2.")

    def build_optimizers(self):
        """
        SAC：3 个 optimizer（actor / critic / alpha）
        critic 可以一个 optimizer 管 Q1+Q2（推荐），也可分开。
        """
        b = self._unwrap(self.bundle)

        # actor optimizer
        self.actor_optimizer = torch.optim.Adam([
            {"params": [p for p in b.actor_encoder.parameters() if p.requires_grad], "lr": self.configs.lr_backbone},
            {"params": [p for p in b.actor_point_encoder.parameters() if p.requires_grad], "lr": self.configs.lr_actor},
            {"params": [p for p in b.actor_net.parameters() if p.requires_grad], "lr": self.configs.lr_actor},
            {"params": [b.log_std], "lr": self.configs.lr_log_std},
        ], eps=self.configs.adam_epsilon)

        # critic1 optimizer (Q1)
        self.critic_optimizer1 = torch.optim.Adam([
            {"params": [p for p in b.q1_encoder.parameters() if p.requires_grad], "lr": self.configs.lr_backbone},
            {"params": [p for p in b.q1_point_encoder.parameters() if p.requires_grad], "lr": self.configs.lr_critic},
            {"params": [p for p in b.q1_net.parameters() if p.requires_grad], "lr": self.configs.lr_critic},
        ], eps=self.configs.adam_epsilon)

        # critic2 optimizer (Q2)
        self.critic_optimizer2 = torch.optim.Adam([
            {"params": [p for p in b.q2_encoder.parameters() if p.requires_grad], "lr": self.configs.lr_backbone},
            {"params": [p for p in b.q2_point_encoder.parameters() if p.requires_grad], "lr": self.configs.lr_critic},
            {"params": [p for p in b.q2_net.parameters() if p.requires_grad], "lr": self.configs.lr_critic},
        ], eps=self.configs.adam_epsilon)

        # alpha optimizer
        self.log_alpha = torch.tensor(np.log(self.configs.initial_temperature), device=self.device, requires_grad=True)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.configs.lr_alpha, eps=self.configs.adam_epsilon)

    def _init_network(self, pretrain_ckpt_path=None):
        # 1) build bundle
        self.bundle = SACBundle(self.cfg, self.configs, self.device).to(self.device)

        # 2) load / freeze（对齐 PPO 风格）
        if pretrain_ckpt_path is not None:
            self.load_pretrained_from_parkingmodelreal_ckpt(pretrain_ckpt_path, strict=False)
            self.freeze_multi_encoder_embed_img()

        # 3) DDP wrap（如果 torchrun 启动）
        self.distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if self.distributed:
            local_rank = int(os.environ["LOCAL_RANK"])
            self.bundle = DDP(
                self.bundle,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

        # 4) build optimizers
        self.build_optimizers()

        # 5) checklist（保存/加载时 unwrap）
        self.check_list = [
            ("configs", self.configs, 0),
            ("bundle", self.bundle, 1),

            ("actor_optimizer", self.actor_optimizer, 1),
            ("critic_optimizer1", self.critic_optimizer1, 1),
            ("critic_optimizer2", self.critic_optimizer2, 1),

            ("log_alpha", self.log_alpha, 1),
            ("log_alpha_optimizer", self.log_alpha_optimizer, 1),
        ]



    def _actor_forward(self, obs, predict_pose_list) -> torch.distributions.Distribution:
        observation = deepcopy(obs)
        gt_traj_point = deepcopy(predict_pose_list)
        gt_traj_point = torch.from_numpy(np.array(gt_traj_point, dtype=np.float32)).to(self.device)
        traj_point_start = gt_traj_point.unsqueeze(1).transpose(0, 1)
        if self.configs.state_norm:
            observation = self.state_normalize.state_norm(observation)
        observation = self.obs2tensor(observation)
        b = self._unwrap(self.bundle)
        b.actor_encoder.eval()
        b.actor_point_encoder.eval()
        b.actor_net.eval()
        
        with torch.no_grad():
            encoder_out, point_out = b.encode_actor_obs(observation)
            length = torch.tensor(traj_point_start.size(1), device=traj_point_start.device)
            _, policy_out = b.actor_net(encoder_out, point_out, traj_point_start, length)
            if len(policy_out.shape) > 1 and policy_out.shape[0] > 1:
                raise NotImplementedError
            a_mean = torch.clamp(policy_out, -0.999, 0.999)
            mu = self.atanh(a_mean)   
            log_std = b.log_std.expand_as(mu)  # To make 'log_std' have the same dimension as 'mean'
            log_std = torch.clamp(log_std, min=-2.0, max=0.0)
            std = torch.exp(log_std)
            dist = Normal(mu, std)
    
        return dist
    
    def _post_process_action(self, action_dist:torch.distributions.Distribution , action_mask=None):
        if action_mask is not None:
            mean, std = action_dist.mean, action_dist.stddev
            action = self.action_filter.choose_action(mean, std, action_mask)
            action = torch.FloatTensor(action).to(self.device)
            action = torch.clamp(action, -0.999, 0.999)
            u = self.atanh(action)
        else:
            u = action_dist.sample()
            action = torch.tanh(u)

        if not self.discrete and self.configs.dist_type == "gaussian":
            action = torch.clamp(action, -0.999, 0.999)
        log_prob_t = action_dist.log_prob(u)
        log_prob_t = log_prob_t.sum(dim=-1)
        log_prob_t = log_prob_t - torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1)

      
        action_np = action.detach().cpu().numpy().astype(np.float32).reshape(-1)
        log_prob_val = log_prob_t.detach().cpu()
        if log_prob_val.numel() == 1:
            log_prob = float(log_prob_val.item())
        else:
            log_prob = float(log_prob_val.view(-1)[0].item())
        return action_np, log_prob

    def choose_action(self, obs, predict_pose_list):

        dist = self._actor_forward(obs, predict_pose_list)
        action_mask = obs['action_mask']
        action, other_info = self._post_process_action(dist, action_mask)

        return action, other_info
    

    def choose_action_eval(self, obs, predict_pose_list):
        observation = deepcopy(obs)
        gt_traj_point = deepcopy(predict_pose_list)
        gt_traj_point = torch.from_numpy(np.array(gt_traj_point, dtype=np.float32)).to(self.device)
        traj_point_start = gt_traj_point.unsqueeze(1).transpose(0, 1)
        if self.configs.state_norm:
            observation = self.state_normalize.state_norm(observation)
        observation = self.obs2tensor(observation)
        b = self._unwrap(self.bundle)
        b.actor_encoder.eval()
        b.actor_point_encoder.eval()
        b.actor_net.eval()
            
        with torch.no_grad():
            encoder_out, point_out = b.encode_actor_obs(observation)
            length = torch.tensor(traj_point_start.size(1), device=traj_point_start.device)
            _, policy_out = b.actor_net(encoder_out, point_out, traj_point_start, length)
            a_mean = torch.clamp(policy_out, -0.999, 0.999)
            a_mean = a_mean.detach().cpu().numpy().astype(np.float32).reshape(-1)
            
        return a_mean, None

    def get_action(self, obs: np.ndarray, predict_pose_list):
        '''Take action based on one observation. 

        Args:
            observation(np.ndarray): np.ndarray with the same shape of self.state_dim.

        Returns:
            action: If self.discrete, the action is an (int) index. 
                If the action space is continuous, the action is an (np.ndarray).
            log_prob(np.ndarray): the log probability of taken action.
        '''
        dist = self._actor_forward(obs, predict_pose_list)
        action, log_prob = self._post_process_action(dist)
                
        return action, log_prob
    

    def atanh(self,x):
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def get_log_prob(self, obs: np.ndarray, action: np.ndarray, predict_pose_list):
        '''get the log probability for given action based on current policy

        Args:
            observation(np.ndarray): np.ndarray with the same shape of self.state_dim.

        Returns:
            log_prob(np.ndarray): the log probability of taken action.
        '''
         # get u-space Normal distribution
        dist_u = self._actor_forward(obs, predict_pose_list)   # Normal(mean_u, std)

        # a-space action -> tensor
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        action = torch.clamp(action, -0.999, 0.999)  # safety for atanh

        # a -> u
        u = self.atanh(action)

        # log π(a) = log N(u) - log|detJ|
        log_prob = dist_u.log_prob(u)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob = log_prob - torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        # return scalar
        return float(log_prob.detach().cpu().item())

    def push_memory(self, observations):
        '''
        Args:
            observations(tuple): (obs, action, reward, done, log_prob, next_obs)
        '''
        obs, action, reward, done, log_prob, next_obs, predict_pose_list, seg_done = deepcopy(observations)
        if self.configs.state_norm:
            obs = self.state_normalize.state_norm(obs)
            next_obs = self.state_normalize.state_norm(next_obs,update=True)
        observations = (obs, action, reward, done, log_prob, next_obs, predict_pose_list, seg_done)
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
    
    def _merge_state_action(self, state:dict, action:torch.tensor):
        state['action'] = action
        return state
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def _get_action_and_log_prob(self, obs, gt_traj_point_batch, length_batch):
        observation = obs
        b = self._unwrap(self.bundle)
        encoder_out, point_out = b.encode_actor_obs(observation)
        len_curr = length_batch
        _, policy_dist = b.actor_net(encoder_out, point_out, gt_traj_point_batch, len_curr)

        a_mean = torch.clamp(policy_dist, -0.999, 0.999)
        mean_u = self.atanh(a_mean)
        log_std = b.log_std.expand_as(mean_u)
        log_std = torch.clamp(log_std, -2.0, 0.0)
        std = torch.exp(log_std)
        dist_u = Normal(mean_u, std)
        action_batch_u = dist_u.rsample()

        action_batch = torch.tanh(action_batch_u)

        action_batch = torch.clamp(action_batch, -0.999, 0.999)

        log_prob = dist_u.log_prob(action_batch_u).sum(dim=1, keepdim=True)
        log_prob = log_prob - torch.log(1.0 - action_batch.pow(2) + 1e-6).sum(dim=1, keepdim=True)
        return action_batch, log_prob
    

    def _q1_forward(self, state_batch: dict, action: torch.Tensor, tgt_sq, length):
        b = self._unwrap(self.bundle)
        enc = b.q1_encoder(state_batch, action)           # adapter: 内部会把 action 塞进 dict（记得用 dict(state) 防污染）
        pt  = b.q1_point_encoder(state_batch["park_target_point"])  # 具体 key 按你数据结构来
        return b.q1_net(enc, pt, tgt_sq, length)                          # 具体签名按你 ValueDecoder 实现来

    def _q2_forward(self, state_batch: dict, action: torch.Tensor, tgt_sq, length):
        b = self._unwrap(self.bundle)
        enc = b.q2_encoder(state_batch, action)
        pt  = b.q2_point_encoder(state_batch["park_target_point"])
        return b.q2_net(enc, pt, tgt_sq, length)
    

    def _q1_target_forward(self, state_batch: dict, action: torch.Tensor, tgt_sq, length):
        b = self._unwrap(self.bundle)
        # target encoder / point encoder
        enc = b.q1_target_encoder(state_batch, action)                 # adapter里记得用 dict(state) 防污染
        pt  = b.q1_target_point_encoder(state_batch["park_target_point"])   # key 按你的实际命名
        return b.q1_target_net(enc, pt, tgt_sq, length)

    def _q2_target_forward(self, state_batch: dict, action: torch.Tensor, tgt_sq, length):
        b = self._unwrap(self.bundle)
        enc = b.q2_target_encoder(state_batch, action)
        pt  = b.q2_target_point_encoder(state_batch["park_target_point"])
        return b.q2_target_net(enc, pt, tgt_sq, length)
    
    def _set_train_mode(self):
        """让在线网络进入训练模式（actor + q1/q2），target 不需要 train。"""
        b = self._unwrap(self.bundle)
        b.actor_encoder.train()
        b.actor_point_encoder.train()
        b.actor_net.train()

        b.q1_encoder.train()
        b.q1_point_encoder.train()
        b.q1_net.train()

        b.q2_encoder.train()
        b.q2_point_encoder.train()
        b.q2_net.train()

        # target 网络一般保持 eval（它们不需要 dropout 行为；也不需要更新 BN）
        b.q1_target_encoder.eval()
        b.q1_target_point_encoder.eval()
        b.q1_target_net.eval()

        b.q2_target_encoder.eval()
        b.q2_target_point_encoder.eval()
        b.q2_target_net.eval()


    def _keep_frozen_embed_img_eval(self):
        """把所有 embed_img（包括 adapter.net.embed_img）强制 eval，并确保 requires_grad=False。"""
        b = self._unwrap(self.bundle)

        def _freeze_embed_img_of(module):
            # 兼容 adapter：真正的 MultiObsEmbedding 在 .net 里
            core = module.net if hasattr(module, "net") else module
            if hasattr(core, "embed_img"):
                core.embed_img.eval()
                for p in core.embed_img.parameters():
                    p.requires_grad = False

        # 你希望冻结哪些就列哪些（actor + q1/q2 + target）
        for m in [
            b.actor_encoder,
            b.q1_encoder, b.q2_encoder,
            b.q1_target_encoder, b.q2_target_encoder,
        ]:
            _freeze_embed_img_of(m)

    def update(self, step):
        for _ in range(self.configs.mini_epoch):
            batches = self.memory.sample(self.configs.batch_size)
            state_batch = self.obs2tensor(batches["state"])
            action_np = np.asarray(batches["action"], dtype=np.float32)
            action_batch = torch.from_numpy(action_np).to(self.device)
            reward_np = np.asarray(batches["reward"], dtype=np.float32).reshape(-1, 1)
            rewards = torch.from_numpy(reward_np).to(self.device)
            reward_batch = self._reward_norm(rewards) if self.configs.reward_norm else rewards
            done_np = np.asarray(batches["done"], dtype=np.float32).reshape(-1, 1)
            done_batch = torch.from_numpy(done_np).to(self.device)
            next_state_batch = self.obs2tensor(batches["next_obs"])
            seg_done_np = np.asarray(batches["seg_done"], dtype=np.float32).reshape(-1, 1)
            seg_done_batch = torch.from_numpy(seg_done_np).to(self.device)  # [B,1] 0/1
            PAD_TOKEN = 602.0
            def pad_pose_seqs(pose_seqs, pad_token, device):
                ts = []
                lengths = torch.empty(len(pose_seqs), dtype=torch.long, device=device)

                for i, s in enumerate(pose_seqs):
                    a = np.asarray(s, dtype=np.float32).reshape(-1, 4)  # [Li,4]
                    t = torch.from_numpy(a).to(device)                  # [Li,4]
                    ts.append(t)
                    lengths[i] = t.size(0)

                tgt = pad_sequence(ts, batch_first=True, padding_value=float(pad_token))  # [B,T,4]
                return tgt, lengths
            
            pose_seqs = batches["predict_pose_list"]
            gt_traj_point_batch, length_batch = pad_pose_seqs(pose_seqs, PAD_TOKEN, self.device)

            # 0) 固定模式：online train、target eval
            b = self._unwrap(self.bundle)
            self._set_train_mode()

            # 1) 永久冻结模块强制 eval（防 BN/Dropout 漂）
            self._keep_frozen_embed_img_eval()

            if self.start_traj_point_np is None:
                raise RuntimeError("start_traj_point_np is None. Call set_start_traj_point() in train script first.")
            
            tgt_next = gt_traj_point_batch.clone()        # [B,T,4]
            len_next = length_batch.clone()               # [B]
            next_state_for_td = {k: v.clone() for k, v in state_batch.items()}  # 独立副本

            # 找到边界样本 idx
            boundary_mask = (seg_done_batch.squeeze(1) > 0.5)  # [B] bool
            if boundary_mask.any():
                idx = torch.nonzero(boundary_mask, as_tuple=False).squeeze(1)

                # next_state = next_obs（边界时）
                # 注意：next_state_batch 已经 obs2tensor 过
                for k in next_state_for_td.keys():
                    next_state_for_td[k][idx] = next_state_batch[k][idx]

                # tgt_next reset：第一 token = start，其余 padding；len_next=1
                start = torch.as_tensor(self.start_traj_point_np, device=self.device, dtype=torch.float32)  # [4]
                tgt_next[idx, :, :] = PAD_TOKEN
                tgt_next[idx, 0, :] = start
                len_next[idx] = 1
            
            # soft Q loss
            with torch.no_grad():
                with temporary_eval(b.actor_encoder, b.actor_point_encoder, b.actor_net):
                    next_action_batch, next_log_prob = self._get_action_and_log_prob(next_state_for_td, tgt_next, len_next)
                q1_target = self._q1_target_forward(next_state_for_td, next_action_batch, tgt_next, len_next)
                q2_target = self._q2_target_forward(next_state_for_td, next_action_batch, tgt_next, len_next)
                q_target = reward_batch + (1 - done_batch) * self.configs.gamma * (
                    torch.min(q1_target, q2_target) - self.alpha.detach() * next_log_prob
                )

            tgt_train = gt_traj_point_batch
            len_curr = (length_batch - 1).clamp(min=1) 

            current_q1 = self._q1_forward(state_batch, action_batch, tgt_train, len_curr)
            current_q2 = self._q2_forward(state_batch, action_batch, tgt_train, len_curr)

            q1_loss = F.mse_loss(current_q1, q_target.detach())
            q2_loss = F.mse_loss(current_q2, q_target.detach())

            # update the critic networks
            self.critic_optimizer1.zero_grad()
            q1_loss.backward()
            self.critic_optimizer1.step()
            self.critic_optimizer2.zero_grad()
            q2_loss.backward()
            self.critic_optimizer2.step()

            q1_parts = [b.q1_encoder, b.q1_point_encoder, b.q1_net]
            q2_parts = [b.q2_encoder, b.q2_point_encoder, b.q2_net]


            with temporary_freeze_modules(q1_parts + q2_parts),temporary_eval(b.q1_encoder, b.q1_point_encoder, b.q1_net,
                                                                  b.q2_encoder, b.q2_point_encoder, b.q2_net):
                # policy loss
                action_, log_prob = self._get_action_and_log_prob(state_batch, tgt_train, len_curr)

                q1_value = self._q1_forward(state_batch, action_, tgt_train, len_curr)
                q2_value = self._q2_forward(state_batch, action_, tgt_train, len_curr)
                actor_loss = (self.alpha.detach() * log_prob - torch.min(q1_value, q2_value)).mean()

                # update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.actor_loss_list.append(actor_loss.mean().item())

            # optimize alpha
            if self.configs.learn_temperature:
                alpha_loss = (self.alpha * (-log_prob - self.configs.target_entropy).detach()).mean()
                self.log_alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.log_alpha_optimizer.step()

            if step % 1000 ==0 and ((not self.distributed) or dist_gpu.get_rank() == 0):
                def safe_grad_norm(params):
                    gs = [p.grad.detach().norm() for p in params if p.grad is not None]
                    return torch.norm(torch.stack(gs)).item() if len(gs) > 0 else 0.0
                
                actor_grad_ar = safe_grad_norm(b.actor_net.parameters())
                actor_grad_enc = safe_grad_norm(b.actor_encoder.parameters()) if hasattr(b, "actor_encoder") else 0.0
                actor_grad_pt  = safe_grad_norm(b.actor_point_encoder.parameters()) if hasattr(b, "actor_point_encoder") else 0.0

                critic_grad_q1 = safe_grad_norm(b.q1_net.parameters())
                critic_grad_enc = safe_grad_norm(b.q1_encoder.parameters()) if hasattr(b, "q1_encoder") else 0.0
                critic_grad_pt  = safe_grad_norm(b.q1_point_encoder.parameters()) if hasattr(b, "q1_point_encoder") else 0.0
                seg_mask = seg_done_batch.squeeze(1) > 0.5
                with torch.no_grad():
                    log = {
                         # critic
                        "Q1_mean": current_q1.mean().item(),
                        "Q2_mean": current_q2.mean().item(),
                        "Q_std": current_q1.std().item(),
                        "q12_gap": (current_q1 - current_q2).abs().mean().item(),
                        "q_target_mean": q_target.mean().item(),
                        "q_target_std": q_target.std().item(),

                        # loss
                        "critic_loss": q1_loss.item(),
                        "actor_loss": actor_loss.item(),

                        # policy / entropy
                        "logp_mean": log_prob.mean().item(),
                        "entropy": (-log_prob).mean().item(),
                        "alpha": self.alpha.item(),

                        # gradients
                        "actor_grad_ar": actor_grad_ar,
                        "actor_grad_enc": actor_grad_enc,
                        "actor_grad_pt": actor_grad_pt,

                        "critic_grad_q1": critic_grad_q1,
                        "critic_grad_enc": critic_grad_enc,
                        "critic_grad_pt": critic_grad_pt,

                        # prefix / segmentation
                        "len_curr_mean": len_curr.float().mean().item(),
                        "len_curr_min": len_curr.min().item(),
                        "len_curr_max": len_curr.max().item(),
                        "seg_done_ratio": seg_done_batch.float().mean().item(),
                        "log_Q_seg": current_q1[seg_mask].mean().item() if seg_mask.any() else 0.0,
                        "log_Q_nonseg": current_q1[~seg_mask].mean().item() if (~seg_mask).any() else 0.0,

                        # action distribution
                        "action_std_steer": action_[:,0].std().item(),
                        "action_std_speed": action_[:,1].std().item(),
                        "action_mean_steer": action_[:,0].mean().item(),
                        "action_mean_speed": action_[:,1].mean().item(),
                    }
                    print("========== TRAIN LOG ==========")
                    for k, v in log.items():
                        print(f"{k}: {v}")
                    print("================================")

            # soft update target networks
            # Q1 target update
            self._soft_update(
                [b.q1_target_encoder, b.q1_target_point_encoder, b.q1_target_net],
                [b.q1_encoder,        b.q1_point_encoder,        b.q1_net]
            )

            # Q2 target update
            self._soft_update(
                [b.q2_target_encoder, b.q2_target_point_encoder, b.q2_target_net],
                [b.q2_encoder,        b.q2_point_encoder,        b.q2_net]
            )
            self.critic_loss_list.append(q1_loss.mean().item())


        # for debug
        a = actor_loss.detach().cpu().numpy()
        b = q1_loss.item()
        return a, b

    def save(self, path: str = None, params_only: bool = None) -> None:
        """Store the model structure and corresponding parameters to a file.
        (aligned with PPO-style check_list save)
        """
        if params_only is not None:
            self.save_params = params_only

        if self.save_params and len(self.check_list) > 0:
            # 只在主进程保存
            if (not self.distributed) or dist_gpu.get_rank() == 0:
                checkpoint = {}
                for name, item, save_state_dict in self.check_list:
                    if save_state_dict:
                        if isinstance(item, nn.Module):
                            # 保存时去掉DDP的module前缀
                            checkpoint[name] = self._unwrap(item).state_dict()
                        elif isinstance(item, torch.optim.Optimizer):
                            checkpoint[name] = item.state_dict()
                        elif isinstance(item, nn.Parameter):
                            checkpoint[name] = item.detach().cpu()
                        elif torch.is_tensor(item):
                            checkpoint[name] = item.detach().cpu()
                        else:
                            checkpoint[name] = item
                    else:
                        checkpoint[name] = item

                # 可选：保存 state_norm
                if getattr(self.configs, "state_norm", False) and hasattr(self, "state_normalize"):
                    checkpoint["state_norm"] = self.state_normalize

                torch.save(checkpoint, path)

            # 等待所有进程
            if self.distributed:
                dist_gpu.barrier()

        else:
            # 不建议保存整个对象（和 PPO 一致），但保留这个分支以兼容你之前的接口
            torch.save(self, path)

        if getattr(self, "verbose", False) and ((not self.distributed) or dist_gpu.get_rank() == 0):
            print(f"Save current model to {path}")

    def load(self, path: str = None, params_only: bool = None) -> None:
        """Load the model structure and corresponding parameters from a file.
        (aligned with PPO-style check_list load)
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
                        missing, unexpected = self._unwrap(item).load_state_dict(ckpt_val, strict=False)
                        if getattr(self, "verbose", False) and (missing or unexpected):
                            print(f"[load][WARN] {name}: missing={len(missing)} unexpected={len(unexpected)}")

                    elif isinstance(item, torch.optim.Optimizer):
                        item.load_state_dict(ckpt_val)

                    # (b) nn.Parameter：用 data.copy_ 恢复
                    elif isinstance(item, nn.Parameter):
                        t = ckpt_val.to(self.device)
                        if item.data.shape != t.shape:
                            raise RuntimeError(
                                f"[load][ERR] {name} shape mismatch: "
                                f"param {tuple(item.data.shape)} vs ckpt {tuple(t.shape)}"
                            )
                        item.data.copy_(t)

                    # (c) Tensor：copy_
                    elif torch.is_tensor(item):
                        t = ckpt_val.to(self.device)
                        if item.shape != t.shape:
                            raise RuntimeError(
                                f"[load][ERR] {name} shape mismatch: "
                                f"tensor {tuple(item.shape)} vs ckpt {tuple(t.shape)}"
                            )
                        item.copy_(t)

                    else:
                        # 兜底
                        setattr(self, name, ckpt_val)

                else:
                    # 直接赋值到 self 上（configs 这类）
                    setattr(self, name, ckpt_val)

            # 2) 恢复 state normalize
            if "state_norm" in checkpoint:
                self.state_normalize = checkpoint["state_norm"]

            # 3) 加载后可选：再次冻结 embed_img（如果你希望）
            if hasattr(self, "freeze_multi_encoder_embed_img"):
                try:
                    self.freeze_multi_encoder_embed_img()
                except Exception as e:
                    if getattr(self, "verbose", False):
                        print(f"[load][WARN] freeze_multi_encoder_embed_img failed: {e}")

        else:
            obj = torch.load(path, map_location=self.device)
            raise RuntimeError("Loading full object is not supported safely here. Use params_only=True.")

        if getattr(self, "verbose", False):
            print(f"Load the model from {path}")


    def set_start_traj_point(self, start_traj_point_np: np.ndarray):
        self.start_traj_point_np = np.asarray(start_traj_point_np, dtype=np.float32).reshape(4,)
