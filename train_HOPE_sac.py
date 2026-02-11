import sys
sys.path.append("..")
sys.path.append(".")
import time
import os
from shutil import copyfile
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from model_interface.model.agent.sac_agent import SACAgent as SAC
from model_interface.model.agent.parking_agent import ParkingAgent, RsPlanner
from env.car_parking_base import CarParking
from env.env_wrapper import CarParkingWrapper
from env.vehicle import VALID_SPEED,Status
from evaluation.eval_utils import eval
from vehicle_config import *
from utils.config import get_train_config_obj
import torch.distributed as dist
import threading



def setup_distributed():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

        device = torch.device(f"cuda:{local_rank}")
        distributed = True
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        distributed = False
        rank = 0
        world_size = 1
    return device, distributed, rank, world_size

# def gpu_burn(device=0, iters_per_burst=5, sleep_ms=8):
#     torch.cuda.set_device(device)
#     x = torch.randn(2048, 2048, device='cuda')

#     start = torch.cuda.Event(enable_timing=False)
#     end = torch.cuda.Event(enable_timing=False)

#     while True:
#         start.record()
#         for _ in range(iters_per_burst):
#             x = x @ x
#         end.record()

#         # 等这“一小串”GPU任务真正跑完（只阻塞 burn 线程）
#         end.synchronize()

#         # 让出 CPU（释放 GIL），避免 burn 线程抢占 CPU
#         time.sleep(sleep_ms / 1000.0)

def gpu_burn(stop_flag, run_flag, device=0, iters_per_burst=5, sleep_ms=8):
    torch.cuda.set_device(device)
    x = torch.randn(3000, 3000, device='cuda')
    end = torch.cuda.Event(enable_timing=False)

    while not stop_flag.is_set():

        # —— 暂停逻辑（核心）——
        if not run_flag.is_set():
            time.sleep(0.01)    # 睡一下，避免 CPU 忙等
            continue

        # —— GPU burn —— 
        for _ in range(iters_per_burst):
            x = x @ x
        end.record()
        end.synchronize()

        time.sleep(sleep_ms / 1000.0)

def reserve_gpu_memory(device, reserve_mb=4096):
    # reserve_mb: 预占用多少 MB 显存
    n_bytes = reserve_mb * 1024 * 1024
    dummy = torch.empty(n_bytes // 4, dtype=torch.float32, device=device)
    return dummy

class GpuBurnWorker:
    def __init__(self, device=0, iters_per_burst=5, sleep_ms=5):
        self.device = device
        self.iters_per_burst = iters_per_burst
        self.sleep_ms = sleep_ms

        self.stop_flag = threading.Event()
        self.run_flag = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self.stop_flag.clear()
        self.run_flag.set()  # 默认开跑
        self._thread = threading.Thread(target=self._loop, daemon=False)
        self._thread.start()

    def pause(self):
        self.run_flag.clear()

    def resume(self):
        self.run_flag.set()

    def stop(self, join_timeout=None):
        """停止 burn，并回收资源；join_timeout=None 表示一直等到线程退出。"""
        self.stop_flag.set()
        # 为了让 pause 状态下也能立刻退出（不一直睡）
        self.run_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)

    def _loop(self):
        # 线程内创建 CUDA 资源，线程退出时统一释放
        torch.cuda.set_device(self.device)

        # 用 stream 让 burn 更可控（可选）
        stream = torch.cuda.Stream(device=self.device)

        # 提前分配，避免循环里不断生成新 tensor
        with torch.cuda.stream(stream):
            x = torch.randn(3000, 3000, device="cuda")
            y = torch.empty_like(x)  # 用 out 复用内存（减少显存抖动）
            end = torch.cuda.Event(enable_timing=False)

        # 确保初始化完成
        stream.synchronize()

        try:
            while not self.stop_flag.is_set():
                # 若暂停：用 wait(timeout) 代替 sleep + busy loop，且可被 stop 立刻打断
                if not self.run_flag.is_set():
                    # 等待 run_flag 置位或 stop_flag 置位
                    self.run_flag.wait(timeout=0.01)
                    continue

                # —— GPU burn ——（尽量避免分配）
                with torch.cuda.stream(stream):
                    for _ in range(self.iters_per_burst):
                        # y = x @ x, then swap (避免 x=x@x 产生新 tensor)
                        torch.matmul(x, x, out=y)
                        x, y = y, x
                    end.record()

                # 让 burst 真正完成（如果你想 stop 更“立刻”，这里可以缩小 burst 或减少同步频率）
                stream.synchronize()

                # sleep 也做成可被 stop 打断
                if self.stop_flag.wait(self.sleep_ms / 1000.0):
                    break

        finally:
            # —— 资源释放 ——（尽力回收本 worker 占用的显存）
            try:
                # 先确保 stream 上没有未完成任务
                stream.synchronize()
            except Exception:
                pass

            # 删除 GPU tensor / event / stream 的引用
            del x, y, end, stream

            # 尽力释放缓存（注意：不会销毁 CUDA context，但会把缓存还给 PyTorch allocator）
            torch.cuda.empty_cache()
            # 如果你有大量碎片，偶尔加这个（代价较大）：
            # torch.cuda.ipc_collect()


class SceneChoose():
    def __init__(self) -> None:
        self.scene_types = {0:'Normal', 
                            1:'Complex',
                            2:'Extrem',
                            3:'dlp',
                            }
        self.target_success_rate = np.array([0.95, 0.95, 0.9, 0.99])
        self.success_record = {}
        for scene_name in self.scene_types:
            self.success_record[scene_name] = []
        self.scene_record = []
        self.history_horizon = 200
        
        
    def choose_case(self,):
        # if len(self.scene_record) < self.history_horizon:
        #     scene_chosen = self._choose_case_uniform()
        # else:
        #     if np.random.random() > 0.5:
        #         scene_chosen = self._choose_case_worst_perform()
        #     else:
        #         scene_chosen = self._choose_case_uniform()
        scene_chosen = 1
        self.scene_record.append(scene_chosen)
        return self.scene_types[scene_chosen]
    
    def update_success_record(self, success:int):
        self.success_record[self.scene_record[-1]].append(success)

    def _choose_case_uniform(self,):
        case_count = np.zeros(len(self.scene_types))
        for i in range(min(len(self.scene_record), self.history_horizon)):
            scene_id = self.scene_record[-(i+1)]
            case_count[scene_id] += 1
        return np.argmin(case_count)
    
    def _choose_case_worst_perform(self,):
        success_rate = []
        for i in self.success_record.keys():
            idx = int(i)
            recent_success_record = self.success_record[idx][-min(250, len(self.success_record[idx])):]
            success_rate.append(np.sum(recent_success_record)/len(recent_success_record))
        fail_rate = self.target_success_rate - np.array(success_rate)
        fail_rate = np.clip(fail_rate, 0.01, 1)
        fail_rate = fail_rate/np.sum(fail_rate)
        return np.random.choice(np.arange(len(fail_rate)), p=fail_rate)

class DlpCaseChoose():
    def __init__(self) -> None:
        self.dlp_case_num = 248
        self.case_record = []
        self.case_success_rate = {}
        for i in range(self.dlp_case_num):
            self.case_success_rate[str(i)] = []
        self.horizon = 500
    
    def choose_case(self,):
        if np.random.random()<0.2 or len(self.case_record)<self.horizon:
            return np.random.randint(0, self.dlp_case_num)
        success_rate = []
        for i in range(self.dlp_case_num):
            idx = str(i)
            if len(self.case_success_rate[idx]) <= 1:
                success_rate.append(0)
            else:
                recent_success_record = self.case_success_rate[idx][-min(10, len(self.case_success_rate[idx])):]
                success_rate.append(np.sum(recent_success_record)/len(recent_success_record))
        fail_rate = 1-np.array(success_rate)
        fail_rate = np.clip(fail_rate, 0.005, 1)
        fail_rate = fail_rate/np.sum(fail_rate)
        return np.random.choice(np.arange(len(fail_rate)), p=fail_rate)
    
    def update_success_record(self, success:int, case_id:int):
        self.case_success_rate[str(case_id)].append(success)
        self.case_record.append(case_id)


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--agent_ckpt', type=str, default='./rl_model/SAC2_14999.pt') # './model/ckpt/SAC.pt'
    parser.add_argument('--img_ckpt', type=str, default='./model/ckpt/autoencoder.pt')
    parser.add_argument('--train_episode', type=int, default=100000)
    parser.add_argument('--eval_episode', type=int, default=20)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--visualize', type=bool, default=True)
    parser.add_argument('--config', default='./config/training_real.yaml', type=str)
    args = parser.parse_args()
    config_path = args.config
    config_obj = get_train_config_obj(config_path)

    verbose = args.verbose
    # import warnings
    # warnings.filterwarnings("error", message="Mean of empty slice.*", category=RuntimeWarning)
    # warnings.filterwarnings("error", message="invalid value encountered in double_scalars.*", category=RuntimeWarning)


    if args.visualize:
        raw_env = CarParking(fps=100, verbose=verbose,)
    else:
        raw_env = CarParking(fps=100, verbose=verbose, render_mode='rgb_array')
    env = CarParkingWrapper(raw_env)
    scene_chooser = SceneChoose()
    dlp_case_chooser = DlpCaseChoose()

    # the path to log and save model
    relative_path = '.'
    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    save_path = relative_path+'/log/exp/sac_%s/' % timestamp
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    if not os.path.exists(save_path) and rank == 0:
        os.makedirs(save_path)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    device, distributed, rank, world_size = setup_distributed()
    torch.cuda.set_device(0)
    # gpu_reserver = reserve_gpu_memory(device, reserve_mb=20000)
    writer = SummaryWriter(save_path) if rank == 0 else None
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            print("DDP world_size =", dist.get_world_size())
    # configs log
    if rank == 0:
        copyfile('./vehicle_config.py', save_path+'vehicle_config.txt')
    if rank == 0:
        print("You can track the training process by command 'tensorboard --log-dir %s'" % save_path)

    seed = SEED + rank
    # env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    actor_params = ACTOR_CONFIGS
    critic_params = CRITIC_CONFIGS
    configs = {
        "discrete": False,
        "observation_shape": env.observation_shape,
        "action_dim": env.action_space.shape[0],
        "hidden_size": 64,
        "activation": "tanh",
        "dist_type": "gaussian",
        "save_params": False,
        "actor_layers": actor_params,
        "critic_layers": critic_params,
    }
    print('observation_space:',env.observation_space)

    rl_agent = SAC(config_obj, configs)
    checkpoint_path = args.agent_ckpt
    if checkpoint_path is not None:
        rl_agent.load(checkpoint_path, params_only=True)
        rl_agent.reset_optimizers(reinit_alpha=True)
        rl_agent.freeze_multi_encoder_embed_img()
        print('load pre-trained model!')
    # img_encoder_checkpoint =  args.img_ckpt if USE_IMG else None
    # if img_encoder_checkpoint is not None and os.path.exists(img_encoder_checkpoint):
    #     rl_agent.load_img_encoder(img_encoder_checkpoint, require_grad=UPDATE_IMG_ENCODE)

    step_ratio = env.vehicle.kinetic_model.step_len*env.vehicle.kinetic_model.n_step*1.0
    rs_planner = RsPlanner(step_ratio)
    parking_agent = ParkingAgent(rl_agent, rs_planner)


    reward_list = []
    reward_per_state_list = []
    reward_info_list = []
    case_id_list = []
    succ_record = []
    total_step_num = 0
    best_success_rate = [0, 0, 0, 0]
    best_success_ratio = 0
    best_reward_averge = 0
    regressive_step = REGRESSIVE_STEP

    def rs_mix_prob(episode: int) -> float:
        start_episode = 14999
        end_episode = 50000
        start_p = 0.865
        end_p = 0.1
        if episode < start_episode:
            return 0.865
        if episode >= end_episode:
            return end_p

        ratio = (episode - start_episode) / (end_episode - start_episode)
        return start_p + ratio * (end_p - start_p)
    traj = [[0.0,0.0,0.0]]
    traj = np.array(traj, dtype=np.float32)   # [T,3] = (x,y,yaw)
            # x,y 归一化到 [-1,1]
    traj_x = np.clip(traj[:,0] / TRAJXRANGE, -1.0, 1.0)
    traj_y = np.clip(traj[:,1] / TRAJYRANGE, -1.0, 1.0)
    traj_yaw = traj[:,2]   # 假设是弧度

            # [T,4] = (x_norm, y_norm, cos(yaw), sin(yaw))
    start_traj_point_np = np.stack(
        [traj_x, traj_y, np.cos(traj_yaw), np.sin(traj_yaw)],
        axis=-1
    ).reshape(4,)   # [T,4]

    parking_agent.agent.set_start_traj_point(start_traj_point_np)
    # stop_flag = threading.Event()   # 控制“彻底退出”
    # run_flag  = threading.Event()   # 控制“是否运行”
    # run_flag.set()                  # 默认启动就运行
    # t = threading.Thread(target=gpu_burn, args=(stop_flag, run_flag), daemon=True)
    # t.start()
    # # ====== burn 调度参数 ======
    # BURN_ON_STEPS  = 4000   # 开启持续多少 step
    # BURN_OFF_STEPS = 6000   # 暂停持续多少 step
    # BURN_PERIOD    = BURN_ON_STEPS + BURN_OFF_STEPS
    # t = threading.Thread(target=gpu_burn, daemon=True)
    # t.start()
    burn = GpuBurnWorker(device=0, iters_per_burst=5, sleep_ms=5)
    burn.start()
    release = None

    for i in range(args.train_episode):
        scene_chosen = scene_chooser.choose_case()
        if scene_chosen == 'dlp':
            case_id = dlp_case_chooser.choose_case()
        else:
            case_id = None
        obs = env.reset(case_id, None, scene_chosen)
        sample_mask = obs['action_mask']
        parking_agent.reset()
        case_id_list.append(env.map.case_id)
        done = False
        total_reward = 0
        step_num = 0
        reward_info = []
        xy = []
        predict_pose_list = []
        predict_pose_list.append(start_traj_point_np)
        noisy_a = np.random.normal(0, 0.5, 2)
        mode = "macro"        # or "policy"
        mode_left = 0
        macro = None
        warmup_steps = int(parking_agent.configs.memory_size * 1.3)  # 你要的阈值
        p_macro = 0.95       # warmup阶段选macro的概率（你自己调）
        hold_min = 6       # 模式最短保持步数
        hold_max = 30       # 模式最长保持步数
        macro_period = 7    # macro刷新周期（保持你原来习惯）
        clip_macro = 0.9
        clip_policy = 0.95
        step_num_threshold = np.random.randint(10, 60)
        while not done:
            step_num += 1
            total_step_num += 1
            if (step_num >= step_num_threshold) and (total_step_num <= warmup_steps) and (not parking_agent.executing_rs):
                # --- 1) decide / refresh mode only when expired ---
                if mode_left <= 0:
                    # 按概率选模式
                    if np.random.rand() < p_macro:
                        mode = "macro"
                    else:
                        mode = "policy"
                    # 选定后保持一段时间（随机一个长度更好，避免周期性偏差）
                    mode_left = np.random.randint(hold_min, hold_max + 1)

                mode_left -= 1

                # --- 2) execute selected mode ---
                if mode == "macro":
                    # macro 只在固定周期刷新一次，其余步保持不变
                    if (macro is None) or (step_num % macro_period == 1):
                        macro = parking_agent.sample_action_with_mask(sample_mask)
                    action = np.clip(macro, -clip_macro, clip_macro)
                    log_prob = 0.0

                else:  # mode == "policy"
                    action_raw, log_prob = parking_agent.get_action(obs, predict_pose_list)
                    action = np.clip(action_raw, -clip_policy, clip_policy)

            else:
                # warmup结束 or 正在执行RS：保持你原来的逻辑
                if step_num < step_num_threshold and (total_step_num <= warmup_steps):
                    action_raw, log_prob = parking_agent.choose_action_eval(obs, predict_pose_list)
                else:
                    action_raw, log_prob = parking_agent.get_action(obs, predict_pose_list)
                if not parking_agent.executing_rs:
                    action = np.clip(action_raw, -clip_policy, clip_policy)
                else:
                    action = action_raw
            next_obs, reward, done, info = env.step(action)
            sample_mask = next_obs['action_mask']
            reward_info.append(list(info['reward_info'].values()))
            total_reward += reward
            reward_per_state_list.append(reward)
            next_pose = parking_agent.vcs_action_step(predict_pose_list[-1], action)
            predict_pose_list.append(next_pose)
            pose_seq_cpu = torch.as_tensor(
                np.asarray(predict_pose_list, dtype=np.float32).reshape(-1, 4),
                dtype=torch.float32
            )  # CPU tensor, [L,4]
            seg_done = (step_num % regressive_step == 0)
            success_flag = bool(done and info.get("status", None) == Status.ARRIVED)

            parking_agent.push_memory((obs, action, reward, done, log_prob, next_obs, pose_seq_cpu, seg_done, success_flag))
            # parking_agent.push_memory((obs, action, reward, done, log_prob, next_obs, pose_seq_cpu, seg_done))
            if seg_done:
                obs = next_obs
                predict_pose_list.clear()
                predict_pose_list.append(start_traj_point_np)
                
            if release is None and total_step_num > warmup_steps:
                burn.stop()
                release = True
            # obs = next_obs
            if total_step_num > warmup_steps and total_step_num%10==0: 
                actor_loss, critic_loss = parking_agent.update(total_step_num, i)
                # if total_step_num%1000==0 and (rank == 0):
                
                #     writer.add_scalar("actor_loss", actor_loss, i)
                #     writer.add_scalar("critic_loss", critic_loss, i)
            
            rs_path_avail = (info.get('path_to_dest', None) is not None)

            # 退火混合：rs_path_avail 时才抽样是否执行 RS
            if rs_path_avail:
                p_rs = rs_mix_prob(i)
                use_rs = (np.random.random() < p_rs)
                # use_rs = True
            else:
                p_rs = 0.0
                use_rs = False
            
            if use_rs:
                parking_agent.set_planner_path(info['path_to_dest'], True)
            else:
                parking_agent.reset()

            if ((not parking_agent.distributed) or rank == 0) and (total_step_num % 1000 == 0):
                print(f"rs/p_rs: {p_rs:.3f} rs/use_rs: {use_rs}  episode: {i}  total_step_num: {total_step_num}")
    
            # —— 状态切换检测 & 打印 ——
            last = parking_agent._last_use_rs

            if last is None:
                # 第一次进入
                # print(f"{total_step_num}: {'use_rs_path' if use_rs else 'not_use_rs_path'} (start)")
                parking_agent._state_start_frame = total_step_num

            elif last != use_rs:
                # 状态发生切换
                duration = total_step_num - parking_agent._state_start_frame
                # print(
                #     f"{total_step_num}: "
                #     f"{'use_rs_path' if last else 'not_use_rs_path'} "
                #     f"lasted {duration} frames"
                # )

                # print(
                #     f"{total_step_num}: "
                #     f"{'use_rs_path' if use_rs else 'not_use_rs_path'} (start)"
                # )

                parking_agent._state_start_frame = total_step_num

            # 更新状态
            parking_agent._last_use_rs = use_rs

            if done:
                if info['status']==Status.ARRIVED:
                    succ_record.append(1)
                    scene_chooser.update_success_record(1)
                    if scene_chosen == 'dlp':
                        dlp_case_chooser.update_success_record(1, case_id)
                else:
                    succ_record.append(0)
                    scene_chooser.update_success_record(0)
                    if scene_chosen == 'dlp':
                        dlp_case_chooser.update_success_record(0, case_id)

        # if total_step_num > parking_agent.configs.memory_size:
        #     phase = total_step_num % BURN_PERIOD
        #     if phase < BURN_ON_STEPS:
        #         run_flag.set()    # 开启 burn
        #     else:
        #         run_flag.clear()  # 暂停 burn

        if total_step_num%1000==0 and ((not parking_agent.distributed) or rank == 0):     
            # writer.add_scalar("total_reward", total_reward, i)
            # writer.add_scalar("avg_reward", np.mean(reward_per_state_list[-1000:]), i)
            bundle = parking_agent.agent._unwrap(parking_agent.agent.bundle)
            log_std = bundle.log_std.detach().cpu().numpy().reshape(-1)
            # writer.add_scalar("action_std0", log_std[0],i)
            # writer.add_scalar("action_std1", log_std[1],i)
            # writer.add_scalar("alpha", parking_agent.alpha.detach().cpu().numpy().reshape(-1)[0],i)
            # for type_id in scene_chooser.scene_types:
            #     vals = scene_chooser.success_record[type_id][-100:]
            #     mean_success = float(np.mean(vals)) if len(vals) > 0 else 0.0
            #     writer.add_scalar(
            #         f"success_rate_{scene_chooser.scene_types[type_id]}",
            #         mean_success,
            #         i
            #     )
            # writer.add_scalar("step_num", step_num, i)
        reward_list.append(total_reward)
        reward_info = np.sum(np.array(reward_info), axis=0)
        reward_info = np.round(reward_info,2)
        reward_info_list.append(list(reward_info))

        if verbose and i%10==0 and i>0 and rank == 0:
            print('success rate:',np.sum(succ_record),'/',len(succ_record))
            print('success rate ratio: {:.6f}'.format(np.sum(succ_record) / len(succ_record)))
            bundle = parking_agent.agent._unwrap(parking_agent.agent.bundle)
            log_std = bundle.log_std.detach().cpu().numpy().reshape(-1)
            print("log_std: ", log_std)
            print("alpha: ", parking_agent.alpha.detach().cpu().numpy().reshape(-1))
            print("episode:%s  average reward:%s"%(i,np.mean(reward_list[-50:])))
            print(np.mean(parking_agent.actor_loss_list[-100:]),np.mean(parking_agent.critic_loss_list[-100:]))
            print('time_cost ,rs_dist_reward ,dist_reward ,angle_reward ,box_union_reward ,gear_shift_reward ,abs_shape ,near_bonus, low_speed, risk_reward, high_speed, big_steer','u_turn')
            for j in range(10):
                print(case_id_list[-(10-j)],reward_list[-(10-j)],reward_info_list[-(10-j)])
            print("")

        def safe_mean_last(lst, n=100, default=0.0):
            vals = lst[-n:] if lst is not None else []
            return float(np.mean(vals)) if len(vals) > 0 else float(default)

        # save best model
        for type_id in scene_chooser.scene_types:
            success_rate_normal  = safe_mean_last(scene_chooser.success_record.get(0, []), 100)
            success_rate_complex = safe_mean_last(scene_chooser.success_record.get(1, []), 100)
            success_rate_extreme = safe_mean_last(scene_chooser.success_record.get(2, []), 100)
            success_rate_dlp     = safe_mean_last(scene_chooser.success_record.get(3, []), 100)
        if success_rate_normal >= best_success_rate[0] and success_rate_complex >= best_success_rate[1] and\
            success_rate_extreme >= best_success_rate[2] and i>100:
            raw_best_success_rate = np.array([success_rate_normal, success_rate_complex, success_rate_extreme, success_rate_dlp])
            best_success_rate = list(np.minimum(raw_best_success_rate, scene_chooser.target_success_rate))
            if distributed:
                dist.barrier()
            # if rank == 0:
            #     parking_agent.save("%s/SAC_best.pt" % (save_path),params_only=True)
            #     f_best_log = open(save_path+'best.txt', 'w')
            #     f_best_log.write('epoch: %s, success rate: %s %s %s %s'%(i+1, raw_best_success_rate[0],
            #                         raw_best_success_rate[1], raw_best_success_rate[2], raw_best_success_rate[3]))
            #     f_best_log.close()
            if distributed:
                dist.barrier()
        if (i+1) % 1000 == 0:
            if distributed:
                dist.barrier()
            if rank == 0:
                parking_agent.save("%s/SAC2_%s.pt" % (save_path, i),params_only=True)
            if distributed:
                dist.barrier()
        

        # if verbose and i%20==0 and rank == 0:
        #     episodes = [j for j in range(len(reward_list))]
        #     mean_reward = [np.mean(reward_list[max(0,j-50):j+1]) for j in range(len(reward_list))]
        #     plt.plot(episodes,reward_list)
        #     plt.plot(episodes,mean_reward)
        #     plt.xlabel('episodes')
        #     plt.ylabel('reward')
        #     f = plt.gcf()
        #     f.savefig('%s/reward.png'%save_path)
        #     f.clear()

    
        if (i+1) % 1000== 0 and ((not parking_agent.distributed) or rank == 0):
            eval_episode = args.eval_episode
            choose_action = True
            with torch.no_grad():
                # eval on dlp
                # env.set_level('dlp')
                # log_path = save_path+'/dlp'
                # if not os.path.exists(log_path):
                #     os.makedirs(log_path)
                # eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)
                
                # # eval on extreme
                # env.set_level('Extrem')
                # log_path = save_path+'/extreme'
                # if not os.path.exists(log_path):
                #     os.makedirs(log_path)
                # eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)
                
                # eval on complex
                env.set_level('Complex')
                log_path = save_path+'/complex'
                if not os.path.exists(log_path):
                    os.makedirs(log_path)
                success_ratio, reward_avg = eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)
                if success_ratio>=best_success_ratio and reward_avg >= best_reward_averge:
                    parking_agent.save("%s/SAC_best.pt" % (save_path),params_only=True)
                    f_best_log = open(save_path+'best.txt', 'w')
                    f_best_log.write('epoch: %s, success rate: %s, reward_avg: %s '%(i+1, success_ratio, reward_avg))
                    f_best_log.close()
                best_success_ratio = success_ratio
                best_reward_averge = reward_avg
                
                # # eval on normalize
                # env.set_level('Normal')
                # log_path = save_path+'/normalize'
                # if not os.path.exists(log_path):
                #     os.makedirs(log_path)
                # eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)

    env.close()