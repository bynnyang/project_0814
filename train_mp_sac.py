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
import torch.multiprocessing as mp
from queue import Empty
from collections import defaultdict
import queue



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

def actor_worker(
    actor_id,
    args,
    config_path,
    configs,
    start_traj_point_np,
    traj_queue,
    weight_queue,
    stop_event,
    seed,
):
    # 每个 actor 自己的 env
    verbose = False
    raw_env = CarParking(fps=100, verbose=verbose, render_mode="rgb_array")
    env = CarParkingWrapper(raw_env)

    env.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    config_obj = get_train_config_obj(config_path)

    # Actor 用轻量 agent：不建 replay，不建优化器也行（这里先不动优化器，只不建 replay）
    actor_agent = SAC(
        config_obj,
        configs,
        force_device=args.actor_device,
        init_memory=False,          # ✅ actor 不需要 replay
    )

    step_ratio = env.vehicle.kinetic_model.step_len * env.vehicle.kinetic_model.n_step * 1.0
    rs_planner = RsPlanner(step_ratio)
    parking_agent = ParkingAgent(actor_agent, rs_planner)
    parking_agent.agent.set_start_traj_point(start_traj_point_np)

    def rs_mix_prob(episode: int) -> float:
        start_episode = 10000
        end_episode = 50000
        start_p = 1.0
        end_p = 0.1
        if episode < start_episode:
            return 1.1
        if episode >= end_episode:
            return end_p

        ratio = (episode - start_episode) / (end_episode - start_episode)
        return start_p + ratio * (end_p - start_p)

    # 接收初始权重（阻塞等一次）
    state = weight_queue.get()
    if state is not None:
        actor_agent.load_actor_state(state)

    regressive_step = REGRESSIVE_STEP
    local_episode = 0
    total_step_num = 0
    warmup_steps = int(configs.get("memory_size", 30000) * 1.3 / 2.0)

    while not stop_event.is_set():
        local_episode += 1

        # 非阻塞尝试更新权重（如果 learner 推了新的）
        try:
            while True:
                st = weight_queue.get_nowait()
                if st is None:
                    stop_event.set()
                    break
                actor_agent.load_actor_state(st)
        except Empty:
            pass

        obs = env.reset(None, None, "Complex")
        sample_mask = obs["action_mask"]
        parking_agent.reset()
        case_id = env.map.case_id

        done = False
        step_num = 0
        reward_info_step_list = []
        predict_pose_list = [start_traj_point_np]

        # warmup mode 相关变量（照你原逻辑拷贝）
        mode = "macro"
        mode_left = 0
        macro = None
        p_macro = 0.95
        hold_min, hold_max = 6, 30
        macro_period = 7
        clip_macro, clip_policy = 0.9, 0.95
        step_num_threshold = np.random.randint(10, 60)

        total_reward = 0.0
        success = 0

        while not done and not stop_event.is_set():
            step_num += 1
            total_step_num += 1

            # -------- 选 action（基本照你原训练脚本）--------
            if (step_num >= step_num_threshold) and (total_step_num <= warmup_steps) and (not parking_agent.executing_rs):
                if mode_left <= 0:
                    mode = "macro" if np.random.rand() < p_macro else "policy"
                    mode_left = np.random.randint(hold_min, hold_max + 1)
                mode_left -= 1

                if mode == "macro":
                    if (macro is None) or (step_num % macro_period == 1):
                        macro = parking_agent.sample_action_with_mask(sample_mask)
                    action = np.clip(macro, -clip_macro, clip_macro)
                    log_prob = 0.0
                else:
                    action_raw, log_prob = parking_agent.get_action(obs, predict_pose_list)
                    action = np.clip(action_raw, -clip_policy, clip_policy)
            else:
                if step_num < step_num_threshold and (total_step_num <= warmup_steps):
                    action_raw, log_prob = parking_agent.choose_action_eval(obs, predict_pose_list)
                else:
                    action_raw, log_prob = parking_agent.get_action(obs, predict_pose_list)
                action = action_raw if parking_agent.executing_rs else np.clip(action_raw, -clip_policy, clip_policy)

            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            sample_mask = next_obs["action_mask"]
            reward_info_step_list.append(list(info['reward_info'].values()))

            next_pose = parking_agent.vcs_action_step(predict_pose_list[-1], action)
            predict_pose_list.append(next_pose)
            pose_seq_cpu = torch.as_tensor(np.asarray(predict_pose_list, dtype=np.float32).reshape(-1, 4), dtype=torch.float32)

            seg_done = (step_num % regressive_step == 0)
            success_flag = bool(done and info.get("status", None) == Status.ARRIVED)
            if success_flag:
                success = 1

            # ✅ 发送 transition 给 learner
            traj_queue.put((
                "transition",
                actor_id,
                (obs, action, reward, done, log_prob, next_obs, pose_seq_cpu, seg_done, success_flag)
            ))

            if seg_done:
                obs = next_obs
                predict_pose_list = [start_traj_point_np]

            rs_path_avail = (info.get('path_to_dest', None) is not None)

            # 退火混合：rs_path_avail 时才抽样是否执行 RS
            if rs_path_avail:
                p_rs = rs_mix_prob(local_episode)
                use_rs = (np.random.random() < p_rs)
                # use_rs = True
            else:
                p_rs = 0.0
                use_rs = False
            
            if use_rs:
                parking_agent.set_planner_path(info['path_to_dest'], True)
            else:
                parking_agent.reset()

            if (actor_id == 0)  and (total_step_num % 1000 == 0):
                print(f"rs/p_rs: {p_rs:.3f} rs/use_rs: {use_rs}  episode: {local_episode}  total_step_num: {total_step_num}")
    
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

            reward_info_sum = np.round(np.sum(np.array(reward_info_step_list), axis=0), 2).tolist()

        # episode 结束：发统计信息给 learner
        traj_queue.put((
            "episode_end",
            actor_id,
            {
                "total_reward": total_reward,
                "success": success,
                "case_id": int(case_id),
                "reward_info_sum": reward_info_sum
            }
        ))

    env.close()

def put_latest(q, item):
    try:
        while True:
            q.get_nowait()
    except Exception:
        pass
    q.put(item)


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_ckpt', type=str, default='./rl_model/SAC2_6999.pt') # './model/ckpt/SAC.pt'
    parser.add_argument('--img_ckpt', type=str, default='./model/ckpt/autoencoder.pt')
    parser.add_argument('--train_episode', type=int, default=800000)
    parser.add_argument('--eval_episode', type=int, default=20)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--visualize', type=bool, default=True)
    parser.add_argument('--config', default='./config/training_real.yaml', type=str)
    parser.add_argument("--num_actors", type=int, default=2)
    parser.add_argument("--actor_device", type=str, default="cpu")  # 建议 cpu，避免多个进程抢同一张 GPU
    parser.add_argument("--sync_interval", type=int, default=200)  # learner 每多少步同步一次 actor 权重
    parser.add_argument("--queue_size", type=int, default=4096)
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
    reward_info_list = []
    case_id_list = []
    succ_record = []

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
    burn = GpuBurnWorker(device=0, iters_per_burst=5, sleep_ms=5)
    burn.start()
    burn_released = threading.Event()   # ✅ 只释放一次


    if args.num_actors > 1:
        ctx = mp.get_context("spawn")
        traj_queue = ctx.Queue(maxsize=args.queue_size)
        stop_event = ctx.Event()
        weight_queues = [ctx.Queue(maxsize=1) for _ in range(args.num_actors)]

        # 初始 actor 权重
        init_state = parking_agent.agent.get_actor_state()
        for q in weight_queues:
            q.put(init_state)

        procs = []
        for aid in range(args.num_actors):
            p = ctx.Process(
                target=actor_worker,
                args=(aid, args, config_path, configs, start_traj_point_np,
                    traj_queue, weight_queues[aid], stop_event, seed + 1000 * (aid+1)),
                daemon=True
            )
            p.start()
            procs.append(p)

    
        # ========== 线程队列 ==========
        # transitions：高频，建议有界，避免内存无限涨
        TRANS_Q_MAX = 2000  # 视 obs 体积调整：如果每条很大就调小
        trans_q = queue.Queue(maxsize=TRANS_Q_MAX)

        # ctrl：低频（timing/episode_end），用有界+大点即可
        ctrl_q = queue.Queue(maxsize=10000)

        # ========== 共享状态 ==========
        state_lock = threading.Lock()
        model_lock = threading.Lock()  # 保存/评测时暂停训练，避免参数读写冲突

        shared = {
            "global_episode": 0,
            "total_step_num": 0,     # 以“写入 replay 的 transition 数”为准
            "last_sync_step": 0,
            "stop": False,
        }

        # ========== PERF 统计（沿用你原来的打印口径） ==========

        warmup_steps = int(parking_agent.configs.memory_size * 1.3)

        # 想“更频繁更新”就设小一点；设为 10 等价你原来 %10
        UPDATE_EVERY_TRANS = 1

        # ========== 队列放入：满了就丢最旧的 transition（保证 collector 永不阻塞） ==========
        def put_trans_drop_oldest(item):
            try:
                trans_q.put_nowait(item)
            except queue.Full:
                # 丢一条最旧的，再放最新的
                try:
                    trans_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    trans_q.put_nowait(item)
                except queue.Full:
                    # 极端情况下再丢就算了
                    pass

        # ========== 三个线程 ==========
        def collector_loop():
            # 只做：从 mp 队列取消息，快速分发
            while not stop_event.is_set():
                try:
                    msg = traj_queue.get(timeout=0.5)  # 0.5s 无消息就超时，便于退出
                except Empty:
                    continue

                tag = msg[0]
                if tag == "transition":
                    _, aid, exp = msg
                    put_trans_drop_oldest((aid, exp))
                else:
                    # timing / episode_end 等交给主线程处理
                    try:
                        ctrl_q.put_nowait(msg)
                    except queue.Full:
                        # ctrl_q 满了很少见；满了就丢 timing，但不要丢 episode_end
                        if tag == "episode_end":
                            ctrl_q.put(msg)

        def pusher_loop():
            # 只做：从 trans_q 取 transition → push_memory 写 replay
            while not stop_event.is_set():
                try:
                    aid, exp = trans_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                parking_agent.push_memory(exp, actor_id=aid)

                with state_lock:
                    shared["total_step_num"] += 1

        def trainer_loop():
            # 只做：update（纯训练）+ 周期性同步权重
            while not stop_event.is_set():
                with state_lock:
                    ts = shared["total_step_num"]
                    ge = shared["global_episode"]
                    last_sync = shared["last_sync_step"]

                # warmup 前不更新
                if ts <= warmup_steps:
                    time.sleep(0.002)
                    continue

                # 如果你希望“严格每来 N 条新数据才更新一次”，用这个门控：
                # （否则 trainer 会尽量一直跑 update，GPU 更满，但 UTD 会更高）
                # if ts % UPDATE_EVERY_TRANS != 0:
                #     time.sleep(0.001)
                #     continue

                # ✅ warmup 结束后，立刻停掉 burn（只执行一次）
                if not burn_released.is_set():
                    burn.stop(join_timeout=None)  # 或者给个 1~2s timeout
                    burn_released.set()
                    # print(f"gpu burn stopped at ts={ts}")

          
                with model_lock:
                    actor_loss, critic_loss = parking_agent.update(ts, ge)
        

                # 同步权重（不属于训练，但轻量，放这里最方便）
                if (ts - last_sync) >= args.sync_interval:
                    with model_lock:
                        st = parking_agent.agent.get_actor_state()
                    for q in weight_queues:
                        put_latest(q, st)
                    with state_lock:
                        shared["last_sync_step"] = ts

        # 启动线程（建议 daemon=False，便于 join 干净退出）
        t_col = threading.Thread(target=collector_loop, daemon=False)
        t_push = threading.Thread(target=pusher_loop, daemon=False)
        t_trn = threading.Thread(target=trainer_loop, daemon=False)

        t_col.start()
        t_push.start()
        t_trn.start()

        # ========== 主线程：处理 ctrl_q（episode_end/timing）+ 打印 PERF + save/eval ==========
        best_success_ratio = 0.0
        best_reward_averge = -1e9

        while not stop_event.is_set():
            # 1) 处理控制消息（episode_end/timing）
            try:
                msg = ctrl_q.get(timeout=0.2)
                tag = msg[0]
                if tag == "timing":
                    _, aid, payload = msg
                elif tag == "episode_end":
                    _, aid, payload = msg
                    with state_lock:
                        shared["global_episode"] += 1
                        ge = shared["global_episode"]

                    succ_record.append(payload["success"])
                    reward_list.append(payload["total_reward"])
                    reward_info_list.append(payload["reward_info_sum"])
                    case_id_list.append(payload["case_id"])

                    # 训练结束
                    if ge >= args.train_episode:
                        stop_event.set()
                        break

                    # 你原来的 verbose 打印逻辑可以放这里（按 ge 判断即可）
                    if verbose and (ge % (10 * args.num_actors) == 0) and rank == 0:
                        print('success rate:', np.sum(succ_record), '/', len(succ_record))
                        print('success rate ratio: {:.6f}'.format(np.sum(succ_record) / len(succ_record)))
                        bundle = parking_agent.agent._unwrap(parking_agent.agent.bundle)
                        log_std = bundle.log_std.detach().cpu().numpy().reshape(-1)
                        print("log_std: ", log_std)
                        print("alpha: ", parking_agent.alpha.detach().cpu().numpy().reshape(-1))
                        print("episode:%s  average reward:%s" % (ge, np.mean(reward_list[-50:])))
                        print(np.mean(parking_agent.actor_loss_list[-100:]), np.mean(parking_agent.critic_loss_list[-100:]))
                        print('time_cost ,rs_dist_reward ,dist_reward ,angle_reward ,box_union_reward ,gear_shift_reward ,abs_shape ,near_bonus, low_speed, risk_reward, high_speed, big_steer','u_turn')
                        for cid, rew, rinfo in zip(case_id_list[-10:], reward_list[-10:], reward_info_list[-10:]):
                            print(cid, rew, rinfo)
                        print("")
            except queue.Empty:
                pass

            # 2) 周期性 PERF 打印


            # 3) save / eval（建议用 model_lock 暂停训练，避免并发读写模型）
            with state_lock:
                ge = shared["global_episode"]

            if (ge + 1) % (1000 * args.num_actors) == 0 and rank == 0:
                with model_lock:
                    if distributed:
                        dist.barrier()
                    parking_agent.save(f"{save_path}/SAC2_{ge}.pt", params_only=True)
                    if distributed:
                        dist.barrier()

                # eval 很慢，会暂停 trainer（因为 model_lock），但不会卡住仿真（collector仍在跑）
                with torch.no_grad():
                    with model_lock:
                        env.set_level('Complex')
                        log_path = save_path + '/complex'
                        os.makedirs(log_path, exist_ok=True)
                        success_ratio, reward_avg = eval(
                            env, parking_agent, episode=args.eval_episode, log_path=log_path, post_proc_action=True
                        )
                        if success_ratio >= best_success_ratio and reward_avg >= best_reward_averge:
                            parking_agent.save(f"{save_path}/SAC_best.pt", params_only=True)
                            with open(save_path + 'best.txt', 'w') as f_best_log:
                                f_best_log.write(f'epoch: {ge+1}, success rate: {success_ratio}, reward_avg: {reward_avg} ')
                            best_success_ratio = success_ratio
                            best_reward_averge = reward_avg

        # ========== 收尾 ==========
        stop_event.set()

        # 先停 actors
        for q in weight_queues:
            put_latest(q, None)
        for p in procs:
            p.join()

        # 再停线程
        t_col.join()
        t_push.join()
        t_trn.join()
        burn.stop()
