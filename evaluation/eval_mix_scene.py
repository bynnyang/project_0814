import sys
sys.path.append("..")
sys.path.append(".")
import time
import os
import argparse

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# from model.MultiModalPPO_AF import PPO
from model_interface.model.agent.ppo_agent import PPOAgent as PPO
from model_interface.model.agent.sac_agent import SACAgent as SAC
from model_interface.model.agent.parking_agent import ParkingAgent, RsPlanner
from env.car_parking_base import CarParking
from env.env_wrapper import CarParkingWrapper
from env.vehicle import VALID_SPEED
from evaluation.eval_utils import eval
from vehicle_config import *
from utils.config import get_train_config_obj


if __name__=="__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='./rl_model/SAC2_1999.pt') # './model/ckpt/HOPE_SAC0.pt'
    parser.add_argument('--eval_episode', type=int, default=2000)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--visualize', type=bool, default=True)
    parser.add_argument('--config', default='./config/training_real.yaml', type=str)
    args = parser.parse_args()
    config_path = args.config
    config_obj = get_train_config_obj(config_path)

    checkpoint_path = args.ckpt_path
    print('ckpt path: ',checkpoint_path)
    verbose = args.verbose

    if args.visualize:
        raw_env = CarParking(fps=100, verbose=verbose)
    else:
        raw_env = CarParking(fps=100, verbose=verbose, render_mode='rgb_array')
    env = CarParkingWrapper(raw_env)

    relative_path = '.'
    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    save_path = relative_path+'/log/eval/%s/' % timestamp
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    configs_file = os.path.join(save_path, 'configs.txt')
    with open(configs_file, 'w') as f:
        f.write(str(checkpoint_path))
    Agent_type = PPO if 'ppo' in checkpoint_path.lower() else SAC
    writer = SummaryWriter(save_path)
    print("You can track the training process by command 'tensorboard --log-dir %s'" % save_path)

    seed = SEED
    # env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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

    rl_agent = Agent_type(config_obj, configs)
    if checkpoint_path is not None:
        rl_agent.load(checkpoint_path, params_only=True)
        print('load pre-trained model!')

    step_ratio = env.vehicle.kinetic_model.step_len*env.vehicle.kinetic_model.n_step*1.0
    rs_planner = RsPlanner(step_ratio)
    parking_agent = ParkingAgent(rl_agent, rs_planner)

    eval_episode = args.eval_episode
    choose_action = True if isinstance(rl_agent, PPO) else False
    choose_action = True
    with torch.no_grad():
        # eval on extreme
        # env.set_level('Extrem')
        # log_path = save_path+'/extreme'
        # if not os.path.exists(log_path):
        #     os.makedirs(log_path)
        # eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)

        # # eval on dlp
        # env.set_level('dlp')
        # log_path = save_path+'/dlp'
        # if not os.path.exists(log_path):
        #     os.makedirs(log_path)
        # eval(env, parking_agent, episode=eval_episode, log_path=log_path, multi_level=True, post_proc_action=choose_action)
        
        # eval on complex
        env.set_level('Complex')
        log_path = save_path+'/complex'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)
        
        # # eval on normalize
        # env.set_level('Normal')
        # log_path = save_path+'/normalize'
        # if not os.path.exists(log_path):
        #     os.makedirs(log_path)
        # eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)

    env.close()