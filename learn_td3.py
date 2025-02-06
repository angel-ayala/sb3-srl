#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:45:59 2025

@author: angel
"""
import argparse
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise

from sb3_srl.td3 import SRLTD3Policy, SRLTD3

from utils import parse_crazyflie_env_args
from utils import parse_memory_args
from utils import parse_srl_args
from utils import parse_training_args
from utils import parse_utils_args
from utils import instance_env
from utils import wrap_env
from utils import args2ae_config
from utils import args2logpath
from utils import ArgsSaveCallback


def parse_agent_args(parser):
    arg_agent = parser.add_argument_group('Agent')
    arg_agent.add_argument("--lr", type=float, default=1e-3,
                           help='Critic function Adam learning rate.')
    arg_agent.add_argument("--tau", type=float, default=0.005,
                           help='Soft target update \tau.')
    arg_agent.add_argument("--exploration-noise", type=float, default=0.1,
                           help='Action noise during learning.')
    arg_agent.add_argument("--discount-factor", type=float, default=0.99,
                           help='Discount factor \gamma.')
    arg_agent.add_argument("--policy-freq", type=int, default=2,
                           help='Steps interval for actor batch training.')
    arg_agent.add_argument("--policy-noise", type=list, default=[0.2, 0.2, 0.1, 0.2],
                           help='Policy noise value.')
    arg_agent.add_argument("--noise-clip", type=list, default=[0.5, 0.5, 0.3, 0.5],
                           help='Policy noise clip value.')
    return arg_agent


def parse_args():
    # Argument parser
    parser = argparse.ArgumentParser()

    parse_crazyflie_env_args(parser)
    parse_agent_args(parser)
    parse_memory_args(parser)
    parse_srl_args(parser)
    parse_training_args(parser)
    parse_utils_args(parser)

    return parser.parse_args()


args = parse_args()
# th.manual_seed(args.seed)
# np.random.seed(args.seed)

# Environment
environment_name = 'webots_drone:webots_drone/CrazyflieEnvContinuous-v0'
env, env_params = instance_env(args, environment_name, seed=args.seed)
env, env_params = wrap_env(env, env_params)  # observation preprocesing

if args.is_srl:
    algo, policy = SRLTD3, SRLTD3Policy
    # Autoencoder parameters
    ae_config = list(args2ae_config(args, env_params).items())
    if len(ae_config) == 0:
        raise ValueError("No SRL model selected")
    ae_type, ae_params = ae_config[0]

    # Policy args
    policy_args = {
        'ae_type': ae_type,
        'ae_params': ae_params,
        }
else:
    algo, policy = TD3, TD3Policy
    policy_args = None

# Save a checkpoint every N steps
log_path, exp_name, run_id = args2logpath(args, 'td3')
agents_path = f"./{log_path}/{exp_name}_{run_id+1}/agents"

checkpoint_callback = CheckpointCallback(
  save_freq=args.eval_interval,
  save_path=agents_path,
  name_prefix="rl_model",
  save_replay_buffer=False,
  save_vecnormalize=False,
)

# Save experiment arguments
args_callback = ArgsSaveCallback(
    args, f"./{log_path}/{exp_name}_{run_id+1}/arguments.json")

# Create action noise because TD3 and DDPG use a deterministic policy
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=args.exploration_noise * np.ones(n_actions))

# Create RL model
model = algo(policy, env,
             learning_rate=args.lr,
             buffer_size=args.memory_capacity,  # 1e6
             learning_starts=args.memory_steps,
             batch_size=args.batch_size,
             tau=args.tau,
             gamma=args.discount_factor,
             train_freq=1,
             gradient_steps=1,
             action_noise=action_noise,
             policy_delay=args.policy_freq,
             target_policy_noise=args.policy_noise,
             target_noise_clip=args.noise_clip,
             stats_window_size=100,
             tensorboard_log=log_path,
             policy_kwargs=policy_args,
             verbose=0,
             seed=args.seed,
             device="auto",
             _init_setup_model=True)

# Train the agent
model.learn(total_timesteps=args.steps,
            callback=[checkpoint_callback, args_callback],
            log_interval=4,
            progress_bar=True,
            tb_log_name=exp_name)
model.save(f"./{agents_path}/rl_model_final")
