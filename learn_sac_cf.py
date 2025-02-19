#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:45:59 2025

@author: angel
"""
import argparse

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.callbacks import CheckpointCallback

from sb3_srl.sac_srl import SRLSACPolicy, SRLSAC

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
    arg_agent.add_argument("--lr", type=float, default=3e-4,
                           help='Critic function Adam learning rate.')
    arg_agent.add_argument("--tau", type=float, default=0.005,
                           help='Soft target update \tau.')
    arg_agent.add_argument("--discount-factor", type=float, default=0.99,
                           help='Discount factor \gamma.')
    arg_agent.add_argument("--train-freq", type=int, default=1,
                           help='Steps interval for actor batch training.')
    arg_agent.add_argument("--target-update-freq", type=int, default=1,
                           help='Steps interval for target network update.')
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
    algo, policy = SRLSAC, SRLSACPolicy
    # Autoencoder parameters
    ae_config = list(args2ae_config(args, env_params).items())
    if len(ae_config) == 0:
        raise ValueError("No SRL model selected")
    ae_type, ae_params = ae_config[0]

    # Policy args
    policy_args = {
        'ae_type': ae_type,
        'ae_params': ae_params,
        'encoder_tau': args.encoder_tau,
        }
else:
    algo, policy = SAC, SACPolicy
    policy_args = None

# Save a checkpoint every N steps
log_path, exp_name, run_id = args2logpath(args, 'sac')
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

# Create RL model
model = algo(policy, env,
             learning_rate=args.lr,
             buffer_size=args.memory_capacity,  # 1e6
             learning_starts=args.memory_steps,
             batch_size=args.batch_size,
             tau=args.tau,
             gamma=args.discount_factor,
             train_freq=args.train_freq,
             gradient_steps=1,
             action_noise=None,
             ent_coef="auto",
             target_update_interval=args.target_update_freq,
             target_entropy="auto",
             use_sde=False,
             sde_sample_freq=-1,
             use_sde_at_warmup=False,
             stats_window_size=100,
             tensorboard_log=log_path,
             policy_kwargs=policy_args,
             verbose=0,
             seed=args.seed,
             device="auto",
             _init_setup_model=True)

# Train the agent
model.learn(total_timesteps=(args.steps + args.memory_steps),
            callback=[checkpoint_callback, args_callback],
            log_interval=4,
            progress_bar=True,
            tb_log_name=exp_name)
model.save(f"./{agents_path}/rl_model_final")
