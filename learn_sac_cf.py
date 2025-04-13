#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:45:59 2025

@author: angel
"""
import argparse

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy, CnnPolicy, MultiInputPolicy

from sb3_srl.sac_srl import SRLSACPolicy, SRLSAC

from utils import (
    DroneEnvMonitor,
    DroneExperimentCallback,
    args2ae_config,
    args2env_params,
    args2logpath,
    instance_env,
    parse_crazyflie_env_args,
    parse_memory_args,
    parse_srl_args,
    parse_training_args,
    parse_utils_args
)


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


if __name__ == '__main__':
    args = parse_args()
    # th.manual_seed(args.seed)
    # np.random.seed(args.seed)
    env_params = args2env_params(args)

    # Environment
    environment_name = 'webots_drone:webots_drone/CrazyflieEnvContinuous-v0'
    env = instance_env(environment_name, env_params, seed=args.seed)

    if args.is_srl:
        algo, policy = SRLSAC, SRLSACPolicy
        # Autoencoder parameters
        ae_config = args2ae_config(args, env_params)
        # Policy args
        policy_args = {
            'ae_config': ae_config,
            'encoder_tau': args.encoder_tau
            }
    else:
        algo, policy = SAC, SACPolicy
        policy_args = None
        if args.is_pixels:
            policy = CnnPolicy
            if args.is_vector:
                policy = MultiInputPolicy

    # Output log path
    log_path, exp_name, run_id = args2logpath(args, 'sac')
    outpath = f"{log_path}/{exp_name}_{run_id+1}"

    # Experiment data log
    env = DroneEnvMonitor(env, store_path=f"{outpath}/history_training.csv", n_sensors=4)

    # Save a checkpoint every N steps
    agents_path = f"{outpath}/agents"
    experiment_callback = DroneExperimentCallback(
      env=env,
      save_freq=args.eval_interval,
      save_path=agents_path,
      name_prefix="rl_model",
      save_replay_buffer=False,
      save_vecnormalize=False,
      exp_args=args,
      out_path=f"{outpath}/arguments.json",
      memory_steps=args.memory_steps
    )

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
                callback=experiment_callback,
                log_interval=4,
                progress_bar=True,
                tb_log_name=exp_name)
    model.save(f"{agents_path}/rl_model_final")
