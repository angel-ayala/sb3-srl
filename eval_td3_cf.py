#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:45:59 2025

@author: angel
"""
import argparse
from pathlib import Path
from natsort import natsorted

from stable_baselines3 import TD3
from stable_baselines3.td3.policies import TD3Policy

from sb3_srl.td3_srl import SRLTD3Policy, SRLTD3

from utils import (
    DroneEnvMonitor,
    args2ae_config,
    args2env_params,
    args2target,
    evaluate_agent,
    instance_env,
    load_json_dict,
    parse_crazyflie_env_args,
    parse_memory_args,
    parse_srl_args,
    parse_utils_args
)


def parse_eval_args(parser):
    arg_eval = parser.add_argument_group('Evaluation')
    arg_eval.add_argument('--episode', type=int, default=-1,
                          help='Indicate the episode number to execute, set -1 for all of them')
    arg_eval.add_argument('--eval-steps', type=int, default=60,
                          help='Number of evaluation steps.')
    arg_eval.add_argument('--eval-episodes', type=int, default=10,
                          help='Number of evaluation episodes.')
    # arg_eval.add_argument('--record', action='store_true',
    #                       help='Specific if record or not a video simulation.')
    return arg_eval


def parse_args():
    # Argument parser
    parser = argparse.ArgumentParser()
    parse_crazyflie_env_args(parser)
    parse_memory_args(parser)
    parse_srl_args(parser)
    parse_eval_args(parser)
    parse_utils_args(parser)
    return parser.parse_args()


def iterate_agents_evaluation(env, algorithm, args):
    logs_path = Path(args.logspath)
    agent_models = natsorted(logs_path.glob('agents/rl_model_*'), key=str)

    for log_ep, agent_path in enumerate(agent_models):
        if args.episode > -1 and log_ep != args.episode:
            continue
        # custom agent episodes selection
        elif args.episode == -1 and log_ep not in [5, 10, 20, 35, 50]:
            continue

        print('Loading', agent_path)
        model = algorithm.load(agent_path)
        def action_selection(observations):
            actions, states = model.predict(
                observations,  # type: ignore[arg-type]
                state=None,
                episode_start=None,
                deterministic=True,
            )
            return actions[0]

        # Target position for evaluation
        targets_pos = args2target(env, args.target_pos)
        # Log eval data
        ep_name = agent_path.stem.replace('rl_model_', '')
        csv_path = logs_path / 'eval' / f"history_{ep_name}.csv"
        csv_path.parent.mkdir(exist_ok=True)
        monitor_env = DroneEnvMonitor(env, store_path=csv_path, n_sensors=4,
                                      reset_keywords=['target_pos'])
        monitor_env.init_store()
        monitor_env.set_eval()
        # Iterate over goal position
        for tpos in targets_pos:
            monitor_env.set_episode(log_ep)
            evaluate_agent(action_selection, monitor_env, args.eval_episodes,
                           args.eval_steps, tpos)
        monitor_env.close()


if __name__ == '__main__':
    args = parse_args()
    env_params = load_json_dict(args.logspath + '/arguments.json')
    env_params = args2env_params(env_params)

    # Environment
    environment_name = 'webots_drone:webots_drone/CrazyflieEnvContinuous-v0'
    env = instance_env(environment_name, env_params, seed=args.seed)

    # Algorithm
    if args.is_srl:
        algo, policy = SRLTD3, SRLTD3Policy
        # Autoencoder parameters
        ae_config = list(args2ae_config(args, env_params).items())
        if len(ae_config) == 0:
            raise ValueError("No SRL model selected")
        ae_type, ae_params = ae_config[0]

        # Policy args
        policy_args = {
            'net_arch': [256, 256],
            'ae_type': ae_type,
            'ae_params': ae_params,
            }
    else:
        algo, policy = TD3, TD3Policy
        policy_args = None

    # Evaluation loop
    iterate_agents_evaluation(env, algo, args)
