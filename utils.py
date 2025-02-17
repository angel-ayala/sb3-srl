#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:48:04 2025

@author: angel
"""
import json
import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_latest_run_id

from webots_drone.envs.preprocessor import MultiModalObservation
from webots_drone.envs.preprocessor import CustomVectorObservation
from webots_drone.stack import ObservationStack


def list_of_float(arg):
    return list(map(float, arg.split(',')))


def list_of_int(arg):
    return list(map(int, arg.split(',')))


def uav_data_list(arg):
    avlbl_data = ['imu', 'gyro', 'gps', 'gps_vel', 'north', 'dist_sensors']
    sel_data = list()
    for d in arg.lower().split(','):
        if d in avlbl_data:
            sel_data.append(d)
    return sel_data


def parse_crazyflie_env_args(parser):
    arg_env = parser.add_argument_group('Environment')
    arg_env.add_argument("--time-limit", type=int, default=60,  # 1m
                         help='Max time (seconds) of the mission.')
    arg_env.add_argument("--time-no-action", type=int, default=5,
                         help='Max time (seconds) with no movement.')
    arg_env.add_argument("--frame-skip", type=int, default=6,  # 192ms
                         help='Number of simulation steps for a RL step')
    arg_env.add_argument("--frame-stack", type=int, default=1,
                         help='Number of RL step to stack as observation.')
    arg_env.add_argument("--goal-threshold", type=float, default=0.25,
                         help='Minimum distance from the target.')
    arg_env.add_argument("--init-altitude", type=float, default=1.0,
                         help='Minimum height distance to begin the mission.')
    arg_env.add_argument("--altitude-limits", type=list_of_float,
                         default=[0.25, 2.], help='Vertical flight limits.')
    arg_env.add_argument("--target-pos", type=int, default=None,
                         help='Cuadrant number for target position.')
    arg_env.add_argument("--target-dim", type=list_of_float, default=[0.05, 0.02],
                         help="Target's dimension size.")
    arg_env.add_argument("--zone-steps", type=int, default=0,
                         help='Max number on target area to end the episode with found target.')
    arg_env.add_argument("--is-pixels", action='store_true',
                         help='Whether if reconstruct an image-based observation.')
    arg_env.add_argument("--is-vector", action='store_true',
                         help='Whether if reconstruct a vector-based observation.')
    arg_env.add_argument("--add-target-pos", action='store_true',
                         help='Whether if add the target position to vector state.')
    arg_env.add_argument("--add-target-dist", action='store_true',
                         help='Whether if add the target distance to vector state.')
    arg_env.add_argument("--add-target-dim", action='store_true',
                         help='Whether if add the target dimension to vector state.')
    arg_env.add_argument("--add-action", action='store_true',
                         help='Whether if add the previous action to vector state.')
    arg_env.add_argument("--uav-data", type=uav_data_list,
                         default=['imu', 'gyro', 'gps', 'gps_vel', 'north', 'dist_sensors'],
                         help='Select the UAV sensor data as state, available'
                         ' options are: imu, gyro, gps, gps_vel, north, dist_sensors')
    return arg_env


def parse_memory_args(parser):
    arg_mem = parser.add_argument_group('Memory buffer')
    arg_mem.add_argument("--memory-capacity", type=int, default=65536,  # 2**16
                           help='Maximum number of transitions in the Experience replay buffer.')
    arg_mem.add_argument("--memory-prioritized", action='store_true',
                           help='Whether if memory buffer is Prioritized experiencie replay or not.')
    arg_mem.add_argument("--prioritized-alpha", type=float, default=0.6,
                           help='Alpha prioritization exponent for PER.')
    arg_mem.add_argument("--prioritized-initial-beta", type=float, default=0.4,
                           help='Beta bias for sampling for PER.')
    arg_mem.add_argument("--beta-steps", type=float, default=112500,
                           help='Beta bias steps to reach 1.')
    return arg_mem

def parse_srl_args(parser):
    arg_srl = parser.add_argument_group(
        'State representation learning variation')
    arg_srl.add_argument("--is-srl", action='store_true',
                         help='Whether if method is SRL-based or not.')
    arg_srl.add_argument("--latent-dim", type=int, default=32,
                         help='Number of features in the latent representation Z.')
    arg_srl.add_argument("--hidden-dim", type=int, default=512,
                         help='Number of units in the hidden layers.')
    arg_srl.add_argument("--num-filters", type=int, default=32,
                         help='Number of filters in the CNN hidden layers.')
    arg_srl.add_argument("--num-layers", type=int, default=1,
                         help='Number of hidden layers.')
    arg_srl.add_argument("--encoder-lr", type=float, default=1e-3,
                         help='Encoder function Adam learning rate.')
    arg_srl.add_argument("--encoder-tau", type=float, default=0.999,
                         help='Encoder \tau polyak update.')
    arg_srl.add_argument("--decoder-lr", type=float, default=1e-3,
                         help='Decoder function Adam learning rate.')
    arg_srl.add_argument("--decoder-latent-lambda", type=float, default=1e-6,
                         help='Decoder regularization \lambda value.')
    arg_srl.add_argument("--decoder-weight-decay", type=float, default=1e-7,
                         help='Decoder function Adam weight decay value.')
    arg_srl.add_argument("--reconstruct-frequency", type=int, default=1,
                         help='Steps interval for AE batch training.')
    arg_srl.add_argument("--encoder-only", action='store_true',
                         help='Whether if use the SRL loss.')
    arg_srl.add_argument("--model-vector", action='store_true',
                         help='Whether if use the Vector reconstruction model.')
    arg_srl.add_argument("--model-vector-spr", action='store_true',
                         help='Whether if use the VectorSPR model.')
    arg_srl.add_argument("--model-vector-target-dist", action='store_true',
                         help='Whether if use the VectorTargetDist reconstruction model.')
    arg_srl.add_argument("--model-advantage", action='store_true',
                         help='Whether if use the Advantage reconstruction model.')
    # arg_srl.add_argument("--model-vector-difference", action='store_true',
    #                      help='Whether if use the VectorDifference reconstruction model.')
    # arg_srl.add_argument("--model-rgb", action='store_true',
    #                      help='Whether if use the RGB reconstruction model.')
    return arg_srl


def parse_training_args(parser):
    arg_training = parser.add_argument_group('Training')
    arg_training.add_argument("--steps", type=int, default=450000,  # 25h at 25 frames
                              help='Number of training steps.')
    arg_training.add_argument('--memory-steps', type=int, default=2048,
                              help='Number of steps for initial population of the Experience replay buffer.')
    arg_training.add_argument("--batch-size", type=int, default=128,
                              help='Minibatch size for training.')
    arg_training.add_argument('--eval-interval', type=int, default=9000,  # 30m at 25 frames
                              help='Steps interval for progress evaluation.')
    arg_training.add_argument('--eval-steps', type=int, default=60,  # 1m at 25 frames
                              help='Number of evaluation steps.')
    return arg_training


def parse_utils_args(parser):
    arg_utils = parser.add_argument_group('Utils')
    arg_utils.add_argument('--use-cuda', action='store_true',
                           help='Flag specifying whether to use the GPU.')
    arg_utils.add_argument('--seed', type=int, default=666,
                           help='Seed valu for torch and nummpy.')
    arg_utils.add_argument('--logspath', type=str, default=None,
                           help='Specific output path for training results.')
    return arg_utils


def instance_env(args, name='webots_drone:webots_drone/DroneEnvDiscrete-v0',
                 seed=666):
    env_params = dict()
    if isinstance(args, dict):
        env_params = args.copy()
        if 'state_shape' in env_params.keys():
            del env_params['state_shape']
        if 'action_shape' in env_params.keys():
            del env_params['action_shape']
        if 'is_vector' in env_params.keys():
            del env_params['is_vector']
        if 'frame_stack' in env_params.keys():
            del env_params['frame_stack']
        if 'target_pos2obs' in env_params.keys():
            del env_params['target_pos2obs']
        if 'target_dist2obs' in env_params.keys():
            del env_params['target_dist2obs']
        if 'target_dim2obs' in env_params.keys():
            del env_params['target_dim2obs']
        if 'action2obs' in env_params.keys():
            del env_params['action2obs']
        if 'uav_data' in env_params.keys():
            del env_params['uav_data']
        if 'is_multimodal' in env_params.keys():
            del env_params['is_multimodal']
        if 'vector_shape' in env_params.keys():
            del env_params['vector_shape']
        if 'image_shape' in env_params.keys():
            del env_params['image_shape']
        if 'obs_space' in env_params.keys():
            del env_params['obs_space']
        if 'target_quadrants' in env_params.keys():
            del env_params['target_quadrants']
        if 'flight_area' in env_params.keys():
            del env_params['flight_area']
    else:
        env_params = dict(
            time_limit_seconds=args.time_limit,
            max_no_action_seconds=args.time_no_action,
            frame_skip=args.frame_skip,
            goal_threshold=args.goal_threshold,
            init_altitude=args.init_altitude,
            altitude_limits=args.altitude_limits,
            target_pos=args.target_pos,
            target_dim=args.target_dim,
            is_pixels=args.is_pixels,
            zone_steps=args.zone_steps)

    # Create the environment
    env = gym.make(name, **env_params)
    env.seed(seed)

    if not isinstance(args, dict):
        env_params['frame_stack'] = args.frame_stack
        env_params['is_multimodal'] = args.is_pixels and args.is_vector
        env_params['is_vector'] = args.is_vector
        env_params['target_dist2obs'] = args.add_target_dist
        env_params['target_pos2obs'] = args.add_target_pos
        env_params['target_dim2obs'] = args.add_target_dim
        env_params['action2obs'] = args.add_action
        env_params['uav_data'] = args.uav_data

    env_params['state_shape'] = env.observation_space.shape
    if len(env.action_space.shape) == 0:
        env_params['action_shape'] = (env.action_space.n, )
    else:
        env_params['action_shape'] = env.action_space.shape

    return env, env_params


def wrap_env(env, env_params):
    env_range = {
        'angles_range': [np.pi/6, np.pi/6, np.pi],
        'avel_range': [np.pi/3, np.pi/3, 2*np.pi],
        'speed_range': [0.8, 0.8, 0.6]
        }
    if env_params['is_multimodal']:
        env = MultiModalObservation(env, uav_data=env_params['uav_data'],
                                    frame_stack=env_params['frame_stack'],
                                    target_pos=env_params['target_pos2obs'],
                                    target_dim=env_params['target_dim2obs'],
                                    target_dist=env_params['target_dist2obs'],
                                    add_action=env_params['action2obs'],
                                    **env_range)
        env_params['image_shape'] = env.observation_space[0].shape
        env_params['vector_shape'] = env.observation_space[1].shape

    else:
        if env_params['is_vector']:
            env = CustomVectorObservation(env, uav_data=env_params['uav_data'],
                                          target_dist=env_params['target_dist2obs'],
                                          target_pos=env_params['target_pos2obs'],
                                          target_dim=env_params['target_dim2obs'],
                                          add_action=env_params['action2obs'],
                                          **env_range)

        if env_params['frame_stack'] > 1:
            env = ObservationStack(env, k=env_params['frame_stack'])

        env_params['vector_shape'] = env.observation_space.shape if env_params['is_vector'] else None
        env_params['image_shape'] = env.observation_space.shape if env_params['is_pixels'] else None
        env_params['state_shape'] = env.observation_space.shape

    env_params['obs_space'] = (env_params['image_shape'], env_params['vector_shape'])

    return env, env_params


def args2ae_config(args, env_params):
    ae_models = {}
    # image_shape = env_params['image_shape']
    model_params = {
        'latent_dim': args.latent_dim,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'encoder_lr': args.encoder_lr,
        'decoder_lr': args.decoder_lr,
        'encoder_only': args.encoder_only,
        'decoder_latent_lambda': args.decoder_latent_lambda,
        'decoder_weight_decay': args.decoder_weight_decay
        }

    if args.model_vector:
        assert env_params['is_vector'], 'Vector model requires is_vector flag.'
        ae_models['Vector'] = model_params.copy()
        ae_models['Vector'].update({'vector_shape': env_params['vector_shape']})
    if args.model_vector_spr:
        assert env_params['is_vector'], 'VectorSPR model requires is_vector flag.'
        ae_models['VectorSPR'] = model_params.copy()
        ae_models['VectorSPR'].update({
            'vector_shape': env_params['vector_shape'],
            'action_shape': env_params['action_shape'],
            })
    if args.model_vector_target_dist:
        assert env_params['is_vector'], 'Vector model requires is_vector flag.'
        ae_models['VectorTargetDist'] = model_params.copy()
        ae_models['VectorTargetDist'].update({
            'vector_shape': env_params['vector_shape']})
    if args.model_advantage:
        ae_models['Advantage'] = model_params.copy()
        ae_models['Advantage'].update({
            'vector_shape': env_params['vector_shape'],
            'action_shape': env_params['action_shape'],
            })

    return ae_models


def args2logpath(args, algo):
    if args.logspath is None:
        if args.is_pixels and args.is_vector:
            path_prefix = 'multi'
        else:
            path_prefix = 'pixels' if args.is_pixels else 'vector'
        # Summary folder
        outfolder = f"logs_cf_{path_prefix}"
    else:
        outfolder = args.logspath

    path_suffix = ''
    if args.is_srl:
        path_suffix += '-srl'
    if args.model_vector_spr:
        path_suffix += '-spr'
    if args.model_vector_target_dist:
        path_suffix += '-tdist'
    # if args.model_vector_difference:
    #     path_suffix += '-diff'
    exp_name = f"{algo}{path_suffix}"

    latest_run_id = get_latest_run_id(outfolder, exp_name)

    return outfolder, exp_name, latest_run_id

def save_dict_json(dict2save, json_path):
    proc_dic = dict2save.copy()
    dict_json = json.dumps(proc_dic,
                           indent=4,
                           default=lambda o: str(o))
    with open(json_path, 'w') as jfile:
        jfile.write(dict_json)
    return dict_json


def load_json_dict(json_path):
    json_dict = dict()
    with open(json_path, 'r') as jfile:
        json_dict = json.load(jfile)
    return json_dict


class ArgsSaveCallback(BaseCallback):
    """
    A callback to save the arguments after creating the log output folder.

    :param args: The ArgumentParser object to be save.
    :param out_path: The json file to be writen.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, args, out_path, verbose: int = 0):
        super().__init__(verbose)
        self.args = vars(args)
        self.out_path = out_path

    def _on_training_start(self) -> None:
        save_dict_json(self.args, self.out_path)

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass
