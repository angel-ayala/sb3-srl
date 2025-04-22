#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:48:04 2025

@author: angel
"""
from typing import Any, Callable, Optional, List, SupportsFloat, Union
from gymnasium.core import ActType, ObsType
import time
import json
import sys
import numpy as np
import gymnasium as gym
from pathlib import Path
from natsort import natsorted

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_latest_run_id

from webots_drone.data import StoreStepData
from webots_drone.envs.preprocessor import MultiModalObservation
from webots_drone.envs.preprocessor import CustomVectorObservation
from webots_drone.envs.preprocessor import UAV_DATA
from webots_drone.stack import ObservationStack


def list_of_float(arg):
    return list(map(float, arg.split(',')))


def list_of_int(arg):
    return list(map(int, arg.split(',')))


def list_of_targets(arg):
    if 'random' in arg or 'sample' in arg:
        return arg
    return list_of_int(arg)


def uav_data_list(arg):
    global UAV_DATA
    sel_data = list()
    for d in arg.lower().split(','):
        if d in UAV_DATA:
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
    arg_env.add_argument("--target-pos", type=list_of_targets, default=None,
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
    arg_env.add_argument("--uav-data", type=uav_data_list, default=UAV_DATA,
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
                         help='Encoder tau polyak update.')
    arg_srl.add_argument("--encoder-steps", type=int, default=9000,
                         help='Steps of no improvement to stop Encoder gradient.')
    arg_srl.add_argument("--decoder-lr", type=float, default=1e-3,
                         help='Decoder function Adam learning rate.')
    arg_srl.add_argument("--decoder-latent-lambda", type=float, default=1e-6,
                         help='Decoder regularization lambda value.')
    arg_srl.add_argument("--decoder-weight-decay", type=float, default=1e-7,
                         help='Decoder function Adam weight decay value.')
    arg_srl.add_argument("--representation-freq", type=int, default=1,
                         help='Steps interval for AE batch training.')
    arg_srl.add_argument("--encoder-only", action='store_true',
                         help='Whether if use the SRL loss.')
    arg_srl.add_argument("--model-reconstruction", action='store_true',
                         help='Whether if use the Reconstruction model.')
    arg_srl.add_argument("--model-spr", action='store_true',
                         help='Whether if use the SelfPredictive model.')
    arg_srl.add_argument("--model-reconstruction-dist", action='store_true',
                         help='Whether if use the ReconstructionDist reconstruction model.')
    arg_srl.add_argument("--model-ispr", action='store_true',
                         help='Whether if use the InfoNCE SimpleSPR version model.')
    arg_srl.add_argument("--model-i2spr", action='store_true',
                         help='Whether if use the Introspective InfoNCE SimpleSPR model.')
    arg_srl.add_argument("--introspection-lambda", type=float, default=0,
                         help='Introspection loss function \lambda value, >0 to use introspection.')
    arg_srl.add_argument("--joint-optimization", action='store_true',
                         help='Whether if jointly optimize representation with RL updates.')
    arg_srl.add_argument("--model-ispr-mumo", action='store_true',
                         help='Whether if use the InfoNCE SimpleSPR Multimodal version model.')
    arg_srl.add_argument("--model-proprio", action='store_true',
                         help='Whether if use the Proprioceptive version model.')
    # arg_srl.add_argument("--model-vector-difference", action='store_true',
    #                      help='Whether if use the VectorDifference reconstruction model.')
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


def args2env_params(args):
    _args = args
    if not isinstance(_args, dict):
        _args = vars(_args)
    env_params = {
        'time_limit_seconds': _args.get('time_limit', 60),
        'max_no_action_seconds': _args.get('time_no_action', 5),
        'frame_skip': _args.get('frame_skip', 6),
        'goal_threshold': _args.get('goal_threshold', 0.25),
        'init_altitude': _args.get('init_altitude', 0.3),
        'altitude_limits': _args.get('altitude_limits', [0.25, 2.]),
        'target_pos': _args.get('target_pos', None),
        'target_dim': _args.get('target_dim', [.05, .02]),
        'is_pixels': _args.get('is_pixels', False),
        'is_vector': _args.get('is_vector', False),
        'frame_stack': _args.get('frame_stack', 1),
        'target_pos2obs': _args.get('add_target_pos', False),
        'target_dist2obs': _args.get('add_target_dist', False),
        'target_dim2obs': _args.get('add_target_dim', False),
        'action2obs': _args.get('add_action', False),
        'uav_data': _args.get('uav_data', UAV_DATA),
        'norm_obs': True
        }
    env_params['is_multimodal'] = env_params['is_pixels'] and env_params['is_vector']
    return env_params


def instance_env(name='webots_drone:webots_drone/DroneEnvDiscrete-v0',
                 env_params={}, seed=666):
    _env_params = {
        'time_limit_seconds': env_params.get('time_limit', 60),
        'max_no_action_seconds': env_params.get('time_no_action', 5),
        'frame_skip': env_params.get('frame_skip', 6),
        'goal_threshold': env_params.get('goal_threshold', 0.25),
        'init_altitude': env_params.get('init_altitude', 0.3),
        'altitude_limits': env_params.get('altitude_limits', [0.25, 2.]),
        'target_pos': env_params.get('target_pos', None),
        'target_dim': env_params.get('target_dim', [.05, .02]),
        'is_pixels': env_params.get('is_pixels', False),
        }

    if isinstance(_env_params['target_pos'], list):
        if len(_env_params['target_pos']) > 1:
            print('WARNING: Multiple target positions were defined, taking the first one during training.')
        _env_params['target_pos'] = env_params['target_pos'][0]
    # Create the environment
    env = gym.make(name, **_env_params)
    env.seed(seed)

    env_params['state_shape'] = env.observation_space.shape
    if len(env.action_space.shape) == 0:
        env_params['action_shape'] = (env.action_space.n, )
    else:
        env_params['action_shape'] = env.action_space.shape

    # wrap observations
    env_range = {
        'angles_range': [np.pi/2, np.pi/2, np.pi],
        'avel_range': [np.pi, np.pi, 2*np.pi],
        'speed_range': [0.5, 0.5, 0.5]
        }
    if env_params['is_multimodal']:
        env = MultiModalObservation(env, uav_data=env_params['uav_data'],
                                    frame_stack=env_params['frame_stack'],
                                    target_pos=env_params['target_pos2obs'],
                                    target_dim=env_params['target_dim2obs'],
                                    target_dist=env_params['target_dist2obs'],
                                    add_action=env_params['action2obs'],
                                    norm_obs=env_params['norm_obs'],
                                    **env_range)
        env_params['state_shape'] = (env.observation_space['vector'].shape,
                                     env.observation_space['pixel'].shape)

    else:
        if env_params['is_vector']:
            env = CustomVectorObservation(env, uav_data=env_params['uav_data'],
                                          target_dist=env_params['target_dist2obs'],
                                          target_pos=env_params['target_pos2obs'],
                                          target_dim=env_params['target_dim2obs'],
                                          add_action=env_params['action2obs'],
                                          norm_obs=env_params['norm_obs'],
                                          **env_range)

        if env_params['frame_stack'] > 1:
            env = ObservationStack(env, k=env_params['frame_stack'])

        env_params['state_shape'] = env.observation_space.shape

    return env


def args2ae_config(args, env_params):
    _args = args
    if not isinstance(_args, dict):
        _args = vars(_args)
    model_params = {
        'action_shape': env_params['action_shape'],
        'state_shape': env_params['state_shape'],
        'latent_dim': _args.get('latent_dim', 32),
        'layers_dim': [_args.get('hidden_dim', 256)] * _args.get('num_layers', 2),
        'layers_filter': [_args.get('num_filters', 32)] * _args.get('num_layers', 2),
        'encoder_lr': _args.get('encoder_lr', 1e-3),
        'decoder_lr': _args.get('decoder_lr', 1e-3),
        'encoder_only': _args.get('encoder_only', False),
        'encoder_steps': _args.get('encoder_steps', 9000),
        'decoder_lambda': _args.get('decoder_lambda', 1e-6),
        'decoder_weight_decay': _args.get('decoder_weight_decay', 1e-7),
        'joint_optimization': _args.get('joint_optimization', False),
        'introspection_lambda': _args.get('introspection_lambda', 0.),
        'is_pixels': _args.get('is_pixels', False),
        'is_multimodal': _args.get('is_pixels', False) and _args.get('is_vector', False),
        }

    if _args.get('model_reconstruction', False):
        model_name = 'Reconstruction'
    elif _args.get('model_spr', False):
        model_name = 'SelfPredictive'
    elif _args.get('model_reconstruction_dist', False):
        model_name = 'ReconstructionDist'
    elif _args.get('model_ispr', False):
        model_name = 'InfoSPR'
    elif _args.get('model_i2spr', False):
        model_name = 'IntrospectiveInfoSPR'
    elif _args.get('model_proprio', False):
        model_name = 'Proprioceptive'
    else:
        raise ValueError('SRL model not recognized...')

    return model_name, model_params


def args2logpath(args, algo):
    if args.logspath is None:
        if args.is_pixels and args.is_vector:
            path_prefix = 'multi'
        else:
            path_prefix = 'pixel' if args.is_pixels else 'vector'
        # Summary folder
        outfolder = f"logs_cf_{path_prefix}"
    else:
        outfolder = args.logspath

    path_suffix = ''
    # method labels
    if args.model_reconstruction:
        path_suffix += '-rec'
    if args.model_spr:
        path_suffix += '-spr'
    if args.model_reconstruction_dist:
        path_suffix += '-drec'
    if args.model_ispr:
        path_suffix += '-ispr'
    if args.model_i2spr:
        path_suffix += '-i2spr'
    if args.model_ispr_mumo:
        path_suffix += '-ispr-custom'
    if args.model_proprio:
        path_suffix += '-proprio'
    # extra labels
    if args.introspection_lambda != 0.:
        path_suffix += '-intr'
    if args.joint_optimization:
        path_suffix += '-joint'
    # if args.model_vector_difference:
    #     path_suffix += '-diff'
    exp_name = f"{algo}{path_suffix}"

    latest_run_id = get_latest_run_id(outfolder, exp_name)

    return outfolder, exp_name, latest_run_id


def args2target(env, arg_tpos):
    target_pos = arg_tpos
    if arg_tpos is None:
        target_pos = list(range(len(env.quadrants)))
    elif 'sample' in arg_tpos:
        target_pos = np.random.choice(range(len(env.quadrants)),
                                      int(target_pos.replace('sample-', '')),
                                      replace=False)
    elif arg_tpos == 'random':
        target_pos = [env.vtarget.get_random_position(env.flight_area)]
    elif 'random-' in arg_tpos:
        n_points = int(target_pos.replace('random-', ''))
        target_pos = [env.vtarget.get_random_position(env.flight_area)
                      for _ in range(n_points)]
    return target_pos


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


class DroneExperimentCallback(CheckpointCallback):
    """
    A callback to save the arguments after creating the log output folder.

    :param exp_args: The Parser object to be save.
    :param out_path: The json file path to be writen.
    :param memory_steps: The number of steps to initialize the memory.
    :param data_store: The StepDataStore object.
    """

    def __init__(self, *args,
                 env: gym.Env,
                 exp_args: dict,
                 out_path: str,
                 memory_steps: int,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.exp_args = vars(exp_args)
        self.out_path = out_path
        self.env = env
        # apply a offset to ensure saving agents after save_freq without memory fill
        self.n_calls = -memory_steps
        self.save_freq_tmp = self.save_freq
        self.save_freq = float('inf')

    def _on_training_start(self) -> None:
        save_dict_json(self.exp_args, self.out_path)
        self.env.init_store()

    def _on_step(self) -> bool:
        if self.n_calls == 0:
            self.env.set_learning()
            self.save_freq = self.save_freq_tmp
        if self.n_calls % self.save_freq == 0:
            self.env.new_episode()
            self.training_env.reset()
        return super()._on_step()


class DroneEnvMonitor(Monitor):
    def __init__(self, *args,
                 store_path: Union[Path, str],
                 n_sensors: int = 0,
                 extra_info: bool = True,
                 epsilon_value: Optional[Callable] = None,
                 other_cols: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._data_store = StoreStepData(
            store_path, n_sensors, epsilon_value, extra_info, other_cols)

    def reset(self, **kwargs) -> tuple[ObsType, dict[str, Any]]:
        obs = super().reset(**kwargs)
        self._data_store.set_init_state(None, obs[1])
        return obs

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action)
        sample = (None, action, reward, None, terminated, truncated)
        self._data_store(sample, info)
        return observation, reward, terminated, truncated, info

    def export_env_data(self, outpath: Optional[Union[str, Path]] = None) -> None:
        env_data = {}
        env_data['target_quadrants'] = str(self.env.unwrapped.quadrants.tolist())
        env_data['flight_area'] = str(self.env.unwrapped.flight_area.tolist())
        if outpath is None:
            json_path = self._data_store.store_path.parent / 'environment.json'
        else:
            json_path = outpath
        save_dict_json(env_data, json_path)

    def new_episode(self, episode: int = -1) -> None:
        self._data_store.new_episode(episode)

    def set_eval(self) -> None:
        self._data_store.set_eval()

    def set_learning(self) -> None:
        self._data_store.set_learning()

    def init_store(self) -> None:
        self._data_store.init_store()
        self.export_env_data()


def evaluate_agent(agent_select_action: Callable,
                   env: gym.Env,
                   n_episodes: int,
                   n_steps: int,
                   target_quadrant: int):
    steps = []
    rewards = []
    times = []

    for i in range(n_episodes):
        timemark = time.time()
        state, info = env.reset(target_pos=target_quadrant)
        ep_reward = 0
        ep_steps = 0
        end = False

        while not end:
            action = agent_select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            end = done or truncated
            ep_steps += 1
            ep_reward += reward
            state = next_state
            prefix = f"Run {i+1:02d}/{n_episodes:02d}"
            sys.stdout.write(f"\r{prefix} | Reward: {ep_reward:.4f} | "
                             f"Length: {ep_steps}  ")
            if ep_steps == n_steps or truncated:
                end = True

        elapsed_time = time.time() - timemark

        steps.append(ep_steps)
        rewards.append(ep_reward)
        times.append(elapsed_time)

    if isinstance(target_quadrant, int):
        target_str = f"{target_quadrant:02d}"
    elif isinstance(target_quadrant, np.ndarray):
        target_str = str(target_quadrant)
    else:
        target_str = 'Random'
    ttime = np.sum(times).round(3)
    tsteps = np.mean(steps)
    treward = np.mean(rewards).round(4)
    sys.stdout.write(f"\r- Evaluated in {ttime:.3f} seconds | "
                     f"Target Position {target_str} | "
                     f"Mean reward: {treward:.4f} | "
                     f"Mean lenght: {tsteps}\n")
    sys.stdout.flush()

    return ep_reward, ep_steps, elapsed_time


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
            if type(observations) is dict:
                for k in observations.keys():
                    observations[k] = np.array(observations[k], dtype=np.float32)
                    if observations[k].shape[0] != 1:
                        observations[k] = observations[k][np.newaxis, ...]
            else:
                observations = np.array(observations, dtype=np.float32)
                if observations.shape[0] != 1:
                    observations = observations[np.newaxis, ...]
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
        monitor_env.new_episode(log_ep)
        # Iterate over goal position
        for tpos in targets_pos:
            evaluate_agent(action_selection, monitor_env, args.eval_episodes,
                           args.eval_steps, tpos)
        monitor_env.close()
