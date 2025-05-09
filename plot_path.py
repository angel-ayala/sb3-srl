#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 20:38:44 2023

@author: Angel Ayala
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from webots_drone.data import ExperimentData
from sb3_srl.plot_utils import plot_trajectory
from sb3_srl.plot_utils import draw_trajectory
from sb3_srl.plot_utils import plot_metric
from sb3_srl.plot_utils import draw_metric


def create_output_path(exp_folder, out_name, exist_ok=False):
    out_path = exp_folder / out_name
    out_path.mkdir(exist_ok=exist_ok)
    return out_path


def smooth(scalars: list[float], weight: float) -> list[float]:
    """
    EMA implementation according to
    https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699
    """
    last = 0
    num_acc = 0
    smoothed = np.zeros_like(scalars)
    for i, next_val in enumerate(scalars):
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - np.power(weight, num_acc)
        smoothed_val = last / debias_weight
        smoothed[i] = smoothed_val

    return smoothed


# %% Read ALM data
if __name__ == '__main__':
    base_path = Path('/home/angel/desarrollo/sb3-srl/logs_cf_vector_alm')
    exp_list = []
    exp_paths = [
        'alm-ours-l2-ema-v-10.0',
        'alm-ours-l2-ema-v-10.0_2',
        'alm-ours-l2-ema-v-10.0_3'
        ]

    for epath in exp_paths:
        exp_data = ExperimentData(base_path / epath, exp_args='flags.yml', eval_regex=r"eval/history_*.csv")
        # set constant
        exp_data.exp_args['is_srl'] = True
        # correct episode offset and filter extra data
        phase_df = exp_data.history_df[exp_data.history_df['phase'] == 'eval']
        phase_df.loc[:, 'ep'] += 1
        ep_df = phase_df[phase_df['ep'] != 51]
        ep_df = ep_df[ep_df['ep'] != 5]
        exp_data.history_df = ep_df
        exp_list.append(exp_data)

    out_path = base_path / 'assets_sim' / 'plot_trajectories'
    if out_path is not None and not out_path.exists():
        out_path.mkdir(parents=True)

    # %% Read proposal data
    base_path = Path('/home/angel/desarrollo/sb3-srl/logs_cf_vector_paper')
    exp_list = []
    exp_paths = [
        'sac_4',
        'sac-ispr-joint_1',
        'td3_5',
        'td3-ispr-joint_5',
        ]

    for epath in exp_paths:
        exp_data = ExperimentData(base_path / epath, eval_regex=r"eval/history_*.csv")
        exp_list.append(exp_data)

    out_path = base_path / 'assets' / 'plot_trajectories'
    if out_path is not None and not out_path.exists():
        out_path.mkdir(parents=True)
    # %% Read proposal pixel data
    base_path = Path('/home/angel/desarrollo/sb3-srl/logs_cf_pixel_paper')
    exp_list = []
    exp_paths = [
        'td3_1',
        'spr_1',
        'td3-spr-joint_1',
        'td3-ispr-joint_1',
        ]

    for epath in exp_paths:
        exp_data = ExperimentData(base_path / epath, eval_regex=r"eval/history_*.csv")
        exp_list.append(exp_data)

    out_path = base_path / 'assets' / 'plot_trajectories'
    if out_path is not None and not out_path.exists():
        out_path.mkdir(parents=True)
    # %% Plot trajectories
    phase = 'eval'
    episode = 49
    iteration = 160

    # Create figure and axis
    for _data in exp_list:
        fig = plt.figure(layout='constrained', figsize=(9, 5))
        # fig.suptitle(f"{_data.alg_name} trajectory in episode {episode + 1}", fontsize='xx-large')
        axes = plot_trajectory(fig, _data, phase, episode, iteration)
        if out_path is not None:
            fig_name = f"{_data.alg_name}_{phase}_ep_{episode+1:03d}_iter_{iteration+1:03d}.pdf"
            fig.savefig(out_path / fig_name)
        fig.show()

    # %% Plot navigation metrics
    # filter data
    phase = 'eval'

    plots = [  # metric_id, y_label, plt_title, is_percent
        ('SR', 'SR', 'Success rate comparison', True),
        ('SPL', 'SPL', 'Success path length comparison', True),
        ('SSPL', 'SSPL', 'Soft success path length comparison', True),
        ('DTS', 'DTS (meters)', 'Distance to success comparison', False)
    ]

    # variables
    metric_id = plots[0]
    metric_key = metric_id[0]
    alg_metrics = {}
    alg_episodes = {}
    for exp in exp_list:
        nav_metrics = exp.get_nav_metrics(phase)
        episodes = exp.get_phase_eps(phase)
        metric_values = np.asarray(nav_metrics[metric_key])
        alg_metrics[exp.alg_name] = metric_values
        alg_episodes[exp.alg_name] = episodes

    # fig, ax = plot_nav_metrics(episodes, metric_values, exp_data.alg_name, *metric_id[1:])
    with plot_metric(title=metric_id[2], label=metric_id[1], is_percent=metric_id[-1], layout='constrained', figsize=(6, 5)) as fig:
        ax = fig.add_subplot(1, 1, 1)
        for label, values in alg_metrics.items():
            eps = np.asarray(alg_episodes[label]) + 1
            draw_metric(ax, label, eps, values, metric_id[-1])

    fig.show()
