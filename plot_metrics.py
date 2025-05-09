#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 20:38:44 2023

@author: Angel Ayala
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import threading

from sb3_srl.plot_utils import draw_metric
from sb3_srl.plot_utils import plot_metric
from sb3_srl.plot_utils import plot_aggregated_metrics
from sb3_srl.plot_utils import plot_performance_profile
from sb3_srl.plot_utils import plot_probability_improvement
from sb3_srl.plot_utils import plot_sample_efficiency
from sb3_srl.plot_utils import set_color_palette
from webots_drone.data import ExperimentData


def append_list2dict(dict_elm, key, value):
    if key not in dict_elm.keys():
        dict_elm[key] = []
    dict_elm[key].append(value)


def append_info(exp_data, phase):
    global exp_summ
    exp_info = exp_data.get_info(phases=[phase])
    if exp_summ is None:
        exp_summ = pd.DataFrame([exp_info])
    else:
        exp_summ = pd.concat((exp_summ, pd.DataFrame([exp_info])))


def append_reward(exp_data, phase):
    global rewards
    append_list2dict(rewards, exp_data.alg_name, exp_data.get_reward_curve(phase)[:, 0])


def append_reward_quadrants(exp_data, phase):
    global rewards_tpos
    append_list2dict(rewards_tpos, exp_data.alg_name, exp_data.get_reward_curve(phase, by_quadrant=True)[:, :, 0])


def append_episodes(exp_data, phase):
    global episodes
    append_list2dict(episodes, exp_data.alg_name, exp_data.get_phase_eps(phase))


def append_metrics(exp_data, phase):
    global nav_metrics
    append_list2dict(nav_metrics, exp_data.alg_name, pd.DataFrame(exp_data.get_nav_metrics(phase)))


def process_exp_data(exp_data, phase, pbar):
    # create threads
    alive_threads = 0
    threads = []

    # try:
    threads.append(
        threading.Thread(target=append_info, args=(exp_data, phase)))
    threads.append(
        threading.Thread(target=append_reward, args=(exp_data, phase)))
    threads.append(
        threading.Thread(target=append_episodes, args=(exp_data, phase)))
    threads.append(
        threading.Thread(target=append_metrics, args=(exp_data, phase)))
    threads.append(
        threading.Thread(target=append_reward_quadrants, args=(exp_data, phase)))

    # init threads
    for thrd in threads:
        thrd.start()
        alive_threads += 1

    # monitor end and update
    while alive_threads > 0:
        for thrd in threads:
            if not thrd.is_alive():
                pbar.update()
                alive_threads -= 1
                threads.remove(thrd)
        time.sleep(0.5)

    # except AssertionError:
    #     print(exp_path, "does not have", phase, "data")

    # finally:
    return exp_data


def filter_metrics(metrics, key_lambda):
    # Split SAC and TD3 results
    filt_metrics = {}
    for k, v in metrics.items():
        if key_lambda(k):
            filt_metrics[k] = v
    return filt_metrics


# %% Data reading and preprocessing
# base_path = Path('/home/angel/desarrollo/uav_navigation/rl_experiments/logs_cf_vector_new')
base_path = Path('/home/angel/desarrollo/sb3-srl/logs_cf_vector_paper')
base_path = Path('/home/angel/desarrollo/sb3-srl/logs_cf_pixel_paper')
# base_path = Path('/home/angel/desarrollo/sb3-srl/logs_cf_vector_alm')
ni_et_al = 'TD3 (Ni et al. 2024)'
proposal_td3_key = 'TD3-ISPR (ours)'
proposal_sac_key = 'SAC-ISPR (ours)'
exp_paths = []
# filter
for p in base_path.iterdir():
    if 'random' in str(p) or 'assets' in str(p):
        continue
    # if 'sac' in str(p):  # filter SAC
    # if 'td3' in str(p):  # filter TD3
    #     continue
    exp_paths.append(p)
exp_paths.sort(reverse=False)

phase = 'eval'
exp_summ = None
rewards = {}
rewards_step = {}
nav_metrics = {}
episodes = {}
rewards_tpos = {}

pbar = tqdm(total=len(exp_paths) * 5, desc="Processing")
for exp_path in exp_paths:
    pbar.set_description("Processing " + exp_path.name)
    # real_world results
    # if not (exp_path / 'eval_real').exists():
    #     continue
    # exp_data = ExperimentData(exp_path, eval_regex=r"eval_real/history_*.csv")

    # read and preprocess self-predictive-rl results
    if 'alm' in exp_path.name:  # TD3
        exp_data = ExperimentData(exp_path, exp_args='flags.yml', eval_regex=r"eval/history_*.csv")
        exp_data.exp_args['is_srl'] = True  # set as constant since is experiment-based
        # correct episode offset and filter extra data
        phase_df = exp_data.history_df[exp_data.history_df['phase'] == 'eval']
        phase_df.loc[:, 'ep'] += 1
        ep_df = phase_df[phase_df['ep'] != 51]
        ep_df = ep_df[ep_df['ep'] != 5]
        exp_data.history_df = ep_df
        exp_data.alg_name = ni_et_al
    else:
        # read normal experiment
        exp_data = ExperimentData(exp_path, eval_regex=r"eval/history_*.csv")

    # proposal results
    if 'ispr' in exp_path.name: # TD3
        exp_data.alg_name += ' (ours)'

    # multi-thread processing
    process_exp_data(exp_data, phase, pbar=pbar)
pbar.close()


# %% Ensure save data
out_path = base_path / 'assets'
# out_path = None
if out_path is not None and not out_path.exists():
    out_path.mkdir()

# %% Overall results
if out_path is not None:
    exp_summ.to_csv(out_path / 'summary.csv', index=False)

# %% Navigation metrics
plots = [  # metric_id, y_label, plt_title, is_percent
    ('SR', 'Success rate', 'Success rate comparison', True),
    ('SPL', 'Success Path Length', 'Success path length comparison', True),
    ('SSPL', 'Soft Success Path Length', 'Soft success path length comparison', True),
    ('DTS', 'Distance to success (meters)', 'Distance to success comparison', False)
]

# _, plot_metrics = split_metric(nav_metrics)
for metric_id in plots:
    print('Plotting', metric_id[0])
    out_fig = out_path / f"nav_metric_{metric_id[0]}_proposal.pdf" if out_path is not None else out_path
    with plot_metric(metric_id[2], metric_id[1], metric_id[-1], figsize=(7,5)) as fig:
        ax = fig.add_subplot(1, 1, 1)
        set_color_palette(ax)
        for i, (label, values) in enumerate(nav_metrics.items()):
            # if 'SAC' in label or 'et al.' in label:
            #     continue
            # if 'TD3' in label or 'et al.' in label:
            #     continue
            # if 'ISPR' not in label and 'et al.' not in label:
            #     continue
            eps = np.asarray(episodes[label][0]) + 1
            plot_values = np.vstack([v[metric_id[0]].values for v in values])
            draw_metric(ax, label, eps, plot_values, metric_id[-1])
    if out_fig is not None:
        fig.savefig(out_fig)
    fig.show()

# %% Reward metrics
print('Plotting reward curve')
out_fig = out_path / f"reward_{phase}.pdf" if out_path is not None else out_path
with plot_metric(f"Reward curve during {phase}", 'Accumulative reward', False) as fig:
    ax = fig.add_subplot(1, 1, 1)
    set_color_palette(ax)
    for i, (label, values) in enumerate(rewards.items()):
        eps = np.asarray(episodes[label][0]) + 1
        plot_values = np.asarray(values)
        draw_metric(ax, label, eps, plot_values, False)
if out_fig is not None:
    fig.savefig(out_fig)
fig.show()

# %% Reward metrics by target quadrant
quadrant_idx = 20
print('Plotting reward curve - TargetQuadrant:', metric_id[0])
out_fig = out_path / f"reward_{phase}_tq_{quadrant_idx}.pdf" if out_path is not None else out_path
with plot_metric(f"Reward curve during {phase} for TargetQuadrant{quadrant_idx}",
                 'Accumulative reward', False) as fig:
    ax = fig.add_subplot(1, 1, 1)
    set_color_palette(ax)
    for i, (label, values) in enumerate(rewards_tpos.items()):
        eps = np.asarray(episodes[label][0]) + 1
        plot_values = np.asarray(values)[:, quadrant_idx]
        draw_metric(ax, label, eps, plot_values, False)
if out_fig is not None:
    fig.savefig(out_fig)
fig.show()

# %% Aggregated metrics with 95% Stratified Bootstrap CIs
# IQM, Optimality Gap, Median, Mean
print('Plotting aggregated reward')
algos = list(rewards_tpos.keys())
# alg_norm = 'SAC'
# alg_norm = ni_et_al
alg_norm = 'TD3'

metric2plot = ['Median', 'IQM', 'Mean', 'Optimality Gap']
fig, axes = plot_aggregated_metrics(algos, rewards_tpos, alg_norm, metric2plot)

if out_path is not None:
    fig.savefig(out_path / "rliable_aggregated_metrics.pdf", bbox_inches='tight', pad_inches=0.1)
fig.show()

# %% Probability of Improvement
print('Plotting Probability of Improvement')
algos = list(rewards_tpos.keys())
# alg_norm = 'SAC'
algo_pairs = [
    # baseline
    (proposal_td3_key, 'TD3'),
    ('TD3-SPR',    'TD3'),
    (ni_et_al,     'TD3'),
    ]
fig, axes = plot_probability_improvement(algos, rewards_tpos, alg_norm, algo_pairs)
if out_path is not None:
    fig.savefig(out_path / "rliable_probability_improvement_TD3.pdf", bbox_inches='tight')
fig.show()

algo_pairs = [
    # baseline
    (proposal_sac_key, 'SAC'),
    ('SAC-SPR',  'SAC'),
    (ni_et_al,   'SAC'),
    ]
fig, axes = plot_probability_improvement(algos, rewards_tpos, alg_norm, algo_pairs)
if out_path is not None:
    fig.savefig(out_path / "rliable_probability_improvement_SAC.pdf", bbox_inches='tight')
fig.show()

algo_pairs = [
    # cross methods
    (proposal_td3_key,  ni_et_al),
    (proposal_sac_key,  ni_et_al),
    (proposal_td3_key,  proposal_sac_key),
    ('TD3-SPR', 'SAC-SPR'),
    ('TD3',   'SAC'),
    ]
fig, axes = plot_probability_improvement(algos, rewards_tpos, alg_norm, algo_pairs)
if out_path is not None:
    fig.savefig(out_path / "rliable_probability_improvement_proposal.pdf", bbox_inches='tight')
fig.show()

# algo_pairs = [
#     # cross methods
#     ('SPR',   'TD3-SPR'),
#     ('SPR',   proposal_td3_key),
#     # baseline
#     (proposal_td3_key,   'TD3'),
#     ('TD3-SPR',   'TD3'),
#     ('SPR',   'TD3'),
#     ]
# fig, axes = plot_probability_improvement(algos, rewards_tpos, alg_norm, algo_pairs)
# if out_path is not None:
#     fig.savefig(out_path / "rliable_probability_improvement.pdf", bbox_inches='tight')
# fig.show()



# %% Performance profile
print('Plotting Performance profile')
# alg_norm = 'SAC'
# algos = list(rewards_tpos.keys())
# fig, axes = plot_performance_profile(algos, rewards_tpos, alg_norm)
# if out_path is not None:
#     fig.savefig(out_path / "rliable_performance_profile.pdf", bbox_inches='tight')
# fig.show()

algos = ['TD3-ISPR (ours)',
         'TD3-SPR',
         'TD3']
fig, axes = plot_performance_profile(algos, rewards_tpos, alg_norm)
if out_path is not None:
    fig.savefig(out_path / "rliable_performance_profile_TD3.pdf", bbox_inches='tight')
fig.show()

algos = ['SAC-ISPR (ours)',
         'SAC-SPR',
         'SAC']
fig, axes = plot_performance_profile(algos, rewards_tpos, alg_norm)
if out_path is not None:
    fig.savefig(out_path / "rliable_performance_profile_SAC.pdf", bbox_inches='tight')
fig.show()

algos = ['TD3 (Ni et al. 2024)',
         'SAC-ISPR (ours)',
         'TD3-ISPR (ours)']
fig, axes = plot_performance_profile(algos, rewards_tpos, alg_norm)
if out_path is not None:
    fig.savefig(out_path / "rliable_performance_profile_proposal.pdf", bbox_inches='tight')
fig.show()

# %% Sample efficiency curve
print('Plotting Sample efficiency curve')
# algos = list(rewards_tpos.keys())
# alg_norm = 'SAC'
# fig, ax = plot_sample_efficiency(algos, rewards_tpos, alg_norm, np.asarray(episodes[alg_norm][0]))
# if out_path is not None:
#     fig.savefig(out_path / "rliable_efficiency_curve.pdf", bbox_inches='tight')
# fig.show()

algos = ['TD3-ISPR (ours)',
         'TD3-SPR',
         'TD3']
fig, ax = plot_sample_efficiency(algos, rewards_tpos, alg_norm, np.asarray(episodes[alg_norm][0]))
if out_path is not None:
    fig.savefig(out_path / "rliable_efficiency_curve_TD3.pdf", bbox_inches='tight')
fig.show()

algos = ['SAC-ISPR (ours)',
         'SAC-SPR',
         'SAC']
fig, ax = plot_sample_efficiency(algos, rewards_tpos, alg_norm, np.asarray(episodes[alg_norm][0]))
if out_path is not None:
    fig.savefig(out_path / "rliable_efficiency_curve_SAC.pdf", bbox_inches='tight')
fig.show()

algos = ['TD3 (Ni et al. 2024)',
         'SAC-ISPR (ours)',
         'TD3-ISPR (ours)']
fig, ax = plot_sample_efficiency(algos, rewards_tpos, alg_norm, np.asarray(episodes[alg_norm][0]))
if out_path is not None:
    fig.savefig(out_path / "rliable_efficiency_curve_proposal.pdf", bbox_inches='tight')
fig.show()
