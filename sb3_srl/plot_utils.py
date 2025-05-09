#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 12:10:14 2024

@author: Angel Ayala
"""
from contextlib import contextmanager
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from rliable import library as rly
from rliable import metrics as rly_metrics
from rliable import plot_utils as rly_plot

from webots_drone.target import VirtualTarget


def draw_flight_area(ax, flight_area, is_3d=False):
    """
    Draws a flight area as a rectangle in 2D or a cuboid in 3D.

    Parameters:
        ax: Matplotlib axis (2D or 3D).
        flight_area: List defining the bounds of the area. Format:
            [[x_min, y_min, z_min], [x_max, y_max, z_max]]
            If is_3d is False, only x_min, y_min, x_max, y_max are used.
        is_3d: Boolean, whether to draw in 3D or 2D.

    Returns:
        ax: Updated axis with the flight area plotted.
    """
    if is_3d:
        # Define the 3D vertices of the cuboid
        vertices = [
            [flight_area[0][0], flight_area[0][1], flight_area[0][2]],  # Bottom face
            [flight_area[1][0], flight_area[0][1], flight_area[0][2]],
            [flight_area[1][0], flight_area[1][1], flight_area[0][2]],
            [flight_area[0][0], flight_area[1][1], flight_area[0][2]],
            [flight_area[0][0], flight_area[0][1], flight_area[1][2]],  # Top face
            [flight_area[1][0], flight_area[0][1], flight_area[1][2]],
            [flight_area[1][0], flight_area[1][1], flight_area[1][2]],
            [flight_area[0][0], flight_area[1][1], flight_area[1][2]],
        ]

        # Define the edges of the cuboid
        edges = [
            # Bottom face edges
            [vertices[0], vertices[1]],
            [vertices[1], vertices[2]],
            [vertices[2], vertices[3]],
            [vertices[3], vertices[0]],
            # Top face edges
            [vertices[4], vertices[5]],
            [vertices[5], vertices[6]],
            [vertices[6], vertices[7]],
            [vertices[7], vertices[4]],
            # Vertical edges connecting top and bottom faces
            [vertices[0], vertices[4]],
            [vertices[1], vertices[5]],
            [vertices[2], vertices[6]],
            [vertices[3], vertices[7]],
        ]

        # Plot each edge of the cuboid
        for edge in edges:
            x, y, z = zip(*edge)
            ax.plot(x, y, z, color='b', alpha=0.5)

        # Set 3D labels
        ax.set_zlabel('Z')
    else:
        # Define the 2D vertices of the rectangle
        vertices = [
            [flight_area[0][0], flight_area[0][1]],
            [flight_area[1][0], flight_area[0][1]],
            [flight_area[1][0], flight_area[1][1]],
            [flight_area[0][0], flight_area[1][1]],
        ]

        # Define the edges of the rectangle
        edges = [
            [vertices[0], vertices[1]],
            [vertices[1], vertices[2]],
            [vertices[2], vertices[3]],
            [vertices[3], vertices[0]],
        ]

        # Plot each edge of the rectangle
        for edge in edges:
            x, y = zip(*edge)
            ax.plot(x, y, color='b', alpha=0.5)

    # Set common axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    return ax


def draw_flight_area_sideview(ax, flight_area):
    """
    Draws a rectangular boundary of the flight area projected onto the X-Z plane.
    flight_area is expected to be [xmin, xmax, ymin, ymax, zmin, zmax]
    """
    (xmin, _, zmin), (xmax, _, zmax) = flight_area

    # Rectangle in X-Z plane
    corners_x = [xmin, xmax, xmax, xmin, xmin]
    corners_z = [zmin, zmin, zmax, zmax, zmin]

    ax.plot(corners_x, corners_z, color='b', alpha=0.5)
    ax.set_xlim(xmin - 0.5, xmax + 0.5)
    ax.set_ylim(zmin - 0.5, zmax + 0.5)


# draw sphere
def draw_sphere(ax, center, dimension, color='r', alpha=0.5):
    """
    Draws a sphere on the given 3D axis.

    Parameters:
        ax: Matplotlib 3D axis.
        center: List or tuple of the sphere's center coordinates [x, y, z].
        dimension: Dimension of the sphere with [height, radius].
        color: Color of the sphere surface (default: 'b').
        alpha: Transparency of the sphere surface (default: 0.5).
    """
    # Generate a meshgrid for spherical coordinates
    u = np.linspace(0, 2 * np.pi, 10)
    v = np.linspace(0, np.pi, 10)
    x = center[0] + dimension[1] * np.outer(np.cos(u), np.sin(v))
    y = center[1] + dimension[1] * np.outer(np.sin(u), np.sin(v))
    z = center[2] + dimension[0] * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface of the sphere
    ax.plot_surface(x, y, z, color=color, alpha=alpha)


def draw_3d_path(ax, points, orientations, rewards, color='k', linewidth=1,
                 arrow_color='k', arrow_length=0.25):
    """
    Plots a 3D path on the given axis with optional orientation arrows.

    Parameters:
        ax: Matplotlib 3D axis.
        points: Sequence of 3D points as a list or NumPy array of shape (n, 3),
                where n is the number of points, and each point is [x, y, z].
        orientations: Optional list or NumPy array of orientation vectors of shape (n, 3),
                      where each vector [dx, dy, dz] represents the direction at a point.
                      Must be the same length as `points`.
        linewidth: Width of the line (default: 2).
        marker: Marker style for the points (default: 'o').
        arrow_color: Color of the orientation arrows (default: 'b').
    """
    points = np.array(points)  # Ensure points is a NumPy array
    if points.shape[1] != 3:
        raise ValueError("Each point must have three coordinates [x, y, z].")

    # Extract x, y, and z coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Plot the path
    ax.plot(x, y, z, linewidth=linewidth)

    # Plot rewards
    ax.scatter(x, y, z, c=rewards, cmap='viridis')

    # Plot vehicle's dimension
    for p in points:
        draw_sphere(ax, p, [0.05, 0.05], color='b', alpha=0.2)

    # Plot orientation arrows if provided
    if orientations is not None:
        if len(orientations) != len(points):
            raise ValueError("Orientations must have the same length as points.")

        # Compute direction vectors based on radian angles
        dx = np.cos(orientations) * arrow_length
        dy = np.sin(orientations) * arrow_length
        dz = np.zeros_like(dx)  # No orientation change in the Z direction

        # Add quivers to represent the orientation
        ax.quiver(x, y, z, dx, dy, dz, color=arrow_color)


def draw_3d_target(ax, position, dimension, goal_threshold):
    # Define target
    vtarget = VirtualTarget(dimension=dimension)
    vtarget.set_position(position)
    # Define areas
    risk_dist = vtarget.get_risk_distance()
    goal_dist = risk_dist + goal_threshold
    # Plot target
    draw_sphere(ax, vtarget.position, vtarget.dimension, alpha=1.)
    # Plot risk zone
    draw_sphere(ax, vtarget.position, [risk_dist, risk_dist], alpha=0.2)
    # Plot goal zone
    draw_sphere(ax, vtarget.position, [goal_dist, goal_dist], color='g', alpha=0.2)


def draw_2d_target(ax, position, dimension, goal_threshold, is_side=False):
    # Define target
    vtarget = VirtualTarget(dimension=dimension)
    vtarget.set_position(position)
    # Define areas
    if is_side:
        t_pos = vtarget.position[0], vtarget.position[2]
    else:
        t_pos = vtarget.position[:2]
    risk_dist = vtarget.get_risk_distance()
    goal_dist = risk_dist + goal_threshold
    # Plot target and risk zone
    target_spot = plt.Circle(t_pos, vtarget.dimension[1], color='r')
    risk_zone = plt.Circle(t_pos, risk_dist, color='r', alpha=0.2)
    goal_zone = plt.Circle(t_pos, goal_dist, color='g', fill=False)
    ax.add_patch(target_spot)
    ax.add_patch(risk_zone)
    ax.add_patch(goal_zone)


def draw_scene_elements(ax, scene_size=(100, 100)):
    tower = plt.Rectangle((-5.1, -7.7), 10.2, 15.4, color='b', alpha=0.3)
    road_pos = (-5, scene_size[1] / 2)
    road_lanes = plt.Rectangle(road_pos, 10, road_pos[1] - 7.7, color='k', alpha=0.1)
    ax.add_patch(road_lanes)
    ax.add_patch(tower)


# def plot_trajectory(fig, exp_data, phase='eval', episode=0, iteration=0):
#     ax = fig.add_subplot(111, projection='3d')
#     # Draw flight area
#     draw_flight_area(ax, exp_data.flight_area, is_3d=True)

#     # Filter trajectory
#     trj_data = exp_data.get_ep_trajectories(episode, phase, iteration)[0]

#     # Draw target
#     draw_3d_target(ax, trj_data['target_pos'],
#                    exp_data.exp_args['target_dim'],
#                    exp_data.exp_args['goal_threshold'])

#     # Draw 3D path
#     path = trj_data['states']
#     draw_3d_path(ax, path[:, :3], path[:, 5], trj_data['rewards'])
#     return ax

def draw_trajectory(axes, trj_data, color='k'):
    ax_top, ax_side = axes
    path = trj_data['states']
    x, y, z = path[:, 0], path[:, 1], path[:, 2]
    orientations = path[:, 5]
    rewards = trj_data['rewards']
    # Plot trajectory (projected)
    ax_top.plot(x, y, color=color, linewidth=1)
    ax_top.scatter(x, y, c=rewards, s=20, alpha=0.6)

    ax_side.plot(x, z, color=color, linewidth=1)
    ax_side.scatter(x, z, c=rewards, s=20, alpha=0.6)

    # Plot orientation arrows in both views
    arrow_length = 0.25
    dx = np.cos(orientations) * arrow_length
    dy = np.sin(orientations) * arrow_length
    dz = np.zeros_like(dx)  # Z direction not used for now

    # Top view (X-Y)
    ax_top.quiver(x, y, dx, dy, color=color)#, angles='xy', scale_units='xy', scale=1)

    # Side view (X-Z): we use dx and dz
    ax_side.quiver(x, z, dx, dz, color=color)#, angles='xy', scale_units='xy', scale=1)
    return ax_top, ax_side


def plot_trajectory(fig, exp_data, phase='eval', episode=0, iteration=0):
    # Create two 2D subplots: top view (X-Y) and side view (X-Z)
    ax_top = fig.add_subplot(1, 2, 1)
    ax_side = fig.add_subplot(1, 2, 2)

    # Get trajectory data
    trj_data = exp_data.get_ep_trajectories(episode, phase, iteration)[0]

    # Draw flight area
    draw_flight_area(ax_top, exp_data.flight_area, is_3d=False)
    draw_flight_area_sideview(ax_side, exp_data.flight_area)

    # Draw target in both views
    target_pos = trj_data['target_pos']
    draw_2d_target(ax_top, target_pos, exp_data.exp_args['target_dim'],
                   exp_data.exp_args['goal_threshold'])
    draw_2d_target(ax_side, target_pos, exp_data.exp_args['target_dim'],
                   exp_data.exp_args['goal_threshold'], is_side=True)  # X-Z projection
    # Plot trajectory
    draw_trajectory([ax_top, ax_side], trj_data)

    # Configure axes
    ax_top.set_title('Top View', fontsize='xx-large')
    ax_top.set_xlabel('X', fontsize='x-large')
    ax_top.set_ylabel('Y', fontsize='x-large')
    ax_top.axis('equal')
    x_min, x_max = exp_data.flight_area[0][0], exp_data.flight_area[1][0]
    y_min, y_max = exp_data.flight_area[0][1], exp_data.flight_area[1][1]
    ax_top.set_xlim(x_min-0.1, x_max+0.1)
    ax_top.set_ylim(y_min-0.1, y_max+0.1)
    ax_top.tick_params(axis='x', labelsize='large')
    ax_top.tick_params(axis='y', labelsize='large')
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['top'].set_visible(False)

    ax_side.set_title('Side View', fontsize='xx-large')
    ax_side.set_xlabel('X', fontsize='x-large')
    ax_side.set_ylabel('Z', fontsize='x-large')
    ax_side.axis('equal')
    z_min, z_max = 0, exp_data.flight_area[1][2]
    x_margin = (abs(x_min) + abs(x_max)) / 2.
    x_min = max(x_min, target_pos[0] - x_margin)
    x_max = min(x_max, target_pos[0] + x_margin)
    ax_side.set_xlim(x_min, x_max)
    ax_side.set_ylim(z_min, z_max)
    ax_side.tick_params(axis='x', labelsize='large')
    ax_side.tick_params(axis='y', labelsize='large')
    ax_side.spines['right'].set_visible(False)
    ax_side.spines['top'].set_visible(False)

    return ax_top, ax_side


@contextmanager
def plot_metric(title, label, is_percent=False, layout='constrained', figsize=(6, 5)):
    # instantiate a new figure
    fig = plt.figure(layout='constrained', figsize=figsize)
    yield fig
    # format
    ax = fig.gca()
    ax.set_xlabel('Episodes', fontsize='xx-large')
    ax.set_ylabel(label, fontsize='xx-large')
    if is_percent:
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(
            lambda x, _: '{:,.0f}%'.format(x * 100)))
        ax.set_ylim((-0.01, 1.01))
    # else:
    #     ax.set_ylim((-0.01))

    ax.tick_params(axis='x', labelsize='x-large')
    ax.tick_params(axis='y', labelsize='x-large')
    ax.grid(True, alpha=0.2)
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)


def set_color_palette(ax):
    ax.set_prop_cycle('color', sns.color_palette("colorblind"))


def draw_metric(ax, label, episodes, metric_values, is_percent):
    if metric_values.shape[0] > 1:
        # add std area
        mean_values = metric_values.mean(axis=0)
        std_values = metric_values.std(axis=0)
        max_values = mean_values + std_values
        min_values = mean_values - std_values
        plt.fill_between(episodes, min_values, max_values, alpha=0.3)
    else:
        mean_values = metric_values[0]
    ax.plot(episodes, mean_values, label=label, marker='o')


def get_norm_rewards(algorithms, rewards_dict, alg_norm='SAC'):
    # preprocess data
    normalized_score = {}
    max_reward = np.asarray(rewards_dict[alg_norm]).sum(axis=-1).max()
    for alg in algorithms:
        normalized_score[alg] = np.asarray(rewards_dict[alg]).sum(axis=-1)
        normalized_score[alg] /= max_reward

    return normalized_score


def plot_aggregated_metrics(algorithms, rewards_dict, alg_norm='SAC',
                            metric_names=['Median', 'IQM', 'Mean', 'Optimality Gap']):
    score_normalized = get_norm_rewards(algorithms, rewards_dict, alg_norm)

    # compute metrics
    def aggregate_func(x):
        out = []
        if 'Median' in metric_names:
            out.append(rly_metrics.aggregate_median(x))
        if 'IQM' in metric_names:
            out.append(rly_metrics.aggregate_iqm(x))
        if 'Mean' in metric_names:
            out.append(rly_metrics.aggregate_mean(x))
        if 'Optimality Gap' in metric_names:
            out.append(rly_metrics.aggregate_optimality_gap(x))
        out_stack = np.hstack(out)
        return out_stack

    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
        score_normalized, aggregate_func, reps=50000)
    # plot
    fig, axes = rly_plot.plot_interval_estimates(
        aggregate_scores, aggregate_score_cis,
        metric_names=metric_names, algorithms=algorithms,
        xlabel=f"Max-{alg_norm} Normalized Score", xlabel_y_coordinate=-.35)
    return fig, axes


def plot_probability_improvement(algorithms, rewards_dict, alg_norm='SAC', algo_pairs=[]):
    score_normalized = get_norm_rewards(algorithms, rewards_dict, alg_norm)
    # preprocess
    algorithm_pairs = {}
    if len(algo_pairs) == 0:
        for alg_a in algorithms:
            for alg_b in algorithms:
                if alg_a == alg_b:
                    continue
                algorithm_pairs[f"{alg_a},{alg_b}"] = score_normalized[alg_a], score_normalized[alg_b]
    else:
        for alg_a, alg_b in algo_pairs:
            algorithm_pairs[f"{alg_a},{alg_b}"] = score_normalized[alg_a], score_normalized[alg_b]
    # Plot score distributions
    fig, ax = plt.subplots(ncols=1, figsize=(7, len(algorithm_pairs.keys())))
    average_probabilities, average_prob_cis = rly.get_interval_estimates(
        algorithm_pairs, rly_metrics.probability_of_improvement, reps=2000)
    rly_plot.plot_probability_of_improvement(average_probabilities, average_prob_cis, ax=ax)
    return fig, ax


def plot_performance_profile(algorithms, rewards_dict, alg_norm='SAC', figsize=(7, 5)):
    score_normalized = get_norm_rewards(algorithms, rewards_dict, alg_norm)

    # Human normalized score thresholds
    score_thresholds = np.linspace(0.0, 1.5, 100)
    distributions, distributions_cis = rly.create_performance_profile(
        score_normalized, score_thresholds)
    # Plot score distributions
    fig, ax = plt.subplots(ncols=1, figsize=figsize)
    rly_plot.plot_performance_profiles(
      distributions, score_thresholds, performance_profile_cis=distributions_cis,
      colors=dict(zip(algorithms, sns.color_palette('colorblind'))),
      xlabel=fr'Max-{alg_norm} Normalized Score $(\tau)$', ax=ax)
    ax.legend()
    return fig, ax


def plot_sample_efficiency(algorithms, rewards_dict, alg_norm, episodes, figsize=(7, 5)):
    # preprocess data
    score_normalized = {}
    max_reward = np.asarray(rewards_dict[alg_norm]).max()
    for alg in algorithms:
        score_normalized[alg] = np.asarray(rewards_dict[alg])
        score_normalized[alg] /= max_reward
    # compute values
    eps_idx = list(range(len(episodes)))
    scores_ep = {algorithm: score[:, :, eps_idx]
                 for algorithm, score in score_normalized.items()}
    iqm = lambda scores: np.array([rly_metrics.aggregate_iqm(scores[..., ep])
                                   for ep in range(scores.shape[-1])])
    iqm_scores, iqm_cis = rly.get_interval_estimates(
      scores_ep, iqm, reps=50000)
    # Plot efficiency cruve
    fig, ax = plt.subplots(ncols=1, figsize=figsize)
    rly_plot.plot_sample_efficiency_curve(
        episodes+1, iqm_scores, iqm_cis, algorithms=algorithms,
        xlabel='Episodes',
        ylabel=f'IQM Max-{alg_norm} Normalized Score', ax=ax)
    ax.legend()
    return fig, ax
