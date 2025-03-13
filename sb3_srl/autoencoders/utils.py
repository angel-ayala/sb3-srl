#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:33:38 2025

@author: angel
"""
import torch as th
from torch.nn import functional as F
from sklearn.feature_selection import mutual_info_regression
from info_nce import InfoNCE


def preprocess_pixel_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039.

    taken from https://github.com/denisyarats/pytorch_sac_ae"""
    bins = 2**bits
    assert obs.dtype == th.float32
    if bits < 8:
        obs = th.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + th.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


def destack(obs_stack, len_hist=3, is_rgb=False):
    orig_shape = obs_stack.shape
    if is_rgb:
        n_stack = (orig_shape[1] // len_hist) * orig_shape[0]
        obs_destack = obs_stack.reshape((n_stack, 3) + orig_shape[-2:])
    else:
        obs_destack = obs_stack.reshape(
            (orig_shape[0] * len_hist, orig_shape[-1]))
    return obs_destack, orig_shape


def latent_l2_loss(latent_value):
    # add L2 penalty on latent representation
    # see https://arxiv.org/pdf/1903.12436.pdf
    latent_value = (0.5 * latent_value.pow(2).sum(1)).mean()
    return latent_value


def compute_mutual_information(latents, q_values):
    mi = mutual_info_regression(
        latents.detach().cpu().numpy(), q_values.cpu().numpy().reshape(-1))
    return mi.mean()


def dist2orientation(distance: th.Tensor) -> th.Tensor:
    """Compute the orientation towards the target."""
    orientation = th.arctan2(distance[:, 1], distance[:, 0])
    # """Apply UAV sensor offset."""
    orientation -= th.pi / 2.
    # Normalize to the range [-pi, pi]
    orientation = (orientation + th.pi) % (2 * th.pi) - th.pi
    return -orientation


def orientation_diff(orientation_ref, orientation):
    """Compute the normalized difference orientation."""
    orientation_diff = th.cos(orientation_ref - orientation)
    return orientation_diff


def obs2positions(obs_1d):
    if obs_1d.dim() == 3:
        uav_pos = obs_1d[:, :, 6:9]
        target_pos = obs_1d[:, :, -3:]
    elif obs_1d.dim() == 2:
        uav_pos = obs_1d[:, 6:9]
        target_pos = obs_1d[:, -3:]
    return uav_pos, target_pos


def obs2orientation(obs_1d):
    if obs_1d.dim() == 3:
        orientation = obs_1d[:, :, 12]
    elif obs_1d.dim() == 2:
        orientation = obs_1d[:, 12]
    return orientation


def obs2target_dist(obs_1d):
    # replace target_pos by target_dist
    uav_pos, target_pos = obs2positions(obs_1d)
    distance = target_pos - uav_pos
    uav_orientation = obs2orientation(obs_1d)
    target_orientation = dist2orientation(distance)
    orientation = orientation_diff(uav_orientation, target_orientation)
    return distance, orientation


def compute_distance(coord1: th.Tensor, coord2: th.Tensor) -> th.Tensor:
    """Compute Euclidean distance.

    :param th.Tensor coord1: The first coordinates.
    :param th.Tensor coord2: The second coordinates.

    :return th.Tensor: Euclidean distance between the coordinates, rounded
        to 4 decimal points.
    """
    return th.linalg.norm(coord1 - coord2).round(4)


def compute_elevation_angle(reference: th.Tensor, target: th.Tensor) -> th.Tensor:
    """Compute the normalized elevation angle between two 3D points.

    Args:
        reference (th.Tensor): A tensor of shape (3,) representing the reference point (x, y, z).
        target (th.Tensor): A tensor of shape (3,) representing the target point (x, y, z).

    Returns:
        th.Tensor: A tensor representing the normalized elevation angle, scaled to the range [-1, 1].
    """
    # Horizontal distance
    h_dist = compute_distance(reference[:2], target[:2])  # Use Euclidean distance
    # Vertical difference
    delta_z = target[2] - reference[2]
    # Elevation angle
    angle = th.arctan2(delta_z, h_dist)
    angle /= (th.pi / 2)  # Normalize to the range [-1, 1]
    return angle


def info_nce_loss(query, positives):
    # Use InfoNCE to align encoded states with the value predictions
    loss_fn = InfoNCE()
    return loss_fn(query, positives)
