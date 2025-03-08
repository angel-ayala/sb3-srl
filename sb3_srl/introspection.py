#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 17:47:50 2025

@author: angel
"""
import torch as th
from torch import nn
import torch.nn.functional as F


def compute_probability_of_success(Q_sa, V_s_prime, eps=1e-6):
    r"""
    Computes the probability of success loss for optimization.

    Updates the proposed equation (1) in https://arxiv.org/abs/2108.08911 of
    \hat{P}_S \equiv 0.5 * log(Q/R^T) + 1, clip(0, 1) by
    \sigmoid((Q(s,a)/V(s')) * 9.).

    Args:
        Q_sa (tuple): (Q1, Q2) from critics.
        V_s_prime (tuple): (Q1_target, Q2_target) from target critics.
        eps (float): numeric value to avoid division by zero.

    Returns:
        torch.Tensor: Loss to maximize probability of success.
    """
    # Compute probability of success values
    # originally 0.5 * log(Q/R^T) + 1 (clipped 0,1)
    ratio = Q_sa.squeeze() / (V_s_prime.squeeze() + eps)
    return th.sigmoid(ratio * 9.).unsqueeze(1)


def compute_Ps(current_q_values, next_v_values, dones):
    success_prob = compute_probability_of_success(current_q_values, next_v_values)
    success_prob = success_prob * (1.  - dones)
    return success_prob


def compute_mse_loss(y_true, y_hat):
    return F.mse_loss(y_true, y_hat)


def compute_nll_loss(y_true, y_hat):
    # clamp to prevent log(0)
    y_true = th.clamp(y_true, 0.0001, 0.9999)
    y_hat = th.clamp(y_hat, 0.0001, 0.9999)
    # compute Negative Log-Likelihood loss
    loss = -(y_true * th.log(y_hat) + (1 - y_true) * th.log(1 - y_hat))
    return loss.mean()


class ProbabilityModel(nn.Module):
    def __init__(self, action_shape: tuple, input_dim: int, hidden_dim: int):
        super(ProbabilityModel, self).__init__()
        self.success_prob = nn.Sequential(
            nn.Linear(input_dim + action_shape[-1], hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid())

    def stop_gradient(self):
        self.requires_grad_(False)

    def forward(self, observation, action):
        return self.success_prob(th.cat([observation, action], dim=1))


class IntrospectionBelief:
    def __init__(self,
                 action_shape: tuple,
                 input_dim: int = 50,
                 hidden_dim: int = 256,
                 latent_lambda: float = 1e-3,
                 early_stop: bool = True):
        self.probability = ProbabilityModel(action_shape, input_dim, hidden_dim)
        self.introspection_lambda = latent_lambda

    def prob_optimizer(self, learning_rate, optim_class=th.optim.Adam,
                       **optim_kwargs):
        self.probability_optim = optim_class(self.probability.parameters(),
                                      lr=learning_rate, **optim_kwargs)

    def prob_zero_grad(self):
        self.probability_optim.zero_grad()

    def prob_step(self):
        self.probability_optim.step()

    def update_prob(self, prob_loss):
        self.prob_zero_grad()
        prob_loss.backward()
        self.prob_step()

    def apply(self, function):
        self.probability.apply(function)

    def to(self, device):
        self.probability.to(device)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the model in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.probability.train(mode)

    def compute_Ps(self, current_q_values, next_v_values, dones):
        return compute_Ps(current_q_values, next_v_values, dones)

    def infer_Ps(self, observations_z, actions):
        return self.probability(observations_z, actions)

    def compute_nll_loss(self, success_prob, success_hat):
        return compute_nll_loss(success_prob, success_hat)
