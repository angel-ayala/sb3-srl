#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 11:22:57 2025

@author: angel
"""
import gymnasium as gym
import torch as th
from torch import nn

from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class EarlyStopper:
    def __init__(self, patience=4500, min_delta=0., models=[]):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.stop = False
        self.models = models

    def __call__(self, validation_loss):
        if self.stop:
            for m in self.models:
                m.requires_grad_(False)
            return True
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                print('EarlyStopper')
                self.stop = True
        return self.stop


class DictFlattenExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the dict input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space: The observation space of the environment
    """

    def __init__(self, observation_space: gym.Space) -> None:
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()

    def forward(self, observations: dict) -> th.Tensor:
        if isinstance(observations, dict): # it is expected to have dict, but in case not, is a th.Tensor
            obs_stack = []
            for k, obs in observations.items():
                obs_stack.append(obs)
            observations = th.cat(obs_stack, dim=1)
        return self.flatten(observations)
