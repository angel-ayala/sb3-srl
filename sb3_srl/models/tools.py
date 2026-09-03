#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 09:42:52 2026

@author: angel
"""

import torch as th

class GradientBalancer:
    def __init__(self, min_weight=0.01, max_weight=100.0, eps=1e-8):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.eps = eps
        self.srl_weight = 1.0

    @staticmethod
    def gradient_norm(loss, parameters):
        grads = th.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        return th.sqrt(
            sum(g.detach().pow(2).sum() for g in grads if g is not None)
        )

    def update_weight(self, critic_loss, srl_loss, parameters):
        params = [p for p in parameters if p.requires_grad]

        critic_norm = self.gradient_norm(critic_loss, params)
        srl_norm = self.gradient_norm(srl_loss, params)
        # print('critic_norm', critic_norm)
        # print('srl_norm', srl_norm)
        current_ratio = (critic_norm / (srl_norm + self.eps)).item()

        self.srl_weight = current_ratio
        # print('srl_weight', self.srl_weight)
        self.srl_weight = max(self.min_weight, min(self.srl_weight, self.max_weight))

        return self.srl_weight
