#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 22:53:38 2026

@author: angel
"""

from __future__ import annotations

import torch.nn as nn


class BaseFunction(nn.Module):
    """
    Base type for reusable SRL function models.

    No optimization logic belongs here.
    """

    pass
