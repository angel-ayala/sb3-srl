#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 16:32:41 2026

@author: angel

Constant definitions of boolean proprioceptive masks for MuJoCo environments.
True = proprioceptive information, False = Exteroceptive information
"""
from typing import Optional, List


def mujoco_prop_mask(env_id: str) -> Optional[List]:
    prop_mask = None
    env_lower = env_id.lower()

    if "ant" in env_lower:
        prop_mask = ANT
    elif "cheeta" in env_lower:
        prop_mask = CHEETAH
    elif "hopper" in env_lower:
        prop_mask = HOPPER
    else:
        raise NotImplementedError(f"Proprioceptive mask not found for {env_id}")

    return prop_mask


ANT = [
    False,  # 0: z-coordinate of torso
    False,  # 1: w-orientation
    False,  # 2: x-orientation
    False,  # 3: y-orientation
    False,  # 4: z-orientation
    True,   # 5: hip_1 angle
    True,   # 6: ankle_1 angle
    True,   # 7: hip_2 angle
    True,   # 8: ankle_2 angle
    True,   # 9: hip_3 angle
    True,   # 10: ankle_3 angle
    True,   # 11: hip_4 angle
    True,   # 12: ankle_4 angle
    False,  # 13: x-velocity
    False,  # 14: y-velocity
    False,  # 15: z-velocity
    False,  # 16: x-angular velocity
    False,  # 17: y-angular velocity
    False,  # 18: z-angular velocity
    True,   # 19: hip_1 angular velocity
    True,   # 20: ankle_1 angular velocity
    True,   # 21: hip_2 angular velocity
    True,   # 22: ankle_2 angular velocity
    True,   # 23: hip_3 angular velocity
    True,   # 24: ankle_3 angular velocity
    True,   # 25: hip_4 angular velocity
    True,   # 26: ankle_4 angular velocity
]

CHEETAH = [
    False,  # 0: z-coordinate of front tip (absolute position)
    False,  # 1: angle of front tip (orientation, requires external reference)
    True,   # 2: angle of back thigh (joint angle)
    True,   # 3: angle of back shin (joint angle)
    True,   # 4: angle of back foot (joint angle)
    True,   # 5: angle of front thigh (joint angle)
    True,   # 6: angle of front shin (joint angle)
    True,   # 7: angle of front foot (joint angle)
    False,  # 8: velocity of x-coordinate of front tip (requires external reference)
    False,  # 9: velocity of z-coordinate of front tip (requires external reference)
    False,  # 10: angular velocity of front tip (requires external reference)
    True,   # 11: angular velocity of back thigh (joint angular velocity)
    True,   # 12: angular velocity of back shin (joint angular velocity)
    True,   # 13: angular velocity of back foot (joint angular velocity)
    True,   # 14: angular velocity of front thigh (joint angular velocity)
    True,   # 15: angular velocity of front shin (joint angular velocity)
    True,   # 16: angular velocity of front foot (joint angular velocity)
]

HOPPER = [
    False,  # 0: z-coordinate of torso (absolute position)
    False,  # 1: angle of torso (orientation, requires external reference)
    True,   # 2: angle of thigh joint (joint angle)
    True,   # 3: angle of leg joint (joint angle)
    True,   # 4: angle of foot joint (joint angle)
    False,  # 5: velocity of x-coordinate of torso (requires external reference)
    False,  # 6: velocity of z-coordinate of torso (requires external reference)
    False,  # 7: angular velocity of torso (requires external reference)
    True,   # 8: angular velocity of thigh joint (joint angular velocity)
    True,   # 9: angular velocity of leg joint (joint angular velocity)
    True,   # 10: angular velocity of foot joint (joint angular velocity)
]

