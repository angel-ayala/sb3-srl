#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 16:32:41 2026

@author: angel

Constant definitions of boolean proprioceptive masks for MuJoCo v4 environments.
True = proprioceptive information, False = Exteroceptive information
"""
from typing import Optional, List


def mujoco_prop_mask(env_id: str) -> Optional[List]:
    prop_mask = None
    env_lower = env_id.lower()
    
    if "humanoid" in env_lower:
        prop_mask = HUMANOID
    elif "ant" in env_lower:
        prop_mask = ANT
    elif "pusher" in env_lower:
        prop_mask = PUSHER
    elif "cheeta" in env_lower:
        prop_mask = CHEETAH
    elif "walker" in env_lower:
        prop_mask = WALKER
    elif "hopper" in env_lower:
        prop_mask = HOPPER
    elif "reacher" in env_lower:
        prop_mask = REACHER
    elif "doublependulum" in env_lower:
        prop_mask = DOUBLE_PENDULUM
    elif "swimmer" in env_lower:
        prop_mask = SWIMMER
    elif "pendulum" in env_lower:
        prop_mask = PENDULUM
    else:
        raise NotImplementedError(f"Proprioceptive mask not found for {env_id}")

    return prop_mask


PENDULUM = [
    False, # 0: position of the cart along the linear surface slider slide position (m)
    True,  # 1: vertical angle of the pole on the cart hinge hinge angle (rad)
    False, # 2: linear velocity of the cart slider slide velocity (m/s)
    True,  # 3: angular velocity of the pole on the cart hinge hinge angular velocity (rad/s)
]


SWIMMER = [
    False, # 0: angle of the front tip free_body_rot hinge angle (rad)
    True,  # 1: angle of the first rotor motor1_rot hinge angle (rad)
    True,  # 2: angle of the second rotor motor2_rot hinge angle (rad)
    False, # 3: velocity of the tip along the x-axis slider1 slide velocity (m/s)
    False, # 4: velocity of the tip along the y-axis slider2 slide velocity (m/s)
    False, # 5: angular velocity of front tip free_body_rot hinge angular velocity (rad/s)
    True,  # 6: angular velocity of first rotor motor1_rot hinge angular velocity (rad/s)
    True,  # 7: angular velocity of second rotor motor2_rot hinge angular velocity (rad/s)
]


DOUBLE_PENDULUM = [
    False, # 0: position of the cart along the linear surface slider slide position (m)
    True,  # 1: sine of the angle between the cart and the first pole sin(hinge) hinge unitless
    True,  # 2: sine of the angle between the two poles sin(hinge2) hinge unitless
    True,  # 3: cosine of the angle between the cart and the first pole cos(hinge) hinge unitless
    True,  # 4: cosine of the angle between the two poles cos(hinge2) hinge unitless
    False, # 5: velocity of the cart slider slide velocity (m/s)
    True,  # 6: angular velocity of the angle between the cart and the first pole hinge hinge angular velocity (rad/s)
    True,  # 7: angular velocity of the angle between the two poles hinge2 hinge angular velocity (rad/s)
    False, # 8: constraint force - x slider slide Force (N)
    False, # 9: (v4 and old) constraint force - 2 slider slide Force (N)
    False, # 10: (v4 and old) constraint force - 3 slider slide Force (N)
]


REACHER = [
    True,  # 0: cosine of the angle of the first arm cos(joint0) hinge unitless
    True,  # 1: cosine of the angle of the second arm cos(joint1) hinge unitless
    True,  # 2: sine of the angle of the first arm sin(joint0) hinge unitless
    True,  # 3: sine of the angle of the second arm sin(joint1) hinge unitless
    False, # 4: x-coordinate of the target target_x slide position (m)
    False, # 5: y-coordinate of the target target_y slide position (m)
    True,  # 6: angular velocity of the first arm joint0 hinge angular velocity (rad/s)
    True,  # 7: angular velocity of the second arm joint1 hinge angular velocity (rad/s)
    False, # 8: x-value of position_fingertip - position_target NA slide position (m)
    False, # 9: y-value of position_fingertip - position_target NA slide position (m)
    False, # 10: (v4 and old) z-value of position_fingertip - position_target NA slide position (m)
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


WALKER = [
    False, # 0: z-coordinate of the torso (height of Walker2d) rootz slide position (m)
    False, # 1: angle of the torso rooty hinge angle (rad)
    True,  # 2: angle of the thigh joint thigh_joint hinge angle (rad)
    True,  # 3: angle of the leg joint leg_joint hinge angle (rad)
    True,  # 4: angle of the foot joint foot_joint hinge angle (rad)
    True,  # 5: angle of the left thigh joint thigh_left_joint hinge angle (rad)
    True,  # 6: angle of the left leg joint leg_left_joint hinge angle (rad)
    True,  # 7: angle of the left foot joint foot_left_joint hinge angle (rad)
    False, # 8: velocity of the x-coordinate of the torso rootx slide velocity (m/s)
    False, # 9: velocity of the z-coordinate (height) of the torso rootz slide velocity (m/s)
    False, # 10: angular velocity of the angle of the torso rooty hinge angular velocity (rad/s)
    True,  # 11: angular velocity of the thigh hinge thigh_joint hinge angular velocity (rad/s)
    True,  # 12: angular velocity of the leg hinge leg_joint hinge angular velocity (rad/s)
    True,  # 13: angular velocity of the foot hinge foot_joint hinge angular velocity (rad/s)
    True,  # 14: angular velocity of the thigh hinge thigh_left_joint hinge angular velocity (rad/s)
    True,  # 15: angular velocity of the leg hinge leg_left_joint hinge angular velocity (rad/s)
    True,  # 16: angular velocity of the foot hinge foot_left_joint hinge angular velocity (rad/s)
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


PUSHER = [
    True,  # 0: Rotation of the panning the shoulder r_shoulder_pan_joint hinge angle (rad)
    True,  # 1: Rotation of the shoulder lifting joint r_shoulder_lift_joint hinge angle (rad)
    True,  # 2: Rotation of the shoulder rolling joint r_upper_arm_roll_joint hinge angle (rad)
    True,  # 3: Rotation of hinge joint that flexed the elbow r_elbow_flex_joint hinge angle (rad)
    True,  # 4: Rotation of hinge that rolls the forearm r_forearm_roll_joint hinge angle (rad)
    True,  # 5: Rotation of flexing the wrist r_wrist_flex_joint hinge angle (rad)
    True,  # 6: Rotation of rolling the wrist r_wrist_roll_joint hinge angle (rad)
    True,  # 7: Rotational velocity of the panning the shoulder r_shoulder_pan_joint hinge angular velocity (rad/s)
    True,  # 8: Rotational velocity of the shoulder lifting joint r_shoulder_lift_joint hinge angular velocity (rad/s)
    True,  # 9: Rotational velocity of the shoulder rolling joint r_upper_arm_roll_joint hinge angular velocity (rad/s)
    True,  # 10: Rotational velocity of hinge joint that flexed the elbow r_elbow_flex_joint hinge angular velocity (rad/s)
    True,  # 11: Rotational velocity of hinge that rolls the forearm r_forearm_roll_joint hinge angular velocity (rad/s)
    True,  # 12: Rotational velocity of flexing the wrist r_wrist_flex_joint hinge angular velocity (rad/s)
    True,  # 13: Rotational velocity of rolling the wrist r_wrist_roll_joint hinge angular velocity (rad/s)
    False, # 14: x-coordinate of the fingertip of the pusher tips_arm slide position (m)
    False, # 15: y-coordinate of the fingertip of the pusher tips_arm slide position (m)
    False, # 16: z-coordinate of the fingertip of the pusher tips_arm slide position (m)
    False, # 17: x-coordinate of the object to be moved object (obj_slidex) slide position (m)
    False, # 18: y-coordinate of the object to be moved object (obj_slidey) slide position (m)
    False, # 19: z-coordinate of the object to be moved object cylinder position (m)
    False, # 20: x-coordinate of the goal position of the object goal (goal_slidex) slide position (m)
    False, # 21: y-coordinate of the goal position of the object goal (goal_slidey) slide position (m)
    False, # 22: z-coordinate of the goal position of the object goal sphere position (m)
]


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

HUMANOID = [
    False, # 0: z-coordinate of the torso (centre) free position (m) 
    False, # 1: w-orientation of the torso (centre) free angle (rad) 
    False, # 2: x-orientation of the torso (centre) free angle (rad) 
    False, # 3: y-orientation of the torso (centre) free angle (rad) 
    False, # 4: z-orientation of the torso (centre) free angle (rad) 
    True,  # 5: z-angle of the abdomen (in lower_waist) abdomen_z hinge angle (rad) 
    True,  # 6: y-angle of the abdomen (in lower_waist) abdomen_y hinge angle (rad) 
    True,  # 7: x-angle of the abdomen (in pelvis) abdomen_x hinge angle (rad) 
    True,  # 8: x-coordinate of angle between pelvis and right hip (in right_thigh) right_hip_x hinge angle (rad) 
    True,  # 9: z-coordinate of angle between pelvis and right hip (in right_thigh) right_hip_z hinge angle (rad) 
    True,  # 10: y-coordinate of angle between pelvis and right hip (in right_thigh) right_hip_y hinge angle (rad) 
    True,  # 11 angle between right hip and the right shin (in right_knee) right_knee hinge angle (rad) 
    True,  # 12 x-coordinate of angle between pelvis and left hip (in left_thigh) left_hip_x hinge angle (rad) 
    True,  # 13 z-coordinate of angle between pelvis and left hip (in left_thigh) left_hip_z hinge angle (rad) 
    True,  # 14 y-coordinate of angle between pelvis and left hip (in left_thigh) left_hip_y hinge angle (rad) 
    True,  # 15 angle between left hip and the left shin (in left_knee) left_knee hinge angle (rad) 
    True,  # 16 coordinate-1 (multi-axis) angle between torso and right arm (in right_upper_arm) right_shoulder1 hinge angle (rad) 
    True,  # 17 coordinate-2 (multi-axis) angle between torso and right arm (in right_upper_arm) right_shoulder2 hinge angle (rad) 
    True,  # 18 angle between right upper arm and right_lower_arm right_elbow hinge angle (rad) 
    True,  # 19 coordinate-1 (multi-axis) angle between torso and left arm (in left_upper_arm) left_shoulder1 hinge angle (rad) 
    True,  # 20 coordinate-2 (multi-axis) angle between torso and left arm (in left_upper_arm) left_shoulder2 hinge angle (rad) 
    True,  # 21 angle between left upper arm and left_lower_arm left_elbow hinge angle (rad) 
    False, # 22 x-coordinate velocity of the torso (centre) root free velocity (m/s) 
    False, # 23 y-coordinate velocity of the torso (centre) root free velocity (m/s) 
    False, # 24 z-coordinate velocity of the torso (centre) root free velocity (m/s) 
    False, # 25 x-coordinate angular velocity of the torso (centre) root free angular velocity (rad/s) 
    False, # 26 y-coordinate angular velocity of the torso (centre) root free angular velocity (rad/s) 
    False, # 27 z-coordinate angular velocity of the torso (centre) root free angular velocity (rad/s) 
    True,  # 28 z-coordinate of angular velocity of the abdomen (in lower_waist) abdomen_z hinge angular velocity (rad/s) 
    True,  # 29 y-coordinate of angular velocity of the abdomen (in lower_waist) abdomen_y hinge angular velocity (rad/s) 
    True,  # 30 x-coordinate of angular velocity of the abdomen (in pelvis) abdomen_x hinge angular velocity (rad/s) 
    True,  # 31 x-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh) right_hip_x hinge angular velocity (rad/s) 
    True,  # 32 z-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh) right_hip_z hinge angular velocity (rad/s) 
    True,  # 33 y-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh) right_hip_y hinge angular velocity (rad/s) 
    True,  # 34 angular velocity of the angle between right hip and the right shin (in right_knee) right_knee hinge angular velocity (rad/s) 
    True,  # 35 x-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh) left_hip_x hinge angular velocity (rad/s) 
    True,  # 36 z-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh) left_hip_z hinge angular velocity (rad/s) 
    True,  # 37 y-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh) left_hip_y hinge angular velocity (rad/s) 
    True,  # 38 angular velocity of the angle between left hip and the left shin (in left_knee) left_knee hinge angular velocity (rad/s) 
    True,  # 39 coordinate-1 (multi-axis) of the angular velocity of the angle between torso and right arm (in right_upper_arm) right_shoulder1 hinge angular velocity (rad/s) 
    True,  # 40 coordinate-2 (multi-axis) of the angular velocity of the angle between torso and right arm (in right_upper_arm) right_shoulder2 hinge angular velocity (rad/s) 
    True,  # 41 angular velocity of the angle between right upper arm and right_lower_arm right_elbow hinge angular velocity (rad/s) 
    True,  # 42 coordinate-1 (multi-axis) of the angular velocity of the angle between torso and left arm (in left_upper_arm) left_shoulder1 hinge angular velocity (rad/s) 
    True,  # 43 coordinate-2 (multi-axis) of the angular velocity of the angle between torso and left arm (in left_upper_arm) left_shoulder2 hinge angular velocity (rad/s) 
    True,  # 44 angular velocity of the angle between left upper arm and left_lower_arm left_elbow hinge angular velocity (rad/s) 
]
# Humanoid also adds cinert: Mass and inertia of a single rigid body relative to the center of mass 
HUMANOID += [True] * 140
# Humanoid also adds cvel: Center of mass based velocity
HUMANOID += [False] * 84
# Humanoid also adds qfrc_actuator: Constraint force generated as the actuator force
HUMANOID += [True] * 23
# Humanoid also adds cfrc_ext: This is the center of mass based external force on the body
HUMANOID += [False] * 84
