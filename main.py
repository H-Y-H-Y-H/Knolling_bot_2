import numpy as np
import pyrealsense2 as rs
import pybullet_data as pd
import math
from function import *
from environment import Arm_env
import socket
import cv2
import torch
from urdfpy import URDF
from env import Env
import shutil

class knolling_main(Arm_env):

    def __init__(self, endnum=None, save_img_flag=None,
                 para_dict=None, init_pos_range=None, init_ori_range=None):
        super(knolling_main, self).__init__(endnum=endnum, save_img_flag=save_img_flag,
                                        para_dict=para_dict, init_pos_range=init_pos_range,
                                        init_ori_range=init_ori_range)

if __name__ == '__main__':

    torch.manual_seed(42)
    general_parameters = {'evaluations': 1,
                          'real_operate': False, 'obs_order': 'sim_image_obj',
                          'check_detection_loss': False,
                          'is_render': True, 'use_knolling_model': False,
                          'reset_pos': np.array([0, 0, 0.12]), 'reset_ori': np.array([0, np.pi / 2, 0]),
                          'urdf_path': '/home/zhizhuo/Creative_Machines_Lab/Knolling_bot_2/urdf/',
                          'yolo_conf': 0.6, 'yolo_iou': 0.8, 'device': 'cuda:0',
                          'img_save_path': './img_temp/', 'save_img_flag': True}

    dynamic_parameters = {'gripper_threshold': 0.004, 'gripper_force': 3, 'gripper_sim_step': 10,
                          'move_threshold': 0.005, 'move_force': 3,
                          'box_lateral_friction': 0.8, 'box_spinning_friction': 0.8, 'box_rolling_friction': 0.0,
                          'box_linear_damping': 0.01, 'box_angular_damping': 0.01, 'box_joint_damping': 0.1,
                          'box_restitution': 0, 'box_contact_damping': 100, 'box_contact_stiffness': 1000000,
                          'gripper_lateral_friction': 1, 'gripper_spinning_friction': 1,
                          'gripper_rolling_friction': 0.001,
                          'gripper_linear_damping': 1, 'gripper_angular_damping': 1, 'gripper_joint_damping': 1,
                          'gripper_restitution': 0, 'gripper_contact_damping': 100,
                          'gripper_contact_stiffness': 1000000,
                          'base_lateral_friction': 1, 'base_spinning_friction': 1, 'base_rolling_friction': 0,
                          'base_restitution': 0, 'base_contact_damping': 10, 'base_contact_stiffness': 1000000}

    env = knolling_main()
    knolling_env = Env(is_render=general_parameters['is_render'])

    evaluations = 1
    for evaluation in range(general_parameters['evaluations']):
        knolling_generate_parameters = {'total_offset': [0.035, -0.17 + 0.016, 0], 'gap_item': 0.015,
                                        'gap_block': 0.015, 'random_offset': False,
                                        'area_num': 2, 'ratio_num': 1,
                                        'boxes_num': np.random.randint(10, 11), 'reset_style': 'pile',
                                        'order_flag': 'confidence',
                                        'item_odd_prevent': True,
                                        'block_odd_prevent': True,
                                        'upper_left_max': True,
                                        'forced_rotate_box': False,
                                        'box_range': [[0.016, 0.048], [0.016], [0.01, 0.02]], 'box_mass': 0.1,
                                        'init_pos_range': [[0.13, 0.17], [-0.03, 0.03], [0.01, 0.02]]}
        env.get_parameters(evaluations=evaluation,
                           knolling_generate_parameters=knolling_generate_parameters,
                           dynamic_parameters=dynamic_parameters,
                           general_parameters=general_parameters,
                           knolling_env=knolling_env)
        env.step()