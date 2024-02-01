from utils import *
import pybullet as p
import pybullet_data as pd
import os
import numpy as np
import random
import time
import gymnasium as gym
from gymnasium import spaces
import cv2
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env

class rl_unstack_model(gym.Env):

    def __init__(self, rl_dict):

        self.rl_dict = rl_dict

        self.x_low_obs = 0.03
        self.x_high_obs = 0.27
        self.y_low_obs = -0.14
        self.y_high_obs = 0.14
        self.z_low_obs = 0.0
        self.z_high_obs = 0.05

        log_path = './RL_motion_model/' + f"logger/{rl_dict['rl_mode']}_{rl_dict['obj_num']}objobs/log{rl_dict['logger_id']}/"
        self.model = SAC.load(log_path + "/ppo_model_best.zip")
    def model_pred(self, obs):

        action, _states = self.model.predict(obs, deterministic=True)

        # from -1 1 to world coordinate
        action[0] = (action[0] + 1) / 2 * (self.x_high_obs - self.x_low_obs) + self.x_low_obs
        action[1] = (action[1] + 1) / 2 * (self.y_high_obs - self.y_low_obs) + self.y_low_obs

        arm_action = np.concatenate((action[:3], [0, np.pi / 2, action[3]]))

        return arm_action

if __name__ == '__main__':

    # np.random.seed(0)
    random.seed(0)
    #center of table: x = 0.12, y = 0
    para_dict = {'reset_pos': np.array([-0.9, 0, 0.005]), 'reset_ori': np.array([0, np.pi / 2, 0]),
                 'save_img_flag': True,
                 'init_pos_range': [[-0.05, 0.05], [-0.05, 0.05], [0.01, 0.02]], 'init_offset_range': [[0.14, 0.16], [-0.01, 0.01]],
                 'init_ori_range': [[-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4]],
                 'boxes_num': 5,
                 'boxes_num_max':5,
                 'is_render': True,
                 'box_range': [[0.016, 0.048], [0.016], [0.01, 0.02]],
                 'box_mass': 0.1,
                 'gripper_force': 3,
                 'move_force': 3,
                 'gripper_lateral_friction': 1, 'gripper_contact_damping': 1, 'gripper_contact_stiffness': 50000,
                 'box_lateral_friction': 1, 'box_contact_damping': 1, 'box_contact_stiffness': 50000,
                 'base_lateral_friction': 1, 'base_contact_damping': 1, 'base_contact_stiffness': 50000,
                 'dataset_path': './knolling_box/',
                 'urdf_path': './urdf/',}

    num_scene = 10
    os.makedirs(para_dict['dataset_path'], exist_ok=True)


    MODE = 0 # RL random or Manual

    if MODE == 0:
        env = Arm_env(para_dict=para_dict, init_scene=num_scene,two_obj_obs=False)

        for i in range(10000):
            # random sample
            action = env.action_space.sample()
            obs, r, done, _, _ = env.step(action)
            print("Obs:", obs, "Reward:", r, 'Action:', action)

            if done:
                env.reset(fix_num_obj=para_dict['boxes_num'])

    elif MODE == 1:
        env = Arm_env(para_dict=para_dict, init_scene=num_scene, offline_data=False, init_num_obj=para_dict['boxes_num'])

        n_samples = 10000
        count = 0

        while 1:
            # obj initialization:
            for k in env.boxes_index: p.removeBody(k)
            env.boxes_index = []
            info_obj = env.create_objects()

            # whether the objects are too closed to each other.
            dist = []
            # calculate the dist of each two objects
            for j in range(env.boxes_num - 1):
                for i in range(j + 1, env.boxes_num):
                    dist.append(np.sqrt(np.sum((info_obj[j][:2] - info_obj[i][:2]) ** 2)))
            dist = np.array(dist)
            if (dist < 0.035).any():
                print(f'successful scene generated {count}.')
                np.savetxt('../obj_init_dataset/%dobj_%d.csv' % (env.boxes_num, env.offline_scene_id), info_obj)
                env.offline_scene_id += 1
                count += 1
            if count == n_samples:
                break

    else:
        env = Arm_env(para_dict=para_dict, init_scene=num_scene, offline_data = False)

        for i in range(10000):
            # control robot arm manually:

            x_ml = p.readUserDebugParameter(env.x_manual_id)
            y_ml = p.readUserDebugParameter(env.y_manual_id)
            z_ml = p.readUserDebugParameter(env.z_manual_id)
            yaw_ml = p.readUserDebugParameter(env.yaw_manual_id)

            action = np.asarray([x_ml, y_ml, z_ml, yaw_ml])
            obs,r,done,_,_ = env.step(action)
            print(r)

