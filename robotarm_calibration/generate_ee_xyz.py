import sys
sys.path.append('../')

import numpy as np
import pybullet_data as pd
import math
import matplotlib.pyplot as plt
from utils import *
import socket
import cv2
import torch
from tqdm import tqdm

from ENV.robot.KnollingRobot import knolling_robot
from ENV.task.CarlibrationEnv import carlibration_env

torch.manual_seed(42)

class calibration_main():

    def __init__(self, para_dict=None, generate_dict=None):

        self.generate_dict = generate_dict
        self.para_dict = para_dict
        self.task = carlibration_env(para_dict=para_dict)
        self.robot = knolling_robot(para_dict=para_dict)

    def reset(self, epoch=None):

        p.resetSimulation()
        self.task.create_scene()
        self.arm_id = self.robot.create_arm()
        # create the gripper mapping from sim to real
        self.robot.calculate_gripper()
        self.conn, self.real_table_height, self.sim_table_height = self.robot.arm_setup()

        self.img_per_epoch = 0

    def cali_motion(self):

        # plt.figure(figsize=(8, 6), dpi=80)
        if self.generate_dict['real_time_flag'] == True:
            plt.ion()
            plt.figure(figsize=(14, 8))

        rest_ori = np.array([0, np.pi / 2, 0])
        last_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
        last_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))

        trajectory_pos_list = np.array([[0.00, 0.14, 0.03],
                                        [0.25, 0.14, 0.03],
                                        [0.25, -0.14, 0.03],
                                        [0.00, -0.14, 0.03]])
        for j in tqdm(range(len(trajectory_pos_list))):

            if self.generate_dict['use_RL_dynamics'] == True:
                if len(trajectory_pos_list[j]) == 3:
                    last_pos = RL_dynamics(last_pos, last_ori, trajectory_pos_list[j], rest_ori)
                    if trajectory_pos_list[j, 2] > 0.01:
                        time.sleep(2)
                        pass
                    else:
                        time.sleep(2)
                        pass
                    last_ori = np.copy(rest_ori)
                elif len(trajectory_pos_list[j]) == 2:
                    self.robot.gripper(trajectory_pos_list[j][0], trajectory_pos_list[j][1])
                    # time.sleep(5)
            else:
                if len(trajectory_pos_list[j]) == 3:
                    last_pos = self.robot.move(last_pos, last_ori, trajectory_pos_list[j], rest_ori)
                    if trajectory_pos_list[j, 2] > 0.01:
                        time.sleep(5)
                        pass
                    else:
                        time.sleep(5)
                        pass
                    last_ori = np.copy(rest_ori)
                elif len(trajectory_pos_list[j]) == 2:
                    self.robot.gripper(trajectory_pos_list[j][0], trajectory_pos_list[j][1])
                    # time.sleep(5)

        if self.generate_dict['real_time_flag'] == True:
            plt.ioff()
            plt.show(block=False)
            plt.savefig(self.para_dict['dataset_path'] + 'motor_analysis.png')

        # back to the reset pos and ori
        ik_angles0 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=[0, 0, 0.06],
                                                  maxNumIterations=200,
                                                  targetOrientation=p.getQuaternionFromEuler(
                                                      [0, math.pi / 2, 0]))
        for motor_index in range(5):
            p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                    targetPosition=ik_angles0[motor_index], maxVelocity=7)
        for i in range(80):
            p.stepSimulation()
            # self.images = self.get_image()
            # time.sleep(1 / 120)

    def step(self):

        self.reset()

        if self.generate_dict['erase_flag'] == True:
            with open(file=self.para_dict['dataset_path'] + "cmd_xyz_nn.txt", mode="a", encoding="utf-8") as f:
                f.truncate(0)
            with open(file=self.para_dict['dataset_path'] + "real_xyz_nn.txt", mode="a", encoding="utf-8") as f:
                f.truncate(0)
            with open(file=self.para_dict['dataset_path'] + "tar_xyz_nn.txt", mode="a", encoding="utf-8") as f:
                f.truncate(0)
            with open(file=self.para_dict['dataset_path'] + "error_xyz_nn.txt", mode="a", encoding="utf-8") as f:
                f.truncate(0)
            with open(file=self.para_dict['dataset_path'] + "cmd_nn.txt", mode="a", encoding="utf-8") as f:
                f.truncate(0)
            with open(file=self.para_dict['dataset_path'] + "real_nn.txt", mode="a", encoding="utf-8") as f:
                f.truncate(0)
            with open(file=self.para_dict['dataset_path'] + "tar_nn.txt", mode="a", encoding="utf-8") as f:
                f.truncate(0)
            with open(file=self.para_dict['dataset_path'] + "error_nn.txt", mode="a", encoding="utf-8") as f:
                f.truncate(0)

        self.cali_motion()

        if self.para_dict['real_operate'] == True:
            end = np.array([0], dtype=np.float32)
            self.conn.sendall(end.tobytes())
        print(f'pipeline over!!!!!')

if __name__ == '__main__':

    para_dict = {'arm_reset_pos': np.array([0, 0, 0.12]), 'arm_reset_ori': np.array([0, np.pi / 2, 0]),
                 'save_img_flag': True,
                 'init_pos_range': [[0.03, 0.27], [-0.13, 0.13], [0.01, 0.02]], 'init_offset_range': [[-0.05, 0.05], [-0.1, 0.1]],
                 'init_ori_range': [[-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4]],
                 'boxes_num': np.random.randint(5, 6),
                 'is_render': True,
                 'gripper_threshold': 0.002, 'gripper_sim_step': 10, 'gripper_force': 3,
                 'move_threshold': 0.001, 'move_force': 3,
                 'gripper_lateral_friction': 1, 'gripper_contact_damping': 1, 'gripper_contact_stiffness': 50000,
                 'box_lateral_friction': 1, 'box_contact_damping': 1, 'box_contact_stiffness': 50000,
                 'base_lateral_friction': 1, 'base_contact_damping': 1, 'base_contact_stiffness': 50000,
                 'dataset_path': '../../knolling_dataset/data_903/',
                 'urdf_path': '../ASSET/urdf/',
                 'yolo_model_path': './627_pile_pose/weights/best.pt',
                 'real_operate': True}

    generate_dict = {'real_time_flag': False, 'erase_flag': False, 'collect_num': 50, 'max_plot_num': 250,
                     'x_range': [0.05, 0.25], 'y_range': [-0.13, 0.13], 'z_range':[0.02, 0.05], 'use_tuning': True,
                     'use_RL_dynamics': False}


    env = calibration_main(para_dict=para_dict, generate_dict=generate_dict)
    evaluation = 1
    env.step()