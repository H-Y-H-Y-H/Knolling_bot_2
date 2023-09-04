import sys
sys.path.append('../')

import numpy as np
# import pyrealsense2 as rs
import pybullet_data as pd
import math
import matplotlib.pyplot as plt
from utils import *
from environment import Arm_env
import socket
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

torch.manual_seed(42)

class calibration_main(Arm_env):

    def __init__(self, para_dict=None, knolling_para=None, lstm_dict=None, generate_dict=None):
        super(calibration_main, self).__init__(para_dict=para_dict, knolling_para=knolling_para, lstm_dict=lstm_dict)
        self.generate_dict = generate_dict

    def planning(self, order, conn, real_height, sim_height, evaluation):
        def move(cur_pos, cur_ori, tar_pos, tar_ori):

            # real_height_offset = np.array([0, 0, real_height])
            send_data = np.concatenate((cur_pos, cur_ori, tar_pos, tar_ori), axis=0).reshape(-1, 3)
            send_data = send_data.astype(np.float32)

            conn.sendall(send_data.tobytes())

            receive_time = 0
            while True:
                buffer = np.frombuffer(conn.recv(8192), dtype=np.float32)
                if receive_time == 0:
                    data_length = int(buffer[0] / 4)
                    recall_data = buffer[1:]
                else:
                    recall_data = np.append(recall_data, buffer)
                if len(recall_data) < data_length:
                    print('continue to receive data')
                else:
                    break
                receive_time += 1
            recall_data = recall_data.reshape(-1 ,36)

            print('this is the shape of final angles real', recall_data.shape)
            cmd_xyz = recall_data[:, :3]
            real_xyz = recall_data[:, 3:6]
            tar_xyz = recall_data[:, 6:9]
            error_xyz = recall_data[:, 9:12]
            cmd_motor = recall_data[:, 12:18]
            real_motor = recall_data[:, 18:24]
            tar_motor = recall_data[:, 24:30]
            error_motor = recall_data[:, 30:]

            cur_pos = real_xyz[-1]
            print('this is cur pos after pid', cur_pos)

            if self.para_dict['data_collection'] == True:
                with open(file=self.para_dict['dataset_path'] + "cmd_xyz_nn.txt", mode="a", encoding="utf-8") as f:
                    np.savetxt(f, cmd_xyz)
                with open(file=self.para_dict['dataset_path'] + "real_xyz_nn.txt", mode="a", encoding="utf-8") as f:
                    np.savetxt(f, real_xyz)
                with open(file=self.para_dict['dataset_path'] + "tar_xyz_nn.txt", mode="a", encoding="utf-8") as f:
                    np.savetxt(f, tar_xyz)
                with open(file=self.para_dict['dataset_path'] + "error_xyz_nn.txt", mode="a", encoding="utf-8") as f:
                    np.savetxt(f, error_xyz)
                with open(file=self.para_dict['dataset_path'] + "cmd_nn.txt", mode="a", encoding="utf-8") as f:
                    np.savetxt(f, cmd_motor)
                with open(file=self.para_dict['dataset_path'] + "real_nn.txt", mode="a", encoding="utf-8") as f:
                    np.savetxt(f, real_motor)
                with open(file=self.para_dict['dataset_path'] + "tar_nn.txt", mode="a", encoding="utf-8") as f:
                    np.savetxt(f, tar_motor)
                with open(file=self.para_dict['dataset_path'] + "error_nn.txt", mode="a", encoding="utf-8") as f:
                    np.savetxt(f, error_motor)
                pass
            if self.generate_dict['use_tuning'] == True:
                print('this is cmd zzz\n', cmd_xyz[-1])
                return cmd_xyz[-1] # return cur pos to let the manipualtor remember the improved pos
            else:
                return tar_pos

        def gripper(gap, obj_width=None):
            obj_width += 0.006
            obj_width_range = np.array([0.021, 0.026, 0.032, 0.039, 0.043, 0.046, 0.050])
            motor_pos_range = np.array([2000, 2100, 2200, 2300, 2400, 2500, 2600])
            formula_parameters = np.polyfit(obj_width_range, motor_pos_range, 3)
            motor_pos = np.poly1d(formula_parameters)

            if gap > 0.5: # close
                pos_real = np.asarray([[gap, 1600]], dtype=np.float32)
            elif gap <= 0.5: # open
                pos_real = np.asarray([[gap, motor_pos(obj_width)]], dtype=np.float32)
            print('gripper', pos_real)
            conn.sendall(pos_real.tobytes())

            real_pos = conn.recv(8192)
            # test_real_pos = np.frombuffer(real_pos, dtype=np.float32)
            real_pos = np.frombuffer(real_pos, dtype=np.float32)
            # print('this is test float from buffer', test_real_pos)

        def knolling():

            rest_ori = np.array([0, np.pi / 2, 0])
            last_pos = self.para_dict['reset_pos']
            last_ori = self.para_dict['reset_ori']

            if self.para_dict['data_collection'] == True:

                trajectory_pos_list = [[0.00, 0.14, 0.04],
                                       [1, 0],
                                        [0.25, 0.14, 0.04],
                                        [0.25, -0.14, 0.04],
                                        [0.00, -0.14, 0.04]]
                trajectory_ori_list = np.repeat([self.para_dict['reset_ori']], len(trajectory_pos_list), axis=0)
                for j in tqdm(range(len(trajectory_pos_list))):
                    if len(trajectory_pos_list[j]) == 3:
                            last_pos = move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j])
                            print('this is pos', trajectory_pos_list[j])
                            if trajectory_pos_list[j][2] > 0.01:
                                time.sleep(2)
                                pass
                            else:
                                time.sleep(2)
                                pass
                            last_ori = np.copy(rest_ori)
                    elif len(trajectory_pos_list[j]) == 2:
                        gripper(trajectory_pos_list[j][0], trajectory_pos_list[j][1])
                        # time.sleep(5)

        if order == 3:
            knolling()

    def step(self):

        if self.para_dict['real_operate'] == True:

            os.makedirs((self.para_dict['dataset_path']), exist_ok=True)

            HOST = "192.168.0.186"  # Standard loopback interface address (localhost)
            PORT = 8880 # Port to listen on (non-privileged ports are > 1023)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((HOST, PORT))
            # It should be an integer from 1 to 65535, as 0 is reserved. Some systems may require superuser privileges if the port number is less than 8192.
            # associate the socket with a specific network interface
            s.listen()
            print(f"Waiting for connection...\n")
            conn, addr = s.accept()
            print(conn)
            print(f"Connected by {addr}")
            table_surface_height = 0.003
            sim_table_surface_height = 0

            # cmd_motor = np.asarray(inverse_kinematic(np.copy(self.para_dict['reset_pos']), np.copy(self.para_dict['reset_ori'])), dtype=np.float32)
            cmd_motor = np.asarray(inverse_kinematic(np.copy(self.para_dict['reset_pos']), np.copy(self.para_dict['reset_ori'])), dtype=np.float32)

            print('this is the reset motor pos', cmd_motor)
            conn.sendall(cmd_motor.tobytes())

            real_motor = conn.recv(8192)
            real_motor = np.frombuffer(real_motor, dtype=np.float32)
            real_motor = real_motor.reshape(-1, 6)
            real_xyz, _ = forward_kinematic(real_motor)
        else:
            conn = None
            # table_surface_height = 0.032
            sim_table_surface_height = 0

        self.planning(3, conn, table_surface_height, sim_table_surface_height, evaluation)


        if self.para_dict['real_operate'] == True:
            end = np.array([0], dtype=np.float32)
            conn.sendall(end.tobytes())
        print(f'evaluation {evaluation} over!!!!!')

if __name__ == '__main__':

    para_dict = {'start_num': 0, 'end_num': 10, 'thread': 9, 'evaluations': 1,
                 'reset_pos': np.array([0, 0, 0.10]), 'reset_ori': np.array([0, np.pi / 2, 0]),
                 'save_img_flag': True,
                 'init_pos_range': [[0.03, 0.27], [-0.13, 0.13], [0.01, 0.02]], 'init_offset_range': [[-0.05, 0.05], [-0.1, 0.1]],
                 'init_ori_range': [[-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4]],
                 'boxes_num': np.random.randint(4, 6),
                 'is_render': False,
                 'box_range': [[0.016, 0.048], [0.016], [0.01, 0.02]],
                 'box_mass': 0.1,
                 'gripper_threshold': 0.002, 'gripper_sim_step': 10, 'gripper_force': 3,
                 'move_threshold': 0.001, 'move_force': 3,
                 'gripper_lateral_friction': 1, 'gripper_contact_damping': 1, 'gripper_contact_stiffness': 50000,
                 'box_lateral_friction': 1, 'box_contact_damping': 1, 'box_contact_stiffness': 50000,
                 'base_lateral_friction': 1, 'base_contact_damping': 1, 'base_contact_stiffness': 50000,
                 'dataset_path': '../../../knolling_dataset/data_903/',
                 'urdf_path': '../urdf/',
                 'yolo_model_path': './627_pile_pose/weights/best.pt',
                 'real_operate': True, 'obs_order': 'real_image_obj', 'data_collection': True,
                 'use_knolling_model': False, 'use_lstm_model': False}

    generate_dict = {'real_time_flag': False, 'erase_flag': True, 'collect_num': 50, 'max_plot_num': 250,
                     'x_range': [0.05, 0.25], 'y_range': [-0.13, 0.13], 'z_range':[0.02, 0.05], 'use_tuning': True,
                     'use_RL_dynamics': True}


    env = calibration_main(para_dict=para_dict, generate_dict=generate_dict)
    evaluation = 1
    for evaluation in range(para_dict['evaluations']):
        env.step()