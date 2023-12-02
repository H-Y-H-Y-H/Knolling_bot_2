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


    def reset(self, epoch=None):

        p.resetSimulation()

        if random.uniform(0, 1) > 0.5:
            p.configureDebugVisualizer(lightPosition=[np.random.randint(1, 2), np.random.uniform(0, 1.5), 2],
                                       shadowMapResolution=8192, shadowMapIntensity=np.random.randint(0, 1) / 10)
        else:
            p.configureDebugVisualizer(lightPosition=[np.random.randint(1, 2), np.random.uniform(-1.5, 0), 2],
                                       shadowMapResolution=8192, shadowMapIntensity=np.random.randint(0, 1) / 10)
        baseid = p.loadURDF(self.urdf_path + "plane_zzz.urdf", useMaximalCoordinates=True)

        ######################################### Draw workspace lines ####################################3
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs - self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_high_obs + self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs - self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_low_obs - self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs + self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_high_obs + self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs + self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_low_obs - self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs])
        ######################################### Draw workspace lines ####################################3

        ######################################## Texture change ########################################
        background = np.random.randint(1, 5)
        textureId = p.loadTexture(self.urdf_path + f"img_{background}.png")
        p.changeVisualShape(baseid, -1, textureUniqueId=textureId, specularColor=[0, 0, 0])

        self.arm_id = p.loadURDF(os.path.join(self.urdf_path, "robot_arm928/robot_arm1_backup.urdf"),
                                 basePosition=[-0.08, 0, 0.02], useFixedBase=True,
                                 flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        ######################################## Texture change ########################################

        #################################### Gripper dynamic change #########################################
        p.changeDynamics(self.arm_id, 7, lateralFriction=self.para_dict['gripper_lateral_friction'],
                                         contactDamping=self.para_dict['gripper_contact_damping'],
                                         contactStiffness=self.para_dict['gripper_contact_stiffness'])
        p.changeDynamics(self.arm_id, 8, lateralFriction=self.para_dict['gripper_lateral_friction'],
                                         contactDamping=self.para_dict['gripper_contact_damping'],
                                         contactStiffness=self.para_dict['gripper_contact_stiffness'])
        #################################### Gripper dynamic change #########################################

        ####################### gripper to the origin position ########################
        ik_angles0 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=self.para_dict['reset_pos'],
                                                  maxNumIterations=200,
                                                  targetOrientation=p.getQuaternionFromEuler(self.para_dict['reset_ori']))
        for motor_index in range(5):
            p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                    targetPosition=ik_angles0[motor_index], maxVelocity=20)
        for _ in range(int(100)):
            # time.sleep(1/480)
            p.stepSimulation()
        ####################### gripper to the origin position ########################

        p.setGravity(0, 0, -10)
        p.changeDynamics(baseid, -1, lateralFriction=self.para_dict['base_lateral_friction'],
                                     contactDamping=self.para_dict['base_contact_damping'],
                                     contactStiffness=self.para_dict['base_contact_stiffness'])

        self.img_per_epoch = 0

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

            if self.generate_dict['real_time_flag'] == True:

                self.plt_cmd_xyz = np.concatenate((self.plt_cmd_xyz, cmd_xyz), axis=0)
                self.plt_real_xyz = np.concatenate((self.plt_real_xyz, real_xyz), axis=0)
                self.plt_tar_xyz = np.concatenate((self.plt_tar_xyz, tar_xyz), axis=0)
                self.plt_cmd_motor = np.concatenate((self.plt_cmd_motor, cmd_motor), axis=0)
                self.plt_real_motor = np.concatenate((self.plt_real_motor, real_motor), axis=0)
                self.plt_tar_motor = np.concatenate((self.plt_tar_motor, tar_motor), axis=0)

                if len(self.plt_cmd_xyz) > self.generate_dict['max_plot_num']:
                    self.plt_cmd_xyz = np.delete(self.plt_cmd_xyz, np.arange(len(cmd_xyz)), axis=0)
                    self.plt_real_xyz = np.delete(self.plt_real_xyz, np.arange(len(real_xyz)), axis=0)
                    self.plt_tar_xyz = np.delete(self.plt_tar_xyz, np.arange(len(tar_xyz)), axis=0)
                    self.plt_cmd_motor = np.delete(self.plt_cmd_motor, np.arange(len(cmd_motor)), axis=0)
                    self.plt_real_motor = np.delete(self.plt_real_motor, np.arange(len(real_motor)), axis=0)
                    self.plt_tar_motor = np.delete(self.plt_tar_motor, np.arange(len(tar_motor)), axis=0)
                x = np.arange(len(self.plt_cmd_xyz))

                plt.clf()

                plt.subplot(2, 3, 1)
                plt.title("Motor 1")
                plt.grid(True)
                plt.xlabel('X-axis')
                plt.ylabel('Angle')
                plt.plot(x, self.plt_cmd_motor[:, 0], label='cmd')
                plt.plot(x, self.plt_tar_motor[:, 0], label='tar')
                plt.plot(x, self.plt_real_motor[:, 0], label='real')
                plt.legend()

                plt.subplot(2, 3, 2)
                plt.title("Motor 2")
                plt.grid(True)
                plt.xlabel('X-axis')
                plt.ylabel('Angle')
                plt.plot(x, self.plt_cmd_motor[:, 1], label='cmd')
                plt.plot(x, self.plt_tar_motor[:, 1], label='tar')
                plt.plot(x, self.plt_real_motor[:, 1], label='real')
                plt.legend()

                plt.subplot(2, 3, 3)
                plt.title("Motor 3")
                plt.grid(True)
                plt.xlabel('X-axis')
                plt.ylabel('Angle')
                plt.plot(x, self.plt_cmd_motor[:, 2], label='cmd')
                plt.plot(x, self.plt_tar_motor[:, 2], label='tar')
                plt.plot(x, self.plt_real_motor[:, 2], label='real')
                plt.legend()

                plt.subplot(2, 3, 4)
                plt.title("Motor 4")
                plt.grid(True)
                plt.xlabel('X-axis')
                plt.ylabel('Angle')
                plt.plot(x, self.plt_cmd_motor[:, 3], label='cmd')
                plt.plot(x, self.plt_tar_motor[:, 3], label='tar')
                plt.plot(x, self.plt_real_motor[:, 3], label='real')
                plt.legend()

                plt.subplot(2, 3, 5)
                plt.title("Motor 5")
                plt.grid(True)
                plt.xlabel('X-axis')
                plt.ylabel('Angle')
                plt.plot(x, self.plt_cmd_motor[:, 4], label='cmd')
                plt.plot(x, self.plt_tar_motor[:, 4], label='tar')
                plt.plot(x, self.plt_real_motor[:, 4], label='real')
                plt.legend()

                plt.suptitle('Motor Data Analysis')
                plt.pause(1)

            if self.para_dict['Data_collection'] == True:
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

        def RL_dynamics(cur_pos, cur_ori, tar_pos, tar_ori):

            last_pos = move(cur_pos, cur_ori, tar_pos, tar_ori)

            return last_pos

        def gripper(gap, obj_width=None):
            obj_width += 0.006
            # obj_width_range = np.array([0.021, 0.026, 0.032, 0.039, 0.045, 0.052, 0.057])
            # motor_pos_range = np.array([2000, 2100, 2200, 2300, 2400, 2500, 2600])
            obj_width_range = np.array([0.021, 0.026, 0.032, 0.039, 0.043, 0.046, 0.050])
            motor_pos_range = np.array([2000, 2100, 2200, 2300, 2400, 2500, 2600])
            formula_parameters = np.polyfit(obj_width_range, motor_pos_range, 3)
            motor_pos = np.poly1d(formula_parameters)

            if self.real_operate == True:
                if gap > 0.0265: # close
                    pos_real = np.asarray([[gap, 1600]], dtype=np.float32)
                elif gap <= 0.0265: # open
                    pos_real = np.asarray([[gap, motor_pos(obj_width)]], dtype=np.float32)
                print('gripper', pos_real)
                conn.sendall(pos_real.tobytes())
                # print(f'this is the cmd pos {pos_real}')
                p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL, targetPosition=gap, force=10)
                p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL, targetPosition=gap, force=10)

                real_pos = conn.recv(8192)
                # test_real_pos = np.frombuffer(real_pos, dtype=np.float32)
                real_pos = np.frombuffer(real_pos, dtype=np.float32)
                # print('this is test float from buffer', test_real_pos)

            else:
                p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL, targetPosition=gap, force=10)
                p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL, targetPosition=gap, force=10)
            for i in range(30):
                p.stepSimulation()
                # time.sleep(1 / 120)

        def knolling():

            # plt.figure(figsize=(8, 6), dpi=80)
            if self.generate_dict['real_time_flag'] == True:
                plt.ion()
                plt.figure(figsize=(14, 8))

            rest_ori = np.array([0, np.pi / 2, 0])
            last_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
            last_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))

            if self.para_dict['Data_collection'] == True:

                trajectory_pos_list = np.array([[0.00, 0.14, 0.04],
                                                [0.25, 0.14, 0.04],
                                                [0.25, -0.14, 0.04],
                                                [0.00, -0.14, 0.04]])
                # pos_x = np.random.uniform(self.generate_dict['x_range'][0], self.generate_dict['x_range'][1], self.generate_dict['collect_num'])
                # pos_y = np.random.uniform(self.generate_dict['y_range'][0], self.generate_dict['y_range'][1], self.generate_dict['collect_num'])
                # pos_z = np.random.uniform(self.generate_dict['z_range'][0], self.generate_dict['z_range'][1], self.generate_dict['collect_num'])
                # trajectory_pos_list = np.concatenate((pos_x.reshape(-1, 1), pos_y.reshape(-1, 1), pos_z.reshape(-1, 1)), axis=1)
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
                            gripper(trajectory_pos_list[j][0], trajectory_pos_list[j][1])
                            # time.sleep(5)
                    else:
                        if len(trajectory_pos_list[j]) == 3:
                            last_pos = move(last_pos, last_ori, trajectory_pos_list[j], rest_ori)
                            if trajectory_pos_list[j, 2] > 0.01:
                                # time.sleep(2)
                                pass
                            else:
                                # time.sleep(2)
                                pass
                            last_ori = np.copy(rest_ori)
                        elif len(trajectory_pos_list[j]) == 2:
                            gripper(trajectory_pos_list[j][0], trajectory_pos_list[j][1])
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

        if order == 3:
            knolling()

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

        if self.para_dict['real_operate'] == True:

            os.makedirs((self.para_dict['dataset_path']), exist_ok=True)

            HOST = "192.168.0.188"  # Standard loopback interface address (localhost)
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
            num_motor = 5
            # ! reset the pos in both real and sim
            reset_pos = np.array([0.015, 0, 0.1])
            reset_ori = np.array([0, np.pi / 2, 0])
            cmd_motor = np.asarray(inverse_kinematic(np.copy(reset_pos), np.copy(reset_ori)), dtype=np.float32)
            print('this is the reset motor pos', cmd_motor)
            conn.sendall(cmd_motor.tobytes())

            real_motor = conn.recv(8192)
            # print('received')
            real_motor = np.frombuffer(real_motor, dtype=np.float32)
            real_motor = real_motor.reshape(-1, 6)
            real_xyz, _ = forward_kinematic(real_motor)

            self.plt_cmd_xyz = reset_pos.reshape(-1, 3)
            self.plt_real_xyz = real_xyz
            self.plt_cmd_motor = cmd_motor.reshape(-1, 6)
            self.plt_real_motor = real_motor
            self.plt_tar_motor = np.copy(self.plt_cmd_motor)
            self.plt_tar_xyz = np.copy(self.plt_cmd_xyz)
            for _ in range(200):
                p.stepSimulation()
                # time.sleep(1 / 48)
        else:
            conn = None
            table_surface_height = 0.032
            sim_table_surface_height = 0

        #######################################################################################
        # self.planning(1, conn, table_surface_height, sim_table_surface_height, evaluation)
        # error = self.planning(5, conn, table_surface_height, sim_table_surface_height, evaluation)
        # self.planning(2, conn, table_surface_height, sim_table_surface_height, evaluation)
        error = self.planning(3, conn, table_surface_height, sim_table_surface_height, evaluation)
        # self.planning(4, conn, table_surface_height, sim_table_surface_height, evaluation)
        #######################################################################################

        if self.para_dict['real_operate'] == True:
            end = np.array([0], dtype=np.float32)
            conn.sendall(end.tobytes())
        print(f'evaluation {evaluation} over!!!!!')

if __name__ == '__main__':

    para_dict = {'start_num': 0, 'end_num': 10, 'thread': 9, 'evaluations': 1,
                 'reset_pos': np.array([0, 0, 0.12]), 'reset_ori': np.array([0, np.pi / 2, 0]),
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
                 'dataset_path': '../../knolling_dataset/data_903/',
                 'urdf_path': '../ASSET/urdf/',
                 'yolo_model_path': './627_pile_pose/weights/best.pt',
                 'real_operate': True, 'obs_order': 'real_image_obj', 'Data_collection': True,
                 'use_knolling_model': False, 'use_lstm_model': False, 'use_yolo_model': False}

    generate_dict = {'real_time_flag': False, 'erase_flag': True, 'collect_num': 50, 'max_plot_num': 250,
                     'x_range': [0.05, 0.25], 'y_range': [-0.13, 0.13], 'z_range':[0.02, 0.05], 'use_tuning': True,
                     'use_RL_dynamics': True}


    env = calibration_main(para_dict=para_dict, generate_dict=generate_dict)
    evaluation = 1
    for evaluation in range(para_dict['evaluations']):
        env.step()