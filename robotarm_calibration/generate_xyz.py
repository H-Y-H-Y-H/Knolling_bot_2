import sys
sys.path.append('../')

import numpy as np
import pyrealsense2 as rs
import pybullet_data as pd
import math
import matplotlib.pyplot as plt
from function import *
from environment import Arm_env
import socket
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

torch.manual_seed(42)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(3, 12)
        self.fc2 = nn.Linear(12, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 12)
        self.fc6 = nn.Linear(12, 3)

    def forward(self, x):
        # define forward pass
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x

    def loss(self, pred, target):
        value = (pred - target) ** 2
        return torch.mean(value)

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

        def Cartesian_offset_nn(xyz_input):

            # input:(n, 3), output: (n, 3)

            input_sc = [[-0.01, -0.201, -0.01],
                        [0.30, 0.201, 0.0601]]
            output_sc = [[-0.01, -0.201, -0.01],
                         [0.30, 0.201, 0.0601]]
            input_sc = np.load('nn_data_xyz/all_distance_free_new/real_scale.npy')
            output_sc = np.load('nn_data_xyz/all_distance_free_new/cmd_scale.npy')

            scaler_output = MinMaxScaler()
            scaler_input = MinMaxScaler()
            scaler_output.fit(output_sc)
            scaler_input.fit(input_sc)

            model = Net().to(device)
            model.load_state_dict(torch.load("model_pt_xyz/all_distance_free_new.pt"))
            # print(model)
            model.eval()
            with torch.no_grad():
                xyz_input_scaled = scaler_input.transform(xyz_input).astype(np.float32)
                xyz_input_scaled = torch.from_numpy(xyz_input_scaled)
                xyz_input_scaled = xyz_input_scaled.to(device)
                pred_xyz = model.forward(xyz_input_scaled)
                # print(pred_angle)
                pred_xyz = pred_xyz.cpu().data.numpy()
                xyz_output = scaler_output.inverse_transform(pred_xyz)

            return xyz_output

        def move(cur_pos, cur_ori, tar_pos, tar_ori):

            if tar_ori[2] > 1.58:
                tar_ori[2] = tar_ori[2] - np.pi
            elif tar_ori[2] < -1.58:
                tar_ori[2] = tar_ori[2] + np.pi

            if self.generate_dict['use_tuning'] == True:

                pass

                d = np.array([0, 0.3])
                d_y = np.array((0, 0.17, 0.21, 0.30))
                d_y = d
                z_bias = np.array([-0.005, 0.01])
                x_bias = np.array([-0.002, 0.00])  # yolo error is +2mm along x axis!
                y_bias = np.array([0, -0.004, -0.001, 0.004])
                y_bias = np.array([0.002, 0.006])
                # z_parameters = np.polyfit(d, z_bias, 3)
                z_parameters = np.polyfit(d, z_bias, 1)
                x_parameters = np.polyfit(d, x_bias, 1)
                y_parameters = np.polyfit(d_y, y_bias, 1)
                new_z_formula = np.poly1d(z_parameters)
                new_x_formula = np.poly1d(x_parameters)
                new_y_formula = np.poly1d(y_parameters)

                # # automatically add z and x bias
                # # d = np.array([0, 0.10, 0.185, 0.225, 0.27])
                # d = np.array([0, 0.3])
                # d_y = np.array((0, 0.17, 0.30))
                # z_bias = np.array([-0.003, 0.008])
                # x_bias = np.array([-0.005, -0.001])
                # y_bias = np.array([0, -0.004, 0.004])
                # # y_bias = np.array([])
                # # z_parameters = np.polyfit(d, z_bias, 3)
                # z_parameters = np.polyfit(d, z_bias, 1)
                # x_parameters = np.polyfit(d, x_bias, 1)
                # y_parameters = np.polyfit(d_y, y_bias, 4)
                # new_z_formula = np.poly1d(z_parameters)
                # new_x_formula = np.poly1d(x_parameters)
                # new_y_formula = np.poly1d(y_parameters)

                distance = tar_pos[0]
                distance_y = tar_pos[0]
                tar_pos[2] = tar_pos[2] + new_z_formula(distance)
                print('this is z', new_z_formula(distance))
                # tar_pos[0] = tar_pos[0] + new_x_formula(distance)
                # print('this is x', new_x_formula(distance))
                # if tar_pos[1] > 0:
                #     tar_pos[1] += new_y_formula(distance_y) * np.clip((6 * (tar_pos[1] + 0.01)), 0, 1)
                # else:
                #     tar_pos[1] -= new_y_formula(distance_y) * np.clip((6 * (tar_pos[1] - 0.01)), 0, 1)
                # print('this is tar pos after manual', tar_pos)

                # distance = tar_pos[0]
                # tar_pos[2] = tar_pos[2] + new_z_formula(distance)
                # print('this is z', new_z_formula(distance))
                # tar_pos[0] = tar_pos[0] + new_x_formula(distance)
                # print('this is x', new_x_formula(distance))
                # # distance_y = np.linalg.norm(tar_pos[:2])
                # if tar_pos[1] > 0:
                #     distance_y = np.linalg.norm(tar_pos[:2])
                #     print('this is y', new_y_formula(distance_y))
                #     tar_pos[1] += new_y_formula(distance_y)
                # else:
                #     distance_y = np.linalg.norm(tar_pos[:2])
                #     print('this is y', new_y_formula(distance_y))
                #     tar_pos[1] -= new_y_formula(distance_y)
                # print('this is tar pos after manual', tar_pos)


            if abs(cur_pos[0] - tar_pos[0]) < 0.001 and abs(cur_pos[1] - tar_pos[1]) < 0.001:
                # vertical, choose a small slice
                move_slice = 0.004
            else:
                # horizontal, choose a large slice
                move_slice = 0.008

            # tar_pos = tar_pos + np.array([0, 0, real_height])
            tar_pos = tar_pos + np.array([0, 0, 0.01])
            target_pos = np.copy(tar_pos)
            target_ori = np.copy(tar_ori)

            vertical_flag = False
            print('this is tar pos', target_pos)
            print('this is cur pos', cur_pos)
            if np.abs(target_pos[2] - cur_pos[2]) > 0.01 \
                    and np.abs(target_pos[0] - cur_pos[0]) < 0.01 \
                    and np.abs(target_pos[1] - cur_pos[1]) < 0.01:
                print('we dont need feedback control')
                mark_ratio = 0.8
                vertical_flag = True
                seg_time = 0
            else:
                if self.generate_dict['use_tuning'] == True:
                    mark_ratio = 0.97
                    seg_time = 0
                else:
                    mark_ratio = 1
                    seg_time = -10

            cmd_motor = []
            # plot_real = []
            cmd_xyz = []
            sim_ori = []
            real_xyz = []

            # divide the whole trajectory into several segment
            seg_time += 1
            seg_pos = mark_ratio * (target_pos - cur_pos) + cur_pos
            seg_ori = mark_ratio * (target_ori - cur_ori) + cur_ori
            distance = np.linalg.norm(seg_pos - cur_pos)
            num_step = np.ceil(distance / move_slice)
            step_pos = (seg_pos - cur_pos) / num_step
            step_ori = (seg_ori - cur_ori) / num_step
            print('this is seg pos', seg_pos)

            while True:
                tar_pos = cur_pos + step_pos
                # print(tar_pos)
                tar_ori = cur_ori + step_ori
                cmd_xyz.append(tar_pos)
                sim_ori.append(tar_ori)

                break_flag = abs(seg_pos[0] - tar_pos[0]) < 0.001 and abs(
                    seg_pos[1] - tar_pos[1]) < 0.001 and abs(seg_pos[2] - tar_pos[2]) < 0.001 and \
                             abs(seg_ori[0] - tar_ori[0]) < 0.001 and abs(
                    seg_ori[1] - tar_ori[1]) < 0.001 and abs(seg_ori[2] - tar_ori[2]) < 0.001
                if break_flag:
                    break
                cur_pos = tar_pos
                cur_ori = tar_ori

            cmd_xyz = np.asarray(cmd_xyz)
            sim_ori = np.asarray(sim_ori)
            cmd_motor = np.asarray(inverse_kinematic(cmd_xyz, sim_ori), dtype=np.float32)
            plot_step = np.arange(num_step)

            print('this is the shape of cmd', cmd_motor.shape)
            print('this is the shape of xyz', cmd_xyz.shape)
            tar_motor = np.copy(cmd_motor)
            tar_xyz = np.copy(cmd_xyz)
            # print('this is the motor pos sent', cmd_motor[-1])
            conn.sendall(cmd_motor.tobytes())
            real_motor = conn.recv(4096)
            # print('received')
            real_motor = np.frombuffer(real_motor, dtype=np.float32)
            real_motor = real_motor.reshape(-1, 6)

            print('this is the shape of angles real', real_motor.shape)
            for i in range(len(real_motor)):
                ik_angles_real = np.asarray(cmd2rad(real_tarpos2cmd(real_motor[i])), dtype=np.float32)
                for motor_index in range(5):
                    p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                            targetPosition=ik_angles_real[motor_index], maxVelocity=100, force=100)
                for i in range(30):
                    p.stepSimulation()
                real_xyz = np.append(real_xyz, np.asarray(p.getLinkState(self.arm_id, 9)[0])).reshape(-1, 3)
            cur_pos = real_xyz[-1]

            if seg_time > 0:
                seg_flag = False
                print('segment fail, try to tune!')
                # ik_angles_sim = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=target_pos,
                #                                              maxNumIterations=500,
                #                                              targetOrientation=p.getQuaternionFromEuler(
                #                                                  target_ori))
                # for motor_index in range(5):
                #     p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                #                             targetPosition=ik_angles_sim[motor_index], maxVelocity=2.5)
                # for i in range(30):
                #     p.stepSimulation()
                #
                # angle_sim = np.asarray(real_cmd2tarpos(rad2cmd(ik_angles_sim[0:5])), dtype=np.float32)
                angle_sim = np.asarray(inverse_kinematic(np.copy(target_pos), np.copy(target_ori)), dtype=np.float32)
                # cmd_motor = np.append(cmd_motor, angle_sim).reshape(-1, 6)
                final_cmd = np.append(angle_sim, 0).reshape(1, -1)
                final_cmd = np.asarray(final_cmd, dtype=np.float32)
                print(final_cmd.shape)
                print(final_cmd)
                conn.sendall(final_cmd.tobytes())

                # get the pos after tune!
                tuning_data = conn.recv(4096)
                # print('received')
                tuning_data = np.frombuffer(tuning_data, dtype=np.float32).reshape(-1, 12)
                print('this is the shape of final angles real', tuning_data.shape)
                real_motor_tuning = tuning_data[:, 6:]
                cmd_motor_tuning = tuning_data[:, :6]
                tar_motor_tuning = np.repeat(final_cmd[:, :6], len(cmd_motor_tuning), axis=0)
                tar_xyz_tuning = np.repeat(target_pos.reshape(1, -1), len(cmd_motor_tuning), axis=0)

                for i in range(len(real_motor_tuning)):
                    ik_angles_real = np.asarray(cmd2rad(real_tarpos2cmd(real_motor_tuning[i])), dtype=np.float32)
                    for motor_index in range(5):
                        p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                                targetPosition=ik_angles_real[motor_index], maxVelocity=100, force=100)
                    for j in range(100):
                        p.stepSimulation()
                    real_xyz = np.append(real_xyz, np.asarray(p.getLinkState(self.arm_id, 9)[0])).reshape(-1, 3)

                    if i == len(real_motor_tuning) - 1:
                        print('here')
                    ik_angles_cmd = np.asarray(cmd2rad(real_tarpos2cmd(cmd_motor_tuning[i])), dtype=np.float32)
                    for motor_index in range(5):
                        p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                                targetPosition=ik_angles_cmd[motor_index], maxVelocity=100, force=100)
                    for j in range(100):
                        p.stepSimulation()

                    cmd_xyz = np.append(cmd_xyz, np.asarray(p.getLinkState(self.arm_id, 9)[0])).reshape(-1, 3)
                # real_motor = np.append(real_motor, tuning_data).reshape(-1, 6)
                tar_motor = np.concatenate((tar_motor, tar_motor_tuning), axis=0)
                tar_xyz = np.concatenate((tar_xyz, tar_xyz_tuning), axis=0)
                real_motor = np.concatenate((real_motor, real_motor_tuning), axis=0)
                cmd_motor = np.concatenate((cmd_motor, cmd_motor_tuning), axis=0)

                cur_pos = real_xyz[-1]
                print('this is cur pos after pid', cur_pos)
            else:
                pass

            # tar_pos = tar_pos + np.array([0, 0, real_height])
            # target_pos = np.copy(tar_pos)
            # target_ori = np.copy(tar_ori)
            #
            # vertical_flag = False
            # print('this is tar pos', target_pos)
            # print('this is cur pos', cur_pos)
            # if np.abs(target_pos[2] - cur_pos[2]) > 0.01 \
            #         and np.abs(target_pos[0] - cur_pos[0]) < 0.01\
            #         and np.abs(target_pos[1] - cur_pos[1]) < 0.01:
            #     print('we dont need feedback control')
            #     mark_ratio = 0.8
            #     vertical_flag = True
            #     seg_time = 0
            # else:
            #     if self.generate_dict['use_tuning'] == True:
            #         mark_ratio = 0.97
            #         seg_time = 0
            #     else:
            #         mark_ratio = 1
            #         seg_time = -10
            #
            # cmd_motor = []
            # # plot_real = []
            # cmd_xyz = []
            # sim_ori = []
            # real_xyz = []
            #
            # # divide the whole trajectory into several segment
            # seg_time += 1
            # seg_pos = mark_ratio * (target_pos - cur_pos) + cur_pos
            # seg_ori = mark_ratio * (target_ori - cur_ori) + cur_ori
            # distance = np.linalg.norm(seg_pos - cur_pos)
            # num_step = np.ceil(distance / move_slice)
            # step_pos = (seg_pos - cur_pos) / num_step
            # step_ori = (seg_ori - cur_ori) / num_step
            # print('this is seg pos', seg_pos)
            #
            # while True:
            #     tar_pos = cur_pos + step_pos
            #     # print(tar_pos)
            #     tar_ori = cur_ori + step_ori
            #     cmd_xyz.append(tar_pos)
            #     sim_ori.append(tar_ori)
            #
            #     ik_test = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=cur_pos,
            #                                               maxNumIterations=500,
            #                                               targetOrientation=p.getQuaternionFromEuler(tar_ori))
            #     angle_test = np.asarray(real_cmd2tarpos(rad2cmd(ik_test[0:5])), dtype=np.float32)
            #
            #     ik_angles_sim = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=tar_pos,
            #                                               maxNumIterations=500,
            #                                               targetOrientation=p.getQuaternionFromEuler(tar_ori))
            #
            #     for motor_index in range(5):
            #         p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
            #                                 targetPosition=ik_angles_sim[motor_index], maxVelocity=100, force = 10)
            #     for i in range(30):
            #         p.stepSimulation()
            #
            #     angle_sim = np.asarray(real_cmd2tarpos(rad2cmd(ik_angles_sim[0:5])), dtype=np.float32)
            #     cmd_motor.append(angle_sim)
            #
            #     break_flag = abs(seg_pos[0] - tar_pos[0]) < 0.001 and abs(
            #         seg_pos[1] - tar_pos[1]) < 0.001 and abs(seg_pos[2] - tar_pos[2]) < 0.001 and \
            #                  abs(seg_ori[0] - tar_ori[0]) < 0.001 and abs(
            #         seg_ori[1] - tar_ori[1]) < 0.001 and abs(seg_ori[2] - tar_ori[2]) < 0.001
            #     if break_flag:
            #         break
            #
            #     cur_pos = tar_pos
            #     cur_ori = tar_ori
            #
            # cmd_xyz = np.asarray(cmd_xyz)
            #
            # plot_step = np.arange(num_step)
            # cmd_motor = np.asarray(cmd_motor)
            # print('this is the shape of cmd', cmd_motor.shape)
            # print('this is the shape of xyz', cmd_xyz.shape)
            # tar_motor = np.copy(cmd_motor)
            # tar_xyz = np.copy(cmd_xyz)
            # # print('this is the motor pos sent', cmd_motor[-1])
            # conn.sendall(cmd_motor.tobytes())
            # real_motor = conn.recv(4096)
            # # print('received')
            # real_motor = np.frombuffer(real_motor, dtype=np.float32)
            # real_motor = real_motor.reshape(-1, 6)
            #
            # print('this is the shape of angles real', real_motor.shape)
            # for i in range(len(real_motor)):
            #     ik_angles_real = np.asarray(cmd2rad(real_tarpos2cmd(real_motor[i])), dtype=np.float32)
            #     for motor_index in range(5):
            #         p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
            #                                 targetPosition=ik_angles_real[motor_index], maxVelocity=100, force=100)
            #     for i in range(30):
            #         p.stepSimulation()
            #     real_xyz = np.append(real_xyz, np.asarray(p.getLinkState(self.arm_id, 9)[0])).reshape(-1, 3)
            # cur_pos = real_xyz[-1]
            #
            # if seg_time > 0:
            #     seg_flag = False
            #     print('segment fail, try to tune!')
            #     ik_angles_sim = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=target_pos,
            #                                                  maxNumIterations=500,
            #                                                  targetOrientation=p.getQuaternionFromEuler(
            #                                                      target_ori))
            #
            #     for motor_index in range(5):
            #         p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
            #                                 targetPosition=ik_angles_sim[motor_index], maxVelocity=2.5)
            #     for i in range(30):
            #         p.stepSimulation()
            #
            #     angle_sim = np.asarray(real_cmd2tarpos(rad2cmd(ik_angles_sim[0:5])), dtype=np.float32)
            #     # cmd_motor = np.append(cmd_motor, angle_sim).reshape(-1, 6)
            #     final_cmd = np.append(angle_sim, 0).reshape(1, -1)
            #     final_cmd = np.asarray(final_cmd, dtype=np.float32)
            #     print(final_cmd.shape)
            #     print(final_cmd)
            #     conn.sendall(final_cmd.tobytes())
            #
            #     # get the pos after tune!
            #     tuning_data = conn.recv(4096)
            #     # print('received')
            #     tuning_data = np.frombuffer(tuning_data, dtype=np.float32).reshape(-1, 12)
            #     print('this is the shape of final angles real', tuning_data.shape)
            #     real_motor_tuning = tuning_data[:, 6:]
            #     cmd_motor_tuning = tuning_data[:, :6]
            #     tar_motor_tuning = np.repeat(final_cmd[:, :6], len(cmd_motor_tuning), axis=0)
            #     tar_xyz_tuning = np.repeat(target_pos.reshape(1, -1), len(cmd_motor_tuning), axis=0)
            #
            #     for i in range(len(real_motor_tuning)):
            #         ik_angles_real = np.asarray(cmd2rad(real_tarpos2cmd(real_motor_tuning[i])), dtype=np.float32)
            #         for motor_index in range(5):
            #             p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
            #                                     targetPosition=ik_angles_real[motor_index], maxVelocity=100, force=100)
            #         for j in range(100):
            #             p.stepSimulation()
            #         real_xyz = np.append(real_xyz, np.asarray(p.getLinkState(self.arm_id, 9)[0])).reshape(-1, 3)
            #
            #         if i == len(real_motor_tuning) - 1:
            #             print('here')
            #         ik_angles_cmd = np.asarray(cmd2rad(real_tarpos2cmd(cmd_motor_tuning[i])), dtype=np.float32)
            #         for motor_index in range(5):
            #             p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
            #                                     targetPosition=ik_angles_cmd[motor_index], maxVelocity=100, force=100)
            #         for j in range(100):
            #             p.stepSimulation()
            #
            #         cmd_xyz = np.append(cmd_xyz, np.asarray(p.getLinkState(self.arm_id, 9)[0])).reshape(-1, 3)
            #     # real_motor = np.append(real_motor, tuning_data).reshape(-1, 6)
            #     tar_motor = np.concatenate((tar_motor, tar_motor_tuning), axis=0)
            #     tar_xyz = np.concatenate((tar_xyz, tar_xyz_tuning), axis=0)
            #     real_motor = np.concatenate((real_motor, real_motor_tuning), axis=0)
            #     cmd_motor = np.concatenate((cmd_motor, cmd_motor_tuning), axis=0)
            #
            #     cur_pos = real_xyz[-1]
            #     print('this is cur pos after pid', cur_pos)
            # else:
            #     pass

            if self.generate_dict['real_time_flag'] == True:

                self.plt_cmd_xyz = np.concatenate((self.plt_cmd_xyz, cmd_xyz), axis=0)
                self.plt_real_xyz = np.concatenate((self.plt_real_xyz, real_xyz), axis=0)
                self.plt_cmd_motor = np.concatenate((self.plt_cmd_motor, cmd_motor), axis=0)
                self.plt_real_motor = np.concatenate((self.plt_real_motor, real_motor), axis=0)

                if len(self.plt_cmd_xyz) > self.generate_dict['max_plot_num']:
                    self.plt_cmd_xyz = np.delete(self.plt_cmd_xyz, np.arange(len(cmd_xyz)), axis=0)
                    self.plt_real_xyz = np.delete(self.plt_real_xyz, np.arange(len(real_xyz)), axis=0)
                    self.plt_cmd_motor = np.delete(self.plt_cmd_motor, np.arange(len(cmd_motor)), axis=0)
                    self.plt_real_motor = np.delete(self.plt_real_motor, np.arange(len(real_motor)), axis=0)
                x = np.arange(len(self.plt_cmd_xyz))

                plt.clf()

                plt.subplot(2, 3, 1)
                plt.title("Motor 1")
                plt.grid(True)
                plt.xlabel('X-axis')
                plt.ylabel('Angle')
                plt.plot(x, self.plt_cmd_motor[:, 0], label='cmd')
                plt.plot(x, self.plt_real_motor[:, 0], label='real')
                plt.legend()

                plt.subplot(2, 3, 2)
                plt.title("Motor 2")
                plt.grid(True)
                plt.xlabel('X-axis')
                plt.ylabel('Angle')
                plt.plot(x, self.plt_cmd_motor[:, 1], label='cmd')
                plt.plot(x, self.plt_real_motor[:, 1], label='real')
                plt.legend()

                plt.subplot(2, 3, 3)
                plt.title("Motor 3")
                plt.grid(True)
                plt.xlabel('X-axis')
                plt.ylabel('Angle')
                plt.plot(x, self.plt_cmd_motor[:, 2], label='cmd')
                plt.plot(x, self.plt_real_motor[:, 2], label='real')
                plt.legend()

                plt.subplot(2, 3, 4)
                plt.title("Motor 4")
                plt.grid(True)
                plt.xlabel('X-axis')
                plt.ylabel('Angle')
                plt.plot(x, self.plt_cmd_motor[:, 3], label='cmd')
                plt.plot(x, self.plt_real_motor[:, 3], label='real')
                plt.legend()

                plt.subplot(2, 3, 5)
                plt.title("Motor 5")
                plt.grid(True)
                plt.xlabel('X-axis')
                plt.ylabel('Angle')
                plt.plot(x, self.plt_cmd_motor[:, 4], label='cmd')
                plt.plot(x, self.plt_real_motor[:, 4], label='real')
                plt.legend()

                plt.suptitle('Motor Data Analysis')

                plt.pause(0.1)

            if self.para_dict['data_collection'] == True:
                with open(file=self.para_dict['dataset_path'] + "cmd_xyz_nn.txt", mode="a", encoding="utf-8") as f:
                    np.savetxt(f, cmd_xyz)
                with open(file=self.para_dict['dataset_path'] + "real_xyz_nn.txt", mode="a", encoding="utf-8") as f:
                    np.savetxt(f, real_xyz)
                with open(file=self.para_dict['dataset_path'] + "tar_xyz_nn.txt", mode="a", encoding="utf-8") as f:
                    np.savetxt(f, tar_xyz)
                with open(file=self.para_dict['dataset_path'] + "cmd_nn.txt", mode="a", encoding="utf-8") as f:
                    np.savetxt(f, cmd_motor)
                with open(file=self.para_dict['dataset_path'] + "real_nn.txt", mode="a", encoding="utf-8") as f:
                    np.savetxt(f, real_motor)
                with open(file=self.para_dict['dataset_path'] + "tar_nn.txt", mode="a", encoding="utf-8") as f:
                    np.savetxt(f, tar_motor)
            if self.generate_dict['use_tuning'] == True:
                print('this is cmd zzz\n', cmd_xyz[-1])
                return target_pos # return cur pos to let the manipualtor remember the improved pos
            else:
                return tar_pos

        def gripper(gap, obj_width=None):
            obj_width += 0.006
            obj_width_range = np.array([0.021, 0.026, 0.032, 0.039, 0.045, 0.052, 0.057])
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

                real_pos = conn.recv(4096)
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

            rest_pos = np.array([0, 0, 0.05])
            rest_ori = np.array([0, np.pi / 2, 0])
            offset_low = np.array([0, 0, 0.002])
            offset_high = np.array([0, 0, 0.035])

            last_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
            last_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))

            if self.para_dict['data_collection'] == True:

                # trajectory_pos_list = np.array([[0.01, 0.016],
                #                                 [0.01, 0.024],
                #                                 [0.01, 0.032],
                #                                 [0.01, 0.040],
                #                                 [0.01, 0.048]])
                trajectory_pos_list = np.array([[0.00, 0.14, 0.03],
                                                [0.25, 0.14, 0.03],
                                                [0.25, -0.14, 0.03],
                                                [0.00, -0.14, 0.03]])
                # pos_x = np.random.uniform(self.generate_dict['x_range'][0], self.generate_dict['x_range'][1], self.generate_dict['collect_num'])
                # pos_y = np.random.uniform(self.generate_dict['y_range'][0], self.generate_dict['y_range'][1], self.generate_dict['collect_num'])
                # pos_z = np.random.uniform(self.generate_dict['z_range'][0], self.generate_dict['z_range'][1], self.generate_dict['collect_num'])
                # trajectory_pos_list = np.concatenate((pos_x.reshape(-1, 1), pos_y.reshape(-1, 1), pos_z.reshape(-1, 1)), axis=1)
                for j in tqdm(range(len(trajectory_pos_list))):

                    if len(trajectory_pos_list[j]) == 3:
                        last_pos = move(last_pos, last_ori, trajectory_pos_list[j], rest_ori)
                        if trajectory_pos_list[j, 2] > 0.01:
                            # time.sleep(5)
                            pass
                        else:
                            pass
                            # time.sleep(3)
                        last_ori = np.copy(rest_ori)

                    elif len(trajectory_pos_list[j]) == 2:
                        gripper(trajectory_pos_list[j][0], trajectory_pos_list[j][1])
                        # time.sleep(5)
            else:
                times = 2
                for j in range(times):

                    trajectory_pos_list = np.array([np.random.uniform(0, 0.28), np.random.uniform(-0.16, 0.16), np.random.uniform(0.032, 0.08)])

                    if len(trajectory_pos_list) == 3:
                        print('ready to move', trajectory_pos_list)
                        last_pos = move(last_pos, last_ori, trajectory_pos_list, rest_ori)
                        # time.sleep(5)
                        # last_pos = np.copy(trajectory_pos_list)
                        last_ori = np.copy(rest_ori)

                    elif len(trajectory_pos_list[j]) == 2:
                        gripper(trajectory_pos_list[j][0])

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
            with open(file=self.para_dict['dataset_path'] + "cmd_nn.txt", mode="a", encoding="utf-8") as f:
                f.truncate(0)
            with open(file=self.para_dict['dataset_path'] + "real_nn.txt", mode="a", encoding="utf-8") as f:
                f.truncate(0)
            with open(file=self.para_dict['dataset_path'] + "tar_nn.txt", mode="a", encoding="utf-8") as f:
                f.truncate(0)

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
            table_surface_height = 0.032
            sim_table_surface_height = 0
            num_motor = 5
            # ! reset the pos in both real and sim
            reset_pos = np.array([0.015, 0, 0.1])
            ik_angles = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=reset_pos,
                                                     maxNumIterations=300,
                                                     targetOrientation=p.getQuaternionFromEuler(
                                                         [0, np.pi / 2, 0]))
            reset_real = np.asarray(real_cmd2tarpos(rad2cmd(ik_angles[0:5])), dtype=np.float32)
            print('this is the reset motor pos', reset_real)
            conn.sendall(reset_real.tobytes())

            real_motor = conn.recv(4096)
            # print('received')
            real_motor = np.frombuffer(real_motor, dtype=np.float32)
            real_motor = real_motor.reshape(-1, 6)

            real_xyz_init = []
            print('this is the shape of angles real', real_motor.shape)
            for i in range(len(real_motor)):
                ik_angles_real = np.asarray(cmd2rad(real_tarpos2cmd(real_motor[i])), dtype=np.float32)
                for motor_index in range(5):
                    p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                            targetPosition=ik_angles_real[motor_index], maxVelocity=25)
                for i in range(30):
                    p.stepSimulation()
                real_xyz_init = np.append(real_xyz_init, np.asarray(p.getLinkState(self.arm_id, 9)[0])).reshape(-1, 3)

            for i in range(num_motor):
                p.setJointMotorControl2(self.arm_id, i, p.POSITION_CONTROL, targetPosition=ik_angles[i],
                                        maxVelocity=3)
            self.plt_cmd_xyz = reset_pos.reshape(-1, 3)
            self.plt_real_xyz = real_xyz_init
            self.plt_cmd_motor = reset_real.reshape(-1, 6)
            self.plt_real_motor = real_motor
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
                 'init_pos_range': [[0.03, 0.27], [-0.13, 0.13], [0.01, 0.02]],
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
                 'dataset_path': '../../knolling_dataset/nn_data_805/',
                 'urdf_path': '../urdf/',
                 'yolo_model_path': './train_pile_overlap_627/weights/best.pt',
                 'real_operate': True, 'obs_order': 'real_image_obj', 'data_collection': True,
                 'use_knolling_model': False, 'use_lstm_model': False}

    generate_dict = {'real_time_flag': False, 'erase_flag': True, 'collect_num': 50, 'max_plot_num': 500,
                     'x_range': [0.05, 0.25], 'y_range': [-0.13, 0.13], 'z_range':[0.02, 0.05], 'use_tuning': True}


    env = calibration_main(para_dict=para_dict, generate_dict=generate_dict)
    evaluation = 1
    for evaluation in range(para_dict['evaluations']):
        env.step()