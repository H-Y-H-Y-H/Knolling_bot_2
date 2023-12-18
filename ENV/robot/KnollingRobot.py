import pybullet as p
import pybullet_data as pd
import numpy as np
import time
import os
import socket
from utils import *

class knolling_robot():

    def __init__(self, para_dict, knolling_para=None):

        self.para_dict = para_dict
        self.knolling_para = knolling_para

    def calculate_gripper(self):
        self.close_open_gap = 0.053
        # close_open_gap = 0.048
        obj_width_range = np.array([0.022, 0.057])
        motor_pos_range = np.array([0.022, 0.010])  # 0.0273
        formula_parameters = np.polyfit(obj_width_range, motor_pos_range, 1)
        self.motor_pos = np.poly1d(formula_parameters)

    def arm_setup(self):

        if self.para_dict['real_operate'] == True:

            HOST = "192.168.1.112"  # Standard loopback interface address (localhost)
            PORT = 8882 # Port to listen on (non-privileged ports are > 1023)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((HOST, PORT))
            # It should be an integer from 1 to 65535, as 0 is reserved. Some systems may require superuser privileges if the port number is less than 8192.
            # associate the socket with a specific network interface
            s.listen()
            print(f"Waiting for connection...\n")
            self.conn, addr = s.accept()
            print(self.conn)
            print(f"Connected by {addr}")
            self.real_table_height = 0.003
            self.sim_table_height = 0
            num_motor = 5
            # ! reset the pos in both real and sim
            reset_pos = np.array([0.015, 0, 0.1])
            reset_ori = np.array([0, np.pi / 2, 0])
            cmd_motor = np.asarray(inverse_kinematic(np.copy(reset_pos), np.copy(reset_ori)), dtype=np.float32)
            print('this is the reset motor pos', cmd_motor)
            self.conn.sendall(cmd_motor.tobytes())

            real_motor = self.conn.recv(8192)
            real_motor = np.frombuffer(real_motor, dtype=np.float32)
            real_motor = real_motor.reshape(-1, 6)

            real_xyz, _ = forward_kinematic(real_motor)
        else:
            self.conn = None
            self.real_table_height = 0.026
            self.sim_table_height = -0.014

        return self.conn, self.real_table_height, self.sim_table_height

    def to_home(self):

        ik_angles0 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=self.para_dict['reset_pos'],
                                                  maxNumIterations=200,
                                                  targetOrientation=p.getQuaternionFromEuler(
                                                      self.para_dict['reset_ori']))

        # after reset the position of the robot arm manually, we should add the force to keep the arm
        for motor_index in range(5):
            p.resetJointState(self.arm_id, motor_index, ik_angles0[motor_index])
        for motor_index in range(5):
            p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                    targetPosition=ik_angles0[motor_index], maxVelocity=20)
        for _ in range(int(30)):
            # time.sleep(1/480)
            p.stepSimulation()

    def create_arm(self):

        self.arm_id = p.loadURDF(os.path.join(self.para_dict['urdf_path'], "robot_arm928/robot_arm1_backup.urdf"),
                                 basePosition=[-0.08, 0, 0.02], useFixedBase=True,
                                 flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        p.changeDynamics(self.arm_id, 7, lateralFriction=self.para_dict['gripper_lateral_friction'],
                         contactDamping=self.para_dict['gripper_contact_damping'],
                         contactStiffness=self.para_dict['gripper_contact_stiffness'])
        p.changeDynamics(self.arm_id, 8, lateralFriction=self.para_dict['gripper_lateral_friction'],
                         contactDamping=self.para_dict['gripper_contact_damping'],
                         contactStiffness=self.para_dict['gripper_contact_stiffness'])
        self.to_home()

        return self.arm_id

    def gripper(self, gap, obj_width):

        if gap > 0.5:
            self.keep_obj_width = obj_width + 0.010
        obj_width += 0.010
        if self.para_dict['real_operate'] == True:
            obj_width_range = np.array([0.021, 0.026, 0.032, 0.039, 0.045, 0.052, 0.057])
            # motor_pos_range = np.array([2050, 2150, 2250, 2350, 2450, 2550, 2650])
            motor_pos_range = np.array([2100, 2200, 2250, 2350, 2450, 2550, 2650])

            formula_parameters = np.polyfit(obj_width_range, motor_pos_range, 3)
            motor_pos = np.poly1d(formula_parameters)
        else:
            close_open_gap = 0.053
            obj_width_range = np.array([0.022, 0.057])
            motor_pos_range = np.array([0.022, 0.010])  # 0.0273
            formula_parameters = np.polyfit(obj_width_range, motor_pos_range, 1)
            motor_pos = np.poly1d(formula_parameters)

        if self.para_dict['real_operate'] == True:
            if gap > 0.5:  # close
                pos_real = np.asarray([[gap, 1600]], dtype=np.float32)
            elif gap <= 0.5:  # open
                pos_real = np.asarray([[gap, motor_pos(obj_width)]], dtype=np.float32)
            print('gripper', pos_real)
            self.conn.sendall(pos_real.tobytes())
            real_pos = self.conn.recv(4096)
            real_pos = np.frombuffer(real_pos, dtype=np.float32)
            # print('this is test float from buffer', test_real_pos)

        else:
            if gap > 0.5:  # close
                p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL,
                                        targetPosition=motor_pos(obj_width) + close_open_gap,
                                        force=self.para_dict['gripper_force'])
                p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL,
                                        targetPosition=motor_pos(obj_width) + close_open_gap,
                                        force=self.para_dict['gripper_force'])
            else:  # open
                p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL,
                                        targetPosition=motor_pos(obj_width),
                                        force=self.para_dict['gripper_force'])
                p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL,
                                        targetPosition=motor_pos(obj_width),
                                        force=self.para_dict['gripper_force'])
            for i in range(self.para_dict['gripper_sim_step']):
                p.stepSimulation()
                if self.para_dict['is_render'] == True:
                    time.sleep(1 / 48)

    def move(self, cur_pos, cur_ori, tar_pos, tar_ori, index=None, task=None):

        if self.para_dict['real_operate'] == True:
            # real_height_offset = np.array([0, 0, real_height])
            send_data = np.concatenate((cur_pos, cur_ori, tar_pos, tar_ori), axis=0).reshape(-1, 3)
            send_data = send_data.astype(np.float32)

            self.conn.sendall(send_data.tobytes())

            receive_time = 0
            while True:
                buffer = np.frombuffer(self.conn.recv(8192), dtype=np.float32)
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
            recall_data = recall_data.reshape(-1, 36)

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
            print('this is cmd zzz\n', cmd_xyz[-1])
            return cmd_xyz[-1]  # return cur pos to let the manipualtor remember the improved pos

        else:
            if tar_ori[2] > 3.1416 / 2:
                tar_ori[2] = tar_ori[2] - np.pi
                print('tar ori is too large')
            elif tar_ori[2] < -3.1416 / 2:
                tar_ori[2] = tar_ori[2] + np.pi
                print('tar ori is too small')
            # print('this is tar ori', tar_ori)

            if abs(cur_pos[0] - tar_pos[0]) < 0.001 and abs(cur_pos[1] - tar_pos[1]) < 0.001:
                # vertical, choose a small slice
                move_slice = 0.004
            else:
                # horizontal, choose a large slice
                move_slice = 0.004

            tar_pos = tar_pos + np.array([0, 0, self.sim_table_height])
            target_pos = np.copy(tar_pos)
            target_ori = np.copy(tar_ori)

            distance = np.linalg.norm(tar_pos - cur_pos)
            num_step = np.ceil(distance / move_slice)
            step_pos = (target_pos - cur_pos) / num_step
            step_ori = (target_ori - cur_ori) / num_step

            print('this is sim tar pos', tar_pos)

            #################### ensure the gripper will not drift while lifting the boxes ###########################
            if index == 5 and task == 'knolling':
                p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL,
                                        targetPosition=self.motor_pos(self.keep_obj_width) + self.close_open_gap,
                                        force=self.para_dict['gripper_force'] * 10)
                p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL,
                                        targetPosition=self.motor_pos(self.keep_obj_width) + self.close_open_gap,
                                        force=self.para_dict['gripper_force'] * 10)
            #################### ensure the gripper will not drift while lifting the boxes ###########################

            while True:
                tar_pos = cur_pos + step_pos
                tar_ori = cur_ori + step_ori
                ik_angles0 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=tar_pos,
                                                          maxNumIterations=200,
                                                          targetOrientation=p.getQuaternionFromEuler(tar_ori))
                for motor_index in range(5):
                    p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                            targetPosition=ik_angles0[motor_index], maxVelocity=100,
                                            force=self.para_dict['move_force'])
                for i in range(10):
                    p.stepSimulation()
                    if self.para_dict['is_render'] == True:
                        time.sleep(1 / 720)
                if abs(target_pos[0] - tar_pos[0]) < 0.001 and abs(target_pos[1] - tar_pos[1]) < 0.001 and abs(
                        target_pos[2] - tar_pos[2]) < 0.001 and \
                        abs(target_ori[0] - tar_ori[0]) < 0.001 and abs(target_ori[1] - tar_ori[1]) < 0.001 and abs(
                    target_ori[2] - tar_ori[2]) < 0.001:
                    break
                cur_pos = tar_pos
                cur_ori = tar_ori
            return cur_pos