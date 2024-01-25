import pybullet as p
import pybullet_data as pd
import numpy as np
import time
import os
import socket
from utils import *

class LSTM_grasp_collection_robot():

    def __init__(self, para_dict, knolling_para=None):

        self.para_dict = para_dict
        self.knolling_para = knolling_para
        self.is_render = para_dict['is_render']

    def calculate_gripper(self):
        self.close_open_gap = 0.053
        # close_open_gap = 0.048
        obj_width_range = np.array([0.022, 0.057])
        motor_pos_range = np.array([0.022, 0.010])  # 0.0273
        formula_parameters = np.polyfit(obj_width_range, motor_pos_range, 1)
        self.motor_pos = np.poly1d(formula_parameters)

    def arm_setup(self):

        if self.para_dict['real_operate'] == True:

            HOST = "192.168.0.187"  # Standard loopback interface address (localhost)
            PORT = 8881 # Port to listen on (non-privileged ports are > 1023)
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

        self.arm_id = p.loadURDF(os.path.join(self.para_dict['urdf_path'], "robot_arm928/robot_arm.urdf"),
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

    def move(self, cur_pos, cur_ori, tar_pos, tar_ori, sim_height=-0.01, origin_left_pos=None, origin_right_pos=None,
             index=None):

        # add the offset manually
        if tar_ori[2] > 3.1416 / 2:
            tar_ori[2] = tar_ori[2] - np.pi
            # print('tar ori is too large')
        elif tar_ori[2] < -3.1416 / 2:
            tar_ori[2] = tar_ori[2] + np.pi
            # print('tar ori is too small')
        # print('this is tar ori', tar_ori)

        #################### use feedback control ###################
        if abs(cur_pos[0] - tar_pos[0]) < 0.001 and abs(cur_pos[1] - tar_pos[1]) < 0.001:
            # vertical, choose a small slice
            move_slice = 0.004
        else:
            # horizontal, choose a large slice
            move_slice = 0.004

        tar_pos = tar_pos + np.array([0, 0, sim_height])
        target_pos = np.copy(tar_pos)
        target_ori = np.copy(tar_ori)

        distance = np.linalg.norm(tar_pos - cur_pos)
        num_step = np.ceil(distance / move_slice)
        step_pos = (target_pos - cur_pos) / num_step
        step_ori = (target_ori - cur_ori) / num_step

        #################### ensure the gripper will not drift while lifting the boxes ###########################
        if index == 6 or index == 5:
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
                                        targetPosition=ik_angles0[motor_index], maxVelocity=25,
                                        force=self.para_dict['move_force'])
            move_success_flag = True
            if index == 3:
                for i in range(20):
                    p.stepSimulation()
                    bar_pos = np.asarray(p.getLinkState(self.arm_id, 6)[0])
                    gripper_left_pos = np.asarray(p.getLinkState(self.arm_id, 7)[0])
                    gripper_right_pos = np.asarray(p.getLinkState(self.arm_id, 8)[0])
                    new_distance_left = np.linalg.norm(bar_pos[:2] - gripper_left_pos[:2])
                    new_distance_right = np.linalg.norm(bar_pos[:2] - gripper_right_pos[:2])
                    # if np.abs(origin_left_pos[1] - gripper_left_pos[1]) > self.para_dict['move_threshold'] or \
                    #         np.abs(origin_right_pos[1] - gripper_right_pos[1]) > self.para_dict['move_threshold']:
                    #     move_success_flag = False
                    #     print('during moving, fail')
                    #     break
                    if np.abs(new_distance_left - self.distance_left) > self.para_dict['gripper_threshold'] or \
                            np.abs(new_distance_right - self.distance_right) > self.para_dict['gripper_threshold']:
                        move_success_flag = False
                        print('during moving, the gripper is disturbed, fail')
                        break

                    if self.is_render:
                        pass
                        time.sleep(1 / 720)
                if move_success_flag == False:
                    break
            else:
                for i in range(10):
                    p.stepSimulation()
                    if self.is_render:
                        pass
                        time.sleep(1 / 720)
            cur_pos = tar_pos
            cur_ori = tar_ori
            if abs(target_pos[0] - tar_pos[0]) < 0.001 and abs(target_pos[1] - tar_pos[1]) < 0.001 and abs(
                    target_pos[2] - tar_pos[2]) < 0.001 and \
                    abs(target_ori[0] - tar_ori[0]) < 0.001 and abs(target_ori[1] - tar_ori[1]) < 0.001 and abs(
                target_ori[2] - tar_ori[2]) < 0.001:
                break
        ee_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
        if ee_pos[2] - target_pos[2] > 0.002 and index == 3 and move_success_flag == True:
            move_success_flag = False
            print('ee can not reach the bottom, fail!')

        self.gripper_left_pos = np.asarray(p.getLinkState(self.arm_id, 7)[0])
        self.gripper_right_pos = np.asarray(p.getLinkState(self.arm_id, 8)[0])
        return cur_pos, self.gripper_left_pos, self.gripper_right_pos, move_success_flag

    def gripper(self, gap, obj_width, left_pos, right_pos, index=None):

        gripper_success_flag = True
        if index == 4:
            self.keep_obj_width = obj_width + 0.01
            bar_pos = np.asarray(p.getLinkState(self.arm_id, 6)[0])
            gripper_left_pos = np.asarray(p.getLinkState(self.arm_id, 7)[0])
            gripper_right_pos = np.asarray(p.getLinkState(self.arm_id, 8)[0])
            new_distance_left = np.linalg.norm(bar_pos[:2] - gripper_left_pos[:2])
            new_distance_right = np.linalg.norm(bar_pos[:2] - gripper_right_pos[:2])
            # if np.abs(self.gripper_left_pos[1] - gripper_left_pos[1]) > self.para_dict['move_threshold'] or \
            #         np.abs(self.gripper_right_pos[1] - gripper_right_pos[1]) > self.para_dict['move_threshold']:
            #     gripper_success_flag = False
            #     print('during moving, fail')

            if np.abs(new_distance_left - self.distance_left) > self.para_dict['gripper_threshold'] or \
                    np.abs(new_distance_right - self.distance_right) > self.para_dict['gripper_threshold']:
                gripper_success_flag = False
                print('gripper is disturbed before grasping, fail')
                return gripper_success_flag

        obj_width += 0.008

        if index == 1:
            num_step = 30
        else:
            num_step = 10

        if gap > 0.5:  # close
            tar_pos = self.motor_pos(obj_width) + self.close_open_gap
            p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL,
                                    targetPosition=self.motor_pos(obj_width) + self.close_open_gap,
                                    force=self.para_dict['gripper_force'])
            p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL,
                                    targetPosition=self.motor_pos(obj_width) + self.close_open_gap,
                                    force=self.para_dict['gripper_force'])
            for i in range(num_step):
                p.stepSimulation()
                gripper_left_pos = np.asarray(p.getLinkState(self.arm_id, 7)[0])
                gripper_right_pos = np.asarray(p.getLinkState(self.arm_id, 8)[0])
                # if gripper_left_pos[1] - left_pos[1] > self.para_dict['gripper_threshold'] or right_pos[1] - gripper_right_pos[1] > self.para_dict['gripper_threshold']:
                #     print('during grasp, fail')
                #     gripper_success_flag = False
                #     break
                # if self.is_render:
                #     pass
                #     # time.sleep(1 / 48)
        else:  # open
            p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL, targetPosition=self.motor_pos(obj_width),
                                    force=self.para_dict['gripper_force'])
            p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL, targetPosition=self.motor_pos(obj_width),
                                    force=self.para_dict['gripper_force'])
            for i in range(num_step):
                p.stepSimulation()
                if self.is_render:
                    pass
                    # time.sleep(1 / 48)
        if index == 1:
            # print('initialize the distance from gripper to bar')
            bar_pos = np.asarray(p.getLinkState(self.arm_id, 6)[0])
            gripper_left_pos = np.asarray(p.getLinkState(self.arm_id, 7)[0])
            gripper_right_pos = np.asarray(p.getLinkState(self.arm_id, 8)[0])
            self.distance_left = np.linalg.norm(bar_pos[:2] - gripper_left_pos[:2])
            self.distance_right = np.linalg.norm(bar_pos[:2] - gripper_right_pos[:2])
        return gripper_success_flag