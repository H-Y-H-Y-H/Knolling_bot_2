import numpy as np
import pyrealsense2 as rs
import pybullet_data as pd
import math
from utils import *
from environment import Arm_env
import socket
import cv2
import torch
from urdfpy import URDF
import shutil

class knolling_main(Arm_env):

    def __init__(self, para_dict=None, knolling_para=None, lstm_dict=None, arrange_dict=None):
        super(knolling_main, self).__init__(para_dict=para_dict, knolling_para=knolling_para, lstm_dict=lstm_dict, arrange_dict=arrange_dict)

        self.success_manipulator_after = []
        self.success_lwh = []
        self.success_num = 0

    def planning(self, order, conn, real_height, sim_height):

        def move(cur_pos, cur_ori, tar_pos, tar_ori, index=None, task=None):

            # add the offset manually
            if self.para_dict['real_operate'] == True:
                # # automatically add z and x bias
                d = np.array([0, 0.3])
                z_bias = np.array([0.002, 0.01])
                x_bias = np.array([-0.004, 0.000])# yolo error is +2mm along x axis!
                y_bias = np.array([0.001, 0.005])
                z_parameters = np.polyfit(d, z_bias, 1)
                x_parameters = np.polyfit(d, x_bias, 1)
                y_parameters = np.polyfit(d, y_bias, 1)
                new_z_formula = np.poly1d(z_parameters)
                new_x_formula = np.poly1d(x_parameters)
                new_y_formula = np.poly1d(y_parameters)

                distance = tar_pos[0]
                # distance_y = tar_pos[0]
                tar_pos[2] = tar_pos[2] + new_z_formula(distance)
                print('this is z add', new_z_formula(distance))
                # tar_pos[0] = tar_pos[0] + new_x_formula(distance)
                # print('this is x add', new_x_formula(distance))
                # if tar_pos[1] > 0:
                #     tar_pos[1] += (new_y_formula(distance_y) * np.clip((6 * (tar_pos[1] + 0.01)), 0, 1) + 0.0015) # 0.003 is manual!
                # else:
                #     tar_pos[1] -= (new_y_formula(distance_y) * np.clip((6 * (tar_pos[1] - 0.01)), 0, 1) - 0.0015) # 0.003 is manual!
                print('this is tar pos after manual', tar_pos)

            if tar_ori[2] > 3.1416 / 2:
                tar_ori[2] = tar_ori[2] - np.pi
                print('tar ori is too large')
            elif tar_ori[2] < -3.1416 / 2:
                tar_ori[2] = tar_ori[2] + np.pi
                print('tar ori is too small')
            # print('this is tar ori', tar_ori)

            if self.para_dict['real_operate'] == True:

                real_height_offset = np.array([0, 0, real_height])
                send_data = np.concatenate((cur_pos, cur_ori, tar_pos, tar_ori, real_height_offset), axis=0).reshape(-1, 3)
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

                return cmd_xyz[-1]

            else:
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

        def gripper(gap, obj_width):

            if gap > 0.5:
                self.keep_obj_width = obj_width + 0.01
            obj_width += 0.010
            if self.para_dict['real_operate'] == True:
                obj_width_range = np.array([0.021, 0.026, 0.032, 0.039, 0.045, 0.052, 0.057])
                motor_pos_range = np.array([2050, 2150, 2250, 2350, 2450, 2550, 2650])
                formula_parameters = np.polyfit(obj_width_range, motor_pos_range, 3)
                motor_pos = np.poly1d(formula_parameters)
            else:
                close_open_gap = 0.053
                obj_width_range = np.array([0.022, 0.057])
                motor_pos_range = np.array([0.022, 0.010]) # 0.0273
                formula_parameters = np.polyfit(obj_width_range, motor_pos_range, 1)
                motor_pos = np.poly1d(formula_parameters)

            if self.para_dict['real_operate'] == True:
                if gap > 0.0265:  # close
                    pos_real = np.asarray([[gap, 1600]], dtype=np.float32)
                elif gap <= 0.0265:  # open
                    pos_real = np.asarray([[gap, motor_pos(obj_width)]], dtype=np.float32)
                print('gripper', pos_real)
                conn.sendall(pos_real.tobytes())
                real_pos = conn.recv(4096)
                real_pos = np.frombuffer(real_pos, dtype=np.float32)
                # print('this is test float from buffer', test_real_pos)

            else:
                if gap > 0.5: # close
                    p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL,
                                            targetPosition=motor_pos(obj_width) + close_open_gap,
                                            force=self.para_dict['gripper_force'])
                    p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL,
                                            targetPosition=motor_pos(obj_width) + close_open_gap,
                                            force=self.para_dict['gripper_force'])
                else: # open
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

        def clean_grasp():
            if self.para_dict['real_operate'] == False:
                gripper_width = 0.024
                gripper_height = 0.034
            else:
                gripper_width = 0.018
                gripper_height = 0.04
            workbench_center = np.array([(self.x_high_obs + self.x_low_obs) / 2,
                                         (self.y_high_obs + self.y_low_obs) / 2])

            offset_low = np.array([0, 0, 0.005])
            offset_high = np.array([0, 0, 0.035])

            ############ Predict the probability of grasp, remember to change the sequence of input #############
            manipulator_before, new_lwh_list, pred_conf = self.get_obs()
            order = change_sequence(manipulator_before)
            manipulator_before = manipulator_before[order]
            new_lwh_list = new_lwh_list[order]
            pred_conf = pred_conf[order]
            crowded_index, prediction, model_output = self.grasp_model.pred(manipulator_before, new_lwh_list, pred_conf)
            self.yolo_pose_model.plot_grasp(manipulator_before, prediction, model_output)
            ############ Predict the probability of grasp, remember to change the sequence of input #############
            tar_success = np.copy(len(manipulator_before))

            restrict_gripper_diagonal = np.sqrt(gripper_width ** 2 + gripper_height ** 2)
            gripper_box_gap = 0.006

            while True:

                ####### knolling only if the number of boxes we can grasp is more than 2 #######
                if len(manipulator_before) - len(crowded_index) >= 1:
                    knolling(manipulator_before=manipulator_before, new_lwh_list=new_lwh_list, crowded_index=crowded_index)
                    ############## exclude boxes which have been knolling ##############
                    manipulator_before = manipulator_before[len(self.success_manipulator_after):]
                    new_lwh_list = new_lwh_list[len(self.success_manipulator_after):]
                    crowded_index = np.setdiff1d(crowded_index, np.arange(len(self.success_manipulator_after)))
                    ############## exclude boxes which have been knolling ##############

                    if len(self.success_manipulator_after) >= tar_success:
                        self.get_obs(look_flag=True)
                        break
                ####### knolling only if the number of boxes we can grasp is more than 2 #######

                crowded_pos = manipulator_before[:, :3]
                crowded_ori = manipulator_before[:, 3:6]
                theta = manipulator_before[:, -1]
                length_box = new_lwh_list[:, 0]
                width_box = new_lwh_list[:, 1]

                trajectory_pos_list = []
                trajectory_ori_list = []
                for i in range(len(crowded_index)):
                    break_flag = False
                    once_flag = False
                    if length_box[i] < width_box[i]:
                        theta[i] += np.pi / 2
                    matrix = np.array([[np.cos(theta[i]), -np.sin(theta[i])],
                                       [np.sin(theta[i]), np.cos(theta[i])]])
                    target_point = np.array([[(length_box[i] + gripper_height + gripper_box_gap) / 2,
                                              (width_box[i] + gripper_width + gripper_box_gap) / 2],
                                             [-(length_box[i] + gripper_height + gripper_box_gap) / 2,
                                              (width_box[i] + gripper_width + gripper_box_gap) / 2],
                                             [-(length_box[i] + gripper_height + gripper_box_gap) / 2,
                                              -(width_box[i] + gripper_width + gripper_box_gap) / 2],
                                             [(length_box[i] + gripper_height + gripper_box_gap) / 2,
                                              -(width_box[i] + gripper_width + gripper_box_gap) / 2]])
                    target_point_rotate = (matrix.dot(target_point.T)).T
                    print('this is target point rotate\n', target_point_rotate)
                    sequence_point = np.concatenate((target_point_rotate, np.zeros((4, 1))), axis=1)

                    t = 0
                    for j in range(len(sequence_point)):
                        vertex_break_flag = False
                        for k in range(len(manipulator_before)):
                            # exclude itself
                            if np.linalg.norm(crowded_pos[i] - manipulator_before[k][:3]) < 0.001:
                                continue
                            restrict_item_k = np.sqrt((new_lwh_list[k][0]) ** 2 + (new_lwh_list[k][1]) ** 2)
                            if 0.001 < np.linalg.norm(sequence_point[0] + crowded_pos[i] - manipulator_before[k][
                                                                                           :3]) < restrict_item_k / 2 + restrict_gripper_diagonal / 2 + 0.001:
                                print(np.linalg.norm(sequence_point[0] + crowded_pos[i] - manipulator_before[k][:3]))
                                p.addUserDebugPoints([sequence_point[0] + crowded_pos[i]], [[0.1, 0, 0]], pointSize=5)
                                p.addUserDebugPoints([manipulator_before[k][:3]], [[0, 1, 0]], pointSize=5)
                                print("this vertex doesn't work")
                                vertex_break_flag = True
                                break
                        if vertex_break_flag == False:
                            print("this vertex is ok")
                            print(break_flag)
                            once_flag = True
                            break
                        else:
                            # should change the vertex and try again
                            sequence_point = np.roll(sequence_point, -1, axis=0)
                            print(sequence_point)
                            t += 1
                        if t == len(sequence_point):
                            # all vertex of this cube fail, should change the cube
                            break_flag = True

                    # problem, change another crowded cube
                    if break_flag == True:
                        if i == len(crowded_index) - 1:
                            print('cannot find any proper vertices to insert, we should unpack the heap!!!')
                            x_high = np.max(self.manipulator_after[:, 0])
                            x_low = np.min(self.manipulator_after[:, 0])
                            y_high = np.max(self.manipulator_after[:, 1])
                            y_low = np.min(self.manipulator_after[:, 1])
                            crowded_x_high = np.max(crowded_pos[:, 0])
                            crowded_x_low = np.min(crowded_pos[:, 0])
                            crowded_y_high = np.max(crowded_pos[:, 1])
                            crowded_y_low = np.min(crowded_pos[:, 1])

                            trajectory_pos_list.append([1, 0])
                            trajectory_pos_list.append([(x_high + x_low) / 2, (y_high + y_low) / 2, offset_high[2]])
                            trajectory_pos_list.append([(x_high + x_low) / 2, (y_high + y_low) / 2, offset_low[2]])
                            trajectory_pos_list.append(
                                [(crowded_x_high + crowded_x_low) / 2, (crowded_y_high + crowded_y_low) / 2,
                                 offset_low[2]])
                            trajectory_pos_list.append(
                                [(crowded_x_high + crowded_x_low) / 2, (crowded_y_high + crowded_y_low) / 2,
                                 offset_high[2]])
                            trajectory_pos_list.append(self.para_dict['reset_pos'])

                            trajectory_ori_list.append(self.para_dict['reset_ori'])
                            trajectory_ori_list.append(self.para_dict['reset_ori'])
                            trajectory_ori_list.append(self.para_dict['reset_ori'])
                            trajectory_ori_list.append(self.para_dict['reset_ori'])
                            trajectory_ori_list.append(self.para_dict['reset_ori'])
                            trajectory_ori_list.append(self.para_dict['reset_ori'])
                        else:
                            pass
                    else:
                        trajectory_pos_list.append([1, 0])
                        print('this is crowded pos', crowded_pos[i])
                        print('this is sequence point', sequence_point)
                        trajectory_pos_list.append(crowded_pos[i] + offset_high + sequence_point[0])
                        trajectory_pos_list.append(crowded_pos[i] + offset_low + sequence_point[0])
                        trajectory_pos_list.append(crowded_pos[i] + offset_low + sequence_point[1])
                        trajectory_pos_list.append(crowded_pos[i] + offset_low + sequence_point[2])
                        trajectory_pos_list.append(crowded_pos[i] + offset_low + sequence_point[3])
                        trajectory_pos_list.append(crowded_pos[i] + offset_low + sequence_point[0])
                        trajectory_pos_list.append(crowded_pos[i] + offset_high + sequence_point[0])
                        # reset the manipulator to read the image
                        trajectory_pos_list.append(self.para_dict['reset_pos'])

                        trajectory_ori_list.append(self.para_dict['reset_ori'])
                        trajectory_ori_list.append(self.para_dict['reset_ori'] + crowded_ori[i])
                        trajectory_ori_list.append(self.para_dict['reset_ori'] + crowded_ori[i])
                        trajectory_ori_list.append(self.para_dict['reset_ori'] + crowded_ori[i])
                        trajectory_ori_list.append(self.para_dict['reset_ori'] + crowded_ori[i])
                        trajectory_ori_list.append(self.para_dict['reset_ori'] + crowded_ori[i])
                        trajectory_ori_list.append(self.para_dict['reset_ori'] + crowded_ori[i])
                        trajectory_ori_list.append(self.para_dict['reset_ori'] + crowded_ori[i])
                        # reset the manipulator to read the image
                        trajectory_ori_list.append([0, math.pi / 2, 0])

                    # only once!
                    if once_flag == True:
                        break
                last_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
                last_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))
                # trajectory_pos_list = np.asarray(trajectory_pos_list)
                # trajectory_ori_list = np.asarray(trajectory_ori_list)

                ######################### add the debug lines for visualization ####################
                line_id = []
                four_points = trajectory_pos_list[2:6]
                line_id.append(p.addUserDebugLine(lineFromXYZ=four_points[0], lineToXYZ=four_points[1]))
                line_id.append(p.addUserDebugLine(lineFromXYZ=four_points[1], lineToXYZ=four_points[2]))
                line_id.append(p.addUserDebugLine(lineFromXYZ=four_points[2], lineToXYZ=four_points[3]))
                line_id.append(p.addUserDebugLine(lineFromXYZ=four_points[3], lineToXYZ=four_points[0]))
                ######################### add the debug line for visualization ####################

                for j in range(len(trajectory_pos_list)):
                    if len(trajectory_pos_list[j]) == 3:
                        last_pos = move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j])
                        last_ori = np.copy(trajectory_ori_list[j])
                    elif len(trajectory_pos_list[j]) == 2:
                        gripper(trajectory_pos_list[j][0], trajectory_pos_list[j][1])

                ######################### remove the debug lines after moving ######################
                for i in line_id:
                    p.removeUserDebugItem(i)
                ######################### remove the debug lines after moving ######################

                ################### Check the results to determine whether to clean again #####################
                manipulator_before, new_lwh_list, pred_conf = self.get_obs()
                order = change_sequence(manipulator_before)
                manipulator_before = manipulator_before[order]
                new_lwh_list = new_lwh_list[order]
                pred_conf = pred_conf[order]
                crowded_index, prediction, model_output = self.grasp_model.pred(manipulator_before, new_lwh_list, pred_conf)
                ############## exclude boxes which have been knolling ##############
                manipulator_before = manipulator_before[len(self.success_manipulator_after):]
                prediction = prediction[len(self.success_manipulator_after):]
                new_lwh_list = new_lwh_list[len(self.success_manipulator_after):]
                crowded_index = np.setdiff1d(crowded_index, np.arange(len(self.success_manipulator_after)))
                model_output = model_output[len(self.success_manipulator_after):]
                ############## exclude boxes which have been knolling ##############
                self.yolo_pose_model.plot_grasp(manipulator_before, prediction, model_output)
                ################### Check the results to determine whether to clean again #####################
                print('here')

                manipulator_before = np.concatenate((self.success_manipulator_after, manipulator_before), axis=0)
                new_lwh_list = np.concatenate((self.success_lwh, new_lwh_list), axis=0)

            return

        def knolling(manipulator_before, new_lwh_list, crowded_index):

            pos_before = manipulator_before[:, :3]
            ori_before = manipulator_before[:, 3:6]
            manipulator_before, manipulator_after, lwh_list = self.manual_knolling(pos_before=pos_before,
                                                                                   ori_before=ori_before,
                                                                                   lwh_list=new_lwh_list,
                                                                                   crowded_index=crowded_index)
            ########### add the offset of the manipulator to avoid collision ###########
            manipulator_after[:, 0] += 0.01
            ########### add the offset of the manipulator to avoid collision ###########

            manipulator_before = manipulator_before[self.success_num:]
            manipulator_after = manipulator_after[self.success_num:]
            lwh_list = lwh_list[self.success_num:]
            # after knolling model, manipulator before and after only contain boxes which can be grasped!
            start_end = np.concatenate((manipulator_before, manipulator_after), axis=1)
            self.success_manipulator_after = np.append(self.success_manipulator_after, manipulator_after).reshape(-1, 6)
            self.success_lwh = np.append(self.success_lwh, lwh_list).reshape(-1, 3)
            self.success_num += len(manipulator_after)

            offset_low = np.array([0, 0, 0.0])
            offset_low_place = np.array([0, 0, 0.0])
            offset_high = np.array([0, 0, 0.035])
            grasp_width = np.min(lwh_list[:, :2], axis=1)
            for i in range(len(start_end)):
                trajectory_pos_list = [[0, grasp_width[i]], # gripper open!
                                       offset_high + start_end[i][:3], # move directly to the above of the target
                                       offset_low + start_end[i][:3], # decline slowly
                                       [1, grasp_width[i]], # gripper close
                                       offset_high + start_end[i][:3], # lift the box up
                                       offset_high + start_end[i][6:9], # to the target position
                                       offset_low_place + start_end[i][6:9], # decline slowly
                                       [0, grasp_width[i]], # gripper open!
                                       offset_high + start_end[i][6:9]] # rise without box
                trajectory_ori_list = [self.para_dict['reset_ori'] + start_end[i][3:6],
                                       self.para_dict['reset_ori'] + start_end[i][3:6],
                                       self.para_dict['reset_ori'] + start_end[i][3:6],
                                       [1, grasp_width[i]],
                                       self.para_dict['reset_ori'] + start_end[i][3:6],
                                       self.para_dict['reset_ori'] + start_end[i][9:12],
                                       self.para_dict['reset_ori'] + start_end[i][9:12],
                                       [0, grasp_width[i]],
                                       self.para_dict['reset_ori'] + start_end[i][9:12]]
                if i == 0:
                    last_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
                    last_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))
                    left_pos = np.asarray(p.getLinkState(self.arm_id, 7)[0])
                    right_pos = np.asarray(p.getLinkState(self.arm_id, 8)[0])
                else:
                    pass

                for j in range(len(trajectory_pos_list)):
                    if len(trajectory_pos_list[j]) == 3:
                        print('ready to move', trajectory_pos_list[j])
                        # print('ready to move cur ori', last_ori)
                        last_pos = move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j], j, task='knolling')
                        last_ori = np.copy(trajectory_ori_list[j])
                        # print('this is last ori after moving', last_ori)

                    elif len(trajectory_pos_list[j]) == 2:
                        gripper(trajectory_pos_list[j][0], trajectory_pos_list[j][1])

            ############### Back to the reset pos and ori ###############
            last_pos = move(last_pos, last_ori, self.para_dict['reset_pos'], self.para_dict['reset_ori'])
            last_ori = np.copy(self.para_dict['reset_ori'])
            ik_angles0 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=self.para_dict['reset_pos'],
                                                      maxNumIterations=200,
                                                      targetOrientation=p.getQuaternionFromEuler(self.para_dict['reset_ori']))
            for motor_index in range(5):
                p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                        targetPosition=ik_angles0[motor_index], maxVelocity=7)
            for i in range(30):
                p.stepSimulation()
                # time.sleep(1 / 48)
            ############### Back to the reset pos and ori ###############

        def check_accuracy_sim(): # need improvement

            manipulator_before, new_xyz_list = self.knolling_env.get_obs(self.para_dict['obs_order'], check='after') # the sequence along 2,3,4
            img_after = self.knolling_env.get_obs('images')
            cv2.imwrite(self.para_dict['img_save_path'] + 'images_%s_after.png' % self.evaluations, img_after)

        def check_accuracy_real():
            manipulator_before, new_xyz_list = self.knolling_env.get_obs(self.para_dict['obs_order'])
            manipulator_knolling = manipulator_before[:, :2]
            xyz_knolling = new_xyz_list
            # don't change the order of xyz in sim!!!!!!!!!!!!!

            order_knolling = np.lexsort((manipulator_before[:, 1], manipulator_before[:, 0]))
            manipulator_knolling_test = np.copy(manipulator_knolling[order_knolling, :])
            for i in range(len(order_knolling) - 1):
                if np.abs(manipulator_knolling_test[i, 0] - manipulator_knolling_test[i + 1, 0]) < 0.003:
                    if manipulator_knolling_test[i, 1] < manipulator_knolling_test[i + 1, 1]:
                        order_knolling[i], order_knolling[i + 1] = order_knolling[i + 1], \
                                                                   order_knolling[i]
                        print('knolling change the order!')
                    else:
                        pass
            print('this is the ground truth order', order_knolling)
            # print('this is the ground truth before changing the order\n', manipulator_knolling)
            manipulator_knolling = manipulator_knolling[order_knolling, :]

            new_pos_before, new_ori_before = [], []
            for i in range(len(self.lego_idx)):
                new_pos_before.append(p.getBasePositionAndOrientation(self.lego_idx[i])[0][:2])
            new_pos_before = np.asarray(new_pos_before)
            manipulator_ground_truth = new_pos_before
            xyz_ground_truth = self.xyz_list
            # don't change the order of xyz in sim!!!!!!!!!!!!!

            order_ground_truth = np.lexsort((manipulator_ground_truth[:, 1], manipulator_ground_truth[:, 0]))
            manipulator_ground_truth_test = np.copy(manipulator_ground_truth[order_ground_truth, :])
            for i in range(len(order_ground_truth) - 1):
                if np.abs(manipulator_ground_truth_test[i, 0] - manipulator_ground_truth_test[i + 1, 0]) < 0.003:
                    if manipulator_ground_truth_test[i, 1] < manipulator_ground_truth_test[i + 1, 1]:
                        order_ground_truth[i], order_ground_truth[i + 1] = order_ground_truth[i + 1], \
                                                                           order_ground_truth[i]
                        print('truth change the order!')
                    else:
                        pass
            print('this is the ground truth order', order_knolling)
            # print('this is the ground truth before changing the order\n', manipulator_knolling)
            manipulator_ground_truth = manipulator_ground_truth[order_ground_truth, :]

            print('this is manipulator ground truth while checking \n', manipulator_ground_truth)
            print('this is manipulator after knolling while checking \n', manipulator_knolling)
            print('this is xyz ground truth while checking \n', xyz_ground_truth)
            print('this is xyz after knolling while checking \n', xyz_knolling)
            for i in range(len(manipulator_ground_truth)):
                if np.linalg.norm(manipulator_ground_truth[i] - manipulator_knolling[i]) < 0.005 and \
                        np.linalg.norm(xyz_ground_truth[i] - xyz_knolling[i]) < 0.005:
                    print('find it!')
                else:
                    print('error!')

            print('this is all distance between messy and neat in real world')

        if order == 1:
            clean_grasp()
        elif order == 4:
            if self.para_dict['real_operate'] == True:
                check_accuracy_real()
            else:
                check_accuracy_sim()
    def step(self):

        self.reset() # reset the table
        self.boxes_id_recover = np.copy(self.boxes_index)
        # self.manual_knolling() # generate the knolling after data based on manual or the model
        self.calculate_gripper()

        if self.para_dict['real_operate'] == True:

            HOST = "192.168.0.186"  # Standard loopback interface address (localhost)
            PORT = 8881  # Port to listen on (non-privileged ports are > 1023)
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
            if self.generate_dict['ik_flag'] == 'manual':
                cmd_motor = np.asarray(inverse_kinematic(np.copy(reset_pos), np.copy(reset_ori)), dtype=np.float32)
                print('this is the reset motor pos', cmd_motor)
            else:
                pass
            conn.sendall(cmd_motor.tobytes())

            real_motor = conn.recv(8192)
            real_motor = np.frombuffer(real_motor, dtype=np.float32)
            real_motor = real_motor.reshape(-1, 6)

            real_xyz, _ = forward_kinematic(real_motor)
        else:
            conn = None
            table_surface_height = 0.026
            sim_table_surface_height = -0.01

        #######################################################################################
        # 1: clean_grasp, 3: knolling, 4: check_accuracy of knolling, 5: get_camera
        self.planning(1, conn, table_surface_height, sim_table_surface_height)
        # error = self.planning(5, conn, table_surface_height, sim_table_surface_height)
        # error = self.planning(3, conn, table_surface_height, sim_table_surface_height)
        # self.planning(4, conn, table_surface_height, sim_table_surface_height)
        #######################################################################################

        if self.para_dict['real_operate'] == True:
            end = np.array([0], dtype=np.float32)
            conn.sendall(end.tobytes())

        # delete_path = self.para_dict['dataset_path']
        # for f in os.listdir(delete_path):
        #     os.remove(delete_path + f)
        # shutil.rmtree(delete_path)
        # os.mkdir(delete_path)

if __name__ == '__main__':

    np.random.seed(21)
    random.seed(21)

    np.set_printoptions(precision=5)
    para_dict = {'start_num': 0, 'end_num': 10, 'thread': 9, 'evaluations': 1,
                 'yolo_conf': 0.6, 'yolo_iou': 0.5, 'device': 'cuda:0',
                 'reset_pos': np.array([0, 0, 0.12]), 'reset_ori': np.array([0, np.pi / 2, 0]),
                 'save_img_flag': True,
                 'init_pos_range': [[0.03, 0.27], [-0.13, 0.13], [0.01, 0.02]],
                 'init_ori_range': [[-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4]],
                 'boxes_num': np.random.randint(5, 6),
                 'is_render': True,
                 'box_range': [[0.016, 0.048], [0.016], [0.01, 0.02]],
                 'box_mass': 0.1,
                 'gripper_threshold': 0.002, 'gripper_sim_step': 10, 'gripper_force': 3,
                 'move_threshold': 0.001, 'move_force': 3,
                 'gripper_lateral_friction': 1, 'gripper_contact_damping': 1, 'gripper_contact_stiffness': 50000,
                 'box_lateral_friction': 1, 'box_contact_damping': 1, 'box_contact_stiffness': 50000,
                 'base_lateral_friction': 1, 'base_contact_damping': 1, 'base_contact_stiffness': 50000,
                 'dataset_path': './knolling_box/',
                 'urdf_path': './urdf/',
                 'yolo_model_path': './828_pile_pose_real_sundry/weights/best.pt',
                 'real_operate': True, 'obs_order': 'real_image_obj', 'data_collection': False,
                 'use_knolling_model': True, 'use_lstm_model': True}

    knolling_para = {'total_offset': [0.035, -0.17 + 0.016, 0], 'gap_item': 0.015,
                     'gap_block': 0.015, 'random_offset': False,
                     'area_num': 2, 'ratio_num': 1,
                     'kind_num': 5,
                     'order_flag': 'confidence',
                     'item_odd_prevent': True,
                     'block_odd_prevent': True,
                     'upper_left_max': True,
                     'forced_rotate_box': False}

    lstm_dict = {'input_size': 6,
                 'hidden_size': 32,
                 'num_layers': 8,
                 'output_size': 2,
                 'hidden_node_1': 32, 'hidden_node_2': 8,
                 'batch_size': 1,
                 'device': 'cuda:0',
                 'set_dropout': 0.1,
                 'threshold': 0.5,
                 'grasp_model_path': './Grasp_pred_model/results/LSTM_730_2_heavy_dropout0/best_model.pt',}

    arrange_dict = {'running_name': 'whole-cherry-11'}

    main_env = knolling_main(para_dict=para_dict, knolling_para=knolling_para, lstm_dict=lstm_dict, arrange_dict=arrange_dict)

    evaluation = 1
    for evaluation in range(para_dict['evaluations']):
        # env.get_parameters(evaluations=evaluation,
        #                    knolling_generate_parameters=knolling_generate_parameters,
        #                    dynamic_parameters=dynamic_parameters,
        #                    general_parameters=general_parameters,
        #                    knolling_env=knolling_env)
        main_env.step()