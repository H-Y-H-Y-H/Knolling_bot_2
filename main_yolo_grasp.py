import numpy as np
import pyrealsense2 as rs
import pybullet_data as pd
import math
from utils import *
from environment_yolo_grasp import Arm_env
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

    def clean_grasp(self):
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
        crowded_index, prediction, model_output = self.grasp_model.pred_yolo(manipulator_before, new_lwh_list,
                                                                             pred_conf)
        self.yolo_pose_model.plot_grasp(manipulator_before, prediction, model_output)
        ############ Predict the probability of grasp, remember to change the sequence of input #############
        tar_success = np.copy(len(manipulator_before))

        restrict_gripper_diagonal = np.sqrt(gripper_width ** 2 + gripper_height ** 2)
        gripper_box_gap = 0.006

        while True:

            ####### knolling only if the number of boxes we can grasp is more than 2 #######
            if len(manipulator_before) - len(crowded_index) >= 1:
                self.knolling(manipulator_before=manipulator_before, new_lwh_list=new_lwh_list, crowded_index=crowded_index)
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
                    last_pos = self.move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j])
                    last_ori = np.copy(trajectory_ori_list[j])
                elif len(trajectory_pos_list[j]) == 2:
                    self.gripper(trajectory_pos_list[j][0], trajectory_pos_list[j][1])

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
            crowded_index, prediction, model_output = self.grasp_model.pred_yolo(manipulator_before, new_lwh_list,
                                                                                 pred_conf)
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

    def gripper(self, gap, obj_width):

        if gap > 0.5:
            self.keep_obj_width = obj_width + 0.01
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

    def unstack(self):

        self.offset_low = np.array([0, 0, 0.0])
        self.offset_high = np.array([0, 0, 0.04])
        while True:
            crowded_index = np.where(self.pred_cls == 0)[0]
            manipulator_before_input = self.manipulator_before[crowded_index]
            lwh_list_input = self.lwh_list[crowded_index]
            rays = self.get_ray(manipulator_before_input, lwh_list_input)[:self.num_rays]
            out_times = 0
            fail_times = 0
            for i in range(len(rays)):

                trajectory_pos_list = [self.para_dict['reset_pos'], # move to the destination
                                       [1, 0],  # gripper close!
                                       self.offset_high + rays[i, :3],  # move directly to the above of the target
                                       self.offset_low + rays[i, :3],  # decline slowly
                                       self.offset_low + rays[i, 6:9],
                                       self.offset_high + rays[i, 6:9],
                                       self.para_dict['reset_pos'],] # unstack

                trajectory_ori_list = [self.para_dict['reset_ori'],
                                       [1, 0],
                                       self.para_dict['reset_ori'] + rays[i, 3:6],
                                       self.para_dict['reset_ori'] + rays[i, 3:6],
                                       self.para_dict['reset_ori'] + rays[i, 9:12],
                                       self.para_dict['reset_ori'] + rays[i, 9:12],
                                       self.para_dict['reset_ori'],]

                if self.para_dict['real_operate'] == True:
                    last_pos = self.para_dict['reset_pos']
                    last_ori = self.para_dict['reset_ori']
                    left_pos = None
                    right_pos = None
                else:
                    last_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
                    last_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))
                    left_pos = np.asarray(p.getLinkState(self.arm_id, 7)[0])
                    right_pos = np.asarray(p.getLinkState(self.arm_id, 8)[0])

                for j in range(len(trajectory_pos_list)):
                    if len(trajectory_pos_list[j]) == 3:
                        last_pos = self.move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j], index=j)
                        last_ori = np.copy(trajectory_ori_list[j])
                    elif len(trajectory_pos_list[j]) == 2:
                        ####################### Detect whether the gripper is disturbed by other objects during closing the gripper ####################
                        self.gripper(trajectory_pos_list[j][0], trajectory_pos_list[j][1])
                        ####################### Detect whether the gripper is disturbed by other objects during closing the gripper ####################

            self.manipulator_before, self.lwh_list, self.pred_cls, self.pred_conf = self.get_obs()
            crowded_index = np.where(self.pred_cls == 0)[0]
            if len(crowded_index) == 0:
                print('unstack success!')
                break

            # if out_times == 0 and fail_times == 0:
            #     break
            # else:
            #     add_num = fail_times + out_times
            #     print('add additional rays', add_num)

        # rewrite this variable to ensure to load data one by one
        return self.img_per_epoch

    def knolling(self):

        crowded_index = np.where(self.pred_cls == 0)[0]
        pos_before = self.manipulator_before[:, :3]
        ori_before = self.manipulator_before[:, 3:6]
        manipulator_before, manipulator_after, lwh_list = self.get_knolling_data(pos_before=pos_before,
                                                                                 ori_before=ori_before,
                                                                                 lwh_list=self.lwh_list,
                                                                                 crowded_index=crowded_index)

        manipulator_before = manipulator_before[self.success_num:]
        manipulator_after = manipulator_after[self.success_num:]
        lwh_list = lwh_list[self.success_num:]
        # after knolling model, manipulator before and after only contain boxes which can be grasped!
        start_end = np.concatenate((manipulator_before, manipulator_after), axis=1)
        self.success_manipulator_after = np.append(self.success_manipulator_after, manipulator_after).reshape(-1, 6)
        # self.success_lwh = np.append(self.success_lwh, lwh_list).reshape(-1, 3)
        self.success_num += len(manipulator_after)

        offset_low = np.array([0, 0, 0.005])
        offset_low_place = np.array([0, 0, 0.007])
        offset_high = np.array([0, 0, 0.040])
        grasp_width = np.min(lwh_list[:, :2], axis=1)
        for i in range(len(start_end)):
            trajectory_pos_list = [[0, grasp_width[i]],  # gripper open!
                                   offset_high + start_end[i][:3],  # move directly to the above of the target
                                   offset_low + start_end[i][:3],  # decline slowly
                                   [1, grasp_width[i]],  # gripper close
                                   offset_high + start_end[i][:3],  # lift the box up
                                   offset_high + start_end[i][6:9],  # to the target position
                                   offset_low_place + start_end[i][6:9],  # decline slowly
                                   [0, grasp_width[i]],  # gripper open!
                                   offset_high + start_end[i][6:9]]  # rise without box
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
                    last_pos = self.move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j], j, task='knolling')
                    last_ori = np.copy(trajectory_ori_list[j])
                    # print('this is last ori after moving', last_ori)

                elif len(trajectory_pos_list[j]) == 2:
                    self.gripper(trajectory_pos_list[j][0], trajectory_pos_list[j][1])

        ############### Back to the reset pos and ori ###############
        last_pos = self.move(last_pos, last_ori, self.para_dict['reset_pos'], self.para_dict['reset_ori'])
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

        self.get_obs(look_flag=True, img_path='main_knolling')

    def step(self):

        self.manipulator_before, self.lwh_list, self.pred_cls, self.pred_conf = self.reset() # reset the table
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
            self.sim_table_height = -0.01

        # self.conn = None
        # self.real_table_height = 0.026
        # self.sim_table_height = -0.01

        #######################################################################################
        # 1: clean_grasp + knolling, 3: knolling, 4: check_accuracy of knolling, 5: get_camera
        self.unstack()
        # self.knolling()
        #######################################################################################

        if self.para_dict['real_operate'] == True:
            end = np.array([0], dtype=np.float32)
            self.conn.sendall(end.tobytes())

if __name__ == '__main__':

    # np.random.seed(21)
    # random.seed(21)

    np.set_printoptions(precision=5)
    para_dict = {'start_num': 0, 'end_num': 10, 'thread': 9, 'evaluations': 1,
                 'yolo_conf': 0.6, 'yolo_iou': 0.8, 'device': 'cuda:0',
                 'reset_pos': np.array([0, 0, 0.12]), 'reset_ori': np.array([0, np.pi / 2, 0]),
                 'save_img_flag': True,
                 'init_pos_range': [[0.13, 0.17], [-0.03, 0.03], [0.01, 0.02]], 'init_offset_range': [[-0.05, 0.05], [-0.1, 0.1]],
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
                 'yolo_model_path': './models/627_pile_pose/weights/best.pt',
                 'real_operate': False, 'obs_order': 'sim_image_obj', 'data_collection': False,
                 'use_knolling_model': True, 'use_lstm_model': False, 'use_yolo_model': True}

    if para_dict['real_operate'] == False:
        para_dict['yolo_model_path'] = './models/924_grasp/weights/best.pt'
    else:
        para_dict['yolo_model_path'] = './models/830_pile_real_box/weights/best.pt'


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
                 'threshold': 0.50,
                 'grasp_model_path': './models/LSTM_918_0/best_model.pt',}

    arrange_dict = {'running_name': 'devoted-terrain-29',
                    'transformer_model_path': './models/devoted-terrain-29',
                    'use_yaml': True}

    main_env = knolling_main(para_dict=para_dict, knolling_para=knolling_para, lstm_dict=lstm_dict, arrange_dict=arrange_dict)

    evaluation = 1
    for evaluation in range(para_dict['evaluations']):
        # env.get_parameters(evaluations=evaluation,
        #                    knolling_generate_parameters=knolling_generate_parameters,
        #                    dynamic_parameters=dynamic_parameters,
        #                    general_parameters=general_parameters,
        #                    knolling_env=knolling_env)
        main_env.step()