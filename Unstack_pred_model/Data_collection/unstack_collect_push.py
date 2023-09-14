from environment import Arm_env
import pybullet as p
import pybullet_data as pd
import numpy as np
import random
import os
import socket
from utils import *

class Unstack_env(Arm_env):

    def __init__(self, para_dict=None, lstm_dict=None):

        super(Unstack_env, self).__init__(para_dict=para_dict, lstm_dict=lstm_dict)
        self.table_center = np.array([0.15, 0])
        # self.begin_pos = np.array([0.04, 0, 0.12])
        # self.begin_ori = np.array([0, np.pi / 2, 0])
        self.num_rays = self.para_dict['num_rays']
        self.tune_range = [-0.01, 0.01]

        self.offset_low = np.array([0, 0, 0.0])
        self.offset_high = np.array([0, 0, 0.04])

        self.angular_distance = np.pi / 8
        if self.para_dict['real_operate'] == True:
            self.ray_height = 0.015
        else:
            self.ray_height = 0.01

        self.grid_size = 5
        self.x_range = (-0.001, 0.001)
        self.y_range = (-0.001, 0.001)
        x_center = 0
        y_center = 0

        x_offset_values = np.linspace(self.x_range[0], self.x_range[1], self.grid_size)
        y_offset_values = np.linspace(self.y_range[0], self.y_range[1], self.grid_size)
        xx, yy = np.meshgrid(x_offset_values, y_offset_values)
        sigma = 0.01
        kernel = np.exp(-((xx - x_center) ** 2 + (yy - y_center) ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
        self.kernel = kernel / np.sum(kernel)

        self.success = 0

    def move(self, cur_pos, cur_ori, tar_pos, tar_ori, sim_height=-0.01, origin_left_pos=None, origin_right_pos=None, index=None):

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

            print('this is cmd zzz\n', cmd_xyz[-1])
            return cmd_xyz[-1], True  # return cur pos to let the manipualtor remember the improved pos

        else:
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
                                        targetPosition=self.motor_pos(self.mark_gripper_pos) + self.close_open_gap,
                                        force=self.para_dict['gripper_force'] * 10)
                p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL,
                                        targetPosition=self.motor_pos(self.mark_gripper_pos) + self.close_open_gap,
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
                                            targetPosition=ik_angles0[motor_index], maxVelocity=25, force=self.para_dict['move_force'])
                move_success_flag = True
                for i in range(10):
                    p.stepSimulation()
                    if self.is_render:
                        time.sleep(1 / 1080)
                cur_pos = tar_pos
                cur_ori = tar_ori
                if abs(target_pos[0] - tar_pos[0]) < 0.001 and abs(target_pos[1] - tar_pos[1]) < 0.001 and abs(
                        target_pos[2] - tar_pos[2]) < 0.001 and \
                        abs(target_ori[0] - tar_ori[0]) < 0.001 and abs(target_ori[1] - tar_ori[1]) < 0.001 and abs(
                    target_ori[2] - tar_ori[2]) < 0.001:
                    break

            ee_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
            if ee_pos[2] - target_pos[2] > 0.005 and (index == 2 or index == 3 or index == 4) and move_success_flag == True:
                move_success_flag = False
                print('ee can not reach the bottom, fail!')

            return cur_pos, move_success_flag

    def gripper(self, gap, obj_width, left_pos, right_pos, index=None):

        if self.para_dict['real_operate'] == True:
            obj_width += 0.006
            obj_width_range = np.array([0.021, 0.026, 0.032, 0.039, 0.043, 0.046, 0.050])
            motor_pos_range = np.array([2000, 2100, 2200, 2300, 2400, 2500, 2600])
            formula_parameters = np.polyfit(obj_width_range, motor_pos_range, 3)
            motor_pos = np.poly1d(formula_parameters)

            if gap > 0.5:  # close
                pos_real = np.asarray([[gap, 1600]], dtype=np.float32)
            elif gap <= 0.5:  # open
                pos_real = np.asarray([[gap, motor_pos(obj_width)]], dtype=np.float32)
            print('gripper', pos_real)
            self.conn.sendall(pos_real.tobytes())

            real_pos = self.conn.recv(8192)
            # test_real_pos = np.frombuffer(real_pos, dtype=np.float32)
            real_pos = np.frombuffer(real_pos, dtype=np.float32)
            # print('this is test float from buffer', test_real_pos)

        else:
            if index == 4:
                self.keep_obj_width = obj_width + 0.01
            obj_width += 0.010
            gripper_success_flag = True
            if index == 1:
                num_step = 30
            else:
                num_step = 10

            if gap > 0.5:
                gripper_pos = self.motor_pos(obj_width) + self.close_open_gap
            else:
                gripper_pos = self.motor_pos(obj_width)
            p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL, targetPosition=gripper_pos, force=self.para_dict['gripper_force'])
            p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL, targetPosition=gripper_pos, force=self.para_dict['gripper_force'])
            for i in range(num_step):
                p.stepSimulation()
                if self.is_render:
                    time.sleep(1 / 48)

            self.mark_gripper_pos = gripper_pos

    def check_bound(self):

        flag = True
        pos_before = []
        ori_before = []
        qua_before = []
        for i in range(len(self.boxes_index)):
            cur_pos = np.asarray(p.getBasePositionAndOrientation(self.boxes_index[i])[0])
            if cur_pos[0] < self.x_low_obs or cur_pos[0] > self.x_high_obs or \
                cur_pos[1] > self.y_high_obs or cur_pos[1] < self.y_low_obs:
                flag = False
            qua_before.append(np.asarray(p.getBasePositionAndOrientation(self.boxes_index[i])[1]))
            ori_before.append(
                np.asarray(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.boxes_index[i])[1])))
            pos_before.append(cur_pos)
        pos_before = np.asarray(pos_before)
        ori_before = np.asarray(ori_before)
        qua_before = np.asarray(qua_before)
        # output_data = np.concatenate((pos_before, ori_before, self.lwh_list, qua_before), axis=1)

        return None, flag

    def get_ray(self, pos_ori, lwh):

        stack_center = np.mean(pos_ori[:, :2], axis=0)
        far_box_index = np.argmax(np.linalg.norm(pos_ori[:, :2] - stack_center, axis=1))
        far_box_pos = pos_ori[far_box_index, :2]
        # radius_start = (np.linalg.norm(stack_center - far_box_pos) +
        #           np.max(np.linalg.norm(lwh[:, :2], axis=1) / 2) +
        #           np.linalg.norm([self.gripper_width, self.gripper_height]) / 2 +
        #           self.gripper_interval)
        radius_end = (np.linalg.norm(stack_center - far_box_pos))
        radius_start = (np.linalg.norm(stack_center - far_box_pos) +
                        np.max(np.linalg.norm(lwh[:, :2], axis=1) / 2) +
                        self.gripper_width / 2)
        # radius_end = 0
        center_list = self.gaussian_center(stack_center)
        angle_center_stack = np.arctan2(stack_center[1] - self.table_center[1], stack_center[0] - self.table_center[0])
        # angle_list = np.random.uniform(-np.pi, np.pi, self.num_rays)
        angle_list = np.random.uniform(angle_center_stack - (self.num_rays // 2) * self.angular_distance,
                                       angle_center_stack + (self.num_rays // 2) * self.angular_distance, self.num_rays)

        x_offset_start = np.cos(angle_list) * radius_start
        y_offset_start = np.sin(angle_list) * radius_start
        x_offset_end = np.cos(angle_list) * radius_end
        y_offset_end = np.sin(angle_list) * radius_end

        angle_list += np.pi / 2
        large_angle_index = np.where(angle_list > np.pi)[0]
        angle_list[large_angle_index] -= np.pi
        small_angle_index = np.where(angle_list < -np.pi)[0]
        angle_list[small_angle_index] += np.pi

        start_pos = np.array([center_list[:, 0] + x_offset_start, center_list[:, 1] + y_offset_start]).T
        end_pos = np.array([center_list[:, 0] - x_offset_end, center_list[:, 1] - y_offset_end]).T

        rays = np.concatenate((start_pos.reshape(-1, 2), np.ones(self.num_rays).reshape(-1, 1) * self.ray_height,
                               np.zeros((self.num_rays, 2)), angle_list.reshape(-1, 1),
                               end_pos.reshape(-1, 2), np.ones(self.num_rays).reshape(-1, 1) * self.ray_height,
                               np.zeros((self.num_rays, 2)), angle_list.reshape(-1, 1)), axis=1)
        return rays

    def gaussian_center(self, stack_center):

        x_blured = stack_center[0] + np.linspace(self.x_range[0], self.x_range[1], self.grid_size ** 2)
        y_blured = stack_center[1] + np.linspace(self.y_range[0], self.y_range[1], self.grid_size ** 2)
        flattened_prob = self.kernel.flatten()
        selected_indices = np.random.choice(len(flattened_prob), self.num_rays, p=flattened_prob)
        selected_x = x_blured[selected_indices].reshape(-1, 1)
        selected_y = y_blured[selected_indices].reshape(-1, 1)
        center_list = np.concatenate((selected_x, selected_y), axis=1)

        return center_list

    def real_world_warmup(self):

        if self.para_dict['real_operate'] == True:

            HOST = "192.168.0.186"  # Standard loopback interface address (localhost)
            PORT = 8880 # Port to listen on (non-privileged ports are > 1023)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((HOST, PORT))
            # It should be an integer from 1 to 65535, as 0 is reserved. Some systems may require superuser privileges if the port number is less than 8192.
            # associate the socket with a specific network interface
            s.listen()
            print(f"Waiting for connection...\n")
            self.conn, addr = s.accept()
            print(self.conn)
            print(f"Connected by {addr}")
            cmd_motor = np.asarray(inverse_kinematic(np.copy(self.para_dict['reset_pos']), np.copy(self.para_dict['reset_ori'])), dtype=np.float32)

            print('this is the reset motor pos', cmd_motor)
            self.conn.sendall(cmd_motor.tobytes())

            real_motor = self.conn.recv(8192)
            real_motor = np.frombuffer(real_motor, dtype=np.float32)
            real_motor = real_motor.reshape(-1, 6)
            real_xyz, _ = forward_kinematic(real_motor)

            self.img_per_epoch = 0

    def try_unstack(self, data_root=None, data_index=None, unstack_data_start_index=None):

        # if self.img_per_epoch + data_index >= self.endnum:
        #     print('END!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #     print('this is success', self.success)
        #     if self.para_dict['real_operate'] == True:
        #         end = np.array([0], dtype=np.float32)
        #         self.conn.sendall(end.tobytes())
        #     quit()

        # manipulator_before_input, new_lwh_list_input, pred_conf_input, crowded_index, prediction, model_output = self.get_obs(epoch=self.img_per_epoch + img_index_start, baseline_flag=True)
        # if self.para_dict['real_operate'] == False:
        #     if len(manipulator_before_input) <= 1 or len(self.boxes_index) == 1:
        #         print('no pile in the environment, try to reset!')
        #         return self.img_per_epoch
        # else:
        #     while len(crowded_index) < len(manipulator_before_input):
        #         manipulator_before_input, new_lwh_list_input, pred_conf_input, crowded_index, prediction, model_output = self.get_obs(
        #             epoch=self.img_per_epoch + img_index_start, baseline_flag=True)
        #         print('There are some boxes can be grasp, try to take another picture!')
        #
        # # initiate the input data
        # input_data = np.concatenate((manipulator_before_input, new_lwh_list_input, pred_conf_input.reshape(-1, 1), model_output), axis=1)
        #
        # if len(crowded_index) < len(manipulator_before_input):
        #     print('There are some boxes can be grasp, try to reset!')
        #     # os.remove(data_root + 'sim_images/%012d_pred.png' % (self.img_per_epoch + img_index_start))
        #     # os.remove(data_root + 'sim_images/%012d_pred_grasp.png' % (self.img_per_epoch + img_index_start))
        #     # os.remove(data_root + 'sim_images/%012d.png' % (self.img_per_epoch + img_index_start))
        #     return self.img_per_epoch

        unstack_data_start_index = np.copy(unstack_data_start_index)

        self.calculate_gripper()
        pred_info = np.loadtxt(data_root + 'pred_info/%012d.txt' % data_index)
        manipulator_before_input = pred_info[:, :6]
        new_lwh_list_input = pred_info[:, 6:9]
        self.state_id = p.saveState()


        add_num = self.num_rays
        while True:

            rays = self.get_ray(manipulator_before_input, new_lwh_list_input)[:add_num]
            out_times = 0
            fail_times = 0
            for i in range(len(rays)):

                trajectory_pos_list = [self.para_dict['reset_pos'], # move to the destination
                                       [1, 0],  # gripper close!
                                       self.offset_high + rays[i, :3],  # move directly to the above of the target
                                       self.offset_low + rays[i, :3],  # decline slowly
                                       self.offset_low + rays[i, 6:9]] # unstack

                trajectory_ori_list = [self.para_dict['reset_ori'],
                                       [1, 0],
                                       self.para_dict['reset_ori'] + rays[i, 3:6],
                                       self.para_dict['reset_ori'] + rays[i, 3:6],
                                       self.para_dict['reset_ori'] + rays[i, 9:12]]

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

                move_success_flag = True
                for j in range(len(trajectory_pos_list)):
                    if len(trajectory_pos_list[j]) == 3:
                        last_pos, move_success_flag = self.move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j], index=j)
                        last_ori = np.copy(trajectory_ori_list[j])
                        if move_success_flag == False:
                            break
                    elif len(trajectory_pos_list[j]) == 2:
                        ####################### Dtect whether the gripper is disturbed by other objects during closing the gripper ####################
                        self.gripper(trajectory_pos_list[j][0], trajectory_pos_list[j][1], left_pos, right_pos, index=j)
                        ####################### Detect whether the gripper is disturbed by other objects during closing the gripper ####################

                if move_success_flag == True:

                    if self.para_dict['real_operate'] == True:
                        check_flag = True
                    else:
                        _, check_flag = self.check_bound()
                    if check_flag == False:
                        print('object out of bound, try another yaw')
                        out_times += 1
                    else:
                        self.to_home()
                        # if self.para_dict['real_operate'] == False:
                        if len(self.boxes_index) == 1:
                            print('no pile in the environment, try to reset!')
                            return self.img_per_epoch
                        else:
                            # print('ok, record the image of this ray')
                            output_data = rays[i].reshape(2, -1)
                            np.savetxt(data_root + 'unstack_rays/%012d.txt' % (self.img_per_epoch + unstack_data_start_index), output_data, fmt='%.04f')
                            img = self.get_obs(epoch=self.img_per_epoch + unstack_data_start_index, look_flag=True)
                            # self.yolo_pose_model.plot_unstack(output_data, img=img, epoch=self.img_per_epoch + unstack_data_start_index)
                            self.img_per_epoch += 1
                else:
                    fail_times += 1
                p.restoreState(self.state_id)

            if out_times == 0 and fail_times == 0:
                break
            else:
                add_num = fail_times + out_times
                print('add additional rays', add_num)

        print('this is record index', unstack_data_start_index)
        # rewrite this variable to ensure to load data one by one
        return self.img_per_epoch

        # if out_times >= len(rays) or fail_times >= len(rays) or max_grasp_num == 0:
        #     print('all methods fail, abort this epoch!')
        #     return self.img_per_epoch
        # else:
        #     # output_data = rays[max_grasp_index].reshape(2, -1)
        #     print(f'this is output index {max_grasp_index}')
        #     print('output data', output_data)
        #     # output: xyz, xyz
        #     # input: box_xyz, box_ori, box_lwh, yolo_conf, grasp_fail_success
        #     if self.para_dict['real_operate'] == True:
        #         np.savetxt(data_root + "real_labels_unstack/%012d.txt" % (img_index_start + self.img_per_epoch), output_data, fmt='%.04f')
        #         np.savetxt(data_root + "real_labels_box/%012d.txt" % (img_index_start + self.img_per_epoch), input_data, fmt='%.04f')
        #     else:
        #         np.savetxt(data_root + "sim_labels_unstack/%012d.txt" % (img_index_start + self.img_per_epoch), output_data, fmt='%.04f')
        #         np.savetxt(data_root + "sim_labels_box/%012d.txt" % (img_index_start + self.img_per_epoch), input_data, fmt='%.04f')
        #     # if self.save_img_flag == False:
        #     #     os.remove(data_root + 'sim_images/%012d.png' % (self.img_per_epoch + img_index_start))
        #     self.img_per_epoch += 1
        #
        #     if max_grasp_num >= 2:
        #         print('try to tune the unstack in the best ray!')
        #         self.tune_unstack(best_ray=output_data, best_num=max_grasp_num, img_index_start=img_index_start, input_data=input_data)
        #
        #     print('this is total num of img after one epoch', self.img_per_epoch + img_index_start)
        #     return self.img_per_epoch

if __name__ == '__main__':

    # np.random.seed(111)
    # random.seed(111)

    # simulation: iou 0.8
    # real world: iou=0.5

    para_dict = {'start_num': 0, 'end_num': 20000, 'thread': 0, 'input_output_label_offset': 100000, 'num_rays': 5,
                 'yolo_conf': 0.6, 'yolo_iou': 0.8, 'device': 'cuda:0',
                 'reset_pos': np.array([0.0, 0, 0.10]), 'reset_ori': np.array([0, np.pi / 2, 0]),
                 'save_img_flag': True,
                 'init_pos_range': [[0.13, 0.17], [-0.03, 0.03], [0.01, 0.02]], 'init_offset_range': [[-0.05, 0.05], [-0.1, 0.1]],
                 'init_ori_range': [[-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4]],
                 'boxes_num': np.random.randint(5, 6),
                 'is_render': False,
                 'box_range': [[0.016, 0.048], [0.016], [0.01, 0.016]],
                 'box_mass': 0.1,
                 'gripper_threshold': 0.002, 'gripper_sim_step': 10, 'gripper_force': 3,
                 'move_threshold': 0.001, 'move_force': 30,
                 'gripper_lateral_friction': 1, 'gripper_contact_damping': 1, 'gripper_contact_stiffness': 50000,
                 'box_lateral_friction': 1, 'box_contact_damping': 1, 'box_contact_stiffness': 50000,
                 'base_lateral_friction': 1, 'base_contact_damping': 1, 'base_contact_stiffness': 50000,
                 'dataset_path': '../../../knolling_dataset/MLP_unstack_908_intensive/',
                 'urdf_path': '../../urdf/',
                 'yolo_model_path': '../../models/627_pile_pose/weights/best.pt',
                 'real_operate': False, 'obs_order': 'sim_image_obj', 'data_collection': True, 'rl_configuration': True,
                 'use_knolling_model': False, 'use_lstm_model': True}

    lstm_dict = {'input_size': 6,
                 'hidden_size': 32,
                 'num_layers': 8,
                 'output_size': 2,
                 'hidden_node_1': 32, 'hidden_node_2': 8,
                 'batch_size': 1,
                 'set_dropout': 0.0,
                 'threshold': 0.6,
                 'device': 'cuda:0',
                 'grasp_model_path': '../../models/LSTM_829_1_heavy_dropout0/best_model.pt', }

    data_root = para_dict['dataset_path']
    with open(para_dict['dataset_path'][:-1] + '_readme.txt', "w") as f:
        for key, value in para_dict.items():
            f.write(key + ': ')
            f.write(str(value) + '\n')

    os.makedirs(data_root, exist_ok=True)

    env = Unstack_env(para_dict=para_dict, lstm_dict=lstm_dict)

    os.makedirs(data_root + 'unstack_images/', exist_ok=True)
    os.makedirs(data_root + 'unstack_rays/', exist_ok=True)

    if para_dict['real_operate'] == True:
        env.real_world_warmup()
    # while True:
    origin_data_start = para_dict['start_num']
    origin_data_end = para_dict['end_num']
    exist_img_num = (origin_data_start + para_dict['input_output_label_offset']) * para_dict['num_rays']

    for i in range(origin_data_start, origin_data_end):
        if para_dict['real_operate'] == False:
            env.reset(epoch=i, recover_flag=True)
        img_per_epoch = env.try_unstack(data_root=data_root, data_index=i, unstack_data_start_index=exist_img_num)
        exist_img_num += img_per_epoch

