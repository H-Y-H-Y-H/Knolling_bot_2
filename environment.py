from yolo_pose_deploy import *

from arrangement import *
# from grasp_model_deploy import *
from arrange_model_deploy import *
from utils import *
import pybullet as p
import pybullet_data as pd
import os
import numpy as np
import random
import math
import cv2
from urdfpy import URDF
from tqdm import tqdm
import time
import torch
# from sklearn.preprocessing import MinMaxScaler

class Arm_env():

    def __init__(self, para_dict, knolling_para=None, lstm_dict=None, arrange_dict=None):


        self.para_dict = para_dict
        self.knolling_para = knolling_para

        self.kImageSize = {'width': 480, 'height': 480}
        self.endnum = para_dict['end_num']
        self.init_pos_range = para_dict['init_pos_range']
        self.init_ori_range = para_dict['init_ori_range']
        self.init_offset_range = para_dict['init_offset_range']
        self.urdf_path = para_dict['urdf_path']
        self.pybullet_path = pd.getDataPath()
        self.is_render = para_dict['is_render']
        self.save_img_flag = para_dict['save_img_flag']
        self.yolo_pose_model = Yolo_pose_model(para_dict=para_dict, lstm_dict=lstm_dict, use_lstm=self.para_dict['use_lstm_model'])
        # self.yolo_pose_model = Yolo_pose_model(None, None)
        # self.yolo_pose_model.yolo_pose_test()
        # self.yolo_seg_model = Yolo_seg_model(para_dict=para_dict)
        self.boxes_sort = Sort_objects(para_dict=para_dict, knolling_para=knolling_para)
        if self.para_dict['use_lstm_model'] == True:
            self.lstm_dict = lstm_dict
            # self.grasp_model = Grasp_model(para_dict=para_dict, lstm_dict=lstm_dict)
        if self.para_dict['use_knolling_model'] == True:
            self.arrange_dict = arrange_dict
            self.arrange_model = Arrange_model(para_dict=para_dict, arrange_dict=arrange_dict)

        self.x_low_obs = 0.03
        self.x_high_obs = 0.27
        self.y_low_obs = -0.14
        self.y_high_obs = 0.14
        self.z_low_obs = 0.0
        self.z_high_obs = 0.05
        x_grasp_accuracy = 0.2
        y_grasp_accuracy = 0.2
        z_grasp_accuracy = 0.2
        self.x_grasp_interval = (self.x_high_obs - self.x_low_obs) * x_grasp_accuracy
        self.y_grasp_interval = (self.y_high_obs - self.y_low_obs) * y_grasp_accuracy
        self.z_grasp_interval = (self.z_high_obs - self.z_low_obs) * z_grasp_accuracy
        self.table_boundary = 0.03

        if self.para_dict['real_operate'] == False:
            self.gripper_width = 0.024
            self.gripper_height = 0.034
        else:
            self.gripper_width = 0.018
            self.gripper_height = 0.04
        self.gripper_interval = 0.01

        if self.is_render:
            p.connect(p.GUI, options="--width=1280 --height=720")
        else:
            p.connect(p.DIRECT)

        self.camera_parameters = {
            'width': 640.,
            'height': 480,
            'fov': 42,
            'near': 0.1,
            'far': 100.,
            'camera_up_vector':
                [1, 0, 0],  # I really do not know the parameter's effect.
            'light_direction': [
                0.5, 0, 1
            ],  # the direction is from the light source position to the origin of the world frame.
        }
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.150, 0, 0], #0.175
                                                               distance=0.4,
                                                               yaw=90,
                                                               pitch = -90,
                                                               roll=0,
                                                               upAxisIndex=2)
        self.projection_matrix = p.computeProjectionMatrixFOV(fov=self.camera_parameters['fov'],
                                                              aspect=self.camera_parameters['width'] /
                                                                     self.camera_parameters['height'],
                                                              nearVal=self.camera_parameters['near'],
                                                              farVal=self.camera_parameters['far'])
        p.resetDebugVisualizerCamera(cameraDistance=0.5,
                                     cameraYaw=45,
                                     cameraPitch=-45,
                                     cameraTargetPosition=[0.1, 0, 0])
        p.setAdditionalSearchPath(pd.getDataPath())
        # p.setPhysicsEngineParameter(numSolverIterations=10)
        p.setTimeStep(1. / 120.)

    def create_scene(self):

        if random.uniform(0, 1) > 0.5:
            p.configureDebugVisualizer(lightPosition=[np.random.randint(1, 2), np.random.uniform(0, 1.5), 2],
                                       shadowMapResolution=8192, shadowMapIntensity=np.random.randint(0, 1) / 10)
        else:
            p.configureDebugVisualizer(lightPosition=[np.random.randint(1, 2), np.random.uniform(-1.5, 0), 2],
                                       shadowMapResolution=8192, shadowMapIntensity=np.random.randint(0, 1) / 10)
        self.baseid = p.loadURDF(self.urdf_path + "plane_zzz.urdf", useMaximalCoordinates=True)

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

        background = np.random.randint(1, 5)
        textureId = p.loadTexture(self.urdf_path + f"img_{background}.png")
        p.changeVisualShape(self.baseid, -1, textureUniqueId=textureId, specularColor=[0, 0, 0])

        p.setGravity(0, 0, -10)
        # wall_id = []
        # wall_pos = np.array([[self.x_low_obs - self.table_boundary, 0, 0],
        #                      [(self.x_low_obs + self.x_high_obs) / 2, self.y_low_obs - self.table_boundary, 0],
        #                      [self.x_high_obs + self.table_boundary, 0, 0],
        #                      [(self.x_low_obs + self.x_high_obs) / 2, self.y_high_obs + self.table_boundary, 0]])
        # wall_ori = np.array([[0, 1.57, 0],
        #                      [0, 1.57, 1.57],
        #                      [0, 1.57, 0],
        #                      [0, 1.57, 1.57]])
        # for i in range(len(wall_pos)):
        #     wall_id.append(p.loadURDF(os.path.join(self.urdf_path, "plane_2.urdf"), basePosition=wall_pos[i],
        #                               baseOrientation=p.getQuaternionFromEuler(wall_ori[i]), useFixedBase=1,
        #                               flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
        #     p.changeVisualShape(wall_id[i], -1, rgbaColor=(1, 1, 1, 0))

    def to_home(self):

        ik_angles0 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=self.para_dict['reset_pos'],
                                                  maxNumIterations=200,
                                                  targetOrientation=p.getQuaternionFromEuler(
                                                      self.para_dict['reset_ori']))
        for motor_index in range(5):
            p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                    targetPosition=ik_angles0[motor_index], maxVelocity=20)
        for _ in range(int(30)):
            # time.sleep(1/480)
            p.stepSimulation()

    def create_arm(self):

        self.arm_id = p.loadURDF(os.path.join(self.urdf_path, "robot_arm928/robot_arm1_backup.urdf"),
                                 basePosition=[-0.08, 0, 0.02], useFixedBase=True,
                                 flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        p.changeDynamics(self.arm_id, 7, lateralFriction=self.para_dict['gripper_lateral_friction'],
                         contactDamping=self.para_dict['gripper_contact_damping'],
                         contactStiffness=self.para_dict['gripper_contact_stiffness'])
        p.changeDynamics(self.arm_id, 8, lateralFriction=self.para_dict['gripper_lateral_friction'],
                         contactDamping=self.para_dict['gripper_contact_damping'],
                         contactStiffness=self.para_dict['gripper_contact_stiffness'])
        self.to_home()


    def create_objects(self, manipulator_after, lwh_after):

        if manipulator_after is None:
            if self.para_dict['real_operate'] == False:
                self.lwh_list = self.boxes_sort.get_data_virtual()
                self.num_boxes = np.copy(len(self.lwh_list))
                rdm_ori_roll  = np.random.uniform(self.init_ori_range[0][0], self.init_ori_range[0][1], size=(self.num_boxes, 1))
                rdm_ori_pitch = np.random.uniform(self.init_ori_range[1][0], self.init_ori_range[1][1], size=(self.num_boxes, 1))
                rdm_ori_yaw   = np.random.uniform(self.init_ori_range[2][0], self.init_ori_range[2][1], size=(self.num_boxes, 1))
                rdm_ori = np.concatenate((rdm_ori_roll, rdm_ori_pitch, rdm_ori_yaw), axis=1)
                rdm_pos_x = np.random.uniform(self.init_pos_range[0][0], self.init_pos_range[0][1], size=(self.num_boxes, 1))
                rdm_pos_y = np.random.uniform(self.init_pos_range[1][0], self.init_pos_range[1][1], size=(self.num_boxes, 1))
                rdm_pos_z = np.random.uniform(self.init_pos_range[2][0], self.init_pos_range[2][1], size=(self.num_boxes, 1))
                x_offset = np.random.uniform(self.init_offset_range[0][0], self.init_offset_range[0][1])
                y_offset = np.random.uniform(self.init_offset_range[1][0], self.init_offset_range[1][1])
                print('this is offset: %.04f, %.04f' % (x_offset, y_offset))
                rdm_pos = np.concatenate((rdm_pos_x + x_offset, rdm_pos_y + y_offset, rdm_pos_z), axis=1)
            else:
                # the sequence here is based on area and ratio!!! must be converted additionally!!!
                # self.lwh_list, rdm_pos, rdm_ori, self.all_index, self.transform_flag = self.boxes_sort.get_data_real(self.yolo_model, self.para_dict['evaluations'])
                manipulator_init, lwh_list_init, _ = self.get_obs()
                rdm_pos = manipulator_init[:, :3]
                rdm_ori = manipulator_init[:, 3:]
                self.lwh_list = lwh_list_init
                self.num_boxes = np.copy(len(self.lwh_list))

        else:
            self.lwh_list = np.copy(lwh_after)
            rdm_pos = np.copy(manipulator_after[:, :3])
            rdm_ori = np.copy(manipulator_after[:, 3:])
            self.num_boxes = len(manipulator_after)

        self.boxes_index = []
        for i in range(self.num_boxes):
            obj_name = f'object_{i}'
            create_box(obj_name, rdm_pos[i], p.getQuaternionFromEuler(rdm_ori[i]), size=self.lwh_list[i])
            self.boxes_index.append(int(i + 2))
            r = np.random.uniform(0, 0.9)
            g = np.random.uniform(0, 0.9)
            b = np.random.uniform(0, 0.9)
            p.changeVisualShape(self.boxes_index[i], -1, rgbaColor=(r, g, b, 1))

        for _ in range(int(100)):
            p.stepSimulation()
            if self.is_render == True:
                time.sleep(1/96)

        p.changeDynamics(self.baseid, -1, lateralFriction=self.para_dict['base_lateral_friction'],
                         contactDamping=self.para_dict['base_contact_damping'],
                         contactStiffness=self.para_dict['base_contact_stiffness'])

    def delete_objects(self, manipulator_after):

        if self.para_dict['real_operate'] == False and manipulator_after is None:
            forbid_range = np.array([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            while True:
                new_num_item = len(self.boxes_index)
                delete_index = []

                self.gt_pos_ori = []
                self.gt_ori_qua = []
                for i in range(len(self.boxes_index)):
                    p.changeDynamics(self.boxes_index[i], -1, lateralFriction=self.para_dict['box_lateral_friction'],
                                                         contactDamping=self.para_dict['box_contact_damping'],
                                                         contactStiffness=self.para_dict['box_contact_stiffness'])
                    cur_qua = np.asarray(p.getBasePositionAndOrientation(self.boxes_index[i])[1])
                    cur_ori = np.asarray(p.getEulerFromQuaternion(cur_qua))
                    cur_pos = np.asarray(p.getBasePositionAndOrientation(self.boxes_index[i])[0])
                    self.gt_pos_ori.append(cur_pos)
                    self.gt_ori_qua.append(cur_qua)
                    roll_flag = False
                    pitch_flag = False
                    # print('this is cur ori:', cur_ori)
                    for j in range(len(forbid_range)):
                        if np.abs(cur_ori[0] - forbid_range[j]) < 0.1:
                            roll_flag = True
                        if np.abs(cur_ori[1] - forbid_range[j]) < 0.1:
                            pitch_flag = True
                    if roll_flag == True and pitch_flag == True and (
                            np.abs(cur_ori[0] - 0) > 0.1 or np.abs(cur_ori[1] - 0) > 0.1) or \
                            cur_pos[0] < self.x_low_obs or cur_pos[0] > self.x_high_obs or cur_pos[
                        1] > self.y_high_obs or \
                            cur_pos[1] < self.y_low_obs:
                        delete_index.append(i)
                        # print('delete!!!')
                        new_num_item -= 1
                self.gt_pos_ori = np.asarray(self.gt_pos_ori)
                self.gt_ori_qua = np.asarray(self.gt_ori_qua)

                delete_index.reverse()
                for i in delete_index:
                    p.removeBody(self.boxes_index[i])
                    self.boxes_index.pop(i)
                    self.lwh_list = np.delete(self.lwh_list, i, axis=0)
                    self.gt_pos_ori = np.delete(self.gt_pos_ori, i, axis=0)
                    self.gt_ori_qua = np.delete(self.gt_ori_qua, i, axis=0)
                for _ in range(int(100)):
                    # time.sleep(1/96)
                    p.stepSimulation()

                if len(delete_index) == 0:
                    break

    def reset(self, epoch=None, manipulator_after=None, lwh_after=None):

        p.resetSimulation()
        self.create_scene()
        self.create_arm()
        self.create_objects(manipulator_after, lwh_after)
        self.delete_objects(manipulator_after)
        self.img_per_epoch = 0
        # return img_per_epoch_result

    def get_knolling_data(self, pos_before, ori_before, lwh_list, crowded_index):  # this is main function!!!!!!!!!

        if self.para_dict['use_knolling_model'] == True:
            input_index = np.setdiff1d(np.arange(len(pos_before)), crowded_index)
            pos_before_input = pos_before.astype(np.float32)
            ori_before_input = ori_before.astype(np.float32)
            lwh_list_input = lwh_list.astype(np.float32)

            pos_after = self.arrange_model.pred(pos_before_input, ori_before_input, lwh_list_input, input_index)
            print('here')
            manipulator_before = np.concatenate((pos_before_input[input_index], ori_before_input[input_index]), axis=1)
            manipulator_after = np.concatenate((pos_after[input_index].astype(np.float32), np.zeros((len(input_index), 3))), axis=1)
            lwh_list_classify = lwh_list_input[input_index]
            rotate_index = np.where(lwh_list_classify[:, 1] > lwh_list_classify[:, 0])[0]
            manipulator_after[rotate_index, -1] += np.pi / 2
        else:
            # determine the center of the tidy configuration
            if len(self.lwh_list) <= 2:
                print('the number of item is too low, no need to knolling!')
            lwh_list_classify, pos_before_classify, ori_before_classify, all_index_classify, transform_flag_classify, crowded_index_classify = self.boxes_sort.judge(
                lwh_list, pos_before, ori_before, crowded_index)

            calculate_reorder = configuration_zzz(lwh_list_classify, all_index_classify, transform_flag_classify, self.knolling_para)
            pos_after_classify, ori_after_classify = calculate_reorder.calculate_block()
            # after this step the length and width of one box in self.lwh_list may exchanged!!!!!!!!!!!
            # but the order of self.lwh_list doesn't change!!!!!!!!!!!!!!
            # the order of pos after and ori after is based on lwh list!!!!!!!!!!!!!!

            ################## change order based on distance between boxes and upper left corner ##################
            order = change_sequence(pos_before_classify)
            pos_before_classify = pos_before_classify[order]
            ori_before_classify = ori_before_classify[order]
            lwh_list_classify = lwh_list_classify[order]
            pos_after_classify = pos_after_classify[order]
            ori_after_classify = ori_after_classify[order]
            crowded_index_classify = crowded_index_classify[order]
            ################## change order based on distance between boxes and upper left corner ##################

            x_low = np.min(pos_after_classify, axis=0)[0]
            x_high = np.max(pos_after_classify, axis=0)[0]
            y_low = np.min(pos_after_classify, axis=0)[1]
            y_high = np.max(pos_after_classify, axis=0)[1]
            center = np.array([(x_low + x_high) / 2, (y_low + y_high) / 2, 0])
            x_length = abs(x_high - x_low)
            y_length = abs(y_high - y_low)
            # print(x_low, x_high, y_low, y_high)
            if self.knolling_para['random_offset'] == True:
                self.knolling_para['total_offset'] = np.array([random.uniform(self.x_low_obs + x_length / 2, self.x_high_obs - x_length / 2),
                                              random.uniform(self.y_low_obs + y_length / 2, self.y_high_obs - y_length / 2), 0.0])
            else:
                pass
            pos_after_classify += np.array([0, 0, 0.006])
            pos_after_classify = pos_after_classify + self.knolling_para['total_offset']

            ########## after generate the neat configuration, pay attention to the difference of urdf ori and manipulator after ori! ############
            items_ori_list_arm = np.copy(ori_after_classify)
            for i in range(len(lwh_list_classify)):
                if lwh_list_classify[i, 0] <= lwh_list_classify[i, 1]:
                    ori_after_classify[i, 2] += np.pi / 2
            ########## after generate the neat configuration, pay attention to the difference of urdf ori and manipulator after ori! ############

            manipulator_before = np.concatenate((pos_before_classify, ori_before_classify), axis=1)
            manipulator_after = np.concatenate((pos_after_classify, ori_after_classify), axis=1)
            print('this is manipulator after\n', manipulator_after)

        return manipulator_before, manipulator_after, lwh_list_classify

    def calculate_gripper(self):
        self.close_open_gap = 0.053
        # close_open_gap = 0.048
        obj_width_range = np.array([0.022, 0.057])
        motor_pos_range = np.array([0.022, 0.010])  # 0.0273
        formula_parameters = np.polyfit(obj_width_range, motor_pos_range, 1)
        self.motor_pos = np.poly1d(formula_parameters)

    def move(self, cur_pos, cur_ori, tar_pos, tar_ori, sim_height=-0.01, origin_left_pos=None, origin_right_pos=None, index=None):

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
                                        targetPosition=ik_angles0[motor_index], maxVelocity=25, force=self.para_dict['move_force'])
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
                    # if np.abs(new_distance_left - self.distance_left) > self.para_dict['move_threshold'] or \
                    #         np.abs(new_distance_right - self.distance_right) > self.para_dict['move_threshold']:
                    #     move_success_flag = False
                    #     print('during moving, fail')
                    #     break

                    if self.is_render:
                        time.sleep(1 / 720)
                if move_success_flag == False:
                    break
            else:
                for i in range(10):
                    p.stepSimulation()
                    if self.is_render:
                        time.sleep(1 / 720)
            cur_pos = tar_pos
            cur_ori = tar_ori
            if abs(target_pos[0] - tar_pos[0]) < 0.001 and abs(target_pos[1] - tar_pos[1]) < 0.001 and abs(
                    target_pos[2] - tar_pos[2]) < 0.001 and \
                    abs(target_ori[0] - tar_ori[0]) < 0.001 and abs(target_ori[1] - tar_ori[1]) < 0.001 and abs(
                target_ori[2] - tar_ori[2]) < 0.001:
                break
        ee_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
        if ee_pos[2] - target_pos[2] > 0.001 and index == 3 and move_success_flag == True:
            move_success_flag = False
            print('ee can not reach the bottom, fail!')

        gripper_left_pos = np.asarray(p.getLinkState(self.arm_id, 7)[0])
        gripper_right_pos = np.asarray(p.getLinkState(self.arm_id, 8)[0])
        return cur_pos, gripper_left_pos, gripper_right_pos, move_success_flag

    def gripper(self, gap, obj_width, left_pos, right_pos, index=None):
        if index == 4:
            self.keep_obj_width = obj_width + 0.01
        obj_width += 0.010
        gripper_success_flag = True
        if index == 1:
            num_step = 30
        else:
            num_step = 10

        if gap > 0.5:  # close
            tar_pos = self.motor_pos(obj_width) + self.close_open_gap
            p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL,
                                    targetPosition=self.motor_pos(obj_width) + self.close_open_gap, force=self.para_dict['gripper_force'])
            p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL,
                                    targetPosition=self.motor_pos(obj_width) + self.close_open_gap, force=self.para_dict['gripper_force'])
            for i in range(num_step):

                p.stepSimulation()
                gripper_left_pos = np.asarray(p.getLinkState(self.arm_id, 7)[0])
                gripper_right_pos = np.asarray(p.getLinkState(self.arm_id, 8)[0])
                if gripper_left_pos[1] - left_pos[1] > self.para_dict['gripper_threshold'] or right_pos[1] - gripper_right_pos[1] > self.para_dict['gripper_threshold']:
                    print('during grasp, fail')
                    gripper_success_flag = False
                    break
                if self.is_render:
                    time.sleep(1 / 48)
        else:  # open
            p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL, targetPosition=self.motor_pos(obj_width), force=self.para_dict['gripper_force'])
            p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL, targetPosition=self.motor_pos(obj_width), force=self.para_dict['gripper_force'])
            for i in range(num_step):
                p.stepSimulation()
                if self.is_render:
                    time.sleep(1 / 48)
        if index == 1:
            # print('initialize the distance from gripper to bar')
            bar_pos = np.asarray(p.getLinkState(self.arm_id, 6)[0])
            gripper_left_pos = np.asarray(p.getLinkState(self.arm_id, 7)[0])
            gripper_right_pos = np.asarray(p.getLinkState(self.arm_id, 8)[0])
            self.distance_left = np.linalg.norm(bar_pos[:2] - gripper_left_pos[:2])
            self.distance_right = np.linalg.norm(bar_pos[:2] - gripper_right_pos[:2])
        return gripper_success_flag

    def get_obs(self, epoch=0, look_flag=False, baseline_flag=False):
        def get_images():
            (width, length, image, image_depth, seg_mask) = p.getCameraImage(width=640,
                                                                             height=480,
                                                                             viewMatrix=self.view_matrix,
                                                                             projectionMatrix=self.projection_matrix,
                                                                             renderer=p.ER_BULLET_HARDWARE_OPENGL)
            far_range = self.camera_parameters['far']
            near_range = self.camera_parameters['near']
            depth_data = far_range * near_range / (far_range - (far_range - near_range) * image_depth)
            top_height = 0.4 - depth_data
            my_im = image[:, :, :3]
            temp = np.copy(my_im[:, :, 0])  # change rgb image to bgr for opencv to save
            my_im[:, :, 0] = my_im[:, :, 2]
            my_im[:, :, 2] = temp
            img = np.copy(my_im)
            return img, top_height

        if look_flag == True:
            if self.para_dict['real_operate'] == False:
                img, _ = get_images()
                cv2.namedWindow('zzz', 0)
                cv2.resizeWindow('zzz', 1280, 960)
                cv2.imshow('zzz', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                img_path = self.para_dict['dataset_path'] + 'sim_images/%012d.png' % (epoch)
                cv2.imwrite(img_path, img)
        else:
            if self.para_dict['real_operate'] == False:

                img, _ = get_images()

                ################### the results of object detection has changed the order!!!! ####################
                # structure of results: x, y, z, length, width, ori
                # results, pred_conf = self.yolo_seg_model.yolo_seg_predict(img_path=img_path, img=img)

                if self.para_dict['use_lstm_model'] == True:

                    manipulator_before, new_lwh_list, pred_conf, crowded_index, prediction, model_output = self.yolo_pose_model.yolo_pose_predict(img=img, epoch=epoch, gt_boxes_num=len(self.boxes_index), first_flag=baseline_flag)
                else:
                    manipulator_before, new_lwh_list, pred_conf = self.yolo_pose_model.yolo_pose_predict(img=img, epoch=epoch, gt_boxes_num=len(self.boxes_index), first_flag=baseline_flag)

                ################### the results of object detection has changed the order!!!! ####################

            if self.para_dict['real_operate'] == True:

                ################### the results of object detection has changed the order!!!! ####################
                # structure of results: x, y, z, length, width, ori
                if self.para_dict['use_lstm_model'] == True:
                    manipulator_before, new_lwh_list, pred_conf, crowded_index, prediction, model_output = self.yolo_pose_model.yolo_pose_predict(real_flag=True, first_flag=baseline_flag, epoch=epoch)
                else:
                    manipulator_before, new_lwh_list, pred_conf = self.yolo_pose_model.yolo_pose_predict(real_flag=True, first_flag=baseline_flag, epoch=epoch)

                ################### the results of object detection has changed the order!!!! ####################

            if self.para_dict['use_lstm_model'] == True:
                return manipulator_before, new_lwh_list, pred_conf, crowded_index, prediction, model_output
            else:
                return manipulator_before, new_lwh_list, pred_conf


if __name__ == '__main__':

    # np.random.seed(183)
    # random.seed(183)

    para_dict = {'start_num': 00, 'end_num': 10000, 'thread': 0,
                 'yolo_conf': 0.6, 'yolo_iou': 0.8, 'device': 'cpu',
                 'save_img_flag': False,
                 'init_pos_range': [[0.13, 0.17], [-0.03, 0.03], [0.01, 0.02]], 'init_offset_range': [[-0.05, 0.05], [-0.1, 0.1]],
                 'init_ori_range': [[-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4], [3 * np.pi / 16, np.pi / 4]],
                 'max_box_num': 2, 'min_box_num': 2,
                 'is_render': False,
                 'box_range': [[0.016, 0.048], [0.016], [0.01, 0.02]],
                 'box_mass': 0.1,
                 'gripper_threshold': 0.002, 'gripper_sim_step': 10, 'gripper_force': 3,
                 'move_threshold': 0.001, 'move_force': 3,
                 'gripper_lateral_friction': 1, 'gripper_linear_damping': 100,
                 'gripper_contact_damping': 1, 'gripper_contact_stiffness': 50000,
                 'box_lateral_friction': 1, 'box_spinning_friction': 1,
                 'box_restitution': 0, 'box_contact_damping': 1, 'box_contact_stiffness': 50000,
                 'base_lateral_friction': 1, 'base_spinning_friction': 1,
                 'base_restitution': 0, 'base_contact_damping': 1, 'base_contact_stiffness': 50000,
                 'dataset_path': '/home/zhizhuo/Creative_Machines_Lab/knolling_dataset/grasp_dataset_721_heavy_test/',
                 'urdf_path': '/home/zhizhuo/Creative_Machines_Lab/Knolling_bot_2/urdf/',
                 'yolo_model_path': '/home/zhizhuo/Creative_Machines_Lab/Knolling_bot_2/627_pile_pose/weights/best.pt'}

    startnum = para_dict['start_num']
    endnum = para_dict['end_num']

    # with open(para_dict['dataset_path'] + 'grasp_dataset_721_heavy_test_readme.txt', "w") as f:
    #     for key, value in para_dict.items():
    #         f.write(key + ': ')
    #         f.write(str(value) + '\n')

    # os.makedirs(para_dict['dataset_path'], exist_ok=True)

    max_box_num = para_dict['max_box_num']
    min_box_num = para_dict['min_box_num']
    mm2px = 530 / 0.34

    env = Arm_env(para_dict=para_dict)

    exist_img_num = startnum
    while True:
        num_item = int(np.random.uniform(min_box_num, max_box_num + 1))
        env.reset(epoch=exist_img_num)
        img_per_epoch = env.try_grasp(img_index_start=exist_img_num)
        exist_img_num += img_per_epoch