from yolo_model_deploy import *
from arrangement import *
from grasp_model_deploy import *
from function import *
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
from sklearn.preprocessing import MinMaxScaler

class Arm_env():

    def __init__(self, para_dict, knolling_para=None, lstm_dict=None):

        self.para_dict = para_dict
        self.knolling_para = knolling_para

        self.kImageSize = {'width': 480, 'height': 480}
        self.endnum = para_dict['end_num']
        self.init_pos_range = para_dict['init_pos_range']
        self.init_ori_range = para_dict['init_ori_range']
        self.urdf_path = para_dict['urdf_path']
        self.pybullet_path = pd.getDataPath()
        self.is_render = para_dict['is_render']
        self.num_boxes = para_dict['boxes_num']
        self.save_img_flag = para_dict['save_img_flag']
        self.yolo_model = Yolo_predict(para_dict=para_dict)
        self.boxes_sort = Sort_objects(para_dict=para_dict, knolling_para=knolling_para)
        if self.para_dict['use_lstm_model'] == True:
            self.lstm_dict = lstm_dict
            self.grasp_model = Grasp_model(para_dict=para_dict, lstm_dict=lstm_dict)
        else:
            pass

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
        for _ in range(int(30)):
            # time.sleep(1/480)
            p.stepSimulation()
        ####################### gripper to the origin position ########################

        p.setGravity(0, 0, -10)
        wall_id = []
        wall_pos = np.array([[self.x_low_obs - self.table_boundary, 0, 0],
                             [(self.x_low_obs + self.x_high_obs) / 2, self.y_low_obs - self.table_boundary, 0],
                             [self.x_high_obs + self.table_boundary, 0, 0],
                             [(self.x_low_obs + self.x_high_obs) / 2, self.y_high_obs + self.table_boundary, 0]])
        wall_ori = np.array([[0, 1.57, 0],
                             [0, 1.57, 1.57],
                             [0, 1.57, 0],
                             [0, 1.57, 1.57]])
        for i in range(len(wall_pos)):
            wall_id.append(p.loadURDF(os.path.join(self.urdf_path, "plane_2.urdf"), basePosition=wall_pos[i],
                                baseOrientation=p.getQuaternionFromEuler(wall_ori[i]), useFixedBase=1,
                                flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
            p.changeVisualShape(wall_id[i], -1, rgbaColor=(1, 1, 1, 0))

        if self.para_dict['real_operate'] == False:
            self.lwh_list = self.boxes_sort.get_data_virtual()
            rdm_ori_roll  = np.random.uniform(self.init_ori_range[0][0], self.init_ori_range[0][1], size=(self.num_boxes, 1))
            rdm_ori_pitch = np.random.uniform(self.init_ori_range[1][0], self.init_ori_range[1][1], size=(self.num_boxes, 1))
            rdm_ori_yaw   = np.random.uniform(self.init_ori_range[2][0], self.init_ori_range[2][1], size=(self.num_boxes, 1))
            rdm_ori = np.concatenate((rdm_ori_roll, rdm_ori_pitch, rdm_ori_yaw), axis=1)
            rdm_pos_x = np.random.uniform(self.init_pos_range[0][0], self.init_pos_range[0][1], size=(self.num_boxes, 1))
            rdm_pos_y = np.random.uniform(self.init_pos_range[1][0], self.init_pos_range[1][1], size=(self.num_boxes, 1))
            rdm_pos_z = np.random.uniform(self.init_pos_range[2][0], self.init_pos_range[2][1], size=(self.num_boxes, 1))
            rdm_pos = np.concatenate((rdm_pos_x, rdm_pos_y, rdm_pos_z), axis=1)
        else:
            # the sequence here is based on area and ratio!!! must be converted additionally!!!
            self.lwh_list, self.pos_before, self.ori_before, self.all_index, self.transform_flag = self.boxes_sort.get_data_real(self.yolo_model, self.para_dict['evaluations'])
            # these data has defined in function change_config, we don't need to define them twice!!!
            sim_pos = np.copy(self.pos_before)
            sim_pos[:, :2] += 0.006

        self.boxes_index = []
        if self.para_dict['data_collection'] == True:
            box_path = self.para_dict['dataset_path'] + "box_urdf/thread_%d/epoch_%d/" % (self.para_dict['thread'], epoch)
        else:
            box_path = self.para_dict['dataset_path']
        os.makedirs(box_path, exist_ok=True)
        temp_box = URDF.load(self.urdf_path + 'box_generator/template.urdf')

        for i in range(self.num_boxes):
            temp_box.links[0].inertial.mass = self.para_dict['box_mass']
            temp_box.links[0].collisions[0].origin[2, 3] = 0
            length = self.lwh_list[i, 0]
            width = self.lwh_list[i, 1]
            height = self.lwh_list[i, 2]
            temp_box.links[0].visuals[0].geometry.box.size = [length, width, height]
            temp_box.links[0].collisions[0].geometry.box.size = [length, width, height]
            temp_box.links[0].visuals[0].material.color = [np.random.random(), np.random.random(), np.random.random(), 1]
            temp_box.save(box_path + 'box_%d.urdf' % (i))
        if self.para_dict['real_operate'] == False:
            for i in range(self.num_boxes):
                self.boxes_index.append(p.loadURDF((box_path + "box_%d.urdf" % i), basePosition=rdm_pos[i],
                                               baseOrientation=p.getQuaternionFromEuler(rdm_ori[i]), useFixedBase=0,
                                               flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
                r = np.random.uniform(0, 0.9)
                g = np.random.uniform(0, 0.9)
                b = np.random.uniform(0, 0.9)
                if random.random() < 0.05:
                    p.changeVisualShape(self.boxes_index[i], -1, rgbaColor=(0.1, 0.1, 0.1, 1))
                else:
                    p.changeVisualShape(self.boxes_index[i], -1, rgbaColor=(r, g, b, 1))
        else:
            for i in range(self.num_boxes):
                self.boxes_index.append(p.loadURDF((box_path + "box_%d.urdf" % i),
                                                   basePosition=self.pos_before[i],
                                                   baseOrientation=p.getQuaternionFromEuler(self.ori_before[i]),
                                                   useFixedBase=False,
                                                   flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
                r = np.random.uniform(0, 0.9)
                g = np.random.uniform(0, 0.9)
                b = np.random.uniform(0, 0.9)
                p.changeVisualShape(self.boxes_index[i], -1, rgbaColor=(r, g, b, 1))

        for _ in range(int(100)):
            p.stepSimulation()
            if self.is_render == True:
                time.sleep(1/96)
        p.changeDynamics(baseid, -1, lateralFriction=self.para_dict['base_lateral_friction'],
                                     contactDamping=self.para_dict['base_contact_damping'],
                                     contactStiffness=self.para_dict['base_contact_stiffness'])

        if self.para_dict['real_operate'] == False:
            forbid_range = np.array([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            while True:
                new_num_item = len(self.boxes_index)
                delete_index = []
                self.pos_before = []
                self.ori_before = []
                for i in range(len(self.boxes_index)):
                    p.changeDynamics(self.boxes_index[i], -1, lateralFriction=self.para_dict['box_lateral_friction'],
                                                         contactDamping=self.para_dict['box_contact_damping'],
                                                         contactStiffness=self.para_dict['box_contact_stiffness'])

                    cur_ori = np.asarray(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.boxes_index[i])[1]))
                    cur_pos = np.asarray(p.getBasePositionAndOrientation(self.boxes_index[i])[0])
                    self.pos_before.append(cur_pos)
                    self.ori_before.append(cur_ori)
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

                self.pos_before = np.asarray(self.pos_before)
                self.ori_before = np.asarray(self.ori_before)
                delete_index.reverse()
                for i in delete_index:
                    p.removeBody(self.boxes_index[i])
                    self.boxes_index.pop(i)
                    self.lwh_list = np.delete(self.lwh_list, i, axis=0)
                    self.pos_before = np.delete(self.pos_before, i, axis=0)
                    self.ori_before = np.delete(self.ori_before, i, axis=0)
                for _ in range(int(100)):
                    # time.sleep(1/96)
                    p.stepSimulation()

                if len(delete_index) == 0:
                    break

                # self.pos_before = []
                # self.ori_before = []
                # check_delete_index = []
                # for i in range(len(self.boxes_index)):
                #     cur_ori = np.asarray(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.boxes_index[i])[1]))
                #     cur_pos = np.asarray(p.getBasePositionAndOrientation(self.boxes_index[i])[0])
                #     self.pos_before.append(cur_pos)
                #     self.ori_before.append(cur_ori)
                #     roll_flag = False
                #     pitch_flag = False
                #     for j in range(len(forbid_range)):
                #         if np.abs(cur_ori[0] - forbid_range[j]) < 0.01:
                #             roll_flag = True
                #         if np.abs(cur_ori[1] - forbid_range[j]) < 0.01:
                #             pitch_flag = True
                #     if roll_flag == True and pitch_flag == True and (
                #             np.abs(cur_ori[0] - 0) > 0.01 or np.abs(cur_ori[1] - 0) > 0.01) or \
                #             cur_pos[0] < self.x_low_obs or cur_pos[0] > self.x_high_obs or cur_pos[
                #         1] > self.y_high_obs or \
                #             cur_pos[1] < self.y_low_obs:
                #         check_delete_index.append(i)
                #         # print('this is cur ori on check:', cur_ori)
                # self.pos_before = np.asarray(self.pos_before)
                # self.ori_before = np.asarray(self.ori_before)
                # if len(check_delete_index) == 0:
                #     break

        self.img_per_epoch = 0
        # return img_per_epoch_result

    def manual_knolling(self, pos_before, ori_before, lwh_list):  # this is main function!!!!!!!!!

        if self.para_dict['use_knolling_model'] == True:
            ######################## knolling demo ###############################

            ################## change order based on distance between boxes and upper left corner ##################
            order = change_sequence(self.pos_before)
            self.pos_before = self.pos_before[order]
            self.ori_before = self.ori_before[order]
            self.lwh_list = self.lwh_list[order]
            knolling_model_input = np.concatenate((self.pos_before[:, :2], self.lwh_list[:, :2],
                                                   self.ori_before[:, 2].reshape(-1, 1)), axis=1).reshape(1, -1)
            ################## change order based on distance between boxes and upper left corner ##################

            ################## input the demo data ##################
            knolling_demo_data = np.loadtxt('./num_10_after_demo_8.txt')[0].reshape(-1, 5)
            ################## input the demo data ##################

            index = []
            after_knolling = []
            after_knolling = np.asarray(after_knolling)

            self.pos_after = np.concatenate((knolling_demo_data[:, :2], np.zeros(len(knolling_demo_data)).reshape(-1, 1)), axis=1)
            self.ori_after = np.concatenate((np.zeros((len(knolling_demo_data), 2)), knolling_demo_data[:, 4].reshape(-1, 1)),
                                            axis=1)
            for i in range(len(knolling_demo_data)):
                if knolling_demo_data[i, 2] < knolling_demo_data[i, 3]:
                    self.ori_after[i, 2] += np.pi / 2

            # self.items_pos_list = np.concatenate((after_knolling[:, :2], np.zeros(len(after_knolling)).reshape(-1, 1)), axis=1)
            # self.items_ori_list = np.concatenate((np.zeros((len(after_knolling), 2)), after_knolling[:, 4].reshape(-1, 1)), axis=1)
            # self.xyz_list = np.concatenate((after_knolling[:, 2:4], (np.ones(len(after_knolling)) * 0.012).reshape(-1, 1)), axis=1)
            ######################## knolling demo ###############################
        else:
            # determine the center of the tidy configuration
            if len(self.lwh_list) <= 2:
                print('the number of item is too low, no need to knolling!')
            lwh_list_classify, pos_before_classify, ori_before_classify, all_index_classify, transform_flag_classify, self.boxes_index = self.boxes_sort.judge(
                lwh_list, pos_before, ori_before)

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
            # boxes_index_classify = boxes_index_classify[order]
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

    def try_grasp(self, img_index_start=None):
        # print('this is img_index start while grasping', img_index_start)
        # manipulator_before, pred_lwh_list, pred_conf = self.get_obs(data_root=data_root, epoch=self.img_per_epoch + img_index_start)
        #
        # if len(manipulator_before) <= 1:
        #     print('no pile in the environment, try to reset!')
        #     return self.img_per_epoch
        # # if np.any(manipulator_before[:, 2].reshape(1, -1) > 0.01) == False:
        # #     print('no pile in the environment, try to reset!')
        # #     return self.img_per_epoch
        #
        # pos_ori_after = np.concatenate((self.reset_pos, np.zeros(3)), axis=0).reshape(-1, 6)
        # manipulator_after = np.repeat(pos_ori_after, len(manipulator_before), axis=0)
        #
        # offset_low = np.array([0, 0, 0.0])
        # offset_high = np.array([0, 0, 0.05])
        #
        # if len(manipulator_before) == 0:
        #     start_end = []
        #     grasp_width = []
        # else:
        #     start_end = np.concatenate((manipulator_before, manipulator_after), axis=1)
        #     grasp_width = np.min(pred_lwh_list[:, :2], axis=1)
        #     box_pos_before = self.gt_pos_ori[:, :3]
        #     box_ori_before = np.copy(self.gt_ori_qua)
        #     if len(start_end) > len(box_pos_before):
        #         print('the yolo model predict additional bounding boxes!')
        #         cut_index = np.arange(len(box_pos_before), len(start_end))
        #         start_end = np.delete(start_end, cut_index, axis=0)
        #         pred_conf = np.delete(pred_conf, cut_index)
        #
        #     state_id = p.saveState()
        #     grasp_flag = []
        #     gt_data = []
        #     exist_gt_index = []
        #
        #     for i in range(len(start_end)):
        #
        #         trajectory_pos_list = [self.reset_pos,
        #                                [0.02, grasp_width[i]],  # open!
        #                                offset_high + start_end[i][:3],
        #                                offset_low + start_end[i][:3],
        #                                [0.0273, grasp_width[i]],  # close
        #                                offset_high + start_end[i][:3],
        #                                start_end[i][6:9]]
        #         trajectory_ori_list = [self.reset_ori,
        #                                self.reset_ori + start_end[i][3:6],
        #                                self.reset_ori + start_end[i][3:6],
        #                                self.reset_ori + start_end[i][3:6],
        #                                [0.0273, grasp_width[i]],
        #                                self.reset_ori + start_end[i][3:6],
        #                                self.reset_ori + start_end[i][9:12]]
        #         last_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
        #         last_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))
        #
        #         success_grasp_flag = True
        #         left_pos = np.asarray(p.getLinkState(self.arm_id, 7)[0])
        #         right_pos = np.asarray(p.getLinkState(self.arm_id, 8)[0])
        #         for j in range(len(trajectory_pos_list)):
        #             if len(trajectory_pos_list[j]) == 3:
        #                 if j == 2:
        #                     last_pos, left_pos, right_pos, _ = self.move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j], index=j)
        #                 elif j == 3:
        #                     last_pos, _, _, success_grasp_flag = self.move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j],
        #                                                                    origin_left_pos=left_pos, origin_right_pos=right_pos, index=j)
        #                     if success_grasp_flag == False:
        #                         break
        #                 else:
        #                     last_pos, _, _, _ = self.move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j])
        #                 last_ori = np.copy(trajectory_ori_list[j])
        #             elif len(trajectory_pos_list[j]) == 2:
        #                 # time.sleep(2)
        #                 success_grasp_flag = self.gripper(trajectory_pos_list[j][0], trajectory_pos_list[j][1], left_pos, right_pos, index=j)
        #
        #         # find which box is moved and judge whether the grasp is success
        #         if success_grasp_flag == False:
        #             print('fail!')
        #             grasp_flag.append(0)
        #             pass
        #         else:
        #             for j in range(len(self.boxes_index)):
        #                 success_grasp_flag = False
        #                 fail_break_flag = False
        #                 box_pos = np.asarray(p.getBasePositionAndOrientation(self.boxes_index[j])[0])  # this is the pos after of the grasped box
        #                 if np.abs(box_pos[0] - last_pos[0]) < 0.02 and np.abs(box_pos[1] - last_pos[1]) < 0.02 and box_pos[2] > 0.06 and \
        #                     np.linalg.norm(box_pos_before[j, :2] - start_end[i, :2]) < 0.005:
        #                     for m in range(len(self.boxes_index)):
        #                         box_pos_after = np.asarray(p.getBasePositionAndOrientation(self.boxes_index[m])[0])
        #                         ori_qua_after = p.getBasePositionAndOrientation(self.boxes_index[m])[1]
        #                         box_ori_after = np.asarray(ori_qua_after)
        #                         upper_limit = np.sum(np.abs(box_ori_after + box_ori_before[m]))
        #                         if box_pos_after[2] > 0.06 and m != j:
        #                             print(f'The {m} boxes have been disturbed, because it is also grasped accidentally, grasp fail!')
        #                             p.addUserDebugPoints([box_pos_before[m]], [[0, 1, 0]], pointSize=5)
        #                             grasp_flag.append(0)
        #                             fail_break_flag = True
        #                             success_grasp_flag = False
        #                             break
        #                         elif m == len(self.boxes_index) - 1:
        #                             grasp_flag.append(1)
        #                             print('grasp success!')
        #                             success_grasp_flag = True
        #                             fail_break_flag = False
        #                     if success_grasp_flag == True or fail_break_flag == True:
        #                         break
        #                 elif j == len(self.boxes_index) - 1:
        #                     print('the target box does not move to the designated pos, grasp fail!')
        #                     success_grasp_flag = False
        #                     grasp_flag.append(0)
        #
        #         # gt_index = np.argmin(np.linalg.norm(box_pos_before[:, :2] - start_end[i, :2], axis=1))
        #
        #         # this is gt data label
        #         gt_index = np.argsort(np.linalg.norm(box_pos_before[:, :2] - start_end[i, :2], axis=1))
        #         gt_index_grasp = gt_index[~np.isin(gt_index, np.asarray(exist_gt_index))][0]
        #         # gt_data.append(np.concatenate((box_pos_before[gt_index_grasp], self.lwh_list[gt_index_grasp], self.gt_pos_ori[gt_index_grasp, 3:])))
        #         exist_gt_index.append(gt_index_grasp)
        #
        #         # this is pred data label, we don't use the simulation data as the dataset because it doesn't work in the real-world environment
        #         gt_data.append(
        #             np.concatenate((manipulator_before[i, :3], pred_lwh_list[i, :3], manipulator_before[i, 3:])))
        #
        #         if success_grasp_flag == True:
        #             print('we should remove this box and try the rest boxes!')
        #             rest_len = len(exist_gt_index)
        #             for m in range(1, len(start_end) - rest_len + 1):
        #                 grasp_flag.append(0)
        #                 gt_data.append(np.concatenate(
        #                     (manipulator_before[i + m, :3], pred_lwh_list[i + m, :3], manipulator_before[i + m, 3:])))
        #                 # gt_index = np.argsort(np.linalg.norm(box_pos_before[:, :2] - start_end[i + m, :2], axis=1))
        #                 # gt_index = gt_index[~np.isin(gt_index, np.asarray(exist_gt_index))][0]
        #                 # gt_data.append(np.concatenate(
        #                 #     (box_pos_before[gt_index], self.lwh_list[gt_index], self.gt_pos_ori[gt_index, 3:])))
        #                 # exist_gt_index.append(gt_index)
        #             # p.restoreState(state_id)
        #             p.removeBody(self.boxes_index[gt_index_grasp])
        #             print('this is len of self.obj', len(self.boxes_index))
        #             del self.boxes_index[gt_index_grasp]
        #             self.lwh_list = np.delete(self.lwh_list, gt_index_grasp, axis=0)
        #
        #             ######## after every grasp, check pos and ori of every box which are out of the field ########
        #             forbid_range = np.array([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        #             delete_index = []
        #             print('this is len of self.obj', len(self.boxes_index))
        #             for m in range(len(self.boxes_index)):
        #                 cur_ori = np.asarray(
        #                     p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.boxes_index[m])[1]))
        #                 cur_pos = np.asarray(p.getBasePositionAndOrientation(self.boxes_index[m])[0])
        #                 roll_flag = False
        #                 pitch_flag = False
        #                 for n in range(len(forbid_range)):
        #                     if np.abs(cur_ori[0] - forbid_range[n]) < 0.01:
        #                         roll_flag = True
        #                     if np.abs(cur_ori[1] - forbid_range[n]) < 0.01:
        #                         pitch_flag = True
        #                 if roll_flag == True and pitch_flag == True and (
        #                         np.abs(cur_ori[0] - 0) > 0.01 or np.abs(cur_ori[1] - 0) > 0.01) or \
        #                         cur_pos[0] < self.x_low_obs or cur_pos[0] > self.x_high_obs or cur_pos[
        #                     1] > self.y_high_obs or cur_pos[1] < self.y_low_obs:
        #                     delete_index.append(m)
        #             delete_index.reverse()
        #             for idx in delete_index:
        #                 print('this is delete index', idx)
        #                 p.removeBody(self.boxes_index[idx])
        #                 self.boxes_index.pop(idx)
        #                 self.lwh_list = np.delete(self.lwh_list, idx, axis=0)
        #             ######## after every grasp, check pos and ori of every box which are out of the field ########
        #
        #             break
        #         else:
        #             p.restoreState(state_id)
        #             print('restore the previous env and try another one')
        #
        #     gt_data = np.asarray(gt_data)
        #     grasp_flag = np.asarray(grasp_flag).reshape(-1, 1)
        #
        #     yolo_label = np.concatenate((grasp_flag, gt_data, pred_conf.reshape(-1, 1)), axis=1)
        #
        # if len(start_end) == 0:
        #     print('No box on the image! This is total num of img after one epoch', self.img_per_epoch)
        #     return self.img_per_epoch
        # elif np.all(grasp_flag == 0):
        #     np.savetxt(os.path.join(data_root, "origin_labels/%012d.txt" % (img_index_start + self.img_per_epoch)), yolo_label, fmt='%.04f')
        #     if self.save_img_flag == False:
        #         os.remove(data_root + 'origin_images/%012d.png' % (self.img_per_epoch + img_index_start))
        #     self.img_per_epoch += 1
        #     print('this is total num of img after one epoch', self.img_per_epoch)
        #     return self.img_per_epoch
        # else:
        #     np.savetxt(os.path.join(data_root, "origin_labels/%012d.txt" % (img_index_start + self.img_per_epoch)), yolo_label, fmt='%.04f')
        #     if self.save_img_flag == False:
        #         os.remove(data_root + 'origin_images/%012d.png' % (self.img_per_epoch + img_index_start))
        #     self.img_per_epoch += 1
        #     return self.try_grasp(data_root=data_root, img_index_start=img_index_start)
        pass

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

    def get_obs(self, epoch=0):

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

        if self.para_dict['data_collection'] == True or self.para_dict['obs_order'] == 'sim_image_obj':
            self.box_pos, self.box_ori, self.gt_ori_qua = [], [], []
            if len(self.boxes_index) == 0:
                return np.array([]), np.array([]), np.array([])
            self.constrain_id = []
            for i in range(len(self.boxes_index)):
                box_pos = np.asarray(p.getBasePositionAndOrientation(self.boxes_index[i])[0])
                box_ori = np.asarray(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.boxes_index[i])[1]))
                self.gt_ori_qua.append(np.asarray(p.getBasePositionAndOrientation(self.boxes_index[i])[1]))
                self.box_pos = np.append(self.box_pos, box_pos).astype(np.float32)
                self.box_ori = np.append(self.box_ori, box_ori).astype(np.float32)
            self.box_pos = self.box_pos.reshape(len(self.boxes_index), 3)
            self.box_ori = self.box_ori.reshape(len(self.boxes_index), 3)
            self.gt_ori_qua = np.asarray(self.gt_ori_qua)
            self.gt_pos_ori = np.concatenate((self.box_pos, self.box_ori), axis=1)
            self.gt_pos_ori = self.gt_pos_ori.astype(np.float32)

            img, _ = get_images()
            os.makedirs(self.para_dict['dataset_path'] + 'origin_images/', exist_ok=True)
            img_path = self.para_dict['dataset_path'] + 'origin_images/%012d' % (epoch)

            ################### the results of object detection has changed the order!!!! ####################
            # structure of results: x, y, z, length, width, ori
            results, pred_conf = self.yolo_model.yolov8_predict(img_path=img_path, img=img)
            if len(results) == 0:
                return np.array([]), np.array([]), np.array([])
            # print('this is the result of yolo-pose\n', results)
            ################### the results of object detection has changed the order!!!! ####################

            manipulator_before = np.concatenate((results[:, :3], np.zeros((len(results), 2)), results[:, 5].reshape(-1, 1)), axis=1)
            new_lwh_list = np.concatenate((results[:, 3:5], np.ones((len(results), 1)) * 0.016), axis=1)
            # print('this is manipulator before after the detection \n', manipulator_before)

            return manipulator_before, new_lwh_list, pred_conf

        if self.para_dict['obs_order'] == 'real_image_obj':
            # # temp useless because of knolling demo
            # img_path = 'Test_images/image_real'
            # # structure: x,y,length,width,yaw
            # results = yolov8_predict(img_path=img_path, real_flag=self.general_parameters['real_operate, target=None)
            # print('this is the result of yolo-pose\n', results)
            #
            # z = 0
            # roll = 0
            # pitch = 0
            # index = []
            # print('this is self.xyz\n', self.xyz_list)
            # for i in range(len(self.xyz_list)):
            #     for j in range(len(results)):
            #         if (np.abs(self.xyz_list[i, 0] - results[j, 2]) <= 0.002 and np.abs(
            #                 self.xyz_list[i, 1] - results[j, 3]) <= 0.002) or \
            #                 (np.abs(self.xyz_list[i, 1] - results[j, 2]) <= 0.002 and np.abs(
            #                     self.xyz_list[i, 0] - results[j, 3]) <= 0.002):
            #             if j not in index:
            #                 print(f"find first xyz{i} in second xyz{j}")
            #                 index.append(j)
            #                 break
            #             else:
            #                 pass
            #
            # manipulator_before = []
            # for i in index:
            #     manipulator_before.append([results[i][0], results[i][1], z, roll, pitch, results[i][4]])
            # # for i in range(len(self.xyz_list)):
            # #     manipulator_before.append([self.pos_before[i][0], self.pos_before[i][1], z, roll, pitch, self.ori_before[i][2]])

            manipulator_before = np.concatenate((self.pos_before, self.ori_before), axis=1)
            manipulator_before = np.asarray(manipulator_before)
            new_xyz_list = self.lwh_list
            print('this is manipulator before after the detection \n', manipulator_before)

            return manipulator_before, new_xyz_list

if __name__ == '__main__':

    # np.random.seed(183)
    # random.seed(183)

    para_dict = {'start_num': 00, 'end_num': 10000, 'thread': 0,
                 'yolo_conf': 0.6, 'yolo_iou': 0.8, 'device': 'cuda:0',
                 'save_img_flag': False,
                 'init_pos_range': [[0.13, 0.17], [-0.03, 0.03], [0.01, 0.02]],
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
                 'yolo_model_path': '/home/zhizhuo/Creative_Machines_Lab/Knolling_bot_2/train_pile_overlap_627/weights/best.pt'}

    startnum = para_dict['start_num']
    endnum = para_dict['end_num']

    # with open(para_dict['dataset_path'] + 'grasp_dataset_721_heavy_test_readme.txt', "w") as f:
    #     for key, value in para_dict.items():
    #         f.write(key + ': ')
    #         f.write(str(value) + '\n')

    os.makedirs(para_dict['dataset_path'], exist_ok=True)

    max_box_num = para_dict['max_box_num']
    min_box_num = para_dict['min_box_num']
    mm2px = 530 / 0.34

    env = Arm_env(para_dict=para_dict)
    os.makedirs(para_dict['dataset_path'] + 'origin_images/', exist_ok=True)
    os.makedirs(para_dict['dataset_path'] + 'origin_labels/', exist_ok=True)

    exist_img_num = startnum
    while True:
        num_item = int(np.random.uniform(min_box_num, max_box_num + 1))
        env.reset(epoch=exist_img_num)
        img_per_epoch = env.try_grasp(img_index_start=exist_img_num)
        exist_img_num += img_per_epoch