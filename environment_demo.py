# from arrangement import *
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

        self.gripper_width = 0.024
        self.gripper_height = 0.034
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
        # self.to_home()

    def get_data_virtual(self):

        xyz_list = []
        length_range = np.round(np.random.uniform(self.para_dict['box_range'][0][0],
                                                  self.para_dict['box_range'][0][1],
                                                  size=(self.para_dict['boxes_num'], 1)), decimals=3)
        width_range = np.round(np.random.uniform(self.para_dict['box_range'][1][0],
                                                 np.minimum(length_range, 0.036),
                                                 size=(self.para_dict['boxes_num'], 1)), decimals=3)
        height_range = np.round(np.random.uniform(self.para_dict['box_range'][2][0],
                                                  self.para_dict['box_range'][2][1],
                                                  size=(self.para_dict['boxes_num'], 1)), decimals=3)
        # random_range = np.concatenate((length_range, width_range, height_range), axis=1)
        # index = np.random.randint(0, self.para_dict['kind_num'], size=(self.para_dict['boxes_num'],))
        xyz_list = np.concatenate((length_range, width_range, height_range), axis=1)
        return xyz_list

    def create_objects(self, manipulator_after, lwh_after):

        self.lwh_list = self.get_data_virtual()
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

        self.num_boxes = np.copy(len(self.lwh_list))
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

    def reset(self, epoch=None, manipulator_after=None, lwh_after=None):

        p.resetSimulation()
        self.create_scene()
        self.create_arm()
        self.create_objects(manipulator_after, lwh_after)

        while True:
            p.stepSimulation()

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

        img, _ = get_images()
        cv2.namedWindow('zzz', 0)
        cv2.resizeWindow('zzz', 1280, 960)
        cv2.imshow('zzz', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        img_path = self.para_dict['dataset_path'] + 'sim_images/%012d.png' % (epoch)
        cv2.imwrite(img_path, img)

if __name__ == '__main__':

    # np.random.seed(183)
    # random.seed(183)

    para_dict = {'start_num': 0, 'end_num': 10, 'thread': 9, 'evaluations': 1,
                 'yolo_conf': 0.6, 'yolo_iou': 0.5, 'device': 'cuda:0',
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
                 'urdf_path': './urdf/',}

    os.makedirs(para_dict['dataset_path'], exist_ok=True)

    env = Arm_env(para_dict=para_dict)
    env.reset()