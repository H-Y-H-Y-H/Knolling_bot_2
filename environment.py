# from models.yolo_pose_deploy import *

# from arrangement import *
# from grasp_model_deploy import *
from models.arrange_model_deploy import *
from models.visual_perception_config import *
from utils import *
import pybullet as p
import pybullet_data as pd
import os
import numpy as np
import random
import cv2
import time

class Sort_objects():

    def __init__(self, para_dict, knolling_para):
        self.error_rate = 0.05
        self.para_dict = para_dict
        self.knolling_para = knolling_para
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
        xyz_list = np.concatenate((length_range, width_range, height_range), axis=1)
        return xyz_list


    def get_data_real(self, yolo_model, evaluations=1, check='before'):

        os.makedirs(self.para_dict['dataset_path'] + 'real_images/', exist_ok=True)
        img_path = self.para_dict['dataset_path'] + 'real_images/%012d' % (evaluations)
        # structure of results: x, y, length, width, ori
        results, pred_conf = yolo_model.yolo_grasp_predict(img_path=img_path, real_flag=True)

        item_pos = results[:, :3]
        item_lw = np.concatenate((results[:, 3:5], (np.ones(len(results)) * 0.016).reshape(-1, 1)), axis=1)
        item_ori = np.concatenate((np.zeros((len(results), 2)), results[:, 5].reshape(-1, 1)), axis=1)

        category_num = int(self.knolling_para['area_num'] * self.knolling_para['ratio_num'] + 1)
        s = item_lw[:, 0] * item_lw[:, 1]
        s_min, s_max = np.min(s), np.max(s)
        s_range = np.linspace(s_max, s_min, int(self.knolling_para['area_num'] + 1))
        lw_ratio = item_lw[:, 0] / item_lw[:, 1]
        ratio_min, ratio_max = np.min(lw_ratio), np.max(lw_ratio)
        ratio_range = np.linspace(ratio_max, ratio_min, int(self.knolling_para['ratio_num'] + 1))

        # ! initiate the number of items
        all_index = []
        new_item_xyz = []
        new_item_pos = []
        new_item_ori = []
        transform_flag = []
        rest_index = np.arange(len(item_lw))
        index = 0

        for i in range(self.knolling_para['area_num']):
            for j in range(self.knolling_para['ratio_num']):
                kind_index = []
                for m in range(len(item_lw)):
                    if m not in rest_index:
                        continue
                    else:
                        if s_range[i] >= s[m] >= s_range[i + 1]:
                            if ratio_range[j] >= lw_ratio[m] >= ratio_range[j + 1]:
                                transform_flag.append(0)
                                # print(f'boxes{m} matches in area{i}, ratio{j}!')
                                kind_index.append(index)
                                new_item_xyz.append(item_lw[m])
                                new_item_pos.append(item_pos[m])
                                new_item_ori.append(item_ori[m])
                                index += 1
                                rest_index = np.delete(rest_index, np.where(rest_index == m))
                if len(kind_index) != 0:
                    all_index.append(kind_index)

        new_item_xyz = np.asarray(new_item_xyz).reshape(-1, 3)
        new_item_pos = np.asarray(new_item_pos)
        new_item_ori = np.asarray(new_item_ori)
        transform_flag = np.asarray(transform_flag)
        if len(rest_index) != 0:
            # we should implement the rest of boxes!
            rest_xyz = item_lw[rest_index]
            new_item_xyz = np.concatenate((new_item_xyz, rest_xyz), axis=0)
            all_index.append(list(np.arange(index, len(item_lw))))
            transform_flag = np.append(transform_flag, np.zeros(len(item_lw) - index))

        # the sequence of them are based on area and ratio!
        return new_item_xyz, new_item_pos, new_item_ori, all_index, transform_flag

    def judge(self, item_xyz, pos_before, ori_before, crowded_index):
        # after this function, the sequence of item xyz, pos before and ori before changed based on ratio and area

        category_num = int(self.knolling_para['area_num'] * self.knolling_para['ratio_num'] + 1)
        s = item_xyz[:, 0] * item_xyz[:, 1]
        s_min, s_max = np.min(s), np.max(s)
        s_range = np.linspace(s_max, s_min, int(self.knolling_para['area_num'] + 1))
        lw_ratio = item_xyz[:, 0] / item_xyz[:, 1]
        ratio_min, ratio_max = np.min(lw_ratio), np.max(lw_ratio)
        ratio_range = np.linspace(ratio_max, ratio_min, int(self.knolling_para['ratio_num'] + 1))
        ratio_range_high = np.linspace(ratio_max, 1, int(self.knolling_para['ratio_num'] + 1))
        ratio_range_low = np.linspace(1 / ratio_max, 1, int(self.knolling_para['ratio_num'] + 1))

        # ! initiate the number of items
        all_index = []
        new_item_xyz = []
        transform_flag = []
        new_pos_before = []
        new_ori_before = []
        new_crowded_index = []
        rest_index = np.arange(len(item_xyz))
        index = 0

        for i in range(self.knolling_para['area_num']):
            for j in range(self.knolling_para['ratio_num']):
                kind_index = []
                for m in range(len(item_xyz)):
                    if m not in rest_index:
                        continue
                    else:
                        if s_range[i] >= s[m] >= s_range[i + 1]:
                            if ratio_range[j] >= lw_ratio[m] >= ratio_range[j + 1]:
                                transform_flag.append(0)
                                # print(f'boxes{m} matches in area{i}, ratio{j}!')
                                kind_index.append(index)
                                new_item_xyz.append(item_xyz[m])
                                new_pos_before.append(pos_before[m])
                                new_ori_before.append(ori_before[m])
                                new_crowded_index.append(crowded_index[m])
                                index += 1
                                rest_index = np.delete(rest_index, np.where(rest_index == m))
                if len(kind_index) != 0:
                    all_index.append(kind_index)

        new_item_xyz = np.asarray(new_item_xyz).reshape(-1, 3)
        new_pos_before = np.asarray(new_pos_before).reshape(-1, 3)
        new_ori_before = np.asarray(new_ori_before).reshape(-1, 3)
        transform_flag = np.asarray(transform_flag)
        new_crowded_index = np.asarray(new_crowded_index)
        if len(rest_index) != 0:
            # we should implement the rest of boxes!
            rest_xyz = item_xyz[rest_index]
            new_item_xyz = np.concatenate((new_item_xyz, rest_xyz), axis=0)
            all_index.append(list(np.arange(index, len(item_xyz))))
            transform_flag = np.append(transform_flag, np.zeros(len(item_xyz) - index))

        return new_item_xyz, new_pos_before, new_ori_before, all_index, transform_flag, new_crowded_index

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
        self.boxes_sort = Sort_objects(para_dict=para_dict, knolling_para=knolling_para)
        if self.para_dict['use_yolo_model'] == True:
            self.yolo_pose_model = Yolo_pose_model(para_dict=para_dict, lstm_dict=lstm_dict, use_lstm=self.para_dict['use_lstm_model'])
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

    def create_objects(self, manipulator_after=None, lwh_after=None):

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
                if self.para_dict['use_lstm_model'] == True:
                    manipulator_init, lwh_list_init, pred_conf, crowded_index, prediction, model_output = self.get_obs()
                else:
                    manipulator_init, lwh_list_init, pred_conf = self.get_obs()

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
                pass
                # time.sleep(1/96)

        p.changeDynamics(self.baseid, -1, lateralFriction=self.para_dict['base_lateral_friction'],
                         contactDamping=self.para_dict['base_contact_damping'],
                         contactStiffness=self.para_dict['base_contact_stiffness'])
        if self.para_dict['real_operate'] == False:
            pass
        else:
            return manipulator_init, lwh_list_init, []

    def delete_objects(self, manipulator_after=None):

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
                    self.gt_pos_ori.append([cur_pos, cur_ori])
                    self.gt_ori_qua.append(cur_qua)
                    roll_flag = False
                    pitch_flag = False
                    # print('this is cur ori:', cur_ori)
                    for j in range(len(forbid_range)):
                        if np.abs(cur_ori[0] - forbid_range[j]) < 0.1:
                            roll_flag = True
                        if np.abs(cur_ori[1] - forbid_range[j]) < 0.1:
                            pitch_flag = True
                    if roll_flag == True and pitch_flag == True and (np.abs(cur_ori[0] - 0) > 0.1 or np.abs(cur_ori[1] - 0) > 0.1) or \
                            cur_pos[0] < self.x_low_obs or cur_pos[0] > self.x_high_obs or cur_pos[1] > self.y_high_obs or cur_pos[1] < self.y_low_obs:
                        # if cur_pos[2] > 0.015: # delete the object with large height although it doesn't incline!
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
                if len(delete_index) != 0:
                    for _ in range(int(50)):
                        # time.sleep(1/96)
                        p.stepSimulation()

                if len(delete_index) == 0:
                    break

    def recover_objects(self, info_path=None, config_data=None):

        if config_data is None:
            config_data = np.loadtxt(info_path)
            pos_data = config_data[:, :3]

            self.new_center = np.array([np.random.uniform(self.para_dict['recover_center_range'][0][0],
                                                       self.para_dict['recover_center_range'][0][1]),
                                    np.random.uniform(self.para_dict['recover_center_range'][1][0],
                                                      self.para_dict['recover_center_range'][1][1])])
            print('this is new center', self.new_center)
            new_center = np.repeat([self.new_center], axis=0, repeats=len(pos_data))
            pos_data[:, 0] -= new_center[:, 0]
            pos_data[:, 1] -= new_center[:, 1]

            ori_data = config_data[:, 3:6]
            lwh_data = config_data[:, 6:9]
            self.lwh_list = np.copy(lwh_data)
            qua_data = config_data[:, 9:]

            self.boxes_index = []
            for i in range(len(pos_data)):
                obj_name = f'object_{i}'
                create_box(obj_name, pos_data[i], p.getQuaternionFromEuler(ori_data[i]), size=lwh_data[i])
                self.boxes_index.append(int(i + 2))
                r = np.random.uniform(0, 0.9)
                g = np.random.uniform(0, 0.9)
                b = np.random.uniform(0, 0.9)
                p.changeVisualShape(self.boxes_index[i], -1, rgbaColor=(r, g, b, 1))

            for _ in range(int(100)):
                p.stepSimulation()
                if self.is_render == True:
                    pass
                    # time.sleep(1 / 96)

            p.changeDynamics(self.baseid, -1, lateralFriction=self.para_dict['base_lateral_friction'],
                             contactDamping=self.para_dict['base_contact_damping'],
                             contactStiffness=self.para_dict['base_contact_stiffness'])
        else:
            pos_data = config_data[:, :3]
            ori_data = config_data[:, 3:6]
            lwh_data = config_data[:, 6:9]

            for i in range(len(self.boxes_index)):
                p.resetBasePositionAndOrientation(self.boxes_index[i], pos_data[i], p.getQuaternionFromEuler(ori_data[i]))

                # obj_name = f'object_{i}'
                # create_box(obj_name, pos_data[i], p.getQuaternionFromEuler(ori_data[i]), size=lwh_data[i])
                # self.boxes_index.append(int(i + 2))
                # r = np.random.uniform(0, 0.9)
                # g = np.random.uniform(0, 0.9)
                # b = np.random.uniform(0, 0.9)
                # p.changeVisualShape(self.boxes_index[i], -1, rgbaColor=(r, g, b, 1))
            pass

    def reset(self, epoch=None, manipulator_after=None, lwh_after=None, recover_flag=False):

        p.resetSimulation()
        self.create_scene()
        self.create_arm()
        if recover_flag == False:
            if self.para_dict['real_operate'] == False:
                self.create_objects()
            else:
                manipulator_before, lwh_list, crowded_index = self.create_objects(manipulator_after, lwh_after)
            self.delete_objects(manipulator_after)
        else:
            info_path = self.para_dict['data_source_path'] + 'sim_info/%012d.txt' % epoch
            self.recover_objects(info_path)
            self.delete_objects()
        self.img_per_epoch = 0

        self.state_id = p.saveState()
        # return img_per_epoch_result

        if recover_flag == False and self.para_dict['real_operate'] == True:
            return manipulator_before, lwh_list, crowded_index

    def get_candidate_index(self, images):

        num_row = 2
        num_col = 4

        for i , image in enumerate(images):
            images[i] = cv2.resize(images[i], dsize=(320, 240), interpolation=cv2.INTER_CUBIC)

        image_height, image_width, _ = images[0].shape

        # Create a canvas to display the images
        canvas_height = num_row * image_height
        canvas_width = num_col * image_width
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Paste the images onto the canvas
        for i, image in enumerate(images):
            row = i // num_col
            col = i % num_col
            canvas[row * image_height:(row + 1) * image_height, col * image_width:(col + 1) * image_width] = image

        # Function to handle mouse clicks
        selected_image_index = None
        def mouse_click(event, x, y, flags, param):
            global selected_image_index
            if event == cv2.EVENT_LBUTTONDOWN:
                print('mouse clicked!')
                col = x // image_width
                row = y // image_height
                print('this is col', col)
                print('this is row', row)
                selected_image_index = row * num_col + col
                print(selected_image_index)

        # Create a window and set the mouse callback function
        cv2.namedWindow('Multiple Images')
        cv2.setMouseCallback('Multiple Images', mouse_click)

        # Display the canvas with all images
        cv2.imshow('Multiple Images', canvas)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if selected_image_index is not None:
                break

        # Close the window
        cv2.destroyAllWindows()

        # Print the selected image index (0-based)
        if selected_image_index is not None:
            print(f"Selected image index: {selected_image_index}")
        else:
            print("No image selected.")

        return selected_image_index

    def get_knolling_data(self, pos_before, ori_before, lwh_list, crowded_index):  # this is main function!!!!!!!!!

        arrangement_num = 8 # provide several candidates to select the best output
        candidate_img = []
        if self.para_dict['use_knolling_model'] == True:

            demo_data = np.loadtxt('./knolling_demo/num_10_after.txt')[0].reshape(-1, 5)

            record_data = np.loadtxt('./knolling_demo/num_10_lwh.txt').reshape(-1, 5)
            lwh_list_classify = lwh_list
            # lwh_list_classify = record_data[:, 2:4]
            # pos_before = np.concatenate((record_data[:, :2], np.ones((len(record_data), 1)) * 0.006), axis=1)
            # ori_before = np.concatenate((np.zeros((len(demo_data), 2)), record_data[:, -1].reshape(len(record_data), 1)), axis=1)

            recover_config = np.concatenate((demo_data[:, :2],
                                             np.ones(len(demo_data)).reshape(len(demo_data), 1) * 0.006,
                                             np.zeros((len(demo_data), 2)),
                                             demo_data[:, -1].reshape(len(demo_data), 1),
                                             demo_data[:, 2:4],
                                             np.ones(len(demo_data)).reshape(len(demo_data), 1) * 0.016), axis=1)
            self.recover_objects(config_data=recover_config)

            manipulator_before = np.concatenate((pos_before, ori_before), axis=1)
            manipulator_after = recover_config[:, :6]

            # for i in range(arrangement_num):
            #     input_index = np.setdiff1d(np.arange(len(pos_before)), crowded_index)
            #     pos_before_input = pos_before.astype(np.float32)
            #     ori_before_input = ori_before.astype(np.float32)
            #     lwh_list_input = lwh_list.astype(np.float32)
            #     ori_after = np.zeros((len(input_index), 3))
            #
            #     #################### exchange the length and width randomly enrich the input ##################
            #     for j in range(len(input_index)):
            #         if np.random.random() < 0.5:
            #             temp = lwh_list_input[j, 0]
            #             lwh_list_input[j, 1] = lwh_list_input[j, 0]
            #             lwh_list_input[j, 0] = temp
            #             ori_after[j, 2] += np.pi / 2
            #     #################### exchange the length and width randomly enrich the input ##################
            #
            #     pos_after = self.arrange_model.pred(pos_before_input, ori_before_input, lwh_list_input, input_index)
            #     manipulator_before = np.concatenate((pos_before_input[input_index], ori_before_input[input_index]), axis=1)
            #     manipulator_after = np.concatenate((pos_after[input_index].astype(np.float32), ori_after), axis=1)
            #     lwh_list_classify = lwh_list_input[input_index]
            #     rotate_index = np.where(lwh_list_classify[:, 1] > lwh_list_classify[:, 0])[0]
            #     manipulator_after[rotate_index, -1] += np.pi / 2
            #
            #     # ##################### add offset to the knolling data #####################
            #     # manipulator_after[:, 0] -=
            #     # ##################### add offset to the knolling data #####################
            #
            #
            #     recover_config = np.concatenate((manipulator_after, lwh_list_classify), axis=1)
            #     self.recover_objects(config_data=recover_config)
            #     candidate_img.append(self.get_obs(look_flag=True, epoch=i, img_path='here'))
            #
            # candidate_index = self.get_candidate_index(candidate_img)

            p.restoreState(self.state_id)

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
                    if np.abs(new_distance_left - self.distance_left) > self.para_dict['gripper_threshold'] or \
                            np.abs(new_distance_right - self.distance_right) > self.para_dict['gripper_threshold']:
                        move_success_flag = False
                        print('during moving, the gripper is disturbed, fail')
                        break

                    if self.is_render:
                        pass
                        # time.sleep(1 / 720)
                if move_success_flag == False:
                    break
            else:
                for i in range(10):
                    p.stepSimulation()
                    if self.is_render:
                        pass
                        # time.sleep(1 / 720)
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
                                    targetPosition=self.motor_pos(obj_width) + self.close_open_gap, force=self.para_dict['gripper_force'])
            p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL,
                                    targetPosition=self.motor_pos(obj_width) + self.close_open_gap, force=self.para_dict['gripper_force'])
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
            p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL, targetPosition=self.motor_pos(obj_width), force=self.para_dict['gripper_force'])
            p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL, targetPosition=self.motor_pos(obj_width), force=self.para_dict['gripper_force'])
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

    def get_obs(self, epoch=0, look_flag=False, baseline_flag=False, sub_index=0, img_path=None):
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
            img, _ = get_images()
            # cv2.namedWindow('zzz', 0)
            # cv2.resizeWindow('zzz', 1280, 960)
            # cv2.imshow('zzz', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            if img_path is None:
                output_img_path = self.para_dict['data_source_path'] + 'sim_images/%012d.png' % (epoch)
            # elif self.para_dict['real_operate'] == False:
            #     output_img_path = self.para_dict['dataset_path'] + 'sim_images/%012d' % (epoch) + '_after.png'
            # else:
            #     output_img_path = self.para_dict['dataset_path'] + 'real_images/%012d' % (epoch) + '_after.png'
            elif self.para_dict['real_operate'] == False:
                output_img_path = img_path
            else:
                output_img_path = img_path
            cv2.imwrite(output_img_path, img)
            return img
        else:
            if self.para_dict['real_operate'] == False:

                img, _ = get_images()
                ################### the results of object detection has changed the order!!!! ####################
                # structure of results: x, y, z, length, width, ori
                if self.para_dict['use_lstm_model'] == True:
                    manipulator_before, new_lwh_list, pred_conf, crowded_index, prediction, model_output\
                        = self.yolo_pose_model.yolo_pose_predict(img=img, epoch=epoch, gt_boxes_num=len(self.boxes_index), first_flag=baseline_flag, sub_index=sub_index)
                    # self.yolo_pose_model.plot_grasp(manipulator_before, prediction, model_output)
                    # cv2.namedWindow('zzz', 0)
                    # cv2.resizeWindow('zzz', 1280, 960)
                    # cv2.imshow('zzz', self.yolo_pose_model.img_output)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     cv2.destroyAllWindows()
                    #     # img_path_output = self.img_path + '_pred.png'
                    #     # cv2.imwrite(img_path_output, origin_img)
                    #     break
                else:
                    manipulator_before, new_lwh_list, pred_conf = self.yolo_pose_model.yolo_pose_predict(img=img, epoch=epoch, gt_boxes_num=len(self.boxes_index), first_flag=baseline_flag)
                ################### the results of object detection has changed the order!!!! ####################

            else:
                ################### the results of object detection has changed the order!!!! ####################
                # structure of results: x, y, z, length, width, ori
                if self.para_dict['use_lstm_model'] == True:
                    manipulator_before, new_lwh_list, pred_conf, crowded_index, prediction, model_output\
                        = self.yolo_pose_model.yolo_pose_predict(real_flag=True, first_flag=baseline_flag, epoch=epoch)
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
                 'is_render': True,
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