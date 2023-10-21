from environment import Arm_env
import pybullet as p
import pybullet_data as pd
import numpy as np
import random
import os
from utils import *
from urdfpy import URDF

class Yolo_seg_env(Arm_env):

    def __init__(self, para_dict=None, lstm_dict=None):

        super(Yolo_seg_env, self).__init__(para_dict=para_dict, lstm_dict=lstm_dict)

        self.test_TP = 0
        self.test_TN = 0
        self.test_FP = 0
        self.test_FN = 0
        self.points_num = self.para_dict['points_range']
        self.total_boxes = self.para_dict['total_boxes']
        self.points_data = []
        self.img_epoch = self.para_dict['start_num']

        for i in range(len(self.points_num)):
            self.points_data.append(np.loadtxt('../../../knolling_dataset/random_polygon/points_%d_%d.txt' % (self.total_boxes * i, self.total_boxes * (i + 1))))

    def reset(self, epoch=None, manipulator_after=None, lwh_after=None):

        if self.img_epoch >= self.para_dict['end_num']:
            quit()
        self.num_boxes = np.random.randint(1, 4)
        # print('this is num boxes', self.num_boxes)
        self.select_index = []
        for i in range(len(self.points_num)):
            self.select_index.append(np.random.choice(np.arange(self.total_boxes * i, self.total_boxes * (i + 1)), self.num_boxes))

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
        ######################################## Texture change ########################################

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
            self.lwh_list = []
            rdm_ori_roll = np.random.uniform(self.init_ori_range[0][0], self.init_ori_range[0][1], size=(self.num_boxes * len(self.points_num), 1))
            rdm_ori_pitch = np.random.uniform(self.init_ori_range[1][0], self.init_ori_range[1][1], size=(self.num_boxes * len(self.points_num), 1))
            rdm_ori_yaw = np.random.uniform(self.init_ori_range[2][0], self.init_ori_range[2][1], size=(self.num_boxes * len(self.points_num), 1))
            rdm_ori = np.concatenate((rdm_ori_roll, rdm_ori_pitch, rdm_ori_yaw), axis=1)
            rdm_pos_x = np.random.uniform(self.init_pos_range[0][0], self.init_pos_range[0][1], size=(self.num_boxes * len(self.points_num), 1))
            rdm_pos_y = np.random.uniform(self.init_pos_range[1][0], self.init_pos_range[1][1], size=(self.num_boxes * len(self.points_num), 1))
            rdm_pos_z = np.random.uniform(self.init_pos_range[2][0], self.init_pos_range[2][1], size=(self.num_boxes * len(self.points_num), 1))
            rdm_pos = np.concatenate((rdm_pos_x, rdm_pos_y, rdm_pos_z), axis=1)

        self.boxes_index = []
        total_boxes = 0
        for j in range(len(self.points_num)):
            for i in range(self.num_boxes):
                self.boxes_index.append(p.loadURDF(('../../../knolling_dataset/' + 'random_polygon/polygon_%d.urdf' % self.select_index[j][i]), basePosition=rdm_pos[total_boxes],
                                                   baseOrientation=p.getQuaternionFromEuler(rdm_ori[total_boxes]), useFixedBase=0,
                                                   flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
                self.lwh_list.append(np.concatenate(([0], self.points_data[j][self.select_index[j][i] - j * self.total_boxes])))
                r = np.random.uniform(0, 0.9)
                g = np.random.uniform(0, 0.9)
                b = np.random.uniform(0, 0.9)
                p.changeVisualShape(self.boxes_index[total_boxes], -1, rgbaColor=(r, g, b, 1))
                total_boxes += 1
        # self.lwh_list = np.asarray(self.lwh_list)

        for _ in range(int(100)):
            p.stepSimulation()
            if self.is_render == True:
                time.sleep(1 / 96)
        p.changeDynamics(baseid, -1, lateralFriction=self.para_dict['base_lateral_friction'],
                         contactDamping=self.para_dict['base_contact_damping'],
                         contactStiffness=self.para_dict['base_contact_stiffness'])

        if self.para_dict['real_operate'] == False and manipulator_after is None:
            forbid_range = np.array([-np.pi, -np.pi / 2, np.pi / 2, np.pi])
            while True:
                new_num_item = len(self.boxes_index)
                delete_index = []
                self.pos_before = []
                self.ori_before = []
                for i in range(len(self.boxes_index)):
                    p.changeDynamics(self.boxes_index[i], -1, lateralFriction=self.para_dict['box_lateral_friction'],
                                     contactDamping=self.para_dict['box_contact_damping'],
                                     contactStiffness=self.para_dict['box_contact_stiffness'])

                    cur_ori = np.asarray(
                        p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.boxes_index[i])[1]))
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
                    if roll_flag == True or pitch_flag == True or \
                            cur_pos[0] < self.x_low_obs or cur_pos[0] > self.x_high_obs or cur_pos[1] > self.y_high_obs or cur_pos[1] < self.y_low_obs or \
                            cur_pos[2] < 0:
                        delete_index.append(i)
                        # print('delete!!!')
                        new_num_item -= 1

                self.pos_before = np.asarray(self.pos_before)
                self.ori_before = np.asarray(self.ori_before)
                delete_index.reverse()
                for i in delete_index:
                    p.removeBody(self.boxes_index[i])
                    self.boxes_index.pop(i)
                    self.lwh_list.pop(i)
                    # self.lwh_list = np.delete(self.lwh_list, i, axis=0)
                    self.pos_before = np.delete(self.pos_before, i, axis=0)
                    self.ori_before = np.delete(self.ori_before, i, axis=0)
                for _ in range(int(100)):
                    # time.sleep(1/96)
                    p.stepSimulation()

                if len(delete_index) == 0:
                    break
        for i in range(len(self.lwh_list)):
            self.lwh_list[i] = np.concatenate((self.pos_before[i, :2], [self.ori_before[i, 2]], self.lwh_list[i]))

        if len(self.lwh_list) != 0:
            with open(file=self.para_dict['dataset_path'] + "origin_labels/%.012d.txt" % self.img_epoch, mode="w") as f:
                for i in range(len(self.lwh_list)):
                    output_data = list(self.lwh_list[i])
                    output = ' '.join(str(item) for item in output_data)
                    f.write(output)
                    f.write('\n')
            self.get_obs(epoch=self.img_epoch, look_flag=True)
            self.img_epoch += 1
            print('this is index', self.img_epoch)
        else:
            pass

if __name__ == '__main__':

    # np.random.seed(185)
    # random.seed(185)
    para_dict = {'start_num': 3500, 'end_num': 4000, 'thread': 0, 'total_boxes': 200, 'points_range': np.array([4, 5, 6]),
                 'yolo_conf': 0.6, 'yolo_iou': 0.8, 'device': 'cuda:0',
                 'reset_pos': np.array([0, 0, 0.12]), 'reset_ori': np.array([0, np.pi / 2, 0]),
                 'save_img_flag': False,
                 'init_pos_range': [[0.10, 0.20], [-0.05, 0.05], [0.01, 0.02]],
                 'init_ori_range': [[-np.pi / 8, np.pi / 8], [-np.pi / 8, np.pi / 8], [-np.pi / 8, np.pi / 8]],
                 'boxes_num': np.random.randint(1, 5),
                 'is_render': False,
                 'box_range': [[0.016, 0.048], [0.016], [0.01, 0.02]],
                 'box_mass': 0.1,
                 'gripper_threshold': 0.002, 'gripper_sim_step': 10, 'gripper_force': 3,
                 'move_threshold': 0.001, 'move_force': 3,
                 'gripper_lateral_friction': 1, 'gripper_contact_damping': 1, 'gripper_contact_stiffness': 50000,
                 'box_lateral_friction': 1, 'box_contact_damping': 1, 'box_contact_stiffness': 50000,
                 'base_lateral_friction': 1, 'base_contact_damping': 1, 'base_contact_stiffness': 50000,
                 'dataset_path': '../../../knolling_dataset/yolo_segmentation_820/',
                 'urdf_path': '../../urdf/',
                 'yolo_model_path': '../../627_pile_pose/weights/best.pt',
                 'real_operate': False, 'obs_order': 'sim_image_obj', 'Data_collection': True,
                 'use_knolling_model': False, 'use_lstm_model': False}

    lstm_dict = {'input_size': 6,
                 'hidden_size': 32,
                 'num_layers': 8,
                 'output_size': 2,
                 'hidden_node_1': 32, 'hidden_node_2': 8,
                 'batch_size': 1,
                 'device': 'cuda:0',
                 'set_dropout': 0.1,
                 'threshold': 0.5,
                 'grasp_model_path': '../results/LSTM_727_2_heavy_multi_dropout0.5/best_model.pt', }

    startnum = para_dict['start_num']

    data_root = para_dict['dataset_path']
    with open(para_dict['dataset_path'][:-1] + '_readme.txt', "w") as f:
        for key, value in para_dict.items():
            f.write(key + ': ')
            f.write(str(value) + '\n')

    os.makedirs(data_root, exist_ok=True)

    env = Yolo_seg_env(para_dict=para_dict, lstm_dict=lstm_dict)
    os.makedirs(data_root + 'origin_images/', exist_ok=True)
    os.makedirs(data_root + 'origin_labels/', exist_ok=True)

    exist_img_num = startnum
    while True:
        env.reset()

