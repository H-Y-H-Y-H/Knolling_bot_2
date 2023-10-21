from environment import Arm_env
import pybullet as p
import pybullet_data as pd
import numpy as np
import random
import os
from utils import *

class Grasp_env(Arm_env):

    def __init__(self, para_dict=None, lstm_dict=None):

        super(Grasp_env, self).__init__(para_dict=para_dict, lstm_dict=lstm_dict)

        self.test_TP = 0
        self.test_TN = 0
        self.test_FP = 0
        self.test_FN = 0

    def get_box_gt(self):

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

        self.gt_pos_ori = np.asarray(self.gt_pos_ori)
        self.gt_ori_qua = np.asarray(self.gt_ori_qua)

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
        output_data = np.concatenate((pos_before, ori_before, self.lwh_list, qua_before), axis=1)

        return output_data, flag

    def try_unstack(self, data_root=None, img_index_start=None):

        if self.img_per_epoch + img_index_start >= self.endnum:
            print('END!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            if self.para_dict['real_operate'] == True:
                end = np.array([0], dtype=np.float32)
                self.conn.sendall(end.tobytes())
            quit()
        else:
            if len(self.boxes_index) <= 1:
                print('no pile in the environment, try to reset!')
                return self.img_per_epoch
            else:
                output_data, check_flag = self.check_bound()

        # output_data: xyz, rpy, lwh, qua
        if check_flag == False:
            print('object out of bound, try another yaw')
            return self.img_per_epoch
        else:
            # self.yolo_pose_model.plot_grasp(manipulator_before_input, prediction, model_output)
            # self.get_obs(look_flag=True, epoch=self.img_per_epoch + img_index_start)
            np.savetxt(os.path.join(data_root, "sim_info/%012d.txt" % (img_index_start + self.img_per_epoch)), output_data, fmt='%.04f')
            self.img_per_epoch += 1
            print('this is total num of img after one epoch', self.img_per_epoch + img_index_start)
            return self.img_per_epoch

if __name__ == '__main__':

    # np.random.seed(185)
    # random.seed(185)
    para_dict = {'start_num': 270000, 'end_num': 300000, 'thread': 0,
                 'yolo_conf': 0.6, 'yolo_iou': 0.8, 'device': 'cuda:1',
                 'reset_pos': np.array([0, 0, 0.12]), 'reset_ori': np.array([0, np.pi / 2, 0]),
                 'save_img_flag': True,
                 'recover_offset_range': [[-0.05, 0.05], [-0.1, 0.1]],
                 'init_pos_range': [[0.13, 0.17], [-0.03, 0.03], [0.01, 0.02]], 'init_offset_range': [[-0.0, 0.0], [-0., 0.]],
                 'init_ori_range': [[-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4]],
                 'boxes_num': np.random.randint(5, 6),
                 'is_render': False,
                 'box_range': [[0.016, 0.048], [0.016], [0.01, 0.02]],
                 'box_mass': 0.1,
                 'gripper_threshold': 0.001, 'gripper_sim_step': 10, 'gripper_force': 3,
                 'move_threshold': 0.001, 'move_force': 3,
                 'gripper_lateral_friction': 1, 'gripper_contact_damping': 1, 'gripper_contact_stiffness': 50000,
                 'box_lateral_friction': 1, 'box_contact_damping': 1, 'box_contact_stiffness': 50000,
                 'base_lateral_friction': 1, 'base_contact_damping': 1, 'base_contact_stiffness': 50000,
                 'data_source_path': '../../../knolling_dataset/base_dataset_crowded/',
                 'urdf_path': '../../urdf/',
                 'yolo_model_path': '../../models/627_pile_pose/weights/best.pt',
                 'real_operate': False, 'obs_order': 'sim_image_obj', 'Data_collection': True,
                 'use_knolling_model': False, 'use_lstm_model': False, 'use_yolo_model': False}

    lstm_dict = {'input_size': 6,
                 'hidden_size': 32,
                 'num_layers': 8,
                 'output_size': 2,
                 'hidden_node_1': 32, 'hidden_node_2': 8,
                 'batch_size': 1,
                 'device': 'cuda:1',
                 'set_dropout': 0.1,
                 'threshold': 0.5,
                 'grasp_model_path': '../results/LSTM_727_2_heavy_multi_dropout0.5/best_model.pt', }

    startnum = para_dict['start_num']

    with open(para_dict['data_source_path'][:-1] + '_readme.txt', "w") as f:
        for key, value in para_dict.items():
            f.write(key + ': ')
            f.write(str(value) + '\n')

    # data_root = '../../../knolling_dataset/grasp_dataset_913/'
    data_root = para_dict['data_source_path']
    os.makedirs(data_root, exist_ok=True)

    env = Grasp_env(para_dict=para_dict, lstm_dict=lstm_dict)
    os.makedirs(data_root + 'sim_images/', exist_ok=True)
    os.makedirs(data_root + 'sim_info/', exist_ok=True)

    exist_img_num = startnum
    while True:
        num_item = para_dict['boxes_num']
        env.reset(epoch=exist_img_num, recover_flag=False)
        img_per_epoch = env.try_unstack(data_root=data_root, img_index_start=exist_img_num)
        exist_img_num += img_per_epoch

