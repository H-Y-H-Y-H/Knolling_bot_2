from environment import Arm_env
import pybullet as p
import pybullet_data as pd
import numpy as np
import random
import os
from utils import *

class Unstack_env(Arm_env):

    def __init__(self, para_dict=None, lstm_dict=None):

        super(Unstack_env, self).__init__(para_dict=para_dict, lstm_dict=lstm_dict)
        self.table_center = np.array([0.15, 0])
        self.begin_pos = np.array([0.05, 0, 0.10])
        self.begin_ori = np.array([0, np.pi / 2, 0])

    def try_unstack(self, data_root=None, img_index_start=None):

        if self.img_per_epoch + img_index_start >= self.endnum:
            print('END!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            quit()

        ############################## Get the information of boxes #################################
        print('this is img_index start while grasping', img_index_start)
        manipulator_before, new_lwh_list, pred_conf = self.get_obs(epoch=self.img_per_epoch + img_index_start)

        if len(manipulator_before) <= 1 or len(self.boxes_index) == 1:
            print('no pile in the environment, try to reset!')
            return self.img_per_epoch
        ############################## Get the information of boxes #################################

        ############## Genarete the results of LSTM model #############
        order = change_sequence(manipulator_before)
        manipulator_before_input = manipulator_before[order]
        new_lwh_list_input = new_lwh_list[order]
        pred_conf_input = pred_conf[order]
        crowded_index, prediction, model_output = self.grasp_model.pred(manipulator_before_input, new_lwh_list_input, pred_conf_input)
        print('this is crowded_index', crowded_index)
        print('this is prediction', prediction)
        self.yolo_pose_model.plot_grasp(manipulator_before_input, prediction, model_output)
        if len(crowded_index) < len(manipulator_before_input):
            print('There are some boxes can be grasp, try to reset!')
            os.remove(data_root + 'sim_images/%012d_pred.png' % (self.img_per_epoch + img_index_start))
            os.remove(data_root + 'sim_images/%012d_pred_grasp.png' % (self.img_per_epoch + img_index_start))
            os.remove(data_root + 'sim_images/%012d.png' % (self.img_per_epoch + img_index_start))
            return self.img_per_epoch
        else:
            if self.para_dict['rl_configuration'] == True:
                pos_before = []
                ori_before = []
                qua_before = []
                for i in range(len(self.boxes_index)):
                    qua_before.append(np.asarray(p.getBasePositionAndOrientation(self.boxes_index[i])[1]))
                    ori_before.append(np.asarray(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.boxes_index[i])[1])))
                    pos_before.append(np.asarray(p.getBasePositionAndOrientation(self.boxes_index[i])[0]))
                pos_before = np.asarray(pos_before)
                ori_before = np.asarray(ori_before)
                qua_before = np.asarray(qua_before)
                output_data = np.concatenate((pos_before, ori_before, self.lwh_list, qua_before), axis=1)
                np.savetxt(os.path.join(data_root, "sim_labels/%012d.txt" % (img_index_start + self.img_per_epoch)),
                           output_data, fmt='%.04f')
                if self.save_img_flag == False:
                    os.remove(data_root + 'sim_images/%012d.png' % (self.img_per_epoch + img_index_start))
                self.img_per_epoch += 1
                print('this is total num of img after one epoch', self.img_per_epoch)
                return self.img_per_epoch
            else:
                pass
        ############## Genarete the results of LSTM model #############

        self.calculate_gripper()

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
        ######################### Back to the reset pos in any cases ##########################



        # if exist_success_num > 0:
        #     # print('exist success boxes, we should remove this box and try the rest boxes!')
        #     # rest_len = len(exist_success_index)
        #     # ############################# Align the data of rest boxes #############################
        #     # for m in range(1, len(start_end) - rest_len + 1):
        #     #     grasp_flag.append(0)
        #     #     box_data.append(np.concatenate((manipulator_before[i + m, :3], new_lwh_list[i + m, :3], manipulator_before[i + m, 3:])))
        #     # ############################# Align the data of rest boxes #############################
        #
        #     random_index = np.random.choice(np.asarray(exist_success_index))
        #     p.removeBody(self.boxes_index[random_index])
        #     del self.boxes_index[random_index]
        #     self.lwh_list = np.delete(self.lwh_list, random_index, axis=0)
        #     for _ in range(int(50)):
        #         # time.sleep(1/96)
        #         p.stepSimulation()
        #
        #     ##################### after every grasp, check pos and ori of every box which are out of the field ####################
        #     forbid_range = np.array([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        #     delete_index = []
        #     for m in range(len(self.boxes_index)):
        #         cur_ori = np.asarray(
        #             p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.boxes_index[m])[1]))
        #         cur_pos = np.asarray(p.getBasePositionAndOrientation(self.boxes_index[m])[0])
        #         roll_flag = False
        #         pitch_flag = False
        #         for n in range(len(forbid_range)):
        #             if np.abs(cur_ori[0] - forbid_range[n]) < 0.01:
        #                 roll_flag = True
        #             if np.abs(cur_ori[1] - forbid_range[n]) < 0.01:
        #                 pitch_flag = True
        #         if roll_flag == True and pitch_flag == True and (
        #                 np.abs(cur_ori[0] - 0) > 0.01 or np.abs(cur_ori[1] - 0) > 0.01) or \
        #                 cur_pos[0] < self.x_low_obs or cur_pos[0] > self.x_high_obs or cur_pos[
        #             1] > self.y_high_obs or cur_pos[1] < self.y_low_obs:
        #             delete_index.append(m)
        #     delete_index.reverse()
        #     for idx in delete_index:
        #         # print('this is delete index', idx)
        #         p.removeBody(self.boxes_index[idx])
        #         self.boxes_index.pop(idx)
        #         self.lwh_list = np.delete(self.lwh_list, idx, axis=0)
        #     for _ in range(int(50)):
        #         # time.sleep(1/96)
        #         p.stepSimulation()
        #     ##################### after every grasp, check pos and ori of every box which are out of the field ####################

        box_data = np.asarray(box_data)
        grasp_flag = np.asarray(grasp_flag).reshape(-1, 1)
        yolo_label = np.concatenate((grasp_flag, box_data, pred_conf.reshape(-1, 1)), axis=1)

        if np.all(grasp_flag == 0):
            np.savetxt(os.path.join(data_root, "origin_labels/%012d.txt" % (img_index_start + self.img_per_epoch)), yolo_label, fmt='%.04f')
            if self.save_img_flag == False:
                os.remove(data_root + 'sim_images/%012d.png' % (self.img_per_epoch + img_index_start))
            self.img_per_epoch += 1
            print('this is total num of img after one epoch', self.img_per_epoch)
            return self.img_per_epoch
        else:
            np.savetxt(os.path.join(data_root, "origin_labels/%012d.txt" % (img_index_start + self.img_per_epoch)), yolo_label, fmt='%.04f')
            if self.save_img_flag == False:
                os.remove(data_root + 'sim_images/%012d.png' % (self.img_per_epoch + img_index_start))
            self.img_per_epoch += 1
            return self.try_unstack(data_root=data_root, img_index_start=img_index_start)

if __name__ == '__main__':

    # np.random.seed(185)
    # random.seed(185)
    para_dict = {'start_num': 250, 'end_num': 500, 'thread': 0,
                 'yolo_conf': 0.6, 'yolo_iou': 0.8, 'device': 'cuda:0',
                 'reset_pos': np.array([0.02, 0, 0.10]), 'reset_ori': np.array([0, np.pi / 2, 0]),
                 'save_img_flag': True,
                 'init_pos_range': [[0.13, 0.17], [-0.03, 0.03], [0.01, 0.02]], 'init_offset_range': [[-0.05, 0.05], [-0.1, 0.1]],
                 'init_ori_range': [[-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4]],
                 'boxes_num': np.random.randint(5, 6),
                 'is_render': False,
                 'box_range': [[0.016, 0.048], [0.016], [0.01, 0.02]],
                 'box_mass': 0.1,
                 'gripper_threshold': 0.002, 'gripper_sim_step': 10, 'gripper_force': 3,
                 'move_threshold': 0.001, 'move_force': 3,
                 'gripper_lateral_friction': 1, 'gripper_contact_damping': 1, 'gripper_contact_stiffness': 50000,
                 'box_lateral_friction': 1, 'box_contact_damping': 1, 'box_contact_stiffness': 50000,
                 'base_lateral_friction': 1, 'base_contact_damping': 1, 'base_contact_stiffness': 50000,
                 'dataset_path': '../../../knolling_dataset/RL_configuration_831/',
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
                 'device': 'cuda:0',
                 'set_dropout': 0.1,
                 'threshold': 0.6,
                 'grasp_model_path': '../../models/LSTM_829_1_heavy_dropout0/best_model.pt', }

    startnum = para_dict['start_num']

    data_root = para_dict['dataset_path']
    with open(para_dict['dataset_path'][:-1] + '_readme.txt', "w") as f:
        for key, value in para_dict.items():
            f.write(key + ': ')
            f.write(str(value) + '\n')

    os.makedirs(data_root, exist_ok=True)

    env = Unstack_env(para_dict=para_dict, lstm_dict=lstm_dict)
    os.makedirs(data_root + 'sim_images/', exist_ok=True)
    os.makedirs(data_root + 'sim_labels/', exist_ok=True)

    exist_img_num = startnum
    while True:
        num_item = para_dict['boxes_num']
        env.reset(epoch=exist_img_num)
        img_per_epoch = env.try_unstack(data_root=data_root, img_index_start=exist_img_num)
        exist_img_num += img_per_epoch

