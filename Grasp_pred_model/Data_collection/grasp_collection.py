from environment import Arm_env
import pybullet as p
import pybullet_data as pd
import numpy as np
import random
import os

class Grasp_env(Arm_env):

    def __init__(self, endnum=None, save_img_flag=None,
                 para_dict=None, init_pos_range=None, init_ori_range=None):

        super(Grasp_env, self).__init__(para_dict=para_dict)

    def try_grasp(self, data_root=None, img_index_start=None):
        print('this is img_index start while grasping', img_index_start)
        manipulator_before, pred_lwh_list, pred_conf = self.get_obs(data_root=data_root, epoch=self.img_per_epoch + img_index_start)

        if len(manipulator_before) <= 1:
            print('no pile in the environment, try to reset!')
            return self.img_per_epoch
        # if np.any(manipulator_before[:, 2].reshape(1, -1) > 0.01) == False:
        #     print('no pile in the environment, try to reset!')
        #     return self.img_per_epoch

        pos_ori_after = np.concatenate((self.reset_pos, np.zeros(3)), axis=0).reshape(-1, 6)
        manipulator_after = np.repeat(pos_ori_after, len(manipulator_before), axis=0)

        offset_low = np.array([0, 0, 0.0])
        offset_high = np.array([0, 0, 0.05])

        if len(manipulator_before) == 0:
            start_end = []
            grasp_width = []
        else:
            start_end = np.concatenate((manipulator_before, manipulator_after), axis=1)
            grasp_width = np.min(pred_lwh_list[:, :2], axis=1)
            box_pos_before = self.gt_pos_ori[:, :3]
            box_ori_before = np.copy(self.gt_ori_qua)
            if len(start_end) > len(box_pos_before):
                print('the yolo model predict additional bounding boxes!')
                cut_index = np.arange(len(box_pos_before), len(start_end))
                start_end = np.delete(start_end, cut_index, axis=0)
                pred_conf = np.delete(pred_conf, cut_index)

            state_id = p.saveState()
            grasp_flag = []
            gt_data = []
            exist_gt_index = []

            for i in range(len(start_end)):

                trajectory_pos_list = [self.reset_pos,
                                       [0.02, grasp_width[i]],  # open!
                                       offset_high + start_end[i][:3],
                                       offset_low + start_end[i][:3],
                                       [0.0273, grasp_width[i]],  # close
                                       offset_high + start_end[i][:3],
                                       start_end[i][6:9]]
                trajectory_ori_list = [self.reset_ori,
                                       self.reset_ori + start_end[i][3:6],
                                       self.reset_ori + start_end[i][3:6],
                                       self.reset_ori + start_end[i][3:6],
                                       [0.0273, grasp_width[i]],
                                       self.reset_ori + start_end[i][3:6],
                                       self.reset_ori + start_end[i][9:12]]
                last_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
                last_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))

                success_grasp_flag = True
                left_pos = np.asarray(p.getLinkState(self.arm_id, 7)[0])
                right_pos = np.asarray(p.getLinkState(self.arm_id, 8)[0])
                for j in range(len(trajectory_pos_list)):
                    if len(trajectory_pos_list[j]) == 3:
                        if j == 2:
                            last_pos, left_pos, right_pos, _ = self.move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j], index=j)
                        elif j == 3:
                            last_pos, _, _, success_grasp_flag = self.move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j],
                                                                           origin_left_pos=left_pos, origin_right_pos=right_pos, index=j)
                            if success_grasp_flag == False:
                                break
                        else:
                            last_pos, _, _, _ = self.move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j])
                        last_ori = np.copy(trajectory_ori_list[j])
                    elif len(trajectory_pos_list[j]) == 2:
                        # time.sleep(2)
                        success_grasp_flag = self.gripper(trajectory_pos_list[j][0], trajectory_pos_list[j][1], left_pos, right_pos, index=j)

                # find which box is moved and judge whether the grasp is success
                if success_grasp_flag == False:
                    print('fail!')
                    grasp_flag.append(0)
                    pass
                else:
                    for j in range(len(self.obj_idx)):
                        success_grasp_flag = False
                        fail_break_flag = False
                        box_pos = np.asarray(p.getBasePositionAndOrientation(self.obj_idx[j])[0])  # this is the pos after of the grasped box
                        if np.abs(box_pos[0] - last_pos[0]) < 0.02 and np.abs(box_pos[1] - last_pos[1]) < 0.02 and box_pos[2] > 0.06 and \
                            np.linalg.norm(box_pos_before[j, :2] - start_end[i, :2]) < 0.005:
                            for m in range(len(self.obj_idx)):
                                box_pos_after = np.asarray(p.getBasePositionAndOrientation(self.obj_idx[m])[0])
                                ori_qua_after = p.getBasePositionAndOrientation(self.obj_idx[m])[1]
                                box_ori_after = np.asarray(ori_qua_after)
                                upper_limit = np.sum(np.abs(box_ori_after + box_ori_before[m]))
                                if box_pos_after[2] > 0.06 and m != j:
                                    print(f'The {m} boxes have been disturbed, because it is also grasped accidentally, grasp fail!')
                                    p.addUserDebugPoints([box_pos_before[m]], [[0, 1, 0]], pointSize=5)
                                    grasp_flag.append(0)
                                    fail_break_flag = True
                                    success_grasp_flag = False
                                    break
                                elif m == len(self.obj_idx) - 1:
                                    grasp_flag.append(1)
                                    print('grasp success!')
                                    success_grasp_flag = True
                                    fail_break_flag = False
                            if success_grasp_flag == True or fail_break_flag == True:
                                break
                        elif j == len(self.obj_idx) - 1:
                            print('the target box does not move to the designated pos, grasp fail!')
                            success_grasp_flag = False
                            grasp_flag.append(0)

                # gt_index = np.argmin(np.linalg.norm(box_pos_before[:, :2] - start_end[i, :2], axis=1))

                # this is gt data label
                gt_index = np.argsort(np.linalg.norm(box_pos_before[:, :2] - start_end[i, :2], axis=1))
                gt_index_grasp = gt_index[~np.isin(gt_index, np.asarray(exist_gt_index))][0]
                # gt_data.append(np.concatenate((box_pos_before[gt_index_grasp], self.gt_lwh_list[gt_index_grasp], self.gt_pos_ori[gt_index_grasp, 3:])))
                exist_gt_index.append(gt_index_grasp)

                # this is pred data label, we don't use the simulation data as the dataset because it doesn't work in the real-world environment
                gt_data.append(
                    np.concatenate((manipulator_before[i, :3], pred_lwh_list[i, :3], manipulator_before[i, 3:])))

                if success_grasp_flag == True:
                    print('we should remove this box and try the rest boxes!')
                    rest_len = len(exist_gt_index)
                    for m in range(1, len(start_end) - rest_len + 1):
                        grasp_flag.append(0)
                        gt_data.append(np.concatenate(
                            (manipulator_before[i + m, :3], pred_lwh_list[i + m, :3], manipulator_before[i + m, 3:])))
                        # gt_index = np.argsort(np.linalg.norm(box_pos_before[:, :2] - start_end[i + m, :2], axis=1))
                        # gt_index = gt_index[~np.isin(gt_index, np.asarray(exist_gt_index))][0]
                        # gt_data.append(np.concatenate(
                        #     (box_pos_before[gt_index], self.gt_lwh_list[gt_index], self.gt_pos_ori[gt_index, 3:])))
                        # exist_gt_index.append(gt_index)
                    # p.restoreState(state_id)
                    p.removeBody(self.obj_idx[gt_index_grasp])
                    print('this is len of self.obj', len(self.obj_idx))
                    del self.obj_idx[gt_index_grasp]
                    self.gt_lwh_list = np.delete(self.gt_lwh_list, gt_index_grasp, axis=0)

                    ######## after every grasp, check pos and ori of every box which are out of the field ########
                    forbid_range = np.array([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
                    delete_index = []
                    print('this is len of self.obj', len(self.obj_idx))
                    for m in range(len(self.obj_idx)):
                        cur_ori = np.asarray(
                            p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.obj_idx[m])[1]))
                        cur_pos = np.asarray(p.getBasePositionAndOrientation(self.obj_idx[m])[0])
                        roll_flag = False
                        pitch_flag = False
                        for n in range(len(forbid_range)):
                            if np.abs(cur_ori[0] - forbid_range[n]) < 0.01:
                                roll_flag = True
                            if np.abs(cur_ori[1] - forbid_range[n]) < 0.01:
                                pitch_flag = True
                        if roll_flag == True and pitch_flag == True and (
                                np.abs(cur_ori[0] - 0) > 0.01 or np.abs(cur_ori[1] - 0) > 0.01) or \
                                cur_pos[0] < self.x_low_obs or cur_pos[0] > self.x_high_obs or cur_pos[
                            1] > self.y_high_obs or cur_pos[1] < self.y_low_obs:
                            delete_index.append(m)
                    delete_index.reverse()
                    for idx in delete_index:
                        print('this is delete index', idx)
                        p.removeBody(self.obj_idx[idx])
                        self.obj_idx.pop(idx)
                        self.gt_lwh_list = np.delete(self.gt_lwh_list, idx, axis=0)
                    ######## after every grasp, check pos and ori of every box which are out of the field ########

                    break
                else:
                    p.restoreState(state_id)
                    print('restore the previous env and try another one')

            gt_data = np.asarray(gt_data)
            grasp_flag = np.asarray(grasp_flag).reshape(-1, 1)

            yolo_label = np.concatenate((grasp_flag, gt_data, pred_conf.reshape(-1, 1)), axis=1)

        if len(start_end) == 0:
            print('No box on the image! This is total num of img after one epoch', self.img_per_epoch)
            return self.img_per_epoch
        elif np.all(grasp_flag == 0):
            np.savetxt(os.path.join(data_root, "origin_labels/%012d.txt" % (img_index_start + self.img_per_epoch)), yolo_label, fmt='%.04f')
            if self.save_img_flag == False:
                os.remove(data_root + 'origin_images/%012d.png' % (self.img_per_epoch + img_index_start))
            self.img_per_epoch += 1
            print('this is total num of img after one epoch', self.img_per_epoch)
            return self.img_per_epoch
        else:
            np.savetxt(os.path.join(data_root, "origin_labels/%012d.txt" % (img_index_start + self.img_per_epoch)), yolo_label, fmt='%.04f')
            if self.save_img_flag == False:
                os.remove(data_root + 'origin_images/%012d.png' % (self.img_per_epoch + img_index_start))
            self.img_per_epoch += 1
            return self.try_grasp(data_root=data_root, img_index_start=img_index_start)

if __name__ == '__main__':

    # np.random.seed(183)
    # random.seed(183)

    para_dict = {'start_num': 00, 'end_num': 10000, 'thread': 0,
                 'yolo_conf': 0.6, 'yolo_iou': 0.8, 'device': 'cuda:0',
                 'save_img_flag': True,
                 'init_pos_range': [[0.13, 0.17], [-0.03, 0.03], [0.01, 0.02]],
                 'init_ori_range': [[-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4]],
                 'max_box_num': 5, 'min_box_num': 4,
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
                 'dataset_path': '../../../knolling_dataset/grasp_dataset_721_heavy_test/',
                 'urdf_path': '../../urdf/',
                 'yolo_model_path': '../../train_pile_overlap_627/weights/best.pt'}

    startnum = para_dict['start_num']
    endnum = para_dict['end_num']
    thread = para_dict['thread']
    save_img_flag = para_dict['save_img_flag']
    init_pos_range = para_dict['init_pos_range']
    init_ori_range = para_dict['init_ori_range']

    data_root = para_dict['dataset_path']
    with open(para_dict['dataset_path'] + 'grasp_dataset_721_heavy_test_readme.txt', "w") as f:
        for key, value in para_dict.items():
            f.write(key + ': ')
            f.write(str(value) + '\n')

    os.makedirs(data_root, exist_ok=True)

    max_box_num = para_dict['max_box_num']
    min_box_num = para_dict['min_box_num']
    mm2px = 530 / 0.34

    env = Grasp_env(para_dict=para_dict)
    os.makedirs(data_root + 'origin_images/', exist_ok=True)
    os.makedirs(data_root + 'origin_labels/', exist_ok=True)

    exist_img_num = startnum
    while True:
        num_item = int(np.random.uniform(min_box_num, max_box_num + 1))
        env.reset(data_root=data_root, num_item=num_item, thread=thread, epoch=exist_img_num)
        img_per_epoch = env.try_grasp(data_root=data_root, img_index_start=exist_img_num)
        exist_img_num += img_per_epoch