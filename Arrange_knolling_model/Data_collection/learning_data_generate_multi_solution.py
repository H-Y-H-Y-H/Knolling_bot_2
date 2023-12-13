import numpy as np
# import pyrealsense2 as rs
from sort_data_collection_multi_solution import Sort_objects
import pybullet_data as pd
import math
import random
# from turdf import *
import socket
import pybullet as p
import os
import cv2
# from cam_obs_yolov8 import *
import torch
import torch.nn as nn
import torch.nn.functional as F
# from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from urdfpy import URDF
import shutil

from arrange_policy import configuration_zzz, random_arrangement

torch.manual_seed(42)

class Arm:

    def __init__(self, is_render=True):

        self.kImageSize = {'width': 480, 'height': 480}
        self.urdf_path = '../../ASSET/urdf/'
        self.pybullet_path = pd.getDataPath()
        self.is_render = is_render
        if self.is_render:
            # p.connect(p.GUI, options="--width=1280 --height=720")
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.num_motor = 5

        self.low_scale = np.array([0.03, -0.14, 0.0, - np.pi / 2, 0])
        self.high_scale = np.array([0.27, 0.14, 0.05, np.pi / 2, 0.4])
        self.low_act = -np.ones(5)
        self.high_act = np.ones(5)
        self.x_low_obs = self.low_scale[0]
        self.x_high_obs = self.high_scale[0]
        self.y_low_obs = self.low_scale[1]
        self.y_high_obs = self.high_scale[1]
        self.z_low_obs = self.low_scale[2]
        self.z_high_obs = self.high_scale[2]
        self.table_boundary = 0.03

        self.lateral_friction = 1
        self.spinning_friction = 1
        self.rolling_friction = 0

        self.camera_parameters = {
            'width': 640.,
            'height': 480,
            'fov': 42,
            'near': 0.1,
            'far': 100.,
            'eye_position': [0.59, 0, 0.8],
            'target_position': [0.55, 0, 0.05],
            'camera_up_vector':
                [1, 0, 0],  # I really do not know the parameter's effect.
            'light_direction': [
                0.5, 0, 1
            ],  # the direction is from the light source position to the origin of the world frame.
        }
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.15, 0, 0],
            distance=0.4,
            yaw=90,
            pitch=-90,
            roll=0,
            upAxisIndex=2)
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_parameters['fov'],
            aspect=self.camera_parameters['width'] / self.camera_parameters['height'],
            nearVal=self.camera_parameters['near'],
            farVal=self.camera_parameters['far'])

        if random.uniform(0, 1) > 0.5:
            p.configureDebugVisualizer(lightPosition=[random.randint(1, 3), random.randint(1, 2), 5])
        else:
            p.configureDebugVisualizer(lightPosition=[random.randint(1, 3), random.randint(-2, -1), 5])
        p.configureDebugVisualizer(lightPosition=[random.randint(1, 3), random.randint(1, 2), 5],
                                   shadowMapResolution=8192, shadowMapIntensity=np.random.randint(5, 8) / 10)
        p.resetDebugVisualizerCamera(cameraDistance=0.5,
                                     cameraYaw=45,
                                     cameraPitch=-45,
                                     cameraTargetPosition=[0.1, 0, 0])
        p.setAdditionalSearchPath(pd.getDataPath())

    def change_sequence(self, pos, flag=None):
        if flag == 'distance':
            origin_point = np.array([0, 0])
            delete_index = np.where(pos == 0)[0]
            distance = np.linalg.norm(pos[:, :2] - origin_point, axis=1)
            order = np.argsort(distance)
        if flag == 'area':

            order = 1
        if flag == 'length':
            order = 1
        return order

    def get_parameters(self, box_num=None, area_ratio=None, solution_num=None, kind_num=None,
                       boxes_index=None, total_offset=None,
                       gap_item=0.03, gap_block=0.02,
                       real_operate=False, obs_order='1',
                       random_offset=False, check_detection_loss=None, obs_img_from=None, use_yolo_pos=True,
                       item_odd_prevent=None, block_odd_prevent=None, upper_left_max = None, forced_rotate_box=None,
                       iteration_time=None):

        # self.lego_num = lego_num
        self.total_offset = total_offset
        self.area_ratio = area_ratio
        self.solution_num = solution_num
        self.kind_num = kind_num
        self.gap_item = gap_item
        self.gap_block = gap_block
        self.real_operate = real_operate
        self.obs_order = obs_order
        self.random_offset = random_offset
        self.num_list = box_num
        self.check_detection_loss = check_detection_loss
        self.obs_img_from = obs_img_from
        self.use_yolo_pos = use_yolo_pos
        self.boxes_index = boxes_index
        self.item_odd_prevent = item_odd_prevent
        self.block_odd_prevent = block_odd_prevent
        self.upper_left_max = upper_left_max
        self.forced_rotate_box = forced_rotate_box
        self.iteration_time = iteration_time

    def get_obs(self, order, evaluation):

        def get_images():
            (width, length, image, _, _) = p.getCameraImage(width=640,
                                                            height=480,
                                                            viewMatrix=self.view_matrix,
                                                            projectionMatrix=self.projection_matrix,
                                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)
            return image

        if order == 'images':
            image = get_images()
            return image

    def label2image(self, labels_data, img_index, save_urdf_path):
        # print(index_flag)
        # index_flag = index_flag.reshape(2, -1)

        total_offset = [0.016, -0.17 + 0.016, 0]

        labels_data = labels_data.reshape(-1, 6)
        pos_data = labels_data[:, :2]
        pos_data = np.concatenate((pos_data, np.zeros(len(pos_data)).reshape(-1, 1)), axis=1)
        pos_data[:, 0] += total_offset[0]
        pos_data[:, 1] += total_offset[1]
        lw_data = labels_data[:, 2:4]
        ori_data = labels_data[:, 4]
        ori_data = np.concatenate((np.zeros((len(ori_data), 2)), ori_data.reshape(-1, 1)), axis=1)
        cls_data = labels_data[:, 5]

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        # Draw workspace lines
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

        baseid = p.loadURDF(self.urdf_path + "plane_zzz.urdf", basePosition=[0, -0.2, 0], useFixedBase=1,
                            flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        # self.arm_id = p.loadURDF(self.urdf_path + "robot_arm928/robot_arm1.urdf",
        #                          basePosition=[-0.08, 0, 0.02], useFixedBase=True,
        #                          flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        textureId = p.loadTexture(self.urdf_path + "img_1.png")
        p.changeDynamics(baseid, -1, lateralFriction=1, spinningFriction=1, rollingFriction=0.002, linearDamping=0.5,
                         angularDamping=0.5)
        # p.changeDynamics(self.arm_id, 7, lateralFriction=1, spinningFriction=1, rollingFriction=0, linearDamping=0, angularDamping=0)
        # p.changeDynamics(self.arm_id, 8, lateralFriction=1, spinningFriction=1, rollingFriction=0, linearDamping=0, angularDamping=0)
        p.changeVisualShape(baseid, -1, textureUniqueId=textureId,
                            rgbaColor=[np.random.uniform(0.9, 1), np.random.uniform(0.9, 1), np.random.uniform(0.9, 1), 1])

        ################### recover urdf boxes based on lw_data ###################
        boxes = []
        xyz_list = []
        new_pos_data = []
        new_ori_data = []
        # for i in range(len(index_flag[0])):
        #     boxes.append(URDF.load('../urdf/box_generator/box_%d.urdf' % index_flag[0, i]))
        #     xyz_list.append(boxes[i].links[0].visuals[0].geometry.box.size)

        temp_box = URDF.load('../../ASSET/urdf/box_generator/template.urdf')
        save_urdf_path_one_img = save_urdf_path + 'img_%d/' % img_index
        os.makedirs(save_urdf_path_one_img, exist_ok=True)
        for i in range(len(lw_data)):
            temp_box.links[0].collisions[0].origin[2, 3] = 0
            length = lw_data[i, 0]
            width = lw_data[i, 1]
            height = 0.012
            temp_box.links[0].visuals[0].geometry.box.size = [length, width, height]
            temp_box.links[0].collisions[0].geometry.box.size = [length, width, height]
            temp_box.links[0].visuals[0].material.color = [np.random.random(), np.random.random(), np.random.random(), 1]
            temp_box.save(save_urdf_path_one_img + 'box_%d.urdf' % (i))

        lego_idx = []
        selected_lw_data = []
        selected_urdf = []
        print('position\n', pos_data)
        print('orietation\n', ori_data)
        print('lwh\n', lw_data)
        for i in range(len(lw_data)):
            print(f'this is matching urdf{i}')
            # print(pos_data[i])
            # print(lw_data[i])
            # print(ori_data[i])
            pos_data[i, 2] += 0.006
            lego_idx.append(p.loadURDF(save_urdf_path_one_img + 'box_%d.urdf' % (i),
                           basePosition=pos_data[i],
                           baseOrientation=p.getQuaternionFromEuler(ori_data[i]), useFixedBase=False,
                           flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
        ################### recover urdf boxes based on lw_data ###################

        # shutil.rmtree(save_urdf_path_one_img)

        return self.get_obs('images', None)

    def change_config(self):  # this is main function!!!!!!!!!

        # get the standard xyz and corresponding index from files in the computer
        items_sort = Sort_objects()
        lwh_list = items_sort.get_data_virtual(self.num_list)

        # determine the center of the tidy configuration
        pos_list_epoch = []
        ori_list_epoch = []
        lwh_list_epoch = []
        cls_list_epoch = []

        for cfg in range(len(self.area_ratio)):

            times = 0
            while True:
                self.xyz_list, self.all_index, self.all_cls = items_sort.judge(lwh_list, self.area_ratio[cfg][0], self.area_ratio[cfg][1])
                calculate_reorder = configuration_zzz(self.xyz_list, self.all_index, self.gap_item, self.gap_block, self.all_cls,
                                                      self.item_odd_prevent, self.block_odd_prevent, self.upper_left_max, self.forced_rotate_box, self.iteration_time)
                self.items_pos_list, self.items_ori_list = calculate_reorder.calculate_block()

                # the sequence of self.items_pos_list, self.items_ori_list are the same as those in self.xyz_list
                x_low = np.min(self.items_pos_list, axis=0)[0]
                x_high = np.max(self.items_pos_list, axis=0)[0]
                y_low = np.min(self.items_pos_list, axis=0)[1]
                y_high = np.max(self.items_pos_list, axis=0)[1]
                center = np.array([(x_low + x_high) / 2, (y_low + y_high) / 2, 0])
                x_length = abs(x_high - x_low)
                y_length = abs(y_high - y_low)
                # print(x_low, x_high, y_low, y_high)
                if self.random_offset == True:
                    self.total_offset = np.array([random.uniform(self.x_low_obs + x_length / 2, self.x_high_obs - x_length / 2),
                                                  random.uniform(self.y_low_obs + y_length / 2, self.y_high_obs - y_length / 2),
                                                  0.0])
                else:
                    pass
                self.items_pos_list = self.items_pos_list + self.total_offset
                self.manipulator_after = np.concatenate((self.items_pos_list, self.items_ori_list), axis=1)

                # return self.get_obs('images', None)

                ################# change the sequence of data based on the max area of single box #####################
                # box_order = np.argsort(self.xyz_list[:, 0] * self.xyz_list[:, 1])[::-1]
                # self.items_pos_list = self.items_pos_list[box_order]
                # self.items_ori_list = self.items_ori_list[box_order]
                # self.xyz_list = self.xyz_list[box_order]

                # for i in range(len(self.xyz_list)):
                #     if self.xyz_list[i, 0] < self.xyz_list[i, 1]:
                #         self.items_ori_list[i, 2] += (np.pi / 2)
                #         temp = self.xyz_list[i, 0]
                #         self.xyz_list[i, 0] = self.xyz_list[i, 1]
                #         self.xyz_list[i, 1] = temp
                ################# change the sequence of data based on the max area of single box #####################

                # # generate rdm pos and ori before the knolling
                # if times == 0:
                #     collect_ori = []
                #     collect_pos = []
                #     restrict = np.max(self.xyz_list)
                #     gripper_height = 0.012
                #     last_pos = np.array([[0, 0, 1]])
                #     for i in range(len(self.xyz_list)):
                #         rdm_pos = np.array([random.uniform(self.x_low_obs, self.x_high_obs),
                #                             random.uniform(self.y_low_obs, self.y_high_obs), 0.0])
                #         ori = [0, 0, random.uniform(0, np.pi)]
                #         # ori = [0, 0, 0]
                #         collect_ori.append(ori)
                #         check_list = np.zeros(last_pos.shape[0])
                #
                #         while 0 in check_list:
                #             rdm_pos = [random.uniform(self.x_low_obs, self.x_high_obs),
                #                        random.uniform(self.y_low_obs, self.y_high_obs), 0.0]
                #             for z in range(last_pos.shape[0]):
                #                 if np.linalg.norm(last_pos[z] - rdm_pos) < restrict + gripper_height:
                #                     check_list[z] = 0
                #                 else:
                #                     check_list[z] = 1
                #         collect_pos.append(rdm_pos)
                #
                #         last_pos = np.append(last_pos, [rdm_pos], axis=0)
                #     collect_pos = np.asarray(collect_pos)[:, :2]
                #     collect_ori = np.asarray(collect_ori)[:, 2]
                # # generate rdm pos and ori before the knolling

                pos_list_epoch.append(self.items_pos_list[:, :2])
                ori_list_epoch.append(self.items_ori_list[:, 2])
                lwh_list_epoch.append(self.xyz_list[:, :2])
                cls_list_epoch.append(self.all_cls)

                times += 1
                if times >= int(self.solution_num / len(self.area_ratio)):
                    break

        # return self.items_pos_list[:, :2], self.items_ori_list[:, 2], self.xyz_list[:, :2], self.transform_flag

        return pos_list_epoch, ori_list_epoch, lwh_list_epoch, cls_list_epoch


if __name__ == '__main__':

    command = 'recover'
    before_after = 'after'

    iteration_time = 10
    start_evaluations = 0
    end_evaluations =   1000
    step_num = 10
    save_point = np.linspace(int((end_evaluations - start_evaluations) / step_num + start_evaluations), end_evaluations, step_num)

    boxes_num = 30
    kind_num = 5

    area_num = 2
    ratio_num = 1

    config_dict = [[2, 1],
                   [1, 2],
                    [1, 1]]

    solution_num = 4 * len(config_dict)

    item_odd_prevent = True
    # always true
    block_odd_prevent = True
    # always true
    upper_left_max = False
    forced_rotate_box = False

    target_path = '../../../knolling_dataset/learning_data_1019/'
    images_log_path = target_path + 'images_%s/' % before_after
    os.makedirs(images_log_path, exist_ok=True)

    if command == 'recover':

        env = Arm(is_render=False)

        # data = np.loadtxt(target_path + 'labels_%s/num_%d.txt' % (before_after, i))

        names = locals()
        # data_before = []
        save_urdf_path = []
        for m in range(solution_num):
            # names['data_' + str(m)] = np.loadtxt(target_path + 'labels_%s_%s/num_%d.txt' % (before_after, m, i))
            names['data_' + str(m)] = np.loadtxt(target_path + 'num_%d_after_%d.txt' % (boxes_num, m))

            if len(names['data_' + str(m)].shape) == 1:
                names['data_' + str(m)] = names['data_' + str(m)].reshape(1, len(names['data_' + str(m)]))

            box_num = boxes_num
            print('this is len data', len(names['data_' + str(m)]))
            save_urdf_path.append(target_path + 'box_urdf/num_%s_%d/' % (m, box_num))
            os.makedirs(save_urdf_path[m], exist_ok=True)

        # new_data = []
        # new_index_flag = []
        for j in range(start_evaluations, end_evaluations):
            env.get_parameters(box_num=boxes_num)
            for m in range(solution_num):
                print(f'this is data {j}')
                one_img_data = names['data_' + str(m)][j].reshape(-1, 6)
                # one_img_index_flag = index_flag[j].reshape(2, -1)
                box_order = np.lexsort((one_img_data[:, 1], one_img_data[:, 0]))
                one_img_data = one_img_data[box_order].reshape(-1, )
                # one_img_index_flag = one_img_index_flag[:, box_order].reshape(-1, )
                # new_data.append(one_img_data)
                # new_index_flag.append(one_img_index_flag)

                image = env.label2image(names['data_' + str(m)][j], j, save_urdf_path[m])
                image = image[..., :3]
                # print('this is shape of image', image.shape)
                # image = np.transpose(image, (2, 0, 1))
                # temp = image[:, :, 2]
                # image[:, :, 2] = image[:, :, 0]
                # image[:, :, 0] = temp
                cv2.namedWindow('zzz', 0)
                cv2.resizeWindow('zzz', 1280, 960)
                cv2.imshow("zzz", image)
                cv2.waitKey()
                cv2.destroyAllWindows()

            # cv2.imwrite(images_log_path + '%d_%d.png' % (i, j), image)

        # new_data = np.asarray(new_data)
        # # new_index_flag = np.asarray(new_index_flag)
        # np.savetxt(preprocess_label_path + 'num_%d.txt' % i, new_data)
        # # np.savetxt(target_path + 'index_flag/num_%s_flag.txt' % i, new_index_flag)

    if command == 'knolling':

        with open(target_path[:-1] + "_readme.txt", "w") as f:
            f.write(f'start_evaluation: {start_evaluations}\n')
            f.write(f'end_evaluation: {end_evaluations}\n')
            f.write(f'step_num: {step_num}\n')
            f.write(f'boxes_num: {boxes_num}\n')
            # f.write(f'kind_num: {kind_num}\n')
            f.write(f'solution_num: {solution_num}\n')
            f.write(f'config: {config_dict}\n')
            f.write(f'item_odd_prevent: {item_odd_prevent}\n')
            f.write(f'block_odd_prevent: {block_odd_prevent}\n')
            f.write(f'upper_left_max: {upper_left_max}\n')
            f.write(f'forced_rotate_box: {forced_rotate_box}\n')

        env = Arm(is_render=False)

        change_cfg_flag = False

        after_path = []

        before_path = target_path + 'labels_before_0/'
        os.makedirs(before_path, exist_ok=True)
        for i in range(solution_num):
            after_path.append(target_path + 'labels_after_%s/' % i)
            # index_flag_path = target_path + 'index_flag/'
            # os.makedirs(before_path, exist_ok=True)
            os.makedirs(after_path[i], exist_ok=True)

        names = locals()
        for m in range(solution_num):
            names['data_after_' + str(m)] = []
        data_before_0 = []
        j = 0
        index_point = 0

        while change_cfg_flag == False:
            total_offset = [0.0, 0, 0]
            # total_offset = [0.016, -0.17 + 0.016, 0]
            gap_item = 0.015
            gap_block = 0.015
            random_offset = False
            real_operate = False
            obs_order = 'sim_image_obj'
            check_detection_loss = False
            obs_img_from = 'env'
            use_yolo_pos = False

            env.get_parameters(box_num=boxes_num, area_ratio=config_dict, solution_num=solution_num, kind_num=kind_num,
                               total_offset=total_offset,
                               gap_item=gap_item, gap_block=gap_block,
                               real_operate=real_operate, obs_order=obs_order,
                               random_offset=random_offset, check_detection_loss=check_detection_loss,
                               obs_img_from=obs_img_from, use_yolo_pos=use_yolo_pos,
                               item_odd_prevent=item_odd_prevent, block_odd_prevent = block_odd_prevent,
                               upper_left_max = upper_left_max, forced_rotate_box=forced_rotate_box,
                               iteration_time=iteration_time)
            pos_after_epoch, ori_after_epoch, xy_after_epoch, cls_after_epoch = env.change_config()

            if j + start_evaluations == int(save_point[-1]):
                print('over!!!!!!!!!!!!')
                quit()

            save_flag = False
            fail_time = 0
            break_flag = False
            # for m in range(solution_num):
            #
            #     pos_after = pos_after_epoch[m]
            #     ori_after = ori_after_epoch[m]
            #     xy_after = xy_after_epoch[m]
            #     order = env.change_sequence(pos_after, flag='distance')
            #     # pos_after = pos_after[order]
            #     # ori_after = ori_after[order]
            #     # xy_after = xy_after[order]
            #     # if m == 0:
            #     #     pos_before = pos_before_epoch[order]
            #     #     ori_before = ori_before_epoch[order]
            #
            #     for i in range(len(pos_after)):
            #         if pos_after[i, 0] > 0.27 or pos_after[i, 1] > 0.19 or pos_after[i, 1] < -0.19:
            #             print(f'num{boxes_num}, solution{m}, evaluation {j} out of the boundary!')
            #             break_flag = True
            #             fail_time += 1
            #     if break_flag == True:
            #         break

            if fail_time == 0:
                for m in range(solution_num):

                    pos_after = pos_after_epoch[m]
                    ori_after = ori_after_epoch[m]
                    xy_after = xy_after_epoch[m]
                    cls_after = cls_after_epoch[m]
                    order = env.change_sequence(pos_after, flag='distance')
                    pos_after = pos_after[order]
                    ori_after = ori_after[order]
                    xy_after = xy_after[order]
                    cls_after = cls_after[order]
                    # if m == 0:
                    #     pos_before = pos_before_epoch[order]
                    #     ori_before = ori_before_epoch[order]

                    names['data_after_' + str(m)].append(np.concatenate((pos_after, xy_after, ori_after.reshape(-1, 1), cls_after.reshape(-1, 1)), axis=1).reshape(-1))
                    # if m == 0:
                    #     data_before_0.append(np.concatenate((pos_before, xy_after, ori_before.reshape(-1, 1)), axis=1).reshape(-1))
                    # data_after.append(np.concatenate((pos_after, xy_after, ori_after.reshape(-1, 1)), axis=1).reshape(-1))
                    # index_flag.append(np.concatenate((boxes_index, transform_after)))

                    if len(names['data_after_' + str(m)]) == int((end_evaluations - start_evaluations) / step_num):
                        names['data_after_' + str(m)] = np.asarray(names['data_after_' + str(m)])
                        np.savetxt(after_path[m] + 'num_%s_%s.txt' % (boxes_num, int(save_point[index_point])), names['data_after_' + str(m)])
                        names['data_after_' + str(m)] = []
                        print('save data in:' + after_path[m] + 'num_%s_%s.txt' % (boxes_num, int(save_point[index_point])))
                        # if m == 0:
                        #     data_before_0 = np.asarray(data_before_0)
                        #     np.savetxt(before_path + 'num_%s_%s.txt' % (boxes_num, int(save_point[index_point])), data_before_0)
                        #     data_before_0 = []
                        #     print('save data in:' + before_path + 'num_%s_%s.txt' % (boxes_num, int(save_point[index_point])))
                        save_flag = True
            if break_flag == False:
                print(j)
                j += 1
            if save_flag == True:
                index_point += 1