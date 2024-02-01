import csv

import numpy as np
import pybullet_data as pd
import random
import pybullet as p
import os
import cv2
import torch
from tqdm import tqdm
# from urdfpy import URDF
import shutil
import json
import csv
import pandas

from arrange_policy import configuration_zzz, arrangement

torch.manual_seed(42)

class Arm:

    def __init__(self, is_render, arrange_policy):

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

        self.arrange_policy = arrange_policy

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

    def get_data_virtual(self):

        # shuffle the class_num and color_num for each scenario
        self.arrange_policy['class_num'] = np.random.randint(2, 11)
        self.arrange_policy['color_num'] = np.random.randint(2, 6)

        if self.arrange_policy['object_type'] == 'box':
            length_data = np.round(np.random.uniform(low=self.arrange_policy['length_range'][0],
                                                      high=self.arrange_policy['length_range'][1],
                                                      size=(self.arrange_policy['object_num'], 1)), decimals=3)
            width_data = np.round(np.random.uniform(low=self.arrange_policy['width_range'][0],
                                                      high=self.arrange_policy['width_range'][1],
                                                      size=(self.arrange_policy['object_num'], 1)), decimals=3)
            height_data = np.round(np.random.uniform(low=self.arrange_policy['height_range'][0],
                                                      high=self.arrange_policy['height_range'][1],
                                                      size=(self.arrange_policy['object_num'], 1)), decimals=3)
            class_data = np.random.randint(low=0,
                                           high=self.arrange_policy['class_num'],
                                           size=(self.arrange_policy['object_num'], 1))
            color_index_data = np.random.randint(low=0,
                                           high=self.arrange_policy['color_num'],
                                           size=(self.arrange_policy['object_num'], 1))
            data = np.concatenate((length_data, width_data, height_data, class_data, color_index_data), axis=1).round(decimals=3)

        elif self.arrange_policy['object_type'] == 'sundry':
            class_index = np.random.choice(a=self.arrange_policy['max_class_num'],
                                          size=self.arrange_policy['class_num'],
                                          replace=False)
            class_index_data = np.random.choice(a=class_index,
                                                size=self.arrange_policy['object_num'])
            # class_index_data = np.random.choice(a=self.arrange_policy['class_num'],
            #                                     size=self.arrange_policy['object_num'],
            #                                     replace=False)
            class_name_list = os.listdir(self.urdf_path + 'OpensCAD_generate/generated_stl/')
            class_name_list.sort()
            class_name = [class_name_list[n] for n in class_index_data]
            object_name_list = []
            object_lwh_list = []
            for i in range(self.arrange_policy['object_num']):
                num_type_each_object = len(os.listdir(self.urdf_path + 'OpensCAD_generate/generated_stl/' + class_name[i] + '/'))
                temp_path = (self.urdf_path + 'OpensCAD_generate/generated_stl/' + class_name[i] + '/'
                             + class_name[i] + '_' + str(np.random.randint(num_type_each_object) + 1) + '/')
                object_name = np.random.choice(os.listdir(temp_path))
                object_name_list.append(object_name)

                object_csv_path = temp_path + object_name + '/' + object_name + '.csv'
                object_lwh_list.append(eval(pandas.read_csv(object_csv_path).loc[0, 'BoundingBoxDimensions (cm)']))

            color_index = np.random.choice(a=self.arrange_policy['max_color_num'],
                                           size=self.arrange_policy['color_num'],
                                           replace=False)
            color_index_data = np.random.choice(a=color_index, size=self.arrange_policy['object_num'])
            # color_index_data = np.random.randint(low=0,
            #                                high=self.arrange_policy['color_num'],
            #                                size=(self.arrange_policy['object_num'], 1))

            object_lwh_list = np.around(np.asarray(object_lwh_list) * 0.001, decimals=4)
            data = np.concatenate((object_lwh_list,
                                   class_index_data.reshape(self.arrange_policy['object_num'], 1),
                                   color_index_data.reshape(self.arrange_policy['object_num'], 1)), axis=1)
            object_name_list = np.asarray(object_name_list)

        return data, object_name_list

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
            temp = np.copy(image[:, :, 0])
            image[:, :, 0] = image[:, :, 2]
            image[:, :, 2] = temp
            return image

    def label2image(self, labels_data, img_index, save_urdf_path, labels_name=None):
        # print(index_flag)
        # index_flag = index_flag.reshape(2, -1)

        total_offset = [0.016, -0.17 + 0.016, 0]

        labels_data = labels_data.reshape(-1, 8)
        pos_data = labels_data[:, :3]
        pos_data[:, 0] += total_offset[0]
        pos_data[:, 1] += total_offset[1]
        lw_data = labels_data[:, 3:6]
        # ori_data = labels_data[:, 3:6]
        ori_data = np.zeros((len(lw_data), 3))
        color_index = labels_data[:, -1]
        color_dict = {'0': [0, 0, 0, 1], '1': [255, 255, 255, 1], '2': [255, 0, 0, 1], '3': [0, 255, 0, 1], '4': [0, 0, 255, 1]}
        class_index = labels_data[:, -2]

        # Converting dictionary keys to integers
        dict_map = {int(k): v for k, v in color_dict.items()}

        # Mapping array values to dictionary values
        mapped_color_values = [dict_map[value] for value in color_index]

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
        # self.arm_id = p.loadURDF(self.urdf_path + "robot_arm928/robot_arm_fixed.urdf",
        #                          basePosition=[-0.08, 0, 0.02], useFixedBase=True,
        #                          flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        textureId = p.loadTexture(self.urdf_path + "floor_1.png")
        p.changeDynamics(baseid, -1, lateralFriction=1, spinningFriction=1, rollingFriction=0.002, linearDamping=0.5,
                         angularDamping=0.5)
        # p.changeDynamics(self.arm_id, 7, lateralFriction=1, spinningFriction=1, rollingFriction=0, linearDamping=0, angularDamping=0)
        # p.changeDynamics(self.arm_id, 8, lateralFriction=1, spinningFriction=1, rollingFriction=0, linearDamping=0, angularDamping=0)
        p.changeVisualShape(baseid, -1, textureUniqueId=textureId,
                            rgbaColor=[np.random.uniform(0.9, 1), np.random.uniform(0.9, 1), np.random.uniform(0.9, 1), 1])

        ################### recover urdf boxes based on lw_data ###################
        if self.arrange_policy['object_type'] == 'box':
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
                temp_box.links[0].visuals[0].material.color = mapped_color_values[i]
                temp_box.save(save_urdf_path_one_img + 'box_%d.urdf' % (i))

            object_idx = []
            print('position\n', pos_data)
            print('orietation\n', ori_data)
            print('lwh\n', lw_data)
            for i in range(len(lw_data)):
                print(f'this is matching urdf{i}')
                pos_data[i, 2] += 0.006
                object_idx.append(p.loadURDF(save_urdf_path_one_img + 'box_%d.urdf' % (i),
                               basePosition=pos_data[i],
                               baseOrientation=p.getQuaternionFromEuler(ori_data[i]), useFixedBase=False,
                               flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
        elif self.arrange_policy['object_type'] == 'sundry':

            urdf_path_one_img = self.urdf_path + 'OpensCAD_generate/urdf_file/'

            object_idx = []
            # ori_data[:, 0] += np.pi / 2
            print('position\n', pos_data)
            print('lwh\n', lw_data)
            for i in range(len(lw_data)):
                print(f'this is matching urdf{i}')
                pos_data[i, 2] += 0.026
                if lw_data[i, 0] < lw_data[i, 1]:
                    ori_data[i, 2] += np.pi / 2
                # labels_name[i] = 'utilityknife_1_L0.65_T0.65'
                object_idx.append(p.loadURDF(urdf_path_one_img + labels_name[i] + '.urdf',
                                             basePosition=pos_data[i],
                                             baseOrientation=p.getQuaternionFromEuler(ori_data[i]), useFixedBase=False,
                                             flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
                p.changeVisualShape(object_idx[i], -1, rgbaColor=mapped_color_values[i])
        ################### recover urdf boxes based on lw_data ###################

        # shutil.rmtree(save_urdf_path_one_img)
        for i in range(100):
            p.stepSimulation()

        return self.get_obs('images', None)

    def change_config(self):  # this is main function!!!!!!!!!

        # get the standard xyz and corresponding index from files in the computer
        arranger = arrangement(self.arrange_policy)
        data_before, object_name_list = self.get_data_virtual()
        data_after_total = []
        object_name_after_total = []

        # type, color, area, ratio


        for i in range(len(policy_switch)):

            arranger.arrange_policy['type_classify_flag'] = policy_switch[i][0]
            arranger.arrange_policy['color_classify_flag'] = policy_switch[i][1]
            arranger.arrange_policy['area_classify_flag'] = policy_switch[i][2]
            arranger.arrange_policy['ratio_classify_flag'] = policy_switch[i][3]

            times = 0
            while True: # generate several results for each configuration, now is 4
                data_after, data_name_after = arranger.generate_arrangement(data=data_before, data_name=object_name_list)
                # the sequence of self.items_pos_list, self.items_ori_list are the same as those in self.xyz_list
                data_after[:, :3] = data_after[:, :3] + self.arrange_policy['total_offset']
                order = self.change_sequence(data_after[:, :2], flag='distance')
                data_after = data_after[order]
                data_after = np.delete(data_after, [2, 3, 4, 5], axis=1)
                data_after_total.append(data_after)

                # new_object_name_list = [object_name_list[order[m]] for m in range(len(order))]
                new_data_name_after = data_name_after[order]
                object_name_after_total.append(new_data_name_after)

                times += 1
                if times >= self.arrange_policy['output_per_cfg']:
                    break

        return data_after_total, object_name_after_total


if __name__ == '__main__':

    command = 'knolling'
    before_after = 'after'

    # np.random.seed(100)

    start_evaluations = 0
    end_evaluations =   100
    step_num = 10
    save_point = np.linspace(int((end_evaluations - start_evaluations) / step_num + start_evaluations), end_evaluations, step_num)

    target_path = '../../../knolling_dataset/learning_data_0131/'
    images_log_path = target_path + 'images_%s/' % before_after
    os.makedirs(images_log_path, exist_ok=True)

    arrange_policy = {
                    'length_range': [0.036, 0.06], 'width_range': [0.016, 0.036], 'height_range': [0.01, 0.02], # objects 3d range
                    'object_num': 10, 'output_per_cfg': 3, 'object_type': 'sundry', # sundry or box
                    'iteration_time': 10,
                    'area_num': None, 'ratio_num': None, 'area_classify_flag': None, 'ratio_classify_flag': None,
                    'class_num': None, 'color_num': None, 'max_class_num': 10, 'max_color_num': 5,
                    'type_classify_flag': None, 'color_classify_flag': None, # classification range
                    'arrangement_policy': 'Type*3, Color*3, Area*3, Ratio*3', # customized setting
                    'object_even': True, 'block_even': True, 'upper_left_max': False, 'forced_rotate_box': False,
                    'total_offset': [0, 0, 0], 'gap_item': 0.016, 'gap_block': 0.016 # inverval and offset of the arrangement

                    }
    policy_switch = [[True, False, False, False],
                     [False, True, False, False],
                     [False, False, True, False],
                     [False, False, False, True]]
    solution_num = int(arrange_policy['output_per_cfg'] * len(policy_switch))

    if command == 'recover':

        env = Arm(is_render=True, arrange_policy=arrange_policy)

        # data = np.loadtxt(target_path + 'labels_%s/num_%d.txt' % (before_after, i))

        names = locals()
        # data_before = []
        save_urdf_path = []
        for m in range(solution_num):
            names['data_' + str(m)] = np.loadtxt(target_path + 'num_%d_after_%d.txt' % (arrange_policy['object_num'], m))
            if arrange_policy['object_type'] == 'sundry':
                names['name_' + str(m)] = np.loadtxt(target_path + 'num_%d_after_name_%d.txt' % (arrange_policy['object_num'], m), dtype=str)

            if len(names['data_' + str(m)].shape) == 1:
                names['data_' + str(m)] = names['data_' + str(m)].reshape(1, len(names['data_' + str(m)]))

            box_num = arrange_policy['object_num']
            print('this is len data', len(names['data_' + str(m)]))
            save_urdf_path.append(target_path + 'box_urdf/num_%s_%d/' % (m, box_num))
            os.makedirs(save_urdf_path[m], exist_ok=True)

        # new_data = []
        # new_index_flag = []
        for j in range(start_evaluations, end_evaluations):
            # env.get_parameters(box_num=boxes_num)
            for m in range(solution_num):
                print(f'this is data {j}')
                one_img_data = names['data_' + str(m)][j].reshape(-1, 8)

                image = env.label2image(names['data_' + str(m)][j], j, save_urdf_path[m], labels_name=names['name_' + str(m)][j])
                image = image[..., :3]

                cv2.namedWindow('zzz', 0)
                cv2.resizeWindow('zzz', 1280, 960)
                cv2.imshow("zzz", image)
                cv2.waitKey()
                cv2.destroyAllWindows()

            # cv2.imwrite(images_log_path + '%d_%d.png' % (i, j), image)

    if command == 'knolling':

        # save the parameters of data collection
        with open(target_path[:-1] + "_readme.json", "w") as f:
            json.dump(arrange_policy, f, indent=4)

        env = Arm(is_render=False, arrange_policy=arrange_policy)

        change_cfg_flag = False

        after_path = []
        for i in range(solution_num):
            after_path.append(target_path + 'labels_after_%s/' % i)
            os.makedirs(after_path[i], exist_ok=True)

        names = locals()
        for m in range(solution_num):
            names['data_after_' + str(m)] = []
            names['name_after_' + str(m)] = []
        j = 0
        index_point = 0

        while change_cfg_flag == False:
            data_after_total, object_name_after_total = env.change_config()

            if j + start_evaluations == int(save_point[-1]):
                print('over!!!!!!!!!!!!')
                quit()

            save_flag = False
            break_flag = False

            for m in range(solution_num):

                names['data_after_' + str(m)].append(data_after_total[m].reshape(-1))
                names['name_after_' + str(m)].append(object_name_after_total[m])

                if len(names['data_after_' + str(m)]) == int((end_evaluations - start_evaluations) / step_num):
                    names['data_after_' + str(m)] = np.asarray(names['data_after_' + str(m)])
                    names['name_after_' + str(m)] = np.asarray(names['name_after_' + str(m)], dtype=str)
                    np.savetxt(after_path[m] + 'num_%s_%s.txt' % (arrange_policy['object_num'], int(save_point[index_point])), names['data_after_' + str(m)])
                    np.savetxt(after_path[m] + 'num_%s_%s_name.txt' % (arrange_policy['object_num'], int(save_point[index_point])), names['name_after_' + str(m)], fmt='%s')

                    # with open(after_path[m] + 'num_%s_%s_name.txt' % (arrange_policy['object_num'], int(save_point[index_point])), 'w') as f:
                    #     for line in names['name_after_' + str(m)]:
                    #
                    #         f.write(str(line) + '\n')

                    names['data_after_' + str(m)] = []
                    names['name_after_' + str(m)] = []

                    print('save data in:' + after_path[m] + 'num_%s_%s.txt' % (arrange_policy['object_num'], int(save_point[index_point])))
                    save_flag = True
            if break_flag == False:
                print(j)
                j += 1
            if save_flag == True:
                index_point += 1