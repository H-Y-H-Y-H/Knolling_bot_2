import numpy as np

from Arrange_knolling_model.Data_collection.sort_data_collection_multi_solution import Sort_objects
import pybullet_data as pd
import random
# from turdf import *
import pybullet as p
import os
import cv2
# from cam_obs_yolov8 import *
import torch

from urdfpy import URDF

from Arrange_knolling_model.Data_collection.arrange_policy import configuration_zzz

torch.manual_seed(42)
np.random.seed(100)
random.seed(100)

# color_list = [[240, 207, 137,1],
#               [240, 207, 137,1],
#               [240, 207, 137,1],
#               [240, 207, 137,1],
#               [200, 162, 200,1],
#               [200, 162, 200,1],
#               [121, 119, 255,1],
#               [144, 238, 144,1],
#               [144, 238, 144,1],
#               [144, 238, 144,1]]
color_list = [[20, 20, 20,1],
              [20, 20, 20,1],
              [20, 20, 20,1],
              [20, 20, 20,1],
              [100,100,100,1],
              [100,100,100,1],
              [100,100,100,1],
              [200, 200,200,1],
              [200, 200,200,1],
              [200, 200,200,1]]
color_list = np.array(color_list,dtype=float)
color_list[:,:3] = color_list[:,:3]/255



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

    def get_parameters(self, box_num=None, area_num=None, ratio_num=None, boxes_index=None,
                       total_offset=None, configuration=None,
                       gap_item=0.03, gap_block=0.02,
                       real_operate=False, obs_order='1',
                       random_offset=False, check_detection_loss=None, obs_img_from=None, use_yolo_pos=True,
                       item_odd_prevent=None, block_odd_prevent=None, upper_left_max = None, forced_rotate_box=None):

        # self.lego_num = lego_num
        self.total_offset = total_offset
        self.area_num = area_num
        self.ratio_num = ratio_num
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
        self.configuration = configuration
        self.item_odd_prevent = item_odd_prevent
        self.block_odd_prevent = block_odd_prevent
        self.upper_left_max = upper_left_max
        self.forced_rotate_box = forced_rotate_box

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
        labels_data = labels_data.reshape(-1, 5)
        pos_data = labels_data[:, :2]
        # pos_data[:,0] = np.random.uniform(0.02,0.25,pos_data[:,0].shape)
        # pos_data[:,1] = np.random.uniform(-0.12,0.12,pos_data[:,1].shape)


        pos_data = np.concatenate((pos_data, np.zeros(len(pos_data)).reshape(-1, 1)), axis=1)
        lw_data = labels_data[:, 2:4]
        ori_data = labels_data[:, 4]
        # ori_data = np.random.random(ori_data.shape)

        ori_data = np.concatenate((np.zeros((len(ori_data), 2)), ori_data.reshape(-1, 1)), axis=1)

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
        # p.changeVisualShape(baseid, -1, textureUniqueId=textureId,
        #                     rgbaColor=[np.random.uniform(0.9, 1), np.random.uniform(0.9, 1), np.random.uniform(0.9, 1),
        #                                1])
        p.changeVisualShape(baseid, -1, textureUniqueId=textureId,
                            rgbaColor=[0.9, 0.9,0.9,1])

        ################### recover urdf boxes based on lw_data ###################
        boxes = []
        xyz_list = []
        new_pos_data = []
        new_ori_data = []
        # for i in range(len(index_flag[0])):
        #     boxes.append(URDF.load('../urdf/box_generator/box_%d.urdf' % index_flag[0, i]))
        #     xyz_list.append(boxes[i].links[0].visuals[0].geometry.box.size)

        temp_box = URDF.load(self.urdf_path + 'box_generator/template.urdf')
        save_urdf_path_one_img = save_urdf_path + 'img_%d/' % img_index
        os.makedirs(save_urdf_path_one_img, exist_ok=True)
        for i in range(len(lw_data)):
            temp_box.links[0].collisions[0].origin[2, 3] = 0
            length = lw_data[i, 0]
            width = lw_data[i, 1]
            height = 0.012
            temp_box.links[0].visuals[0].geometry.box.size = [length, width, height]
            temp_box.links[0].collisions[0].geometry.box.size = [length, width, height]

            # Change the color of the boxes.
            # Random the color
            # temp_box.links[0].visuals[0].material.color = [np.random.random(), np.random.random(), np.random.random(), 1]
            # Use the color list:
            # temp_box.links[0].visuals[0].material.color = color_list[i]

            temp_box.save(save_urdf_path_one_img + 'box_%d.urdf' % (i))

        lego_idx = []
        selected_lw_data = []
        selected_urdf = []
        for i in range(len(lw_data)):
            print(f'this is matching urdf{j}')
            print(pos_data[i])
            print(lw_data[i])
            print(ori_data[i])
            pos_data[i, 2] += 0.006
            lego_idx.append(p.loadURDF(save_urdf_path_one_img + 'box_%d.urdf' % (i),
                           basePosition=pos_data[i],
                           baseOrientation=p.getQuaternionFromEuler(ori_data[i]), useFixedBase=False,
                           flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
        ################### recover urdf boxes based on lw_data ###################

        return self.get_obs('images', None)

    def reset(self):

        # get the standard xyz and corresponding index from files in the computer
        items_sort = Sort_objects(configuration=self.configuration)
        self.obj_idx = []
        if self.real_operate == False:
            self.xyz_list, _, _, self.all_index, self.transform_flag = items_sort.get_data_virtual(self.area_num,
                                                                                                   self.ratio_num,
                                                                                                   self.num_list,
                                                                                                   self.boxes_index)
            restrict = np.max(self.xyz_list)
            gripper_height = 0.012
            last_pos = np.array([[0, 0, 1]])

            ############## collect ori and pos to calculate the error of detection ##############
            collect_ori = []
            collect_pos = []
            ############## collect ori and pos to calculate the error of detection ##############

            for i in range(len(self.all_index)):
                for j in range(len(self.all_index[i])):
                    #         pass
                    # for i in range(len(self.grasp_order)):
                    #     for j in range(self.num_list[self.grasp_order[i]]):

                    rdm_pos = np.array([random.uniform(self.x_low_obs, self.x_high_obs),
                                        random.uniform(self.y_low_obs, self.y_high_obs), 0.0])
                    ori = [0, 0, random.uniform(0, np.pi)]
                    # ori = [0, 0, 0]
                    collect_ori.append(ori)
                    check_list = np.zeros(last_pos.shape[0])

                    while 0 in check_list:
                        rdm_pos = [random.uniform(self.x_low_obs, self.x_high_obs),
                                   random.uniform(self.y_low_obs, self.y_high_obs), 0.0]
                        for z in range(last_pos.shape[0]):
                            if np.linalg.norm(last_pos[z] - rdm_pos) < restrict + gripper_height:
                                check_list[z] = 0
                            else:
                                check_list[z] = 1
                    collect_pos.append(rdm_pos)

                    last_pos = np.append(last_pos, [rdm_pos], axis=0)

            collect_ori = np.asarray(collect_ori)
            collect_pos = np.asarray(collect_pos)
            self.check_ori = collect_ori[:, 2]
            self.check_pos = collect_pos[:, :2]
        for i in range(60):
            p.stepSimulation()
        # return self.get_obs('images', None)

        ################# change the sequence of data based on the max area of single box #####################
        box_order = np.argsort(self.xyz_list[:, 0] * self.xyz_list[:, 1])[::-1]
        self.check_pos = self.check_pos[box_order]
        self.check_ori = self.check_ori[box_order]
        self.xyz_list = self.xyz_list[box_order]
        ################# change the sequence of data based on the max area of single box #####################


        return self.check_pos, self.check_ori, self.xyz_list[:, :2], self.transform_flag

    def change_config(self):  # this is main function!!!!!!!!!

        p.resetSimulation()
        p.setGravity(0, 0, -10)

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
        p.changeDynamics(baseid, -1, lateralFriction=self.lateral_friction, frictionAnchor=True)
        # p.changeDynamics(self.arm_id, 7, lateralFriction=self.lateral_friction, frictionAnchor=True)
        # p.changeDynamics(self.arm_id, 8, lateralFriction=self.lateral_friction, frictionAnchor=True)
        p.changeVisualShape(baseid, -1, textureUniqueId=textureId,
                            rgbaColor=[np.random.uniform(0.9, 1), np.random.uniform(0.9, 1), np.random.uniform(0.9, 1),
                                       1])

        # get the standard xyz and corresponding index from files in the computer
        items_sort = Sort_objects(configuration=self.configuration)
        if self.real_operate == False:
            self.xyz_list, _, _, self.all_index, self.transform_flag = items_sort.get_data_virtual(self.area_num,
                                                                                                   self.ratio_num,
                                                                                                   self.num_list,
                                                                                                   self.boxes_index)
        calculate_reorder = configuration_zzz(self.xyz_list, self.all_index, self.gap_item, self.gap_block, self.transform_flag, self.configuration,
                                              self.item_odd_prevent, self.block_odd_prevent, self.upper_left_max, self.forced_rotate_box)

        # determine the center of the tidy configuration
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
        box_order = np.argsort(self.xyz_list[:, 0] * self.xyz_list[:, 1])[::-1]
        self.items_pos_list = self.items_pos_list[box_order]
        self.items_ori_list = self.items_ori_list[box_order]
        self.xyz_list = self.xyz_list[box_order]
        ################# change the sequence of data based on the max area of single box #####################

        return self.items_pos_list[:, :2], self.items_ori_list[:, 2], self.xyz_list[:, :2], self.transform_flag


if __name__ == '__main__':

    command = 'recover'
    before_after = 'before'
    configuration = '1'

    start_evaluations = 0
    end_evaluations =   1000000
    step_num = 10
    save_point = np.linspace(int((end_evaluations - start_evaluations) / step_num + start_evaluations), end_evaluations, step_num)


    area_num = 2
    ratio_num = 1

    item_odd_prevent = True

    # always true
    block_odd_prevent = True
    # always true

    upper_left_max = True

    forced_rotate_box = False

    # DATAROOT = "C:/Users/yuhan/Downloads/learning_data_804_20w/"
    DATAROOT = "../../../knolling_dataset/learning_data_1019_42w/"

    target_path = DATAROOT + 'cfg_%s/' % configuration
    images_log_path = target_path + 'images_%s/' % before_after
    preprocess_label_path = target_path + 'preprocess_label_%s/' % before_after
    os.makedirs(images_log_path, exist_ok=True)
    os.makedirs(preprocess_label_path, exist_ok=True)

    # name = "devoted-terraqin-29"
    # name = "classic-bush-95"
    name = 'dandy-hill-142'
    name = 'devoted-terrain-29'
    # name = 'fallen-morning-156'
    # name = "radiant-puddle-143"

    show_baseline = 0
    show_results_flag = True

    if command == 'recover':

        env = Arm(is_render=True)
        visual_path = '../train_and_test/results/%s/pred_after'%(name)


        box_num = 10
        if show_baseline == 0:
            data = np.loadtxt(visual_path + '/num_%d_new.txt' % 30)
            savefolder = '../train_and_test/results/%s/pred_after/image/' % (name)
        elif show_baseline == 1:
            data = np.loadtxt('../train_and_test/baseline/mlp_result/outputs.csv')

        test_start = int(len(data)*0.8)
        # test_end = int(len(data)*0.81)
        data = data[:test_start]
        savefolder = '../train_and_test/results/%s/pred_%s/' % (name, before_after)

        os.makedirs(savefolder, exist_ok=True)

        if len(data.shape) == 1:
            data = data.reshape(1, len(data))

        data = data[:,:box_num*6]
        print('this is len data', len(data))
        save_urdf_path = DATAROOT + '/box_urdf/num_%d/' % (box_num)
        os.makedirs(save_urdf_path, exist_ok=True)

        new_data = []
        # new_index_flag = []
        # for j in range(start_evaluations, end_evaluations):
        # for j in [ 81434, 100777,  88176, 148385,  9905,  23617,  95448, 103549, 113927,  17746]:
        # for j in [88176,  9905,  81434]:
        for j in range(len(data)):
            env.get_parameters(box_num=box_num)
            print(f'this is data {j}')
            one_img_data = data[j].reshape(-1, 5)
            # one_img_index_flag = index_flag[j].reshape(2, -1)
            box_order = np.lexsort((one_img_data[:, 1], one_img_data[:, 0]))
            one_img_data = one_img_data[box_order].reshape(-1,)
            # one_img_index_flag = one_img_index_flag[:, box_order].reshape(-1, )
            new_data.append(one_img_data)
            # new_index_flag.append(one_img_index_flag)

            image = env.label2image(data[j], j, save_urdf_path)
            image = image[..., :3]

            cv2.imwrite(savefolder+'%d.png'%j,image)
            cv2.namedWindow('zzz', 0)
            cv2.resizeWindow('zzz', 1280, 960)
            cv2.imshow("zzz", image)
            print('This is the data: \n', data[j])

            cv2.waitKey()
            cv2.destroyAllWindows()

            # cv2.imwrite(images_log_path + '%d_%d.png' % (i, j), image)

        # new_data = np.asarray(new_data)
        # # new_index_flag = np.asarray(new_index_flag)
        # np.savetxt(preprocess_label_path + 'num_%d.txt' % i, new_data)
        # # np.savetxt(target_path + 'index_flag/num_%s_flag.txt' % i, new_index_flag)
