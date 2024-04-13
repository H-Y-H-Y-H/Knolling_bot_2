# from arrangement import *
from ASSET.arrange_model_deploy import *
from ASSET.visual_perception import *
# from ASSET.yolo_grasp_deploy import *
from utils import *
import pybullet as p
import pybullet_data as pd
import os
import numpy as np
import random
import cv2
import pandas


class Sundry_env():

    def __init__(self, para_dict, knolling_para=None, lstm_dict=None, arrange_dict=None):

        self.para_dict = para_dict
        self.knolling_para = knolling_para

        self.kImageSize = {'width': 480, 'height': 480}
        self.init_pos_range = para_dict['init_pos_range']
        self.init_ori_range = para_dict['init_ori_range']
        self.init_offset_range = para_dict['init_offset_range']
        self.urdf_path = para_dict['urdf_path']
        self.object_urdf_path = para_dict['object_urdf_path']
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
        self.table_center = np.array([(self.x_low_obs + self.x_high_obs) / 2, (self.y_low_obs + self.y_high_obs) / 2])

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

        self.grid_size = 5
        self.x_range = (-0.001, 0.001)
        self.y_range = (-0.001, 0.001)
        x_center = 0
        y_center = 0
        x_offset_values = np.linspace(self.x_range[0], self.x_range[1], self.grid_size)
        y_offset_values = np.linspace(self.y_range[0], self.y_range[1], self.grid_size)
        xx, yy = np.meshgrid(x_offset_values, y_offset_values)
        sigma = 0.01
        kernel = np.exp(-((xx - x_center) ** 2 + (yy - y_center) ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
        self.kernel = kernel / np.sum(kernel)

        self.main_demo_epoch = 0

        self.create_entry_num = 0

    def create_scene(self):

        if random.uniform(0, 1) > 0.5:
            p.configureDebugVisualizer(lightPosition=[np.random.randint(1, 2), np.random.uniform(0, 1.5), 2],
                                       shadowMapResolution=8192, shadowMapIntensity=np.random.randint(0, 1) / 10)
        else:
            p.configureDebugVisualizer(lightPosition=[np.random.randint(1, 2), np.random.uniform(-1.5, 0), 2],
                                       shadowMapResolution=8192, shadowMapIntensity=np.random.randint(0, 1) / 10)
        self.baseid = p.loadURDF(self.urdf_path + "plane.urdf", useMaximalCoordinates=True)

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
        textureId = p.loadTexture(self.urdf_path + f"floor_{background}.png")
        p.changeVisualShape(self.baseid, -1, textureUniqueId=textureId, specularColor=[0, 0, 0])

        p.setGravity(0, 0, -10)

    def create_objects(self, pos_data=None, ori_data=None, lwh_data=None):
        # Randomly select 5-10 URDF files from a list of 300
        all_urdf_files = os.listdir(self.object_urdf_path)
        selected_urdf_files = random.sample(all_urdf_files, random.randint(5, 10))

        # Update self.objects_num to match the number of selected URDF files
        self.objects_num = len(selected_urdf_files)
        self.objects_index = []

        # Function to generate random color variations
        def get_random_color_variation(base_color):
            variation_intensity = 0.7  # Adjust this for more or less color variation
            color_variation = np.random.uniform(1 - variation_intensity, 1 + variation_intensity, size=3)
            if np.array_equal(base_color, np.array([0.7, 0.7, 0.7])):
                color_variation = 1.15
            return np.clip(base_color * color_variation, 0, 1)

        # Define base colors
        base_colors = {
            'red': np.array([1, 0, 0]),
            'black': np.array([0, 0, 0]),
            'blue': np.array([0, 0, 1]),
            'green': np.array([0, 1, 0]),
            'grey': np.array([0.7, 0.7, 0.7])
        }

        # Function to check if a new position is too close to existing objects
        def is_too_close(new_pos, existing_objects, min_distance=0.08):
            for _, obj_pos in existing_objects:
                if np.linalg.norm(np.array(new_pos) - np.array(obj_pos)) < min_distance:
                    return True
            return False

        # Generate random orientations for the objects
        # rdm_ori_roll = np.pi / 2 * np.ones((self.objects_num, 1))  # Fixed roll value
        rdm_ori_roll = np.random.uniform(self.init_ori_range[0][0], self.init_ori_range[0][1],
                                          size=(self.objects_num, 1))
        rdm_ori_pitch = np.random.uniform(self.init_ori_range[1][0], self.init_ori_range[1][1],
                                          size=(self.objects_num, 1))
        rdm_ori_yaw = np.random.uniform(self.init_ori_range[2][0], self.init_ori_range[2][1],
                                        size=(self.objects_num, 1))
        rdm_ori = np.hstack([rdm_ori_roll, rdm_ori_pitch, rdm_ori_yaw])

        x_offset = np.random.uniform(self.init_offset_range[0][0], self.init_offset_range[0][1])
        y_offset = np.random.uniform(self.init_offset_range[1][0], self.init_offset_range[1][1])

        # Place the objects, ensuring they don't overlap
        self.obj_gt_lwh = []
        self.obj_info_total = []
        for i in range(self.objects_num):
            placement_successful = False
            while not placement_successful:
                # Generate a new position for the object
                rdm_pos_x = np.random.uniform(self.init_pos_range[0][0], self.init_pos_range[0][1])
                rdm_pos_y = np.random.uniform(self.init_pos_range[1][0], self.init_pos_range[1][1])
                rdm_pos_z = np.random.uniform(self.init_pos_range[2][0], self.init_pos_range[2][1])
                new_pos = [rdm_pos_x + x_offset, rdm_pos_y + y_offset, rdm_pos_z]

                # Check if the new position is too close to any existing objects
                if not is_too_close(new_pos, self.objects_index):
                    urdf_file = self.object_urdf_path + selected_urdf_files[i % len(selected_urdf_files)]
                    # urdf_file = self.object_urdf_path + 'utilityknife_1_L1.02_T0.98.urdf'
                    # urdf_file = self.object_urdf_path + 'chisel_2_L0.67_T0.66.urdf'
                    object_info = selected_urdf_files[i][:-5]
                    self.obj_info_total.append(object_info)
                    object_name = object_info.split('_')[0]
                    object_index = object_info.split('_')[1]
                    csv_path = (self.urdf_path + 'OpensCAD_generate/generated_stl/' + object_name + '/' +
                                   object_name + '_' + object_index + '/' + object_info + '/' + object_info + '.csv')
                    csv_lwh = np.asarray(eval(pandas.read_csv(csv_path).loc[0, 'BoundingBoxDimensions (cm)'])) * 0.001
                    new_pos[2] = csv_lwh[2] / 2
                    self.obj_gt_lwh.append(csv_lwh)
                    obj_id = p.loadURDF(urdf_file,
                                        basePosition=new_pos,
                                        baseOrientation=p.getQuaternionFromEuler(rdm_ori[i]),
                                        flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

                    # Select a random base color and apply variation
                    chosen_color = random.choice(list(base_colors.values()))
                    color_variation = get_random_color_variation(chosen_color)

                    p.changeVisualShape(obj_id, -1, rgbaColor=[*color_variation, 1])
                    self.objects_index.append((obj_id, new_pos))
                    placement_successful = True
        self.obj_gt_lwh = np.asarray(self.obj_gt_lwh)

        # Rest of the existing code for object setup
        for _ in range(int(100)):
            p.stepSimulation()
            if self.is_render == True:
                pass
                # time.sleep(1/96)

        for i in range(self.objects_num):
            p.changeDynamics(self.objects_index[i][0], -1, lateralFriction=self.para_dict['obj_lateral_friction'],
                             contactDamping=self.para_dict['obj_contact_damping'],
                             contactStiffness=self.para_dict['obj_contact_stiffness'])

        p.changeDynamics(self.baseid, -1, lateralFriction=self.para_dict['base_lateral_friction'],
                         contactDamping=self.para_dict['base_contact_damping'],
                         contactStiffness=self.para_dict['base_contact_stiffness'])

    def get_gt_pos_ori(self):

        obj_pos, obj_ori = [], []
        for i in range(len(self.objects_index)):
            obj_pos.append(np.asarray(p.getBasePositionAndOrientation(self.objects_index[i][0])[0]))
            obj_ori.append(np.asarray(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.objects_index[i][0])[1])))
        obj_pos = np.asarray(obj_pos)
        obj_ori = np.asarray(obj_ori)

        return obj_pos, obj_ori

    def get_obs(self, epoch=None, look_flag=False, baseline_flag=False, sub_index=0, img_path=None):

        if epoch is None:
            epoch = self.main_demo_epoch

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
                manipulator_before, new_lwh_list, pred_cls = self.visual_perception_model.model_predict(img=img, epoch=epoch, gt_boxes_num=len(self.objects_index), first_flag=baseline_flag)
                self.main_demo_epoch += 1
                ################### the results of object detection has changed the order!!!! ####################
            else:
                ################### the results of object detection has changed the order!!!! ####################
                # structure of results: x, y, z, length, width, ori
                manipulator_before, new_lwh_list, pred_cls = self.visual_perception_model.model_predict(real_flag=True, first_flag=baseline_flag, epoch=epoch, gt_boxes_num=self.para_dict['boxes_num'])
                self.main_demo_epoch += 1
                ################### the results of object detection has changed the order!!!! ####################

            return manipulator_before, new_lwh_list, pred_cls, None


if __name__ == '__main__':

    # np.random.seed(183)
    # random.seed(183)

    para_dict = {'start_num': 00, 'end_num': 10000, 'thread': 0,
                 'yolo_conf': 0.3, 'yolo_iou': 0.8, 'device': 'cpu',
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

    env = knolling_env(para_dict=para_dict)

    exist_img_num = startnum
    while True:
        num_item = int(np.random.uniform(min_box_num, max_box_num + 1))
        env.reset(epoch=exist_img_num)
        img_per_epoch = env.try_grasp(img_index_start=exist_img_num)
        exist_img_num += img_per_epoch