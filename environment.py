from yolo_model_deploy import *
from function import *
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
from sklearn.preprocessing import MinMaxScaler

class Arm_env():

    def __init__(self, para_dict):

        self.kImageSize = {'width': 480, 'height': 480}
        self.endnum = para_dict['end_num']
        self.init_pos_range = para_dict['init_pos_range']
        self.init_ori_range = para_dict['init_ori_range']
        self.urdf_path = para_dict['urdf_path']
        self.pybullet_path = pd.getDataPath()
        self.is_render = para_dict['is_render']
        self.save_img_flag = para_dict['save_img_flag']
        self.yolo_model = Yolo_predict(save_img_flag=self.save_img_flag, para_dict=para_dict)
        self.para_dict = para_dict

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
        self.reset_pos = np.array([0, 0, 0.12])
        self.reset_ori = np.array([0, np.pi / 2, 0])

        self.slep_t = 1 / 120
        self.joints_index = [0, 1, 2, 3, 4, 7, 8]
        # 5 6 9不用管，固定的！
        self.init_joint_positions = [0, 0, -1.57, 0, 0, 0, 0, 0, 0, 0]

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
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.150, 0, 0], #0.175
            distance=0.4,
            yaw=90,
            pitch = -90,
            roll=0,
            upAxisIndex=2)
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_parameters['fov'],
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

    def reset(self, num_item=None, thread=None, epoch=None, data_root=None):

        p.resetSimulation()
        self.num_item = num_item

        if random.uniform(0, 1) > 0.5:
            p.configureDebugVisualizer(lightPosition=[np.random.randint(1, 2), np.random.uniform(0, 1.5), 2],
                                       shadowMapResolution=8192, shadowMapIntensity=np.random.randint(0, 1) / 10)
        else:
            p.configureDebugVisualizer(lightPosition=[np.random.randint(1, 2), np.random.uniform(-1.5, 0), 2],
                                       shadowMapResolution=8192, shadowMapIntensity=np.random.randint(0, 1) / 10)
        baseid = p.loadURDF(self.urdf_path + "plane_zzz.urdf", useMaximalCoordinates=True)

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

        # Texture change
        background = np.random.randint(1, 5)
        textureId = p.loadTexture(self.urdf_path + f"img_{background}.png")
        p.changeVisualShape(baseid, -1, textureUniqueId=textureId, specularColor=[0, 0, 0])

        self.arm_id = p.loadURDF(os.path.join(self.urdf_path, "robot_arm928/robot_arm1_backup.urdf"),
                                 basePosition=[-0.08, 0, 0.02], useFixedBase=True,
                                 flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        p.changeDynamics(self.arm_id, 7, linearDamping=self.para_dict['gripper_linear_damping'],
                                         lateralFriction=self.para_dict['gripper_lateral_friction'],
                                         contactDamping=self.para_dict['gripper_contact_damping'],
                                         contactStiffness=self.para_dict['gripper_contact_stiffness'])

        p.changeDynamics(self.arm_id, 8, linearDamping=self.para_dict['gripper_linear_damping'],
                                         lateralFriction=self.para_dict['gripper_lateral_friction'],
                                         contactDamping=self.para_dict['gripper_contact_damping'],
                                         contactStiffness=self.para_dict['gripper_contact_stiffness'])


        ik_angles0 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=self.reset_pos,
                                                  maxNumIterations=200,
                                                  targetOrientation=p.getQuaternionFromEuler(self.reset_ori))
        for motor_index in range(5):
            p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                    targetPosition=ik_angles0[motor_index], maxVelocity=20)
        for _ in range(int(30)):
            # time.sleep(1/480)
            p.stepSimulation()

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

        rdm_ori_roll  = np.random.uniform(self.init_ori_range[0][0], self.init_ori_range[0][1], size=(self.num_item, 1))
        rdm_ori_pitch = np.random.uniform(self.init_ori_range[1][0], self.init_ori_range[1][1], size=(self.num_item, 1))
        rdm_ori_yaw   = np.random.uniform(self.init_ori_range[2][0], self.init_ori_range[2][1], size=(self.num_item, 1))
        rdm_ori = np.concatenate((rdm_ori_roll, rdm_ori_pitch, rdm_ori_yaw), axis=1)
        rdm_pos_x = np.random.uniform(self.init_pos_range[0][0], self.init_pos_range[0][1], size=(self.num_item, 1))
        rdm_pos_y = np.random.uniform(self.init_pos_range[1][0], self.init_pos_range[1][1], size=(self.num_item, 1))
        rdm_pos_z = np.random.uniform(self.init_pos_range[2][0], self.init_pos_range[2][1], size=(self.num_item, 1))
        rdm_pos = np.concatenate((rdm_pos_x, rdm_pos_y, rdm_pos_z), axis=1)


        self.gt_lwh_list = []
        self.obj_idx = []
        box_path = data_root + "box_urdf/thread_%d/epoch_%d/" % (thread, epoch)
        os.makedirs(box_path, exist_ok=True)
        temp_box = URDF.load(self.urdf_path + 'box_generator/template.urdf')

        length_range = np.round(np.random.uniform(0.016, 0.048, size=(self.num_item, 1)), decimals=3)
        width_range = np.round(np.random.uniform(0.016, np.minimum(length_range, 0.036), size=(self.num_item, 1)),decimals=3)
        height_range = np.round(np.random.uniform(0.010, 0.020, size=(self.num_item, 1)), decimals=3)

        for i in range(self.num_item):
            temp_box.links[0].inertial.mass = self.para_dict['box_mass']
            temp_box.links[0].collisions[0].origin[2, 3] = 0
            self.gt_lwh_list.append(np.concatenate((length_range[i], width_range[i], height_range[i])))
            temp_box.links[0].visuals[0].geometry.box.size = np.concatenate((length_range[i], width_range[i], height_range[i]))
            temp_box.links[0].collisions[0].geometry.box.size = np.concatenate((length_range[i], width_range[i], height_range[i]))
            temp_box.links[0].visuals[0].material.color = [np.random.random(), np.random.random(), np.random.random(), 1]
            temp_box.save(box_path + 'box_%d.urdf' % (i))
            self.obj_idx.append(p.loadURDF((box_path + "box_%d.urdf" % i), basePosition=rdm_pos[i],
                                           baseOrientation=p.getQuaternionFromEuler(rdm_ori[i]), useFixedBase=0,
                                           flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
            r = np.random.uniform(0, 0.9)
            g = np.random.uniform(0, 0.9)
            b = np.random.uniform(0, 0.9)
            if random.random() < 0.05:
                p.changeVisualShape(self.obj_idx[i], -1, rgbaColor=(0.1, 0.1, 0.1, 1))
            else:
                p.changeVisualShape(self.obj_idx[i], -1, rgbaColor=(r, g, b, 1))

        self.gt_lwh_list = np.asarray(self.gt_lwh_list)
        for _ in range(int(100)):
            p.stepSimulation()
            if self.is_render == True:
                time.sleep(1/48)
        p.changeDynamics(baseid, -1, lateralFriction=self.para_dict['base_lateral_friction'],
                                     spinningFriction=self.para_dict['base_spinning_friction'],
                                     contactDamping=self.para_dict['base_contact_damping'],
                                     contactStiffness=self.para_dict['base_contact_stiffness'],
                                     restitution=self.para_dict['base_restitution'])

        forbid_range = np.array([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        while True:
            new_num_item = len(self.obj_idx)
            delete_index = []
            for i in range(len(self.obj_idx)):
                p.changeDynamics(self.obj_idx[i], -1, lateralFriction=self.para_dict['box_lateral_friction'],
                                                     spinningFriction=self.para_dict['box_spinning_friction'],
                                                     contactDamping=self.para_dict['box_contact_damping'],
                                                     contactStiffness=self.para_dict['box_contact_stiffness'],
                                                     restitution=self.para_dict['box_restitution'])

                cur_ori = np.asarray(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.obj_idx[i])[1]))
                cur_pos = np.asarray(p.getBasePositionAndOrientation(self.obj_idx[i])[0])
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
                    delete_index.append(i)
                    # print('delete!!!')
                    new_num_item -= 1
            delete_index.reverse()
            for i in delete_index:
                p.removeBody(self.obj_idx[i])
                self.obj_idx.pop(i)
                self.gt_lwh_list = np.delete(self.gt_lwh_list, i, axis=0)
            for _ in range(int(100)):
                # time.sleep(1/96)
                p.stepSimulation()

            check_delete_index = []
            for i in range(len(self.obj_idx)):
                cur_ori = np.asarray(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.obj_idx[i])[1]))
                cur_pos = np.asarray(p.getBasePositionAndOrientation(self.obj_idx[i])[0])
                roll_flag = False
                pitch_flag = False
                for j in range(len(forbid_range)):
                    if np.abs(cur_ori[0] - forbid_range[j]) < 0.01:
                        roll_flag = True
                    if np.abs(cur_ori[1] - forbid_range[j]) < 0.01:
                        pitch_flag = True
                if roll_flag == True and pitch_flag == True and (np.abs(cur_ori[0] - 0) > 0.01 or np.abs(cur_ori[1] - 0) > 0.01) or \
                        cur_pos[0] < self.x_low_obs or cur_pos[0] > self.x_high_obs or cur_pos[1] > self.y_high_obs or cur_pos[1] < self.y_low_obs:
                    check_delete_index.append(i)
                    # print('this is cur ori on check:', cur_ori)
            if len(check_delete_index) == 0:
                break

        self.img_per_epoch = 0
        # # manipulator_before, pred_lwh_list = self.get_obs(format='grasp', data_root=data_root, epoch=epoch)
        # img_per_epoch_result = self.try_grasp(data_root=data_root, img_index_start=epoch)
        # return img_per_epoch_result

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
        pass

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
            if index == 3 or index == 5:
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
                        time.sleep(1 / 360)
                if move_success_flag == False:
                    break
            else:
                for i in range(10):
                    p.stepSimulation()
                    if self.is_render:
                        time.sleep(1 / 360)
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
        obj_width += 0.010
        close_open_gap = 0.053
        # close_open_gap = 0.048
        obj_width_range = np.array([0.022, 0.057])
        motor_pos_range = np.array([0.022, 0.010])  # 0.0273
        formula_parameters = np.polyfit(obj_width_range, motor_pos_range, 1)
        motor_pos = np.poly1d(formula_parameters)

        gripper_success_flag = True
        if index == 1:
            num_step = 30
        else:
            num_step = 10

        if gap > 0.0265:  # close
            tar_pos = motor_pos(obj_width) + close_open_gap
            p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL,
                                    targetPosition=motor_pos(obj_width) + close_open_gap, force=self.para_dict['gripper_force'])
            p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL,
                                    targetPosition=motor_pos(obj_width) + close_open_gap, force=self.para_dict['gripper_force'])
            for i in range(num_step):

                p.stepSimulation()
                gripper_left_pos = np.asarray(p.getLinkState(self.arm_id, 7)[0])
                gripper_right_pos = np.asarray(p.getLinkState(self.arm_id, 8)[0])
                if gripper_left_pos[1] - left_pos[1] > self.para_dict['gripper_threshold'] or right_pos[1] - gripper_right_pos[1] > self.para_dict['gripper_threshold']:
                    print('during grasp, fail')
                    gripper_success_flag = False
                    break
                if self.is_render:
                    time.sleep(1 / 24)
        else:  # open
            p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL, targetPosition=motor_pos(obj_width), force=self.para_dict['gripper_force'])
            p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL, targetPosition=motor_pos(obj_width), force=self.para_dict['gripper_force'])
            for i in range(num_step):
                p.stepSimulation()
                if self.is_render:
                    time.sleep(1 / 24)
        if index == 1:
            print('initialize the distance from gripper to bar')
            bar_pos = np.asarray(p.getLinkState(self.arm_id, 6)[0])
            gripper_left_pos = np.asarray(p.getLinkState(self.arm_id, 7)[0])
            gripper_right_pos = np.asarray(p.getLinkState(self.arm_id, 8)[0])
            self.distance_left = np.linalg.norm(bar_pos[:2] - gripper_left_pos[:2])
            self.distance_right = np.linalg.norm(bar_pos[:2] - gripper_right_pos[:2])
        return gripper_success_flag

    def get_obs(self, format=None, data_root=None, epoch=None):

        self.box_pos, self.box_ori, self.gt_ori_qua = [], [], []
        if len(self.obj_idx) == 0:
            return np.array([]), np.array([]), np.array([])
        self.constrain_id = []
        for i in range(len(self.obj_idx)):
            box_pos = np.asarray(p.getBasePositionAndOrientation(self.obj_idx[i])[0])
            box_ori = np.asarray(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.obj_idx[i])[1]))
            # self.constrain_id.append(p.createConstraint(self.obj_idx[i], -1, -1, -1, p.JOINT_FIXED,
            #                                             jointAxis=[1, 1, 1],
            #                                             parentFramePosition=[0, 0, 0],
            #                                             childFramePosition=box_pos,
            #                                             childFrameOrientation=[1, 1, 1]))
            self.gt_ori_qua.append(np.asarray(p.getBasePositionAndOrientation(self.obj_idx[i])[1]))
            self.box_pos = np.append(self.box_pos, box_pos).astype(np.float32)
            self.box_ori = np.append(self.box_ori, box_ori).astype(np.float32)
        self.box_pos = self.box_pos.reshape(len(self.obj_idx), 3)
        self.box_ori = self.box_ori.reshape(len(self.obj_idx), 3)
        self.gt_ori_qua = np.asarray(self.gt_ori_qua)
        self.gt_pos_ori = np.concatenate((self.box_pos, self.box_ori), axis=1)
        self.gt_pos_ori = self.gt_pos_ori.astype(np.float32)

        (width, length, image, image_depth, seg_mask) = p.getCameraImage(width=640,
                                                                height=480,
                                                                viewMatrix=self.view_matrix,
                                                                projectionMatrix=self.projection_matrix,
                                                                renderer=p.ER_BULLET_HARDWARE_OPENGL)
        far_range = self.camera_parameters['far']
        near_range = self.camera_parameters['near']
        # depth_data = far_range * near_range / (far_range - (far_range - near_range) * image_depth)
        # top_height = 0.4 - depth_data
        my_im = image[:, :, :3]
        temp = np.copy(my_im[:, :, 0])  # change rgb image to bgr for opencv to save
        my_im[:, :, 0] = my_im[:, :, 2]
        my_im[:, :, 2] = temp
        img = np.copy(my_im)
        os.makedirs(data_root + 'origin_images/', exist_ok=True)
        img_path = data_root + 'origin_images/%012d' % (epoch)

        ################### the results of object detection has changed the order!!!! ####################
        # structure of results: x, y, z, length, width, ori
        results, pred_conf = self.yolo_model.yolov8_predict(img_path=img_path, img=img)
        if len(results) == 0:
            return np.array([]), np.array([]), np.array([])
        print('this is the result of yolo-pose\n', results)
        ################### the results of object detection has changed the order!!!! ####################

        manipulator_before = np.concatenate((results[:, :3], np.zeros((len(results), 2)), results[:, 5].reshape(-1, 1)), axis=1)
        new_lwh_list = np.concatenate((results[:, 3:5], np.ones((len(results), 1)) * 0.016), axis=1)
        # print('this is manipulator before after the detection \n', manipulator_before)

        return manipulator_before, new_lwh_list, pred_conf

if __name__ == '__main__':

    # np.random.seed(183)
    # random.seed(183)

    para_dict = {'start_num': 00, 'end_num': 10000, 'thread': 0,
                 'yolo_conf': 0.6, 'yolo_iou': 0.8, 'device': 'cuda:0',
                 'save_img_flag': True,
                 'init_pos_range': [[0.13, 0.17], [-0.03, 0.03], [0.01, 0.02]],
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
                 'yolo_model_path': '/home/zhizhuo/Creative_Machines_Lab/Knolling_bot_2/train_pile_overlap_627/weights/best.pt'}

    startnum = para_dict['start_num']
    endnum = para_dict['end_num']
    thread = para_dict['thread']
    save_img_flag = para_dict['save_img_flag']
    init_pos_range = para_dict['init_pos_range']
    init_ori_range = para_dict['init_ori_range']

    data_root = para_dict['dataset_path'] + 'grasp_dataset_721_heavy_test/'
    # with open(para_dict['dataset_path'] + 'grasp_dataset_721_heavy_test_readme.txt', "w") as f:
    #     for key, value in para_dict.items():
    #         f.write(key + ': ')
    #         f.write(str(value) + '\n')

    os.makedirs(data_root, exist_ok=True)

    max_box_num = para_dict['max_box_num']
    min_box_num = para_dict['min_box_num']
    mm2px = 530 / 0.34

    env = Arm_env(endnum=endnum, save_img_flag=save_img_flag, init_pos_range=init_pos_range, init_ori_range=init_ori_range, para_dict=para_dict)
    os.makedirs(data_root + 'origin_images/', exist_ok=True)
    os.makedirs(data_root + 'origin_labels/', exist_ok=True)

    exist_img_num = startnum
    while True:
        num_item = int(np.random.uniform(min_box_num, max_box_num + 1))
        env.reset(data_root=data_root, num_item=num_item, thread=thread, epoch=exist_img_num)
        img_per_epoch = env.try_grasp(data_root=data_root, img_index_start=exist_img_num)
        exist_img_num += img_per_epoch