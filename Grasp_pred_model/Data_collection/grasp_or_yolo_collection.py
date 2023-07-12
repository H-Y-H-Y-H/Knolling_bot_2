import pybullet as p
import pybullet_data as pd
import os
import gym
import numpy as np
import random
import math
import cv2
from urdfpy import URDF
from tqdm import tqdm
import time
import torch
from sklearn.preprocessing import MinMaxScaler
# from Grasp_pred_model.train_model import collate_fn, Generate_Dataset

import sys
sys.path.append('/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_bot/')
from Grasp_pred_model.network import LSTMRegressor

from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

# logger = EasyLog(log_level=logging.INFO)

def change_sequence(pos_before):

    origin_point = np.array([0, -0.2])
    delete_index = np.where(pos_before == 0)[0]
    distance = np.linalg.norm(pos_before[:, :2] - origin_point, axis=1)
    order = np.argsort(distance)
    return order

class PosePredictor(DetectionPredictor):

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes,
                                        nc=len(self.model.names))

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, shape)
            path, _, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            results.append(
                Results(orig_img=orig_img,
                        path=img_path,
                        names=self.model.names,
                        boxes=pred[:, :6],
                        keypoints=pred_kpts))
        return results

class Yolo_predict():

    def __init__(self, save_img_flag):

        self.save_img_flag = save_img_flag

    def find_keypoints(self, xpos, ypos, l, w, ori, mm2px):

        gamma = ori
        rot_z = [[np.cos(gamma), -np.sin(gamma)],
                 [np.sin(gamma), np.cos(gamma)]]
        # rot_z = [[1, 0],
        #          [0, 1]]
        rot_z = np.asarray(rot_z)

        kp1 = np.asarray([l / 2, 0])
        kp2 = np.asarray([0, w / 2])
        kp3 = np.asarray([-l / 2, 0])
        kp4 = np.asarray([0, -w / 2])

        keypoint1 = np.dot(rot_z, kp1)
        keypoint2 = np.dot(rot_z, kp2)
        keypoint3 = np.dot(rot_z, kp3)
        keypoint4 = np.dot(rot_z, kp4)

        keypoint1 = np.array([((keypoint1[1] + ypos) * mm2px + 320) / 640, ((keypoint1[0] + xpos) * mm2px + 6) / 480, 1])
        keypoint2 = np.array([((keypoint2[1] + ypos) * mm2px + 320) / 640, ((keypoint2[0] + xpos) * mm2px + 6) / 480, 1])
        keypoint3 = np.array([((keypoint3[1] + ypos) * mm2px + 320) / 640, ((keypoint3[0] + xpos) * mm2px + 6) / 480, 1])
        keypoint4 = np.array([((keypoint4[1] + ypos) * mm2px + 320) / 640, ((keypoint4[0] + xpos) * mm2px + 6) / 480, 1])
        keypoints = np.concatenate((keypoint1, keypoint2, keypoint3, keypoint4), axis=0).reshape(-1, 3)

        return keypoints

    def gt_data_preprocess(self, xy, lw, ori):

        mm2px = 530 / 0.34  # (1558)
        total_num = len(xy)
        num_item = 15
        label = []
        for j in range(total_num):
            # print(real_world_data[j])
            print('this is index if legos', j)
            xpos1, ypos1 = xy[j, 0], xy[j, 1]
            l, w = lw[j, 0], lw[j, 1]
            yawori = ori[j]

            # ensure the yolo sequence!
            label_y = (xpos1 * mm2px + 6) / 480
            label_x = (ypos1 * mm2px + 320) / 640
            length = l * 3
            width = w * 3
            # ensure the yolo sequence!
            keypoints = self.find_keypoints(xpos1, ypos1, l, w, yawori, mm2px)
            keypoints_order = np.lexsort((keypoints[:, 0], keypoints[:, 1]))
            keypoints = keypoints[keypoints_order]

            element = np.concatenate(([0], [label_x, label_y], [length, width], keypoints.reshape(-1)))
            label.append(element)

        label = np.asarray(label)

        return label

    def plot_and_transform(self, im, box, label='', color=(0, 0, 0), txt_color=(255, 255, 255), index=None,
                           scaled_xylw=None, keypoints=None, cls=None, conf=None, use_xylw=True, truth_flag=None, height_data=None):

        ############### zzz plot parameters ###############
        zzz_lw = 1
        tf = 1 # font thickness
        mm2px = 530 / 0.34
        # x_mm_center = scaled_xylw[1] * 0.3
        # y_mm_center = scaled_xylw[0] * 0.4 - 0.2
        # x_px_center = x_mm_center * mm2px + 6
        # y_px_center = y_mm_center * mm2px + 320
        x_px_center = scaled_xylw[1] * 480
        y_px_center = scaled_xylw[0] * 640
        z_mm_center = height_data[int(x_px_center), int(y_px_center)] - 0.01
        if z_mm_center <= 0.006:
            z_mm_center = 0.006
        # z_mm_center = 0.006

        # this is the knolling sequence, not opencv!!!!
        keypoints_x = ((keypoints[:, 1] * 480 - 6) / mm2px).reshape(-1, 1)
        keypoints_y = ((keypoints[:, 0] * 640 - 320) / mm2px).reshape(-1, 1)
        keypoints_mm = np.concatenate((keypoints_x, keypoints_y), axis=1)
        keypoints_center = np.average(keypoints_mm, axis=0)

        length = max(np.linalg.norm(keypoints_mm[0] - keypoints_mm[-1]),
                     np.linalg.norm(keypoints_mm[1] - keypoints_mm[2]))
        width = min(np.linalg.norm(keypoints_mm[0] - keypoints_mm[-1]),
                    np.linalg.norm(keypoints_mm[1] - keypoints_mm[2]))
        # length = np.linalg.norm(keypoints_mm[0] - keypoints_mm[-1])
        # width = np.linalg.norm(keypoints_mm[1] - keypoints_mm[2])
        c1 = np.array([length / (2), width / (2)])
        c2 = np.array([length / (2), -width / (2)])
        c3 = np.array([-length / (2), width / (2)])
        c4 = np.array([-length / (2), -width / (2)])
        if use_xylw == True:
            # length = scaled_xylw[2] / 3
            # width = scaled_xylw[3] / 3
            # c1 = np.array([length / (2), width / (2)])
            # c2 = np.array([length / (2), -width / (2)])
            # c3 = np.array([-length / (2), width / (2)])
            # c4 = np.array([-length / (2), -width / (2)])
            box_center = np.array([scaled_xylw[0], scaled_xylw[1]])
        else:
            box_center = keypoints_center

        all_distance = np.linalg.norm((keypoints_mm - keypoints_center), axis=1)
        k = 2
        max_index = all_distance.argsort()[-k:]
        lucky_keypoint_index = np.argmax([keypoints_mm[max_index[0], 1], keypoints_mm[max_index[1], 1]])
        lucky_keypoint = keypoints_mm[max_index[lucky_keypoint_index]]
        # print('the ori keypoint is ', keypoints_mm[max_index[lucky_keypoint_index]])
        my_ori = np.arctan2(lucky_keypoint[1] - keypoints_center[1], lucky_keypoint[0] - keypoints_center[0])
        # In order to grasp, this ori is based on the longest side of the box, not the label ori!

        if length < width:
            if my_ori > np.pi / 2:
                my_ori_plot = my_ori - np.pi / 2
            else:
                my_ori_plot = my_ori + np.pi / 2
        else:
            my_ori_plot = my_ori

        rot_z = [[np.cos(my_ori_plot), -np.sin(my_ori_plot)],
                 [np.sin(my_ori_plot), np.cos(my_ori_plot)]]
        corn1 = (np.dot(rot_z, c1)) * mm2px
        corn2 = (np.dot(rot_z, c2)) * mm2px
        corn3 = (np.dot(rot_z, c3)) * mm2px
        corn4 = (np.dot(rot_z, c4)) * mm2px

        corn1 = [corn1[0] + x_px_center, corn1[1] + y_px_center]
        corn2 = [corn2[0] + x_px_center, corn2[1] + y_px_center]
        corn3 = [corn3[0] + x_px_center, corn3[1] + y_px_center]
        corn4 = [corn4[0] + x_px_center, corn4[1] + y_px_center]
        ############### zzz plot parameters ###############


        ############### zzz plot the box ###############
        if isinstance(box, torch.Tensor):
            box = box.cpu().detach().numpy()
        # print(box)
        p1 = np.array([int(box[0] * 640), int(box[1] * 480)])
        # print('this is p1 and p2', p1, p2)

        # cv2.rectangle(self.im, p1, p2, color, thickness=zzz_lw, lineType=cv2.LINE_AA)
        im = cv2.line(im, (int(corn1[1]), int(corn1[0])), (int(corn2[1]), int(corn2[0])), color, 1)
        im = cv2.line(im, (int(corn2[1]), int(corn2[0])), (int(corn4[1]), int(corn4[0])), color, 1)
        im = cv2.line(im, (int(corn4[1]), int(corn4[0])), (int(corn3[1]), int(corn3[0])), color, 1)
        im = cv2.line(im, (int(corn3[1]), int(corn3[0])), (int(corn1[1]), int(corn1[0])), color, 1)
        plot_x = np.copy((scaled_xylw[1] * 480 - 6) / mm2px)
        plot_y = np.copy((scaled_xylw[0] * 640 - 320) / mm2px)
        plot_l = np.copy(length)
        plot_w = np.copy(width)
        label1 = 'cls: %d, conf: %.5f' % (cls, conf)
        label2 = 'index: %d, x: %.4f, y: %.4f' % (index, plot_x, plot_y)
        label3 = 'l: %.4f, w: %.4f, ori: %.4f' % (plot_l, plot_w, my_ori)
        if label:
            w, h = cv2.getTextSize(label, 0, fontScale=zzz_lw / 3, thickness=tf)[0]  # text width, z_mm_center
            outside = p1[1] - h >= 3
            # cv2.rectangle(self.im, p1, p2, color, 0, cv2.LINE_AA)  # filled
            if truth_flag == True:
                txt_color = (0, 0, 255)
                # im = cv2.putText(im, label1, (p1[0] - 50, p1[1] - 32 if outside else p1[1] + h + 2),
                #                  0, zzz_lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
                # im = cv2.putText(im, label2, (p1[0] - 50, p1[1] - 22 if outside else p1[1] + h + 12),
                #                  0, zzz_lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
            else:
                im = cv2.putText(im, label1, (p1[0] - 50, p1[1] + 22 if outside else p1[1] + h + 2),
                                 0, zzz_lw / 3, (0, 0, 255), thickness=tf, lineType=cv2.LINE_AA)
                im = cv2.putText(im, label2, (p1[0] - 50, p1[1] + 32 if outside else p1[1] + h + 12),
                                 0, zzz_lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
                m = cv2.putText(im, label3, (p1[0] - 50, p1[1] + 42 if outside else p1[1] + h + 22),
                                0, zzz_lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
            # im = cv2.putText(im, label1, (c1[0] - 70, c1[1] - 35), 0, tl / 3, color, thickness=tf, lineType=cv2.LINE_AA)
        ############### zzz plot the box ###############

        ############### zzz plot the keypoints ###############
        shape = (640, 640)
        radius = 1
        for i, k in enumerate(keypoints):
            if truth_flag == False:
                if i == 0:
                    color_k = (255, 0, 0)
                else:
                    color_k = (0, 0, 0)
            elif truth_flag == True:
                if i == 0:
                    color_k = (0, 0, 255)
                elif i == 3:
                    color_k = (255, 255, 0)
            x_coord, y_coord = k[0] * 640, k[1] * 480
            # if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
            #     if len(k) == 3:
            #         conf = k[2]
            #         if conf < 0.5:
            #             continue
            im = cv2.circle(im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)
        ############### zzz plot the keypoints ###############

        result = np.concatenate((box_center, [z_mm_center], [round(length, 4)], [round(width, 4)], [my_ori]))

        return im, result

    def yolov8_predict(self, cfg=DEFAULT_CFG, use_python=False, img_path=None, img=None, target=None, boxes_num=None, height_data=None, test_pile_detection=None):

        # model = '/home/zhizhuo/ADDdisk/Create Machine Lab/YOLOv8/runs/pose/train_pile_overlap_627/weights/best.pt'
        model = '/home/ubuntu/Desktop/Knolling_bot_2/train_pile_overlap_627/weights/best.pt'
        # model = 'C:/Users/24356/Desktop/Knolling_bot_2/train_pile_overlap_627/weights/best.pt'
        # img = adjust_img(img)

        cv2.imwrite(img_path + '.png', img)
        img_path_input = img_path + '.png'
        args = dict(model=model, source=img_path_input, conf=0.5, iou=0.8, device='cpu')
        use_python = True
        if use_python:
            from ultralytics import YOLO
            images = YOLO(model)(**args)
        else:
            predictor = PosePredictor(overrides=args)
            predictor.predict_cli()
        device = 'cuda:0'

        origin_img = cv2.imread(img_path_input)

        use_xylw = False # use lw or keypoints to export length and width

        one_img = images[0]

        pred_result = []
        pred_xylws = one_img.boxes.xywhn.cpu().detach().numpy()
        if len(pred_xylws) == 0:
            return [], []
        else:
            pred_cls = one_img.boxes.cls.cpu().detach().numpy()
            pred_conf = one_img.boxes.conf.cpu().detach().numpy()
            pred_keypoints = one_img.keypoints.cpu().detach().numpy()
            pred_keypoints[:, :, :2] = pred_keypoints[:, :, :2] / np.array([640, 480])
            pred_keypoints = pred_keypoints.reshape(len(pred_xylws), -1)
            pred = np.concatenate((np.zeros((len(pred_xylws), 1)), pred_xylws, pred_keypoints), axis=1)

        # ######## order based on distance to draw it on the image!!!
        # mm2px = 530 / 0.34
        # x_px_center = pred_xylws[:, 1] * 480
        # y_px_center = pred_xylws[:, 0] * 640
        # mm_center = np.concatenate(
        #     (((x_px_center - 6) / mm2px).reshape(-1, 1), ((y_px_center - 320) / mm2px).reshape(-1, 1)), axis=1)
        # pred_order = change_sequence(mm_center)

        # pred = pred[pred_order]
        # pred_xylws = pred_xylws[pred_order]
        # pred_keypoints = pred_keypoints[pred_order]
        # pred_cls = pred_cls[pred_order]
        # pred_conf = pred_conf[pred_order]
        # print('this is the pred order', pred_order)

        for j in range(len(pred_xylws)):

            pred_keypoint = pred_keypoints[j].reshape(-1, 3)
            pred_xylw = pred_xylws[j]

            # print('this is pred xylw', pred_xylw)
            origin_img, result = self.plot_and_transform(im=origin_img, box=pred_xylw, label='0:, predic', color=(0, 0, 0), txt_color=(255, 255, 255),
                                                         index=j, scaled_xylw=pred_xylw, keypoints=pred_keypoint,
                                                         cls=pred_cls[j], conf=pred_conf[j],
                                                         use_xylw=use_xylw, truth_flag=False, height_data=height_data)
            pred_result.append(result)

        if test_pile_detection == True:
            for j in range(len(target)):
                tar_xylw = np.copy(target[j, 1:5])
                tar_keypoints = np.copy((target[j, 5:]).reshape(-1, 3)[:, :2])

                # plot target
                print('this is tar xylw', tar_xylw)
                origin_img, _ = self.plot_and_transform(im=origin_img, box=tar_xylw, label='0: target', color=(255, 255, 0), txt_color=(255, 255, 255),
                                                        index=j, scaled_xylw=tar_xylw, keypoints=tar_keypoints,
                                                        cls=0, conf=1,
                                                        use_xylw=use_xylw, truth_flag=True, height_data=height_data)


        if self.save_img_flag == True:
            cv2.namedWindow('zzz', 0)
            cv2.resizeWindow('zzz', 1280, 960)
            cv2.imshow('zzz', origin_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            img_path_output = img_path + '_pred.png'
            cv2.imwrite(img_path_output, origin_img)
        pred_result = np.asarray(pred_result)

        return pred_result, pred_conf

class Arm_env(gym.Env):

    def __init__(self,max_step, is_render=True, x_grasp_accuracy=0.2, y_grasp_accuracy=0.2,
                 z_grasp_accuracy=0.2, endnum=None, save_img_flag=None, urdf_path=None, use_grasp_model=False,
                 para_dict=None, total_error=None, init_pos_range=None):
        self.endnum = endnum
        self.kImageSize = {'width': 480, 'height': 480}

        self.init_pos_range = init_pos_range
        self.urdf_path = urdf_path
        self.pybullet_path = pd.getDataPath()
        self.is_render = is_render
        self.save_img_flag = save_img_flag
        self.yolo_model = Yolo_predict(save_img_flag=self.save_img_flag)
        if use_grasp_model == True:
            self.para_dict = para_dict
            self.model = LSTMRegressor(input_dim=self.para_dict['input_size'], hidden_dim=self.para_dict['hidden_size'], output_dim=self.para_dict['output_size'],
                                      num_layers=self.para_dict['num_layers'], batch_size=self.para_dict['batch_size'], device=self.para_dict['device'])
            self.model.load_state_dict(torch.load(self.para_dict['model_path'] + 'best_model.pt'))
            self.use_grasp_model = use_grasp_model
            self.total_error = total_error
        else:
            self.use_grasp_model = False

        self.x_low_obs = 0.03
        self.x_high_obs = 0.27
        self.y_low_obs = -0.14
        self.y_high_obs = 0.14
        self.z_low_obs = 0.0
        self.z_high_obs = 0.05
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

    def reset_table(self, close_flag=False, use_lego_urdf=None, lego_list=None, num_item=None, thread=None, epoch=None,
                    pile_flag=None, data_root=None, try_grasp_flag=None, test_pile_detection=False):

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

        if try_grasp_flag == True:
            self.arm_id = p.loadURDF(os.path.join(self.urdf_path, "robot_arm928/robot_arm1_backup.urdf"),
                                     basePosition=[-0.08, 0, 0.02], useFixedBase=True,
                                     flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

            p.changeDynamics(self.arm_id, 7, lateralFriction=para_dict['gripper_lateral_friction'],
                             spinningFriction=para_dict['gripper_spinning_friction'],
                             rollingFriction=para_dict['gripper_rolling_friction'],
                             linearDamping=para_dict['gripper_linear_damping'],
                             angularDamping=para_dict['gripper_angular_damping'],
                             jointDamping=para_dict['gripper_joint_damping'],
                             restitution=para_dict['gripper_restitution'],
                             contactDamping=para_dict['gripper_contact_damping'],
                             contactStiffness=para_dict['gripper_contact_stiffness'])

            p.changeDynamics(self.arm_id, 8, lateralFriction=para_dict['gripper_lateral_friction'],
                             spinningFriction=para_dict['gripper_spinning_friction'],
                             rollingFriction=para_dict['gripper_rolling_friction'],
                             linearDamping=para_dict['gripper_linear_damping'],
                             angularDamping=para_dict['gripper_angular_damping'],
                             jointDamping=para_dict['gripper_joint_damping'],
                             restitution=para_dict['gripper_restitution'],
                             contactDamping=para_dict['gripper_contact_damping'],
                             contactStiffness=para_dict['gripper_contact_stiffness'])


            ik_angles0 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=self.reset_pos,
                                                      maxNumIterations=200,
                                                      targetOrientation=p.getQuaternionFromEuler(self.reset_ori))
            for motor_index in range(5):
                p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                        targetPosition=ik_angles0[motor_index], maxVelocity=20)
            for _ in range(int(30)):
                # time.sleep(1/480)
                p.stepSimulation()

        if pile_flag == True:
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
        else:
            if not close_flag:
                p.setGravity(0, 0, -10)
            else:
                if random.random() < 0.5:  # x_mode
                    wall_flag = 0
                    wall_pos = np.random.uniform(0, 0.20)
                    p.setGravity(-10, 0, -15)
                    wallid = p.loadURDF(os.path.join(self.urdf_path, "plane_2.urdf"), basePosition=[wall_pos, 0, 0],
                                        baseOrientation=p.getQuaternionFromEuler([0, 1.57, 0]), useFixedBase=1,
                                        flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
                else:  # y_mode
                    wall_flag = 1
                    wall_pos = np.random.uniform(-0.18, 0.10)
                    p.setGravity(0, -10, -15)
                    wallid = p.loadURDF(os.path.join(self.urdf_path, "plane_2.urdf"), basePosition=[0, wall_pos, 0],
                                        baseOrientation=p.getQuaternionFromEuler([-1.57, 0, 0]), useFixedBase=1,
                                        flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

                p.changeVisualShape(wallid, -1, rgbaColor=[1, 1, 1, 0])

        if pile_flag == True:
            # rdm_ori = np.concatenate((np.ones((self.num_item, 1)) * (np.pi / 2.5),
            #                           np.zeros((self.num_item, 1)),
            #                           np.ones((self.num_item, 1)) * (np.pi / 9)), axis=1)
            rdm_ori = np.random.uniform(-np.pi / 4, np.pi / 4, size=(self.num_item, 3))
            rdm_pos_x = np.random.uniform(self.init_pos_range[0][0], self.init_pos_range[0][1], size=(self.num_item, 1))
            rdm_pos_y = np.random.uniform(self.init_pos_range[1][0], self.init_pos_range[1][1], size=(self.num_item, 1))
            rdm_pos_z = np.random.uniform(self.init_pos_range[2][0], self.init_pos_range[2][1], size=(self.num_item, 1))
            rdm_pos = np.concatenate((rdm_pos_x, rdm_pos_y, rdm_pos_z), axis=1)
        else:
            while True:
                dis_flag = True
                rdm_ori_yaw = np.random.uniform(0, np.pi, size=(self.num_item))

                if close_flag:
                    # print('this is wall pos', wall_pos)
                    # print(wall_flag)
                    if wall_flag == 0:  # x
                        rdm_pos_x = np.random.uniform(wall_pos + 0.02, wall_pos + 0.25, size=(self.num_item, 1))
                        rdm_pos_y = np.random.uniform(-0.15, 0.15, size=(self.num_item, 1))
                    else:  # y
                        rdm_pos_x = np.random.uniform(0.06, 0.25, size=(self.num_item,  1))
                        rdm_pos_y = np.random.uniform(wall_pos + 0.02, wall_pos + 0.25, size=(self.num_item, 1))
                else:
                    rdm_pos_x = np.random.uniform(0.06, 0.25, size=(self.num_item, 1))
                    rdm_pos_y = np.random.uniform(-0.15, 0.15, size=(self.num_item, 1))

                rot_parallel = [[np.cos(rdm_ori_yaw[0]), -np.sin(rdm_ori_yaw[0])],
                                [np.sin(rdm_ori_yaw[0]), np.cos(rdm_ori_yaw[0])]]
                rot_parallel = np.asarray(rot_parallel)

                xy_parallel = np.dot(rot_parallel, np.asarray([np.random.uniform(-0.016, 0.016), 0.040]))

                xy_parallel = np.add(xy_parallel, np.asarray([rdm_pos_x[0], rdm_pos_y[0]]))
                # print(xy_parallel)

                for i in range(self.num_item):
                    if dis_flag == False:
                        break
                    num2 = np.copy(self.num_item)

                    # print(num2)
                    for j in range(i + 1, num2):

                        dis_check = math.dist([rdm_pos_x[i], rdm_pos_y[i]], [rdm_pos_x[j], rdm_pos_y[j]])

                        if dis_check < 0.050:
                            # print(i,"and",j,"gg")
                            dis_flag = False

                #
                if dis_flag == True:
                    break
            rdm_pos = np.concatenate((rdm_pos_x, rdm_pos_y, np.ones((self.num_item, 1)) * 0.01),  axis=1)
            rdm_ori = np.concatenate((np.zeros((self.num_item, 2)), rdm_ori_yaw.reshape(-1, 1)), axis=1)

        self.gt_lwh_list = []
        self.obj_idx = []
        box_path = data_root + "box_urdf/thread_%d/epoch_%d/" % (thread, epoch)
        os.makedirs(box_path, exist_ok=True)
        temp_box = URDF.load(self.urdf_path + 'box_generator/template.urdf')

        length_range = np.round(np.random.uniform(0.016, 0.048, size=(self.num_item, 1)), decimals=3)
        width_range = np.round(np.random.uniform(0.016, np.minimum(length_range, 0.036), size=(self.num_item, 1)),decimals=3)
        height_range = np.round(np.random.uniform(0.010, 0.020, size=(self.num_item, 1)), decimals=3)

        for i in range(self.num_item):
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
        p.changeDynamics(baseid, -1, lateralFriction=para_dict['base_lateral_friction'],
                             spinningFriction=para_dict['base_spinning_friction'],
                             rollingFriction=para_dict['base_rolling_friction'],
                             restitution=para_dict['base_restitution'],
                            contactDamping=para_dict['base_contact_damping'],
                            contactStiffness=para_dict['base_contact_stiffness'])

        # if try_grasp_flag == True:
        #     for i in range(len(self.obj_idx)):
        #         p.changeDynamics(self.obj_idx[i], -1, lateralFriction=0.8, spinningFriction=0.8, rollingFriction=0.00,
        #                          linearDamping=0, angularDamping=0, jointDamping=0,
        #                          restitution=0.0, contactDamping=0.001, contactStiffness=10000)
        forbid_range = np.array([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        while True:
            new_num_item = len(self.obj_idx)
            delete_index = []
            for i in range(len(self.obj_idx)):
                p.changeDynamics(self.obj_idx[i], -1, lateralFriction=para_dict['box_lateral_friction'],
                                 spinningFriction=para_dict['box_spinning_friction'],
                                 rollingFriction=para_dict['box_rolling_friction'],
                                 linearDamping=para_dict['box_linear_damping'],
                                 angularDamping=para_dict['box_angular_damping'],
                                 jointDamping=para_dict['box_joint_damping'],
                                 restitution=para_dict['box_restitution'],
                                 contactDamping=para_dict['box_contact_damping'],
                                 contactStiffness=para_dict['box_contact_stiffness'])

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
            for _ in range(int(200)):
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

        if try_grasp_flag == True or self.use_grasp_model == True:
            self.img_per_epoch = 0
            # manipulator_before, pred_lwh_list = self.get_obs(format='grasp', data_root=data_root, epoch=epoch)
            img_per_epoch_result = self.try_grasp(data_root=data_root, img_index_start=epoch, test_pile_detection=test_pile_detection)
            return img_per_epoch_result
        elif test_pile_detection == True:
            return self.get_obs(format='pile', test_pile_detection=test_pile_detection, data_root=data_root, epoch=epoch), \
                   self.gt_lwh_list, new_num_item

    def try_grasp(self, data_root=None, img_index_start=None, test_pile_detection=None):
        print('this is img_index start while grasping', img_index_start)
        if img_index_start + self.img_per_epoch >= self.endnum:
            if self.use_grasp_model == True:
                self.total_error = np.asarray(self.total_error).reshape(-1, 1)
                np.savetxt(data_root + 'total_error.txt', self.total_error)
            quit()
        manipulator_before, pred_lwh_list, pred_conf = self.get_obs(format='grasp', data_root=data_root, epoch=self.img_per_epoch + img_index_start)

        if len(manipulator_before) <= 1:
            print('no pile in the environment, try to reset!')
            return self.img_per_epoch
        # if np.any(manipulator_before[:, 2].reshape(1, -1) > 0.01) == False:
        #     print('no pile in the environment, try to reset!')
        #     return self.img_per_epoch

        if self.use_grasp_model == True:

            xy = manipulator_before[:, :2]
            yaw = manipulator_before[:, -1].reshape(-1, 1)
            lw = pred_lwh_list[:, :2]
            data = np.concatenate((xy, lw, pred_conf.reshape(-1, 1), yaw), axis=1)

            scaler = MinMaxScaler()
            data_range = np.array([[0, -0.14, 0, 0, 0, 0.5],
                                   [0.3, 0.14, 0.06, 0.06, np.pi, 1]])
            scaler.fit(data_range)
            tar_index = np.where(data[:, -2] < 0)[0]
            data[tar_index, -2] += np.pi
            LSTM_input = torch.from_numpy(np.array([scaler.transform(data)])).to(self.para_dict['device'], dtype=torch.float32)
            LSTM_input = pack_padded_sequence(LSTM_input, list([data.shape[0]]), batch_first=True)
            out = self.model.forward(LSTM_input)
            pred_grasp = out.cpu().detach().numpy().reshape(-1, 1)

        pos_ori_after = np.concatenate((self.reset_pos, np.zeros(3)), axis=0).reshape(-1, 6)
        manipulator_after = np.repeat(pos_ori_after, len(manipulator_before), axis=0)

        rest_ori = np.array([0, 1.57, 0])
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

                trajectory_pos_list = [[0.02, grasp_width[i]],  # open!
                                       offset_high + start_end[i][:3],
                                       offset_low + start_end[i][:3],
                                       [0.0273, grasp_width[i]],  # close
                                       offset_high + start_end[i][:3],
                                       start_end[i][6:9]]
                trajectory_ori_list = [rest_ori + start_end[i][3:6],
                                       rest_ori + start_end[i][3:6],
                                       rest_ori + start_end[i][3:6],
                                       [0.0273, grasp_width[i]],
                                       rest_ori + start_end[i][3:6],
                                       rest_ori + start_end[i][9:12]]
                last_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
                last_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))

                success_grasp_flag = True
                left_pos = np.asarray(p.getLinkState(self.arm_id, 7)[0])
                right_pos = np.asarray(p.getLinkState(self.arm_id, 8)[0])
                for j in range(len(trajectory_pos_list)):
                    if len(trajectory_pos_list[j]) == 3:
                        if j == 1:
                            last_pos, left_pos, right_pos, _ = self.move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j], index=j)
                        elif j == 2:
                            last_pos, _, _, success_grasp_flag = self.move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j],
                                                                           origin_left_pos=left_pos, origin_right_pos=right_pos, index=j)
                            if success_grasp_flag == False:
                                break
                        else:
                            last_pos, _, _, _ = self.move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j])
                        last_ori = np.copy(trajectory_ori_list[j])
                    elif len(trajectory_pos_list[j]) == 2:
                        # time.sleep(2)
                        success_grasp_flag = self.gripper(trajectory_pos_list[j][0], trajectory_pos_list[j][1], left_pos, right_pos)

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

            gt_data = np.asarray(gt_data)
            grasp_flag = np.asarray(grasp_flag).reshape(-1, 1)


            if self.use_grasp_model == True:
                grasp_flag_error = np.mean((grasp_flag - pred_grasp) ** 2)
                yolo_label = np.concatenate((grasp_flag, pred_grasp, gt_data, pred_conf.reshape(-1, 1)), axis=1)
                self.total_error.append(grasp_flag_error)
                print('this is grasp error', grasp_flag_error)
                conf_1_index = np.where(yolo_label[:, 0] == 1)[0]
                if yolo_label[conf_1_index, -1] < 0.70:
                    print(f'index {img_index_start + self.img_per_epoch}, check it!')
            else:
                yolo_label = np.concatenate((grasp_flag, gt_data, pred_conf.reshape(-1, 1)), axis=1)

        if test_pile_detection == True:
            pass
        else:
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
                                        targetPosition=ik_angles0[motor_index], maxVelocity=100, force=para_dict['move_force'])
            move_success_flag = True
            if index == 2 or index == 4:
                for i in range(15):
                    p.stepSimulation()
                    gripper_left_pos = np.asarray(p.getLinkState(self.arm_id, 7)[0])
                    gripper_right_pos = np.asarray(p.getLinkState(self.arm_id, 8)[0])
                    if np.abs(origin_left_pos[1] - gripper_left_pos[1]) > para_dict['move_threshold'] or \
                            np.abs(origin_right_pos[1] - gripper_right_pos[1]) > para_dict['move_threshold']:
                        move_success_flag = False
                        print('during moving, fail')
                        break
                    if self.is_render:
                        time.sleep(1 / 120)
                if move_success_flag == False:
                    break
            else:
                for i in range(3):
                    p.stepSimulation()
                    if self.is_render:
                        time.sleep(1 / 120)
            if abs(target_pos[0] - tar_pos[0]) < 0.001 and abs(target_pos[1] - tar_pos[1]) < 0.001 and abs(
                    target_pos[2] - tar_pos[2]) < 0.001 and \
                    abs(target_ori[0] - tar_ori[0]) < 0.001 and abs(target_ori[1] - tar_ori[1]) < 0.001 and abs(
                target_ori[2] - tar_ori[2]) < 0.001:
                break
            cur_pos = tar_pos
            cur_ori = tar_ori
        gripper_left_pos = np.asarray(p.getLinkState(self.arm_id, 7)[0])
        gripper_right_pos = np.asarray(p.getLinkState(self.arm_id, 8)[0])
        return cur_pos, gripper_left_pos, gripper_right_pos, move_success_flag

    def gripper(self, gap, obj_width, left_pos, right_pos):
        obj_width += 0.010
        close_open_gap = 0.053
        # close_open_gap = 0.048
        obj_width_range = np.array([0.022, 0.057])
        motor_pos_range = np.array([0.022, 0.010])  # 0.0273
        formula_parameters = np.polyfit(obj_width_range, motor_pos_range, 1)
        motor_pos = np.poly1d(formula_parameters)

        gripper_success_flag = True
        # gripper_left_pos = np.asarray(p.getLinkState(self.arm_id, 7)[0])
        # gripper_right_pos = np.asarray(p.getLinkState(self.arm_id, 8)[0])
        # print('left pos', gripper_left_pos)
        # print('right pos', gripper_right_pos)

        num_step = para_dict['gripper_sim_step']
        if gap > 0.0265:  # close
            tar_pos = motor_pos(obj_width) + close_open_gap
            p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL,
                                    targetPosition=motor_pos(obj_width) + close_open_gap, force=para_dict['gripper_force'])
            p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL,
                                    targetPosition=motor_pos(obj_width) + close_open_gap, force=para_dict['gripper_force'])
            for i in range(num_step):

                # step_left = (gripper_left_pos[1] + (motor_pos(obj_width) + close_open_gap)) * i / num_step
                # step_right = (gripper_right_pos[1] + (motor_pos(obj_width) + close_open_gap)) * i / num_step
                # p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL,
                #                         targetPosition=step_left, force=0.3)
                # p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL,
                #                         targetPosition=step_right, force=0.3)
                p.stepSimulation()
                gripper_left_pos = np.asarray(p.getLinkState(self.arm_id, 7)[0])
                gripper_right_pos = np.asarray(p.getLinkState(self.arm_id, 8)[0])
                if gripper_left_pos[1] - left_pos[1] > para_dict['gripper_threshold'] or right_pos[1] - gripper_right_pos[1] > para_dict['gripper_threshold']:
                    print('during grasp, fail')
                    gripper_success_flag = False
                    break
                if self.is_render:
                    time.sleep(1 / 24)
        else:  # open
            p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL, targetPosition=motor_pos(obj_width), force=para_dict['gripper_force'])
            p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL, targetPosition=motor_pos(obj_width), force=para_dict['gripper_force'])
            for i in range(para_dict['gripper_sim_step']):
                p.stepSimulation()
                if self.is_render:
                    time.sleep(1 / 24)
        return gripper_success_flag

    def get_obs(self, format=None, data_root=None, epoch=None, test_pile_detection=False):

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

        if format == 'grasp':
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
            os.makedirs(data_root + 'origin_images/', exist_ok=True)
            img_path = data_root + 'origin_images/%012d' % (epoch)

            ################### the results of object detection has changed the order!!!! ####################
            # structure of results: x, y, z, length, width, ori
            results, pred_conf = self.yolo_model.yolov8_predict(img_path=img_path, img=img, height_data=top_height)
            if len(results) == 0:
                return np.array([]), np.array([]), np.array([])
            print('this is the result of yolo-pose\n', results)
            ################### the results of object detection has changed the order!!!! ####################

            manipulator_before = np.concatenate((results[:, :3], np.zeros((len(results), 2)), results[:, 5].reshape(-1, 1)), axis=1)
            new_lwh_list = np.concatenate((results[:, 3:5], np.ones((len(results), 1)) * 0.016), axis=1)
            # print('this is manipulator before after the detection \n', manipulator_before)

            return manipulator_before, new_lwh_list, pred_conf

        elif format == 'pile':

            os.makedirs(data_root + 'origin_images/', exist_ok=True)
            img_path = data_root + 'origin_images/%012d' % (epoch)
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
            os.makedirs(data_root + 'origin_images/', exist_ok=True)
            img_path = data_root + 'origin_images/%012d' % (epoch)

            if test_pile_detection == True:
                gt_pos = self.gt_pos_ori[:, :2]
                gt_lw = self.gt_lwh_list[:, :2]
                gt_ori = self.gt_pos_ori[:, 5]
                gt_label = self.yolo_model.gt_data_preprocess(gt_pos, gt_lw, gt_ori)
                # results: x, y, z, length, width, ori
                results, pred_conf = self.yolo_model.yolov8_predict(img_path=img_path, img=my_im,
                                                                    height_data=top_height,
                                                                    target=gt_label, test_pile_detection=test_pile_detection)
                crowded_index = []
                for i in range(len(results)):
                    # seg_area = seg_mask[seg_mask == gt_compare_index[i]]
                    for j in range(len(results)):
                        lw_i = results[i, 3:5]
                        lw_j = results[j, 3:5]
                        if np.linalg.norm(results[i, :2] - results[j, :2]) < 0.85 * (np.linalg.norm(lw_i) + np.linalg.norm(lw_j)) / 2 and i != j:
                            # if (np.abs(self.gt_pos_ori[gt_compare_index[i], 3] - 0) > 0.01 or np.abs(self.gt_pos_ori[gt_compare_index[i], 4] - 0) > 0.01):
                            crowded_index.append(i)
                            break
                crowded_index = np.asarray(crowded_index)
                if len(crowded_index) != 0:
                    pred_conf_normal = pred_conf[np.setdiff1d(np.arange(len(results)), crowded_index, assume_unique=True)]
                    pred_conf_crowded = pred_conf[crowded_index]
                    # print('this is pred results\n', results)
                    # print('this is crowded index\n', crowded_index)
                    # print('this is pred confidence normal\n', pred_conf_normal)
                    # print('this is pred confidence crowded\n', pred_conf_crowded)
                    if len(pred_conf_normal) == 0:
                        pred_conf_normal = 0
                else:
                    pred_conf_crowded = 0
                    pred_conf_normal = pred_conf
                    print('all box are normal, confidence:\n', pred_conf)
                return np.mean(pred_conf_crowded), np.mean(pred_conf_normal)
            else:
                return self.gt_pos_ori

    def get_image(self):
        # reset camera
        (width, length, image, _, _) = p.getCameraImage(width=640,
                                                        height=480,
                                                        viewMatrix=self.view_matrix,
                                                        projectionMatrix=self.projection_matrix,
                                                        renderer=p.ER_BULLET_HARDWARE_OPENGL)
                                                        # lightDirection = [light_x, light_y, light_z])
        my_im = image[:, :, :3]
        temp = np.copy(my_im[:, :, 0])  # change rgb image to bgr for opencv to save
        my_im[:, :, 0] = my_im[:, :, 2]
        my_im[:, :, 2] = temp
        return my_im


if __name__ == '__main__':

    para_dict = {'start_num': 0, 'end_num': 100, 'thread': 9,
                 'close_flag': False,
                 'pile_flag': True,
                 'use_lego_urdf': False,
                 'try_grasp_flag': True,
                 'test_pile_detection': False,
                 'save_img_flag': True,
                 'init_pos_range': [[0.13, 0.17], [-0.03, 0.03], [0.01, 0.02]],
                 'max_box_num': 5, 'min_box_num': 4,
                 'is_render': True,
                 'box_range': [[0.016, 0.048], [0.016], [0.01, 0.02]],
                 'gripper_threshold': 0.004, 'gripper_force': 0.3, 'gripper_sim_step': 10,
                 'move_threshold': 0.005, 'move_force': 1.5,
                 'box_lateral_friction': 1, 'box_spinning_friction': 1, 'box_rolling_friction': 0.001,
                 'box_linear_damping': 1, 'box_angular_damping': 1, 'box_joint_damping': 0,
                 'box_restitution': 0, 'box_contact_damping': 0.001, 'box_contact_stiffness': 10000,
                 'gripper_lateral_friction': 1, 'gripper_spinning_friction': 1, 'gripper_rolling_friction': 0.001,
                 'gripper_linear_damping': 1, 'gripper_angular_damping': 1, 'gripper_joint_damping': 1,
                 'gripper_restitution': 0, 'gripper_contact_damping': 10, 'gripper_contact_stiffness': 100000,
                 'base_lateral_friction': 1, 'base_spinning_friction': 1, 'base_rolling_friction': 0,
                 'base_restitution': 0, 'base_contact_damping': 0.001, 'base_contact_stiffness': 100000,
                 'dataset_path': '/home/ubuntu/Desktop/knolling_dataset/'}

    # 'C:/Users/24356/Desktop/knolling_dataset/'
    # np.random.seed(65)
    # random.seed(65)

    startnum = para_dict['start_num']
    endnum =   para_dict['end_num']
    thread = para_dict['thread']
    CLOSE_FLAG = para_dict['close_flag']
    pile_flag = para_dict['pile_flag']
    use_lego_urdf = para_dict['use_lego_urdf']
    try_grasp_flag = para_dict['try_grasp_flag']
    test_pile_detection = para_dict['test_pile_detection']
    save_img_flag = para_dict['save_img_flag']

    init_pos_range = para_dict['init_pos_range']

    if try_grasp_flag == True:
        data_root = para_dict['dataset_path'] + 'grasp_pile_712_lab/'
        with open(para_dict['dataset_path'] + 'grasp_pile_712_lab_readme.txt', "w") as f:
            for key, value in para_dict.items():
                f.write(key + ': ')
                f.write(str(value) + '\n')
    else:
        data_root = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/yolo_pile_overlap_627_test/'
    os.makedirs(data_root, exist_ok=True)

    max_box_num = para_dict['max_box_num']
    min_box_num = para_dict['min_box_num']
    mm2px = 530 / 0.34

    env = Arm_env(max_step=1, is_render=para_dict['is_render'], endnum=endnum, save_img_flag=save_img_flag,
                  urdf_path='/home/ubuntu/Desktop/Knolling_bot_2/urdf/', init_pos_range=init_pos_range)
    os.makedirs(data_root + 'origin_images/', exist_ok=True)
    os.makedirs(data_root + 'origin_labels/', exist_ok=True)
    if try_grasp_flag == True:
        exist_img_num = startnum
        while True:
            num_item = int(np.random.uniform(min_box_num, max_box_num + 1))
            img_per_epoch = env.reset_table(close_flag=CLOSE_FLAG, use_lego_urdf=use_lego_urdf, data_root=data_root,
                                             num_item=num_item, thread=thread, epoch=exist_img_num,
                                             pile_flag=pile_flag,
                                             try_grasp_flag=try_grasp_flag)
            exist_img_num += img_per_epoch
    else:
        conf_crowded_total = []
        conf_normal_total = []
        for epoch in tqdm(range(startnum,endnum)):
            # num_item = random.randint(1, 5)
            num_item = int(np.random.uniform(min_box_num, max_box_num + 1))
            num_2x2 = np.random.randint(1, 5)
            num_2x3 = np.random.randint(1, 5)
            num_2x4 = np.random.randint(1, 5)
            lego_list = np.array([num_2x2, num_2x3, num_2x4])

            if test_pile_detection == False:
                state, lw_list, new_num_list = env.reset_table(close_flag=CLOSE_FLAG, use_lego_urdf=use_lego_urdf, data_root=data_root,
                                             lego_list=lego_list, num_item=num_item, thread=thread, epoch=epoch, pile_flag=pile_flag, try_grasp_flag=try_grasp_flag)
                my_im2 = env.get_image()[:, :, :3]
                if len(state) == 0:
                    cv2.imwrite(os.path.join(data_root, 'origin_images/%012d.png') % epoch, my_im2)
                    np.savetxt(os.path.join(data_root, "origin_labels/%012d.txt" % epoch), np.array([]), fmt='%.04f')
                else:
                    label = np.zeros((new_num_list, 10))
                    all_pos = state[:, :3]
                    all_ori = state[:, 3:]
                    corner_list = []
                    for j in range(new_num_list):
                        if j >= new_num_list:
                            element = np.zeros(9)
                            # element = np.append(element, 0)
                            label.append(element)
                        else:
                            pos = all_pos[j]
                            ori = all_ori[j]
                            element = np.concatenate(([1], pos, lw_list[j], ori))
                        label[j] = element
                    cv2.imwrite(os.path.join(data_root, 'origin_images/%012d.png') % epoch, my_im2)
                    np.savetxt(os.path.join(data_root, "origin_labels/%012d.txt" % epoch), label, fmt='%.04f')
            else:
                conf_test, lw_list, new_num_list = env.reset_table(close_flag=CLOSE_FLAG, use_lego_urdf=use_lego_urdf,
                                                               data_root=data_root,
                                                               lego_list=lego_list, num_item=num_item, thread=thread,
                                                               epoch=epoch, pile_flag=pile_flag,
                                                               try_grasp_flag=try_grasp_flag, test_pile_detection=test_pile_detection)
                conf_crowded_total.append(conf_test[0])
                conf_normal_total.append(conf_test[1])
        conf_crowded_total = np.mean(np.asarray(conf_crowded_total))
        conf_normal_total = np.mean(np.asarray(conf_normal_total))
        print('this is conf crowded total', conf_crowded_total)
        print('this is conf normal total', conf_normal_total)