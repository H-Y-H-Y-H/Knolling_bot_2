# from ASSET.ultralytics.yolo.engine.results import Results
# from ASSET.ultralytics.yolo.utils import ops
# from ASSET.ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2
import numpy as np

from utils import *
import pyrealsense2 as rs
from ASSET.grasp_model_deploy import *


class Yolo_pose_model():

    def __init__(self, para_dict, lstm_dict=False, use_lstm=False):

        self.mm2px = 530 / 0.34
        self.px2mm = 0.34 / 530
        self.para_dict = para_dict
        if use_lstm == False:
            pass
        else:
            self.grasp_model = Grasp_model(para_dict=para_dict, lstm_dict=lstm_dict)

        self.yolo_device = self.para_dict['device']
        print('this is yolo pose device', self.yolo_device)

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
        # x_mm_center = scaled_xylw[1] * 0.3
        # y_mm_center = scaled_xylw[0] * 0.4 - 0.2
        # x_px_center = x_mm_center * mm2px + 6
        # y_px_center = y_mm_center * mm2px + 320
        x_px_center = scaled_xylw[1] * 480
        y_px_center = scaled_xylw[0] * 640
        # z_mm_center = height_data[int(x_px_center), int(y_px_center)] - 0.01
        # if z_mm_center <= 0.006:
        #     z_mm_center = 0.006
        z_mm_center = 0.00

        # this is the knolling sequence, not opencv!!!!
        keypoints_x = ((keypoints[:, 1] * 480 - 6) / self.mm2px).reshape(-1, 1)
        keypoints_y = ((keypoints[:, 0] * 640 - 320) / self.mm2px).reshape(-1, 1)
        keypoints_mm = np.concatenate((keypoints_x, keypoints_y), axis=1)
        keypoints_center = np.average(keypoints_mm, axis=0)

        length = max(np.linalg.norm(keypoints_mm[0] - keypoints_mm[-1]),
                     np.linalg.norm(keypoints_mm[1] - keypoints_mm[2]))
        width = min(np.linalg.norm(keypoints_mm[0] - keypoints_mm[-1]),
                    np.linalg.norm(keypoints_mm[1] - keypoints_mm[2]))
        c1 = np.array([length / (2), width / (2)])
        c2 = np.array([length / (2), -width / (2)])
        c3 = np.array([-length / (2), width / (2)])
        c4 = np.array([-length / (2), -width / (2)])
        if use_xylw == True:
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

        # randomly exchange the length and width to make the yolo result match the expectation of the knolling model

        # # 不要随便交换长宽，grasp model会寄掉！！！！！！
        # if np.random.rand() < 0.5:
        #     # temp = length
        #     # length = width
        #     # width = temp
        #     # if my_ori > np.pi / 2:
        #     #     return_ori = my_ori - np.pi / 2
        #     # else:
        #     #     return_ori = my_ori + np.pi / 2
        #     pass
        # else:
        #     pass
        # # 不要随便交换长宽，grasp model会寄掉！！！！！！

        if my_ori > np.pi:
            my_ori -= np.pi
        elif my_ori < 0:
            my_ori += np.pi

        rot_z = [[np.cos(my_ori), -np.sin(my_ori)],
                 [np.sin(my_ori), np.cos(my_ori)]]
        corn1 = (np.dot(rot_z, c1)) * self.mm2px
        corn2 = (np.dot(rot_z, c2)) * self.mm2px
        corn3 = (np.dot(rot_z, c3)) * self.mm2px
        corn4 = (np.dot(rot_z, c4)) * self.mm2px

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

        im = cv2.line(im, (int(corn1[1]), int(corn1[0])), (int(corn2[1]), int(corn2[0])), color, 1)
        im = cv2.line(im, (int(corn2[1]), int(corn2[0])), (int(corn4[1]), int(corn4[0])), color, 1)
        im = cv2.line(im, (int(corn4[1]), int(corn4[0])), (int(corn3[1]), int(corn3[0])), color, 1)
        im = cv2.line(im, (int(corn3[1]), int(corn3[0])), (int(corn1[1]), int(corn1[0])), color, 1)
        plot_x = np.copy((scaled_xylw[1] * 480 - 6) / self.mm2px)
        plot_y = np.copy((scaled_xylw[0] * 640 - 320) / self.mm2px)
        plot_l = np.copy(length)
        plot_w = np.copy(width)
        label1 = 'cls: %d, conf: %.5f' % (cls, conf)
        label2 = 'index: %d, x: %.4f, y: %.4f' % (index, plot_x, plot_y)
        label3 = 'l: %.4f, w: %.4f, ori: %.4f' % (plot_l, plot_w, my_ori)
        if label:
            w, h = cv2.getTextSize(label, 0, fontScale=zzz_lw / 3, thickness=tf)[0]  # text width, z_mm_center
            outside = p1[1] - h >= 3
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
                im = cv2.putText(im, label3, (p1[0] - 50, p1[1] + 42 if outside else p1[1] + h + 22),
                                0, zzz_lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
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

    def model_predict(self, first_flag=False, real_flag=False, img=None, target=None, gt_boxes_num=None, test_pile_detection=None, epoch=0):

        self.epoch = epoch
        if real_flag == True:
            self.img_path = self.para_dict['data_source_path'] + 'real_images/%012d' % (self.epoch)
            model = self.para_dict['yolo_model_path']

            # # initiate the realsense
            # pipeline = rs.pipeline()
            # config = rs.config()
            #
            # # Get device product line for setting a supporting resolution
            # pipeline_wrapper = rs.pipeline_wrapper(pipeline)
            # pipeline_profile = config.resolve(pipeline_wrapper)
            # device_camera = pipeline_profile.get_device()
            # device_product_line = str(device_camera.get_info(rs.camera_info.product_line))
            #
            # found_rgb = False
            # for s in device_camera.sensors:
            #     if s.get_info(rs.camera_info.name) == 'RGB Camera':
            #         found_rgb = True
            #         break
            # if not found_rgb:
            #     print("The demo requires Depth camera with Color sensor")
            #     exit(0)
            #
            # # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            # # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
            # # Start streaming
            # pipeline.start(config)
            #
            # origin_point = np.array([0, -0.20])
            #
            # while True:
            #     # Wait for a coherent pair of frames: depth and color
            #     frames = pipeline.wait_for_frames()
            #     color_frame = frames.get_color_frame()
            #     color_image = np.asanyarray(color_frame.get_data())
            #     resized_color_image = color_image
            #     resized_color_image = cv2.flip(resized_color_image, -1)

            cap = cv2.VideoCapture(8)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # set the resolution width
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

            while True:

                ret, resized_color_image = cap.read()

                # cv2.imwrite(img_path + '.png', resized_color_image)
                # img_path_input = img_path + '.png'
                args = dict(model=model, source=resized_color_image, conf=self.para_dict['yolo_conf'], iou=self.para_dict['yolo_iou'], device=self.para_dict['device'])
                use_python = True
                if use_python:
                    from ultralytics import YOLO
                    images = YOLO(model)(**args)

                origin_img = np.copy(resized_color_image)
                use_xylw = False  # use lw or keypoints to export length and width
                one_img = images[0]

                pred_result = []
                pred_xylws = one_img.boxes.xywhn.cpu().detach().numpy()[:gt_boxes_num]
                if len(pred_xylws) == 0:
                    continue
                pred_keypoints = one_img.keypoints.cpu().detach().numpy()[:gt_boxes_num]
                pred_keypoints[:, :, :2] = pred_keypoints[:, :, :2] / np.array([640, 480])
                pred_keypoints = pred_keypoints.reshape(len(pred_xylws), -1)
                pred_cls = one_img.boxes.cls.cpu().detach().numpy()[:gt_boxes_num]
                pred_conf = one_img.boxes.conf.cpu().detach().numpy()[:gt_boxes_num]
                pred = np.concatenate((np.zeros((len(pred_xylws), 1)), pred_xylws, pred_keypoints), axis=1)

                ######## order based on distance to draw it on the image!!!
                mm2px = 530 / 0.34
                x_px_center = pred_xylws[:, 1] * 480
                y_px_center = pred_xylws[:, 0] * 640
                mm_center = np.concatenate((((x_px_center - 6) / mm2px).reshape(-1, 1), ((y_px_center - 320) / mm2px).reshape(-1, 1)), axis=1)
                pred_order = change_sequence(mm_center)
                pred = pred[pred_order]
                pred_xylws = pred_xylws[pred_order]
                pred_conf = pred_conf[pred_order]
                pred_keypoints = pred_keypoints[pred_order]
                ######## order based on distance to draw it on the image!!!

                for j in range(len(pred_xylws)):
                    pred_keypoint = pred_keypoints[j].reshape(-1, 3)
                    pred_xylw = pred_xylws[j]

                    origin_img, result = self.plot_and_transform(im=origin_img, box=pred_xylw, label='0:, predic', color=(0, 0, 0),
                                                            txt_color=(255, 255, 255), index=j, cls=pred_cls[j], conf=pred_conf[j],
                                                            scaled_xylw=pred_xylw, keypoints=pred_keypoint, use_xylw=use_xylw,
                                                            truth_flag=False)
                    pred_result.append(result)
                pred_result = np.asarray(pred_result)

                self.img_output = origin_img

                manipulator_before = np.concatenate((pred_result[:, :3], np.zeros((len(pred_result), 2)), pred_result[:, 5].reshape(-1, 1)), axis=1)
                new_lwh_list = np.concatenate((pred_result[:, 3:5], np.ones((len(pred_result), 1)) * 0.016), axis=1)

                if self.para_dict['lstm_enable_flag'] == True:
                    input_data = np.concatenate((manipulator_before[:, :2],
                                                 new_lwh_list[:, :2],
                                                 manipulator_before[:, -1].reshape(-1, 1),
                                                 pred_conf.reshape(-1, 1)), axis=1)
                    prediction, model_output = self.grasp_model.pred_test(input_data)

                    # yolo_baseline_threshold = 0.92
                    # prediction = np.where(pred_conf < yolo_baseline_threshold, 0, 1)
                    # model_output = np.concatenate((np.zeros((len(prediction), 1)), pred_conf.reshape(len(prediction), 1)), axis=1)
                    print('this is prediction', prediction)
                    self.plot_grasp(manipulator_before, prediction, model_output)

                cv2.namedWindow('zzz', 0)
                cv2.resizeWindow('zzz', 1280, 960)
                cv2.imshow('zzz', origin_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    img_path_output = self.img_path + '_pred.png'
                    cv2.imwrite(img_path_output, origin_img)
                    break

        else:
            model = self.para_dict['yolo_model_path']

            if self.para_dict['save_img_flag'] == True and first_flag == True:
                self.img_path = self.para_dict['data_tar_path'] + 'sim_images/%012d' % (self.epoch)
                # cv2.namedWindow('zzz', 0)
                # cv2.resizeWindow('zzz', 1280, 960)
                # cv2.imshow('zzz', origin_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                img_path_output = self.img_path + '.png'
                # cv2.imwrite(img_path_output, img)

            args = dict(model=model, source=img, conf=self.para_dict['yolo_conf'], iou=self.para_dict['yolo_iou'], device=self.yolo_device)
            use_python = True
            if use_python:
                from ultralytics import YOLO
                images = YOLO(model)(**args)

            # origin_img = cv2.imread(img_path_input)
            origin_img = np.copy(img)
            use_xylw = False # use lw or keypoints to export length and width
            one_img = images[0]

            pred_result = []
            if gt_boxes_num is None:
                pred_xylws = one_img.boxes.xywhn.cpu().detach().numpy()
                if len(pred_xylws) <= 1:  # filter the results no more than 1
                    print('yolo no more than 1')
                    if self.para_dict['lstm_enable_flag'] == True:
                        return [], [], []
                    else:
                        return [], [], []
                else:
                    pred_cls = one_img.boxes.cls.cpu().detach().numpy()
                    pred_conf = one_img.boxes.conf.cpu().detach().numpy()
                    pred_keypoints = one_img.keypoints.cpu().detach().numpy()
                    pred_keypoints[:, :, :2] = pred_keypoints[:, :, :2] / np.array([640, 480])
                    pred_keypoints = pred_keypoints.reshape(len(pred_xylws), -1)
                    pred = np.concatenate((np.zeros((len(pred_xylws), 1)), pred_xylws, pred_keypoints), axis=1)
            else:
                pred_xylws = one_img.boxes.xywhn.cpu().detach().numpy()[:gt_boxes_num]
                if len(pred_xylws) <= 1: # filter the results no more than 1
                    print('yolo no more than 1')
                    if self.para_dict['lstm_enable_flag'] == True:
                        return [], [], []
                    else:
                        return [], [], []
                else:
                    pred_cls = one_img.boxes.cls.cpu().detach().numpy()[:gt_boxes_num]
                    pred_conf = one_img.boxes.conf.cpu().detach().numpy()[:gt_boxes_num]
                    pred_keypoints = one_img.keypoints.cpu().detach().numpy()[:gt_boxes_num]
                    pred_keypoints[:, :, :2] = pred_keypoints[:, :, :2] / np.array([640, 480])
                    pred_keypoints = pred_keypoints.reshape(len(pred_xylws), -1)
                    pred = np.concatenate((np.zeros((len(pred_xylws), 1)), pred_xylws, pred_keypoints), axis=1)


            ######## order based on distance to draw it on the image while deploying the model ########
            # if self.para_dict['Data_collection'] == False:
            mm2px = 530 / 0.34
            x_px_center = pred_xylws[:, 1] * 480
            y_px_center = pred_xylws[:, 0] * 640
            mm_center = np.concatenate(
                (((x_px_center - 6) / mm2px).reshape(-1, 1), ((y_px_center - 320) / mm2px).reshape(-1, 1)), axis=1)
            pred_order = change_sequence(mm_center)

            pred = pred[pred_order]
            pred_xylws = pred_xylws[pred_order]
            pred_keypoints = pred_keypoints[pred_order]
            pred_cls = pred_cls[pred_order]
            pred_conf = pred_conf[pred_order]
            print('this is the pred order', pred_order)

            for j in range(len(pred_xylws)):

                pred_keypoint = pred_keypoints[j].reshape(-1, 3)
                pred_xylw = pred_xylws[j]

                # print('this is pred xylw', pred_xylw)
                origin_img, result = self.plot_and_transform(im=origin_img, box=pred_xylw, label='0:, predic', color=(0, 0, 0), txt_color=(255, 255, 255),
                                                             index=j, scaled_xylw=pred_xylw, keypoints=pred_keypoint,
                                                             cls=pred_cls[j], conf=pred_conf[j],
                                                             use_xylw=use_xylw, truth_flag=False)
                pred_result.append(result)

            pred_result = np.asarray(pred_result)
            self.img_output = origin_img

            manipulator_before = np.concatenate((pred_result[:, :3], np.zeros((len(pred_result), 2)), pred_result[:, 5].reshape(-1, 1)), axis=1)
            new_lwh_list = np.concatenate((pred_result[:, 3:5], np.ones((len(pred_result), 1)) * 0.016), axis=1)

            if self.para_dict['lstm_enable_flag'] == True:
                input_data = np.concatenate((manipulator_before[:, :2],
                                             new_lwh_list[:, :2],
                                             manipulator_before[:, -1].reshape(-1, 1),
                                             pred_conf.reshape(-1, 1)), axis=1)
                prediction, model_output = self.grasp_model.pred_test(input_data)


                print('this is prediction', prediction)
                self.plot_grasp(manipulator_before, prediction, model_output)

        if self.para_dict['lstm_enable_flag'] == True:
            return manipulator_before, new_lwh_list, prediction
        else:
            return manipulator_before, new_lwh_list, pred_conf

    def plot_unstack(self, success_ray, img=None, epoch=None):

        start_pos = success_ray[0, :2]
        start_px = [int(start_pos[0] * self.mm2px + 6), int(start_pos[1] * self.mm2px + 320)]
        end_pos = success_ray[1, :2]
        end_px = [int(end_pos[0] * self.mm2px + 6), int(end_pos[1] * self.mm2px + 320)]

        if img is None:
            output_img = self.img_output
            output_epoch = self.epoch
        else:
            output_img = img
            output_epoch = epoch

        output_img = cv2.line(output_img, (start_px[1], start_px[0]), (end_px[1], end_px[0]), (255, 0, 0), 1)
        img_path_output = self.para_dict['data_tar_path'] + 'unstack_rays/%012d' % (output_epoch) + '_grasp.png'
        cv2.imwrite(img_path_output, output_img)

    def plot_grasp(self, manipulator_before, prediction, model_output, img=None, epoch=None):

        x_px_center = manipulator_before[:, 0] * self.mm2px + 6
        y_px_center = manipulator_before[:, 1] * self.mm2px + 320
        zzz_lw = 1
        tf = 1

        if img is None:
            output_img = self.img_output
            output_epoch = self.epoch
        else:
            output_img = img
            output_epoch = epoch

        for i in range(len(manipulator_before)):
            if prediction[i] == 0:
                label = 'False %.03f' % model_output[i, 1]
            else:
                label = 'True %.03f' % model_output[i, 1]
            output_img = cv2.putText(output_img, label, (int(y_px_center[i]) - 10, int(x_px_center[i])),
                             0, zzz_lw / 3, (0, 255, 0), thickness=tf, lineType=cv2.LINE_AA)

        # cv2.namedWindow('zzz', 0)
        # cv2.resizeWindow('zzz', 1280, 960)
        # cv2.imshow('zzz', output_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # img_path_output = self.para_dict['data_tar_path'] + 'sim_images/%012d' % (output_epoch) + '_grasp.png'
        img_path_output = self.para_dict['data_source_path'] + 'sim_images/grasp.png'
        # img_path_output = self.para_dict['data_tar_path'] + 'unstack_images/%012d' % (output_epoch) + '_grasp.png'
        cv2.imwrite(img_path_output, output_img)
        pass

# class Yolo_seg_model():
#
#     def __init__(self, para_dict):
#
#         self.mm2px = 530 / 0.34
#         self.px2mm = 0.34 / 530
#         self.para_dict = para_dict
#
#         self.yolo_device = self.para_dict['device']
#         print('this is yolo seg device', self.yolo_device)
#
#     def plot_and_transform(self, im, box, label='', color=(0, 0, 0), txt_color=(255, 255, 255), index=None,
#                            scaled_xylw=None, keypoints=None, cls=None, conf=None, use_xylw=True, truth_flag=None,
#                            height_data=None):
#
#         ############### zzz plot parameters ###############
#         zzz_lw = 1
#         tf = 1  # font thickness
#         # x_mm_center = scaled_xylw[1] * 0.3
#         # y_mm_center = scaled_xylw[0] * 0.4 - 0.2
#         # x_px_center = x_mm_center * mm2px + 6
#         # y_px_center = y_mm_center * mm2px + 320
#         x_px_center = scaled_xylw[1] * 480
#         y_px_center = scaled_xylw[0] * 640
#         # z_mm_center = height_data[int(x_px_center), int(y_px_center)] - 0.01
#         # if z_mm_center <= 0.006:
#         #     z_mm_center = 0.006
#         z_mm_center = 0.006
#
#         # this is the knolling sequence, not opencv!!!!
#         keypoints_x = ((keypoints[:, 1] - 6) / self.mm2px).reshape(-1, 1)
#         keypoints_y = ((keypoints[:, 0] - 320) / self.mm2px).reshape(-1, 1)
#         keypoints_mm = np.concatenate((keypoints_x, keypoints_y), axis=1)
#         keypoints_center = np.average(keypoints_mm, axis=0)
#
#         length = max(np.linalg.norm(keypoints_mm[0] - keypoints_mm[-1]),
#                      np.linalg.norm(keypoints_mm[1] - keypoints_mm[2]))
#         width = min(np.linalg.norm(keypoints_mm[0] - keypoints_mm[-1]),
#                     np.linalg.norm(keypoints_mm[1] - keypoints_mm[2]))
#         c1 = np.array([length / (2), width / (2)])
#         c2 = np.array([length / (2), -width / (2)])
#         c3 = np.array([-length / (2), width / (2)])
#         c4 = np.array([-length / (2), -width / (2)])
#         if use_xylw == True:
#             box_center = np.array([scaled_xylw[0], scaled_xylw[1]])
#         else:
#             box_center = keypoints_center
#
#         all_distance = np.linalg.norm((keypoints_mm - keypoints_center), axis=1)
#         k = 2
#         max_index = all_distance.argsort()[-k:]
#         lucky_keypoint_index = np.argmax([keypoints_mm[max_index[0], 1], keypoints_mm[max_index[1], 1]])
#         lucky_keypoint = keypoints_mm[max_index[lucky_keypoint_index]]
#         # print('the ori keypoint is ', keypoints_mm[max_index[lucky_keypoint_index]])
#         my_ori = np.arctan2(lucky_keypoint[1] - keypoints_center[1], lucky_keypoint[0] - keypoints_center[0])
#         # In order to grasp, this ori is based on the longest side of the box, not the label ori!
#
#         # randomly exchange the length and width to make the yolo result match the expectation of the knolling model
#
#         if my_ori > np.pi:
#             my_ori -= np.pi
#         elif my_ori < 0:
#             my_ori += np.pi
#
#         rot_z = [[np.cos(my_ori), -np.sin(my_ori)],
#                  [np.sin(my_ori), np.cos(my_ori)]]
#         corn1 = (np.dot(rot_z, c1)) * self.mm2px
#         corn2 = (np.dot(rot_z, c2)) * self.mm2px
#         corn3 = (np.dot(rot_z, c3)) * self.mm2px
#         corn4 = (np.dot(rot_z, c4)) * self.mm2px
#
#         corn1 = [corn1[0] + x_px_center, corn1[1] + y_px_center]
#         corn2 = [corn2[0] + x_px_center, corn2[1] + y_px_center]
#         corn3 = [corn3[0] + x_px_center, corn3[1] + y_px_center]
#         corn4 = [corn4[0] + x_px_center, corn4[1] + y_px_center]
#         ############### zzz plot parameters ###############
#
#         ############### zzz plot the box ###############
#         if isinstance(box, torch.Tensor):
#             box = box.cpu().detach().numpy()
#         # print(box)
#         p1 = np.array([int(box[0] * 640), int(box[1] * 480)])
#         # print('this is p1 and p2', p1, p2)
#
#         im = cv2.line(im, (int(corn1[1]), int(corn1[0])), (int(corn2[1]), int(corn2[0])), color, 1)
#         im = cv2.line(im, (int(corn2[1]), int(corn2[0])), (int(corn4[1]), int(corn4[0])), color, 1)
#         im = cv2.line(im, (int(corn4[1]), int(corn4[0])), (int(corn3[1]), int(corn3[0])), color, 1)
#         im = cv2.line(im, (int(corn3[1]), int(corn3[0])), (int(corn1[1]), int(corn1[0])), color, 1)
#         plot_x = np.copy((scaled_xylw[1] * 480 - 6) / self.mm2px)
#         plot_y = np.copy((scaled_xylw[0] * 640 - 320) / self.mm2px)
#         plot_l = np.copy(length)
#         plot_w = np.copy(width)
#         label1 = 'cls: %d, conf: %.5f' % (cls, conf)
#         label2 = 'index: %d, x: %.4f, y: %.4f' % (index, plot_x, plot_y)
#         label3 = 'l: %.4f, w: %.4f, ori: %.4f' % (plot_l, plot_w, my_ori)
#         if label:
#             w, h = cv2.getTextSize(label, 0, fontScale=zzz_lw / 3, thickness=tf)[0]  # text width, z_mm_center
#             outside = p1[1] - h >= 3
#             if truth_flag == True:
#                 txt_color = (0, 0, 255)
#                 # im = cv2.putText(im, label1, (p1[0] - 50, p1[1] - 32 if outside else p1[1] + h + 2),
#                 #                  0, zzz_lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
#                 # im = cv2.putText(im, label2, (p1[0] - 50, p1[1] - 22 if outside else p1[1] + h + 12),
#                 #                  0, zzz_lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
#             else:
#                 im = cv2.putText(im, label1, (p1[0] - 50, p1[1] + 22 if outside else p1[1] + h + 2),
#                                  0, zzz_lw / 3, (0, 0, 255), thickness=tf, lineType=cv2.LINE_AA)
#                 im = cv2.putText(im, label2, (p1[0] - 50, p1[1] + 32 if outside else p1[1] + h + 12),
#                                  0, zzz_lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
#                 im = cv2.putText(im, label3, (p1[0] - 50, p1[1] + 42 if outside else p1[1] + h + 22),
#                                  0, zzz_lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
#         ############### zzz plot the box ###############
#
#         ############### zzz plot the keypoints ###############
#         shape = (640, 640)
#         radius = 1
#         for i, k in enumerate(keypoints):
#             if truth_flag == False:
#                 if i == 0:
#                     color_k = (255, 0, 0)
#                 else:
#                     color_k = (0, 0, 0)
#             elif truth_flag == True:
#                 if i == 0:
#                     color_k = (0, 0, 255)
#                 elif i == 3:
#                     color_k = (255, 255, 0)
#             x_coord, y_coord = k[0], k[1]
#             im = cv2.circle(im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)
#         ############### zzz plot the keypoints ###############
#
#         result = np.concatenate((box_center, [z_mm_center], [round(length, 4)], [round(width, 4)], [my_ori]))
#
#         return im, result
#
#     def model_predict(self, first_flag=False, real_flag=False, img=None, target=None, gt_boxes_num=None, test_pile_detection=None, epoch=0):
#
#         self.epoch = epoch
#         if real_flag == True:
#             self.img_path = self.para_dict['data_source_path'] + 'real_images/%012d' % (self.epoch)
#             # os.makedirs(img_path, exist_ok=True)
#             model = self.para_dict['yolo_model_path']
#             pipeline = rs.pipeline()
#             config = rs.config()
#
#             # Get device product line for setting a supporting resolution
#             pipeline_wrapper = rs.pipeline_wrapper(pipeline)
#             pipeline_profile = config.resolve(pipeline_wrapper)
#             device_camera = pipeline_profile.get_device()
#             device_product_line = str(device_camera.get_info(rs.camera_info.product_line))
#
#             found_rgb = False
#             for s in device_camera.sensors:
#                 if s.get_info(rs.camera_info.name) == 'RGB Camera':
#                     found_rgb = True
#                     break
#             if not found_rgb:
#                 print("The demo requires Depth camera with Color sensor")
#                 exit(0)
#
#             # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#             config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#             # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
#             # Start streaming
#             pipeline.start(config)
#
#             mean_floor = (160, 160, 160)
#             origin_point = np.array([0, -0.20])
#
#             while True:
#                 # Wait for a coherent pair of frames: depth and color
#                 frames = pipeline.wait_for_frames()
#                 color_frame = frames.get_color_frame()
#                 color_image = np.asanyarray(color_frame.get_data())
#                 color_colormap_dim = color_image.shape
#                 resized_color_image = color_image
#
#                 # cv2.imwrite(img_path + '.png', resized_color_image)
#                 # img_path_input = img_path + '.png'
#                 args = dict(model=model, source=resized_color_image, conf=self.para_dict['yolo_conf'],
#                             iou=self.para_dict['yolo_iou'], device=self.para_dict['device'])
#                 use_python = True
#                 if use_python:
#                     from ultralytics import YOLO
#                     images = YOLO(model)(**args)
#
#                 origin_img = np.copy(resized_color_image)
#                 use_xylw = False  # use lw or keypoints to export length and width
#                 one_img = images[0]
#
#                 pred_result = []
#                 pred_xylws = one_img.boxes.xywhn.cpu().detach().numpy()[:gt_boxes_num]
#                 if len(pred_xylws) == 0:
#                     continue
#                 pred_keypoints = one_img.keypoints.cpu().detach().numpy()[:gt_boxes_num]
#                 pred_keypoints[:, :, :2] = pred_keypoints[:, :, :2] / np.array([640, 480])
#                 pred_keypoints = pred_keypoints.reshape(len(pred_xylws), -1)
#                 pred_cls = one_img.boxes.cls.cpu().detach().numpy()[:gt_boxes_num]
#                 pred_conf = one_img.boxes.conf.cpu().detach().numpy()[:gt_boxes_num]
#                 pred = np.concatenate((np.zeros((len(pred_xylws), 1)), pred_xylws, pred_keypoints), axis=1)
#
#                 ######## order based on distance to draw it on the image!!!
#                 mm2px = 530 / 0.34
#                 x_px_center = pred_xylws[:, 1] * 480
#                 y_px_center = pred_xylws[:, 0] * 640
#                 mm_center = np.concatenate(
#                     (((x_px_center - 6) / mm2px).reshape(-1, 1), ((y_px_center - 320) / mm2px).reshape(-1, 1)), axis=1)
#                 pred_order = change_sequence(mm_center)
#                 pred = pred[pred_order]
#                 pred_xylws = pred_xylws[pred_order]
#                 pred_conf = pred_conf[pred_order]
#                 pred_keypoints = pred_keypoints[pred_order]
#                 ######## order based on distance to draw it on the image!!!
#
#                 for j in range(len(pred_xylws)):
#                     pred_keypoint = pred_keypoints[j].reshape(-1, 3)
#                     pred_xylw = pred_xylws[j]
#
#                     origin_img, result = self.plot_and_transform(im=origin_img, box=pred_xylw, label='0:, predic',
#                                                                  color=(0, 0, 0),
#                                                                  txt_color=(255, 255, 255), index=j, cls=pred_cls[j],
#                                                                  conf=pred_conf[j],
#                                                                  scaled_xylw=pred_xylw, keypoints=pred_keypoint,
#                                                                  use_xylw=use_xylw,
#                                                                  truth_flag=False)
#                     pred_result.append(result)
#                 pred_result = np.asarray(pred_result)
#
#                 self.img_output = origin_img
#
#                 manipulator_before = np.concatenate(
#                     (pred_result[:, :3], np.zeros((len(pred_result), 2)), pred_result[:, 5].reshape(-1, 1)), axis=1)
#                 new_lwh_list = np.concatenate((pred_result[:, 3:5], np.ones((len(pred_result), 1)) * 0.016), axis=1)
#
#                 cv2.namedWindow('zzz', 0)
#                 cv2.resizeWindow('zzz', 1280, 960)
#                 cv2.imshow('zzz', origin_img)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     cv2.destroyAllWindows()
#                     img_path_output = self.img_path + '_pred.png'
#                     cv2.imwrite(img_path_output, origin_img)
#                     break
#
#         else:
#             model = self.para_dict['yolo_model_path']
#
#             if self.para_dict['save_img_flag'] == True and first_flag == True:
#                 self.img_path = self.para_dict['data_tar_path'] + 'sim_images/%012d' % (self.epoch)
#                 # cv2.namedWindow('zzz', 0)
#                 # cv2.resizeWindow('zzz', 1280, 960)
#                 # cv2.imshow('zzz', origin_img)
#                 # cv2.waitKey(0)
#                 # cv2.destroyAllWindows()
#                 img_path_output = self.img_path + '.png'
#                 # cv2.imwrite(img_path_output, img)
#
#             args = dict(model=model, source=img, conf=self.para_dict['yolo_conf'], iou=self.para_dict['yolo_iou'],
#                         device=self.yolo_device)
#             use_python = True
#             if use_python:
#                 from ultralytics import YOLO
#                 images = YOLO(model)(**args)
#
#             # origin_img = cv2.imread(img_path_input)
#             origin_img = np.copy(img)
#             use_xylw = False  # use lw or keypoints to export length and width
#             one_img = images[0]
#
#             pred_result = []
#             if gt_boxes_num is None:
#                 pred_xylws = one_img.boxes.xywhn.cpu().detach().numpy()
#                 if len(pred_xylws) <= 1:  # filter the results no more than 1
#                     print('yolo no more than 1')
#                     return [], [], []
#                 else:
#                     pred_cls = one_img.boxes.cls.cpu().detach().numpy()
#                     pred_conf = one_img.boxes.conf.cpu().detach().numpy()
#                     pred_keypoints = one_img.keypoints.cpu().detach().numpy()
#                     pred_keypoints[:, :, :2] = pred_keypoints[:, :, :2] / np.array([640, 480])
#                     pred_keypoints = pred_keypoints.reshape(len(pred_xylws), -1)
#                     pred = np.concatenate((np.zeros((len(pred_xylws), 1)), pred_xylws, pred_keypoints), axis=1)
#             else:
#                 pred_xylws = one_img.boxes.xywhn.cpu().detach().numpy()[:gt_boxes_num]
#                 if len(pred_xylws) <= 1:  # filter the results no more than 1
#                     print('yolo no more than 1')
#                     return [], [], []
#                 else:
#                     pred_keypoints = one_img.masks.xy
#                     pred_cls = one_img.boxes.cls.cpu().detach().numpy()[:gt_boxes_num]
#                     pred_conf = one_img.boxes.conf.cpu().detach().numpy()[:gt_boxes_num]
#
#             ######## order based on distance to draw it on the image while deploying the model ########
#             # if self.para_dict['Data_collection'] == False:
#             mm2px = 530 / 0.34
#             x_px_center = pred_xylws[:, 1] * 480
#             y_px_center = pred_xylws[:, 0] * 640
#             mm_center = np.concatenate(
#                 (((x_px_center - 6) / mm2px).reshape(-1, 1), ((y_px_center - 320) / mm2px).reshape(-1, 1)), axis=1)
#             pred_order = change_sequence(mm_center)
#
#             pred_xylws = pred_xylws[pred_order]
#             pred_keypoints = [pred_keypoints[i] for i in pred_order]
#             pred_cls = pred_cls[pred_order]
#             pred_conf = pred_conf[pred_order]
#             print('this is the pred order', pred_order)
#
#             for j in range(len(pred_xylws)):
#                 pred_keypoint = pred_keypoints[j]
#                 pred_xylw = pred_xylws[j]
#                 origin_img, result = self.plot_and_transform(im=origin_img, box=pred_xylw, label='0:, predic',
#                                                              color=(0, 0, 0), txt_color=(255, 255, 255),
#                                                              index=j, scaled_xylw=pred_xylw, keypoints=pred_keypoint,
#                                                              cls=pred_cls[j], conf=pred_conf[j],
#                                                              use_xylw=use_xylw, truth_flag=False)
#                 pred_result.append(result)
#
#             pred_result = np.asarray(pred_result)
#             self.img_output = origin_img
#
#             cv2.namedWindow('zzz', 0)
#             cv2.resizeWindow('zzz', 1280, 960)
#             cv2.imshow('zzz', origin_img)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#
#             manipulator_before = np.concatenate(
#                 (pred_result[:, :3], np.zeros((len(pred_result), 2)), pred_result[:, 5].reshape(-1, 1)), axis=1)
#             new_lwh_list = np.concatenate((pred_result[:, 3:5], np.ones((len(pred_result), 1)) * 0.016), axis=1)
#
#         return pred_result, pred_conf
#
#     def plot_grasp(self, manipulator_before, prediction, model_output):
#
#         x_px_center = manipulator_before[:, 0] * self.mm2px + 6
#         y_px_center = manipulator_before[:, 1] * self.mm2px + 320
#         zzz_lw = 1
#         tf = 1
#         img_path = self.para_dict['dataset_path'] + 'origin_images/%012d' % (0)
#         for i in range(len(manipulator_before)):
#             if prediction[i] == 0:
#                 label = 'False %.03f' % model_output[i, 1]
#             else:
#                 label = 'True %.03f' % model_output[i, 1]
#             self.img_output = cv2.putText(self.img_output, label, (int(y_px_center[i]) - 10, int(x_px_center[i])),
#                              0, zzz_lw / 3, (0, 255, 0), thickness=tf, lineType=cv2.LINE_AA)
#
#         if self.para_dict['save_img_flag'] == True:
#             cv2.namedWindow('zzz', 0)
#             cv2.resizeWindow('zzz', 1280, 960)
#             cv2.imshow('zzz', self.img_output)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#             img_path_output = img_path + '_pred_grasp.png'
#             cv2.imwrite(img_path_output, self.img_output)

if __name__ == '__main__':

    zzz_yolo = Yolo_pose_model(None, None)
    zzz_yolo.yolo_pose_test()