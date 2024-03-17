# from ASSET.ultralytics.yolo.engine.results import Results
# from ASSET.ultralytics.yolo.utils import ops
# from ASSET.ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2
import numpy as np
import json
from utils import *
from ASSET.grasp_model_deploy import *
import time

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
        model = self.para_dict['yolo_model_path']
        if self.para_dict['real_operate'] == True:
            self.img_path = self.para_dict['data_source_path'] + 'real_images/%012d' % (self.epoch)


            cap = cv2.VideoCapture(8)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # set the resolution width
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

            for i in range(10):  # Camera warm-up
                ret, resized_color_image = cap.read()

            while True:

                ret, resized_color_image = cap.read()
                resized_color_image = cv2.flip(resized_color_image, -1)

                # cv2.imwrite(img_path + '.png', resized_color_image)
                # img_path_input = img_path + '.png'
                args = dict(model=model, source=resized_color_image, conf=self.para_dict['yolo_conf'], iou=self.para_dict['yolo_iou'], device=self.para_dict['device'])

                from ultralytics import YOLO
                images = YOLO(model)(**args)

                origin_img = np.copy(resized_color_image)
                use_xylw = False  # use lw or keypoints to export length and width
                one_img = images[0]

                pred_result = []
                if len(one_img) == 0:
                    continue
                pred_xylws = one_img.boxes.xywhn.cpu().detach().numpy()[:gt_boxes_num]
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
            # if self.para_dict['save_img_flag'] == True and first_flag == True:
            self.img_path = self.para_dict['data_source_path'] + 'sim_images/%012s' % (self.epoch)
            # cv2.namedWindow('zzz', 0)
            # cv2.resizeWindow('zzz', 1280, 960)
            # cv2.imshow('zzz', origin_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            img_path_output = self.img_path + '.png'
            # cv2.imwrite(img_path_output, img)

            args = dict(model=model, source=img, conf=self.para_dict['yolo_conf'], iou=self.para_dict['yolo_iou'], device=self.yolo_device)
            from ultralytics import YOLO
            images = YOLO(model)(**args)

            # origin_img = cv2.imread(img_path_input)
            origin_img = np.copy(img)
            use_xylw = False # use lw or keypoints to export length and width
            one_img = images[0]

            pred_result = []
            if gt_boxes_num is None:

                if len(one_img) <= 1:  # filter the results no more than 1
                    print('yolo no more than 1')
                    if self.para_dict['lstm_enable_flag'] == True:
                        return [], [], []
                    else:
                        return [], [], []
                else:
                    pred_xylws = one_img.boxes.xywhn.cpu().detach().numpy()
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
        img_path_output = self.para_dict['data_source_path'] + 'unstack_rays/%012d' % (output_epoch) + '_grasp.png'
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

        # img_path_output = self.para_dict['data_source_path'] + 'sim_images/%012d' % (output_epoch) + '_grasp.png'
        img_path_output = self.img_path + '_grasp.png'
        # img_path_output = self.para_dict['data_source_path'] + 'unstack_images/%012d' % (output_epoch) + '_grasp.png'
        cv2.imwrite(img_path_output, output_img)
        pass

class Yolo_seg_model():

    def __init__(self, para_dict, lstm_dict=False, use_lstm=False):

        self.mm2px = 530 / 0.34
        self.px2mm = 0.34 / 530
        self.para_dict = para_dict
        self.yolo_device = self.para_dict['device']
        print('this is yolo seg device', self.yolo_device)

        if use_lstm == False:
            pass
        else:
            self.grasp_model = Grasp_model(para_dict=para_dict, lstm_dict=lstm_dict)

        try:
            with open('./ASSET/urdf/object_color/rgb_info.json') as f:
                self.color_dict = json.load(f)
        except FileNotFoundError:
            with open('./urdf/object_color/rgb_info.json') as f:
                self.color_dict = json.load(f)

    def adjust_box(self, box, center_image, factor):
        center_box = np.mean(box, axis=0)
        move_vector = center_image - center_box
        move_vector *= factor
        new_box = box + move_vector
        return new_box.astype(int)

    def find_rotated_bounding_box(self, mask_channel, center_image=np.array([320, 240]), factor=0.01):
        contours, _ = cv2.findContours(mask_channel.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            adjusted_box = self.adjust_box(box, center_image, factor)
            return adjusted_box, rect
        return None, None

    def calculate_color(self, image, mask):

        kernel = np.ones((10, 10), np.uint8)

        # Apply erosion to shrink the object's margin
        mask = cv2.erode(mask[:, :, 0], kernel, iterations=1).astype(bool)

        # mask = mask[:, :, 0].astype(bool)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masked_rgb = np.mean(image_rgb[mask], axis=0) / 255
        return masked_rgb

    def plot_result(self, result_one_img, info_flag=False):

        result = []

        ############### zzz plot parameters ###############
        txt_color = (255, 255, 255)
        box_color = (0, 0, 0)
        zzz_lw = 1
        tf = 1  # font thickness
        highlight_overlay = np.zeros_like(result_one_img.orig_img)
        highlight_overlay[:, :, 2] = 255  # Set the red channel to 255 (full red), other channels to 0
        ############### zzz plot parameters ###############

        img = result_one_img.orig_img
        masks = result_one_img.masks.data.cpu().detach().numpy()
        pred_cls_idx = result_one_img.boxes.cls.cpu().detach().numpy()
        pred_conf = result_one_img.boxes.conf.cpu().detach().numpy()
        name_dict = result_one_img.names
        pred_xylws = result_one_img.boxes.xywhn.cpu().detach().numpy()

        ######## order based on distance to draw it on the image!!!
        x_px_center = pred_xylws[:, 1] * 480
        y_px_center = pred_xylws[:, 0] * 640
        mm_center = np.concatenate(
            (((x_px_center - 6) / self.mm2px).reshape(-1, 1), ((y_px_center - 320) / self.mm2px).reshape(-1, 1)),
            axis=1)
        pred_order = change_sequence(mm_center)
        pred_cls_idx = pred_cls_idx[pred_order]
        pred_cls_name = [name_dict[i] for i in pred_cls_idx]
        pred_conf = pred_conf[pred_order]
        pred_xylws = pred_xylws[pred_order]
        masks = masks[pred_order]
        ######## order based on distance to draw it on the image!!!

        ################### zzz plot mask ####################
        for i in range(len(masks)):
            # Resize the mask array to match the image dimensions
            resized_mask = cv2.resize(masks[i], (img.shape[1], img.shape[0]))

            # Create a mask with the red overlay only in the "special area"
            special_area_mask = np.stack([resized_mask] * 3, axis=-1)  # Convert to 3-channel mask
            special_area_mask = special_area_mask.astype(np.uint8)

            # calculate color in the special area
            pred_color_rgb = self.calculate_color(image=img, mask=special_area_mask)
            total_color_name = []
            total_color_value = []
            for color_name, value in self.color_dict.items():
                total_color_value.append(np.mean(np.asarray(value), axis=0))
                total_color_name.append(color_name)
            total_color_value = np.asarray(total_color_value)
            pred_color_idx = np.argmin(np.linalg.norm(total_color_value - pred_color_rgb, axis=1))

            # Blend the original image and the red overlay in the "special area"
            if info_flag == False:
                special_area_mask[:, :, :] = special_area_mask[:, :, :] * highlight_overlay
                img = cv2.addWeighted(img, 1, special_area_mask, 0.2, 0)

            correct_box, rect = self.find_rotated_bounding_box(masks[i])
            kpt_x = ((correct_box[:, 1] - 6) / self.mm2px).reshape(-1, 1)
            kpt_y = ((correct_box[:, 0] - 320) / self.mm2px).reshape(-1, 1)
            kpt_mm = np.concatenate((kpt_x, kpt_y), axis=1)
            kpt_center = np.average(kpt_mm, axis=0)

            # calculate length and width
            length_mm = max(np.linalg.norm(kpt_mm[0] - kpt_mm[1]), np.linalg.norm(kpt_mm[0] - kpt_mm[-1]))
            width_mm = min(np.linalg.norm(kpt_mm[0] - kpt_mm[1]), np.linalg.norm(kpt_mm[0] - kpt_mm[-1]))

            # calcualte ori
            length_idx = np.argmax([np.linalg.norm(kpt_mm[0] - kpt_mm[1]), np.linalg.norm(kpt_mm[0] - kpt_mm[-1])])
            temp = np.where(length_idx == 1, [-1], [1])[0]
            ori = np.arctan2(kpt_mm[temp, 1] - kpt_mm[0, 1], kpt_mm[temp, 0] - kpt_mm[0, 0])
            # the output of np.arctan2 is (pi / 2, -pi / 2)
            if ori > np.pi / 2:
                ori -= np.pi
            elif ori < -np.pi / 2:
                ori += np.pi

            # Plot the bounding box
            if info_flag == False:
                img = cv2.line(img, (correct_box[0, 0], correct_box[0, 1]), (correct_box[1, 0], correct_box[1, 1]),
                               box_color, 1)
                img = cv2.line(img, (correct_box[1, 0], correct_box[1, 1]), (correct_box[2, 0], correct_box[2, 1]),
                               box_color, 1)
                img = cv2.line(img, (correct_box[2, 0], correct_box[2, 1]), (correct_box[3, 0], correct_box[3, 1]),
                               box_color, 1)
                img = cv2.line(img, (correct_box[3, 0], correct_box[3, 1]), (correct_box[0, 0], correct_box[0, 1]),
                               box_color, 1)
            else:
                # plot label
                label1 = 'cls: %s, conf: %.5f, color: %s' % (pred_cls_name[i], pred_conf[i], total_color_name[pred_color_idx])
                # label1 = 'cls: %s, conf: %.5f' % (pred_cls_name[i], pred_conf[i])

                label2 = 'index: %d, x: %.4f, y: %.4f' % (i, kpt_center[0], kpt_center[1])
                label3 = 'l: %.4f, w: %.4f, ori: %.4f' % (length_mm, width_mm, ori)
                w, h = cv2.getTextSize('', 0, fontScale=zzz_lw / 3, thickness=tf)[0]  # text width, z_mm_center
                p1 = np.array([int(pred_xylws[i, 0] * 640), int(pred_xylws[i, 1] * 480)])
                outside = p1[1] - h >= 3

                img = cv2.putText(img, label1, (p1[0] - 50, p1[1] + 22 if outside else p1[1] + h + 2),
                                  0, zzz_lw / 3, (0, 0, 0), thickness=tf, lineType=cv2.LINE_AA)
                img = cv2.putText(img, label2, (p1[0] - 50, p1[1] + 32 if outside else p1[1] + h + 12),
                                  0, zzz_lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
                img = cv2.putText(img, label3, (p1[0] - 50, p1[1] + 42 if outside else p1[1] + h + 22),
                                  0, zzz_lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)

            result.append([kpt_center[0], kpt_center[1], length_mm, width_mm, ori, pred_cls_idx[i], pred_color_idx, pred_conf[i]])

        ################### zzz plot mask ####################
        result = np.asarray(result)

        return img, result

    def model_predict(self, start_index=0, end_index=0, use_dataset=False,
                      img=None, epoch=0, gt_boxes_num=None, show_mask=False, capture_video=False):

        from ultralytics import YOLO
        self.epoch = epoch
        model = self.para_dict['yolo_model_path']
        if self.para_dict['real_operate'] == False:

            args = dict(model=model, source=img, conf=self.para_dict['yolo_conf'],
                        iou=self.para_dict['yolo_iou'], device=self.para_dict['device'], agnostic_nms=True)
            from ultralytics import YOLO
            pre_images = YOLO(model)(**args)
            result_one_img = pre_images[0][:gt_boxes_num]
            if len(result_one_img) == 0:
                print('yolo no more than 1')
                return []

            if show_mask == False:
                new_img, results = self.plot_result(result_one_img, info_flag=True)
            else:
                new_img, results = self.plot_result(result_one_img)

            manipulator_before = np.concatenate((results[:, :2], np.zeros((len(results), 3)), results[:, 4].reshape(-1, 1)), axis=1)
            pred_lwh_list = np.concatenate((results[:, 2:4], np.ones((len(results), 1)) * 0.016), axis=1)
            pred_cls = results[:, -3]
            pred_color = results[:, -2]
            pred_conf = results[:, -1]
            self.img_output = img

            # if self.para_dict['lstm_enable_flag'] == True and show_mask == False:
            #     input_data = np.concatenate((manipulator_before[:, :2],
            #                                  pred_lwh_list[:, :2],
            #                                  manipulator_before[:, -1].reshape(-1, 1),
            #                                  pred_conf.reshape(-1, 1)), axis=1)
            #     pred_grasp, model_output = self.grasp_model.pred_test(input_data)
            #
            #     # yolo_baseline_threshold = 0.92
            #     # prediction = np.where(pred_conf < yolo_baseline_threshold, 0, 1)
            #     # model_output = np.concatenate((np.zeros((len(prediction), 1)), pred_conf.reshape(len(prediction), 1)), axis=1)
            #     print('this is prediction', pred_grasp)
            #     self.plot_grasp(manipulator_before, pred_grasp, model_output)

            cv2.namedWindow('zzz', 0)
            cv2.resizeWindow('zzz', 1280, 960)
            cv2.imshow('zzz', new_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            self.img_path = self.para_dict['data_source_path'] + 'real_images/%012d' % (epoch)

            cap = cv2.VideoCapture(8)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # set the resolution width
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

            if capture_video == True:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = 9#int(cap.get(cv2.CAP_PROP_FPS))

                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out_origin = cv2.VideoWriter('origin_video.avi', fourcc, fps, (w, h))
                out_landmark = cv2.VideoWriter('landmark_video.avi', fourcc, fps, (w, h))
                out_info = cv2.VideoWriter('info_video.avi', fourcc, fps, (w, h))
                print('capture warmed up, recording start!')

            for i in range(10):  # Camera warm-up
                ret, resized_color_image = cap.read()
            count = 0
            while True:
                time0 = time.time()
                ret, resized_color_image = cap.read()
                resized_color_image = cv2.flip(resized_color_image, -1)

                args = dict(model=model, source=resized_color_image, conf=self.para_dict['yolo_conf'],
                            iou=self.para_dict['yolo_iou'], device=self.para_dict['device'], agnostic_nms=True, verbose=False)


                pre_images = YOLO(model)(**args)
                result_one_img = pre_images[0][:gt_boxes_num]
                origin_img = np.copy(result_one_img.orig_img)
                if len(result_one_img) == 0:
                    continue

                landmark_img, results = self.plot_result(result_one_img)
                info_img, results = self.plot_result(result_one_img, info_flag=True)
                # if show_mask == False:
                #     new_img, results = self.plot_result(result_one_img)
                # else:
                #     new_img, results = self.plot_result(result_one_img, mask_flag=True)

                manipulator_before = np.concatenate((results[:, :2], np.zeros((len(results), 3)), results[:, 4].reshape(-1, 1)), axis=1)
                pred_lwh_list = np.concatenate((results[:, 2:4], np.ones((len(results), 1)) * 0.016), axis=1)
                pred_cls = results[:, -3]
                pred_color = results[:, -2]
                pred_conf = results[:, -1]
                self.img_output = landmark_img

                if self.para_dict['lstm_enable_flag'] == True:
                    input_ori = np.copy(manipulator_before[:, -1].reshape(-1, 1))
                    for ori in input_ori:
                        if ori > np.pi:
                            ori -= np.pi
                        elif ori < 0:
                            ori += np.pi
                    input_data = np.concatenate((manipulator_before[:, :2],
                                                 pred_lwh_list[:, :2],
                                                 input_ori,
                                                 pred_conf.reshape(-1, 1)), axis=1)
                    pred_grasp, model_output = self.grasp_model.pred_test(input_data)

                    # yolo_baseline_threshold = 0.92
                    # prediction = np.where(pred_conf < yolo_baseline_threshold, 0, 1)
                    # model_output = np.concatenate((np.zeros((len(prediction), 1)), pred_conf.reshape(len(prediction), 1)), axis=1)
                    print('this is prediction', pred_grasp)
                    self.plot_grasp(manipulator_before, pred_grasp, model_output)

                if capture_video == True:
                    # print(count)

                    # key = cv2.waitKey(1) & 0xFF
                    # if key == ord('q'):
                    #     break
                    # elif key == ord('s'):
                    #     print('start recording')
                    #     recording = not recording  # Toggle recording state
                    # elif key == ord('p'):
                    #     print('pause recording')
                    #     recording = not recording  # Toggle recording state

                    out_origin.write(origin_img)
                    out_landmark.write(landmark_img)
                    out_info.write(info_img)
                    t1 = time.time()
                    print(1/(t1-time0))
                else:
                    cv2.namedWindow('zzz_landmark', 0)
                    cv2.resizeWindow('zzz_landmark', 1280, 960)
                    cv2.imshow('zzz_landmark', landmark_img)
                    cv2.namedWindow('zzz_info', 0)
                    cv2.resizeWindow('zzz_info', 1280, 960)
                    cv2.imshow('zzz_info', info_img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('s'):
                        cv2.destroyAllWindows()
                        img_path_landmark = self.img_path + '_pred_landmark.png'
                        cv2.imwrite(img_path_landmark, landmark_img)
                        img_path_info = self.img_path + '_pred_info.png'
                        cv2.imwrite(img_path_info, info_img)
                        break

        return manipulator_before, pred_lwh_list, pred_cls, pred_color, None

    def plot_grasp(self, manipulator_before, prediction, model_output, img=None, epoch=None):

        x_px_center = manipulator_before[:, 0] * self.mm2px + 6
        y_px_center = manipulator_before[:, 1] * self.mm2px + 320
        zzz_lw = 1
        tf = 1

        if img is None:
            output_img = self.img_output
        else:
            output_img = img

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

        # img_path_output = self.para_dict['data_source_path'] + 'sim_images/%012d' % (output_epoch) + '_grasp.png'
        img_path_output = self.para_dict['data_source_path'] + 'sim_images/grasp.png'
        # img_path_output = self.para_dict['data_source_path'] + 'unstack_images/%012d' % (output_epoch) + '_grasp.png'
        cv2.imwrite(img_path_output, output_img)

    def evaluation(self, start_index, end_index):

        none_output = 0
        h_distance = []
        center_offset = []
        category_accuracy = []

        model = self.para_dict['yolo_model_path']

        for i in range(int(end_index - start_index)):
            image_source = self.para_dict['dataset_path'] + 'images/val/%012d.png' % (start_index + i)
            with open(self.para_dict['dataset_path'] + 'labels/val/%012d.txt' % (start_index + i)) as f:
                gt_label = f.readlines()

            args = dict(model=model, source=image_source, conf=self.para_dict['yolo_conf'],
                        iou=self.para_dict['yolo_iou'], device=self.para_dict['device'], agnostic_nms=True)
            from ultralytics import YOLO
            pre_results = YOLO(model)(**args)
            if len(pre_results) == 0:
                print('yolo no more than 1')
                none_output += 1
                continue
            else:
                result_one_img = pre_results[0]
                # results: x, y, length, width, ori, cls
                weighted_h_distance_per_img, center_offset_per_img, category_accuracy_per_img = self.eval_metrics(result_one_img, gt_label)
                h_distance.append(weighted_h_distance_per_img)
                center_offset.append(center_offset_per_img)
                category_accuracy.append(category_accuracy_per_img)
                new_img, results = self.plot_result(result_one_img)

            # cv2.namedWindow('zzz', 0)
            # cv2.resizeWindow('zzz', 1280, 960)
            # cv2.imshow('zzz', new_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        h_distance = np.asarray(h_distance)
        m_h_distance = np.mean(h_distance)
        center_offset = np.asarray(center_offset)
        m_center_offset = np.mean(center_offset)
        category_accuracy = np.asarray(category_accuracy)
        m_category_accuracy = np.mean(category_accuracy)

        log_path = self.para_dict['yolo_model_path'][:-15] + '/evaluate_log/'
        os.makedirs(log_path, exist_ok=True)
        with open(log_path + "log_mask.txt", "w") as f:
            f.write(f'Dataset: {self.para_dict["dataset_path"]}\n')
            f.write(f'mean hamming distance: {m_h_distance}\n')
            f.write(f'mean center offset: {m_center_offset}\n')
            f.write(f'mean category accuracy: {m_category_accuracy}\n')

            for i in range(int(end_index - start_index)):
                f.write(f'index: {start_index + i}, hamming distance: {h_distance[i]}, center offset: {center_offset[i]}, category accuracy: {category_accuracy[i]}\n')

    def eval_metrics(self, results, gt_label):

        results = results[:len(gt_label)]
        img = results.orig_img
        masks = results.masks.data.cpu().detach().numpy()
        pred_cls_idx = results.boxes.cls.cpu().detach().numpy()
        pred_conf = results.boxes.conf.cpu().detach().numpy()
        name_dict = results.names
        pred_xylws = results.boxes.xywhn.cpu().detach().numpy()

        ######## order based on distance to draw it on the image!!!
        x_px_center = pred_xylws[:, 1] * 480
        y_px_center = pred_xylws[:, 0] * 640
        mm_center = np.concatenate(
            (((x_px_center - 6) / self.mm2px).reshape(-1, 1), ((y_px_center - 320) / self.mm2px).reshape(-1, 1)),
            axis=1)
        # pred_order = change_sequence(mm_center)
        # pred_cls_idx = pred_cls_idx[pred_order]
        pred_cls_name = [name_dict[i] for i in pred_cls_idx]
        # pred_conf = pred_conf[pred_order]
        # pred_xylws = pred_xylws[pred_order]
        # masks = masks[pred_order]

        total_h_distance = []
        total_center_offset = []
        total_paired_num = 0

        gt_masks = np.zeros((len(gt_label), 480, 640))
        gt_cls = []
        for i in range(len(gt_label)):
            processed_label = np.asarray(gt_label[i].split(' '), dtype=np.float32)
            gt_cls.append(processed_label[0])
            gt_contour = processed_label[1:].reshape(-1, 2)
            gt_contour[:, 1] *= 480  # from ratio to pixel
            gt_contour[:, 0] *= 640  # from ratio to pixel
            gt_contour = gt_contour.astype(np.int32)
            cv2.fillPoly(gt_masks[i], [gt_contour], color=1)
        gt_cls = np.asarray(gt_cls)
        for i in range(len(masks)):

            # evaluate the similarity between gt and pred mask via hamming distance
            weighted_h_distance = np.min(np.count_nonzero(masks[i] != gt_masks, axis=(1, 2))) / np.count_nonzero(masks[i])
            total_h_distance.append(weighted_h_distance)

            # evaluate the precision of the center
            correct_box, rect = self.find_rotated_bounding_box(masks[i])
            kpt_x = ((correct_box[:, 1] - 6) / self.mm2px).reshape(-1, 1)
            kpt_y = ((correct_box[:, 0] - 320) / self.mm2px).reshape(-1, 1)
            kpt_mm = np.concatenate((kpt_x, kpt_y), axis=1)
            kpt_center = np.average(kpt_mm, axis=0)

            # evaluate category accuracy
            paired_index = np.argmin(np.count_nonzero(masks[i] != gt_masks, axis=(1, 2)))
            selected_gt_cls = gt_cls[paired_index]
            if selected_gt_cls == pred_cls_idx[i]:
                print(f'success pair {selected_gt_cls}, {name_dict[selected_gt_cls]}')
                total_paired_num += 1

            # cv2.namedWindow('zzz', 0)
            # cv2.resizeWindow('zzz', 1280, 960)
            # cv2.imshow('zzz', gt_mask)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        category_accuracy = total_paired_num / len(gt_label)

        h_distance_per_img = np.mean(np.asarray(total_h_distance))
        # center_offset_per_img = np.mean(np.asarray(total_center_offset))
        center_offset_per_img = 0
        return h_distance_per_img, center_offset_per_img, category_accuracy

if __name__ == '__main__':
    para_dict = {'device': 'cuda:0', 'yolo_conf': 0.5, 'yolo_iou': 0.6, 'real_operate': True,
                 'yolo_model_path': '../ASSET/models/301_seg_real_sundry/weights/best.pt',
                 'dataset_path': '../../knolling_dataset/yolo_seg_sim_sundry_301/',
                 'index_begin': 44000,
                 'data_source_path': '../IMAGE/',
                 'lstm_enable_flag': False}

    # model_threshold_start = 0.3
    # model_threshold_end = 0.8
    # check_point = 10
    # valid_num = 20000
    # model_threshold = np.linspace(model_threshold_start, model_threshold_end, check_point)

    zzz_yolo = Yolo_seg_model(para_dict=para_dict)
    zzz_yolo.model_predict(start_index=3200, end_index=4000, use_dataset=False, epoch=0, capture_video=False)
    # zzz_yolo.model_predict(start_index=3200, end_index=4000, use_dataset=False, show_mask=True, epoch=1)
    # zzz_yolo.evaluation(start_index=3200, end_index=4000)