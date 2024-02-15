import os

import torch
from utils import *
import cv2
import pyrealsense2 as rs
import numpy as np
from ASSET.grasp_model_deploy import *
import json

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

        with open('./urdf/object_color/rgb_info.json') as f:
            self.color_dict = json.load(f)

    def adjust_box(self, box, center_image, factor):
        center_box = np.mean(box, axis=0)
        move_vector = center_image - center_box
        move_vector *= factor
        new_box = box + move_vector
        return new_box.astype(int)

    def find_rotated_bounding_box(self, mask_channel, center_image=np.array([320 ,240]), factor=0.01):
        contours, _ = cv2.findContours(mask_channel.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            adjusted_box = self.adjust_box(box, center_image, factor)
            return adjusted_box, rect
        return None, None

    def calculate_color(self, image, mask):

        mask = mask[:, :, 0].astype(bool)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masked_rgb = np.mean(image_rgb[mask], axis=0)
        return masked_rgb

    def plot_result(self, result_one_img, truth_flag=False):

        result = []

        ############### zzz plot parameters ###############
        txt_color = (255, 255, 255)
        box_color = (0, 0, 0)
        zzz_lw = 1
        tf = 1 # font thickness
        highlight_overlay = np.zeros_like(result_one_img.orig_img)
        highlight_overlay[:, :, 2] = 255  # Set the red channel to 255 (full red), other channels to 0
        ############### zzz plot parameters ###############

        img = result_one_img.orig_img
        masks = result_one_img.masks.data.squeeze().cpu().detach().numpy()
        pred_cls_idx = result_one_img.boxes.cls.cpu().detach().numpy()
        pred_conf = result_one_img.boxes.conf.cpu().detach().numpy()
        name_dict = result_one_img.names
        pred_xylws = result_one_img.boxes.xywhn.cpu().detach().numpy()

        ######## order based on distance to draw it on the image!!!
        x_px_center = pred_xylws[:, 1] * 480
        y_px_center = pred_xylws[:, 0] * 640
        mm_center = np.concatenate(
            (((x_px_center - 6) / self.mm2px).reshape(-1, 1), ((y_px_center - 320) / self.mm2px).reshape(-1, 1)), axis=1)
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
            pred_color_idx = np.argmin(np.linalg.norm(total_color_value - (pred_color_rgb / 255), axis=1))

            # Blend the original image and the red overlay in the "special area"
            special_area_mask[:, :, :] = special_area_mask[:, :, :] * highlight_overlay
            # img = cv2.addWeighted(img, 1, special_area_mask, 0.5, 0)

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
            if ori > np.pi:
                ori -= np.pi
            elif ori < 0:
                ori += np.pi

            # Plot the bounding box
            img = cv2.line(img, (correct_box[0, 0], correct_box[0, 1]), (correct_box[1, 0], correct_box[1, 1]), box_color, 1)
            img = cv2.line(img, (correct_box[1, 0], correct_box[1, 1]), (correct_box[2, 0], correct_box[2, 1]), box_color, 1)
            img = cv2.line(img, (correct_box[2, 0], correct_box[2, 1]), (correct_box[3, 0], correct_box[3, 1]), box_color, 1)
            img = cv2.line(img, (correct_box[3, 0], correct_box[3, 1]), (correct_box[0, 0], correct_box[0, 1]), box_color, 1)

            # plot label
            label1 = 'cls: %s, conf: %.5f, color: %s' % (pred_cls_name[i], pred_conf[i], total_color_name[pred_color_idx])
            label2 = 'index: %d, x: %.4f, y: %.4f' % (i, kpt_center[0], kpt_center[1])
            label3 = 'l: %.4f, w: %.4f, ori: %.4f' % (length_mm, width_mm, ori)
            w, h = cv2.getTextSize('', 0, fontScale=zzz_lw / 3, thickness=tf)[0]  # text width, z_mm_center
            p1 = np.array([int(pred_xylws[i, 0] * 640), int(pred_xylws[i, 1] * 480)])
            outside = p1[1] - h >= 3
            if truth_flag == True:
                txt_color = (0, 0, 255)
                pass
            else:
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

    def model_predict(self, start_index=0, end_index=0, use_dataset=False, img=None):

        model = self.para_dict['yolo_model_path']
        if self.para_dict['real_operate'] == False:

            if use_dataset == True:
                mask_dir = '../../knolling_dataset/yolo_seg_sundry_205/masks/'
                os.makedirs(mask_dir, exist_ok=True)
                for i in range(int(end_index - start_index)):
                    image_source = self.para_dict['dataset_path'] + '%012d.png' % (start_index + i)
                    args = dict(model=model, source=image_source, conf=self.para_dict['yolo_conf'],
                                iou=self.para_dict['yolo_iou'], device=self.para_dict['device'])
                    from ultralytics import YOLO
                    pre_images = YOLO(model)(**args)
                    if len(pre_images) == 0:
                        print('yolo no more than 1')
                        continue
                    else:
                        result_one_img = pre_images[0]
                        # results: x, y, length, width, ori, cls
                        new_img, results = self.plot_result(result_one_img)

                    cv2.namedWindow('zzz', 0)
                    cv2.resizeWindow('zzz', 1280, 960)
                    cv2.imshow('zzz', new_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                args = dict(model=model, source=img, conf=self.para_dict['yolo_conf'],
                            iou=self.para_dict['yolo_iou'], device=self.para_dict['device'])
                from ultralytics import YOLO
                pre_images = YOLO(model)(**args)
                result_one_img = pre_images[0]
                pred_xylws = result_one_img.boxes.xywhn.cpu().detach().numpy()
                if len(pred_xylws) == 0:
                    print('yolo no more than 1')
                    return []

                new_img, results = self.plot_result(result_one_img)

                cv2.namedWindow('zzz', 0)
                cv2.resizeWindow('zzz', 1280, 960)
                cv2.imshow('zzz', new_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            self.img_path = '../IMAGE/real_images/%012d' % (0)

            cap = cv2.VideoCapture(8)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # set the resolution width
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

            for i in range(10): # warm-up camera
                ret, resized_color_image = cap.read()

            while True:
                ret, resized_color_image = cap.read()
                resized_color_image = cv2.flip(resized_color_image, -1)

                # cv2.imwrite(img_path + '.png', resized_color_image)
                # img_path_input = img_path + '.png'
                args = dict(model=model, source=resized_color_image, conf=self.para_dict['yolo_conf'],
                            iou=self.para_dict['yolo_iou'], device=self.para_dict['device'])

                from ultralytics import YOLO
                pre_images = YOLO(model)(**args)
                result_one_img = pre_images[0]
                pred_xylws = result_one_img.boxes.xywhn.cpu().detach().numpy()
                if len(pred_xylws) == 0:
                    continue

                new_img, results = self.plot_result(result_one_img)

                cv2.namedWindow('zzz', 0)
                cv2.resizeWindow('zzz', 1280, 960)
                cv2.imshow('zzz', new_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    img_path_output = self.img_path + '_pred.png'
                    cv2.imwrite(img_path_output, new_img)
                    break

        manipulator_before = np.concatenate((results[:, :2], np.zeros((len(results), 3)), results[:, 4].reshape(-1, 1)), axis=1)
        pred_lwh_list = np.concatenate((results[:, 2:4], np.ones((len(results), 1)) * 0.016), axis=1)
        pred_cls = results[:, -3]
        pred_color = results[:, -2]
        pred_conf = results[:, -1]

        if self.para_dict['lstm_enable_flag'] == True:
            input_data = np.concatenate((manipulator_before[:, :2],
                                         pred_lwh_list[:, :2],
                                         manipulator_before[:, -1].reshape(-1, 1),
                                         pred_conf.reshape(-1, 1)), axis=1)
            prediction, model_output = self.grasp_model.pred_test(input_data)

            # yolo_baseline_threshold = 0.92
            # prediction = np.where(pred_conf < yolo_baseline_threshold, 0, 1)
            # model_output = np.concatenate((np.zeros((len(prediction), 1)), pred_conf.reshape(len(prediction), 1)), axis=1)
            print('this is prediction', prediction)
            self.plot_grasp(manipulator_before, prediction, model_output)

if __name__ == '__main__':

    para_dict = {'device': 'cuda:0', 'yolo_conf': 0.7, 'yolo_iou': 0.6, 'real_operate': False,
                 'yolo_model_path': '../ASSET/models/205_seg_sundry/weights/best.pt',
                 'dataset_path': '../../knolling_dataset/yolo_seg_sundry_205/images/val/',
                 'index_begin': 44000}


    model_threshold_start = 0.3
    model_threshold_end = 0.8
    check_point = 10
    valid_num = 20000
    model_threshold = np.linspace(model_threshold_start, model_threshold_end, check_point)

    zzz_yolo = Yolo_seg_model(para_dict=para_dict)
    zzz_yolo.model_predict(start_index=3200, end_index=4000, use_dataset=True)