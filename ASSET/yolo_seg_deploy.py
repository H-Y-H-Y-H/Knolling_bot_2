import os

import torch
from utils import *
import cv2
import pyrealsense2 as rs
import numpy as np
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

class Yolo_seg_model():

    def __init__(self, para_dict):

        self.mm2px = 530 / 0.34
        self.px2mm = 0.34 / 530
        self.para_dict = para_dict

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
                           scaled_xylw=None, mask=None, cls=None, conf=None, use_xylw=True, truth_flag=None):

        ############### zzz plot parameters ###############
        zzz_lw = 1
        tf = 1 # font thickness
        z_mm_center = 0.006
        ############### zzz plot parameters ###############

        ################### zzz plot mask ####################
        red_overlay = np.zeros_like(im)
        red_overlay[:, :, 2] = 255  # Set the red channel to 255 (full red), other channels to 0

        # Resize the mask array to match the image dimensions
        resized_mask = cv2.resize(mask, (im.shape[1], im.shape[0]))

        # Create a mask with the red overlay only in the "special area"
        special_area_mask = np.stack([resized_mask] * 3, axis=-1)  # Convert to 3-channel mask
        special_area_mask = special_area_mask.astype(np.uint8)
        special_area_mask[:, :, :] = special_area_mask[:, :, :] * red_overlay

        # Blend the original image and the red overlay in the "special area"
        im = cv2.addWeighted(im, 1, special_area_mask, 0.5, 0)
        ################### zzz plot mask ####################

        ############### zzz plot the box ###############
        if isinstance(box, torch.Tensor):
            box = box.cpu().detach().numpy()
        p1 = np.array([int(box[0] * 640), int(box[1] * 480)])
        plot_x = np.copy((scaled_xylw[1] * 480 - 6) / self.mm2px)
        plot_y = np.copy((scaled_xylw[0] * 640 - 320) / self.mm2px)
        label1 = 'cls: %d, conf: %.5f' % (cls, conf)
        label2 = 'index: %d, x: %.4f, y: %.4f' % (index, plot_x, plot_y)
        box_center = np.array([plot_x, plot_y])
        if label:
            w, h = cv2.getTextSize(label, 0, fontScale=zzz_lw / 3, thickness=tf)[0]  # text width, z_mm_center
            outside = p1[1] - h >= 3
            im = cv2.putText(im, label1, (p1[0] - 50, p1[1] + 22 if outside else p1[1] + h + 2),
                             0, zzz_lw / 3, (0, 0, 255), thickness=tf, lineType=cv2.LINE_AA)
            im = cv2.putText(im, label2, (p1[0] - 50, p1[1] + 32 if outside else p1[1] + h + 12),
                             0, zzz_lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
            im = cv2.circle(im, (int(scaled_xylw[0] * 640), int(scaled_xylw[1] * 480)), 1, (255, 0, 255), -1, lineType=cv2.LINE_AA)
        ############### zzz plot the box ###############

        result = np.concatenate((box_center, [z_mm_center]))

        return im, result

    def yolo_seg_predict(self, real_flag=False, img_path=None, img=None, target=None, boxes_num=None, height_data=None, test_pile_detection=None):

        if real_flag == True:
            model = self.para_dict['yolo_model_path']

            cap = cv2.VideoCapture(8)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # set the resolution width
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

            total_pred_result = []
            while True:
                # Wait for a coherent pair of frames: depth and color
                ret, resized_color_image = cap.read()
                args = dict(model=model, source=resized_color_image, conf=0.3, iou=0.8)
                from ultralytics import YOLO
                images = YOLO(model)(**args)

                origin_img = cv2.imread(resized_color_image)
                use_xylw = False  # use lw or keypoints to export length and width
                one_img = images[0]

                pred_result = []
                pred_xylws = one_img.boxes.xywhn.cpu().detach().numpy()
                if len(pred_xylws) == 0:
                    continue
                pred_keypoints = one_img.keypoints.cpu().detach().numpy()
                pred_keypoints[:, :, :2] = pred_keypoints[:, :, :2] / np.array([640, 480])
                pred_keypoints = pred_keypoints.reshape(len(pred_xylws), -1)
                pred_cls = one_img.boxes.cls.cpu().detach().numpy()
                pred_conf = one_img.boxes.conf.cpu().detach().numpy()
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
                ############ fill the rest of result with zeros if the number of result is less than 10 #############
                if len(pred_result) < boxes_num:
                    pred_result = np.concatenate((pred_result, np.zeros((int(boxes_num - len(pred_result)), pred_result.shape[1]))), axis=0)
                print('this is result\n', pred_result)
                ############ fill the rest of result with zeros if the number of result is less than 10 #############
                total_pred_result.append(pred_result)

                cv2.namedWindow('zzz', 0)
                cv2.resizeWindow('zzz', 1280, 960)
                cv2.imshow('zzz', origin_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    img_path_output = img_path + '_pred.png'
                    cv2.imwrite(img_path_output, origin_img)
                    break
            total_pred_result = np.asarray(total_pred_result)
            # pred_result = np.concatenate((np.mean(total_pred_result, axis=0), np.max(total_pred_result[:, :, 2:4], axis=0),
            #                               np.mean(total_pred_result[:, :, 4], axis=0).reshape(-1, 1)), axis=1)
            # pred_result = np.mean(total_pred_result, axis=0)
            # print(pred_result)

        else:
            model = self.para_dict['yolo_model_path']

            cv2.imwrite(img_path + '.png', img)
            img_path_input = img_path + '.png'
            args = dict(model=model, source=img_path_input, conf=self.para_dict['yolo_conf'], iou=self.para_dict['yolo_iou'], device=self.para_dict['device'])
            from ultralytics import YOLO
            images = YOLO(model)(**args)

            origin_img = cv2.imread(img_path_input)
            use_xylw = False # use lw or keypoints to export length and width
            one_img = images[0]

            pred_result = []
            pred_mask = one_img.masks.data.cpu().detach().numpy()
            pred_xylws = one_img.boxes.xywhn.cpu().detach().numpy()
            if len(pred_xylws) == 0:
                return [], []
            else:
                pred_cls = one_img.boxes.cls.cpu().detach().numpy()
                pred_conf = one_img.boxes.conf.cpu().detach().numpy()


            mm2px = 530 / 0.34
            x_px_center = pred_xylws[:, 1] * 480
            y_px_center = pred_xylws[:, 0] * 640
            mm_center = np.concatenate(
                (((x_px_center - 6) / mm2px).reshape(-1, 1), ((y_px_center - 320) / mm2px).reshape(-1, 1)), axis=1)
            pred_order = change_sequence(mm_center)

            # pred = pred[pred_order]
            pred_xylws = pred_xylws[pred_order]
            # pred_keypoints = pred_keypoints[pred_order]
            pred_cls = pred_cls[pred_order]
            pred_conf = pred_conf[pred_order]
            print('this is the pred order', pred_order)

            for j in range(len(pred_xylws)):
                # print('this is pred xylw', pred_xylw)
                origin_img, result = self.plot_and_transform(im=origin_img, box=pred_xylws[j], label='0:, predict', color=(0, 0, 0), txt_color=(255, 255, 255),
                                                             index=j, scaled_xylw=pred_xylws[j], mask=pred_mask[j],
                                                             cls=pred_cls[j], conf=pred_conf[j],
                                                             use_xylw=use_xylw, truth_flag=False)
                pred_result.append(result)

            self.img_output = origin_img
            if self.para_dict['save_img_flag'] == True:
                cv2.namedWindow('zzz', 0)
                cv2.resizeWindow('zzz', 1280, 960)
                cv2.imshow('zzz', origin_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                img_path_output = img_path + '_pred.png'
                cv2.imwrite(img_path_output, origin_img)
            pred_result = np.asarray(pred_result)

        return pred_result, pred_conf

    def plot_result(self, result_one_img):

        img = result_one_img.orig_img
        masks = result_one_img.masks
        cls_idx = result_one_img.boxes.cls.cpu().detach().numpy()
        name_dict = result_one_img.names
        cls_name = []

        ############### zzz plot parameters ###############
        zzz_lw = 1
        tf = 1 # font thickness
        ############### zzz plot parameters ###############

        ################### zzz plot mask ####################
        highlight_overlay = np.zeros_like(img)
        highlight_overlay[:, :, 2] = 255  # Set the red channel to 255 (full red), other channels to 0
        for i in range(len(masks)):
            # Resize the mask array to match the image dimensions
            resized_mask = cv2.resize(masks[i].data.squeeze().cpu().detach().numpy(), (img.shape[1], img.shape[0]))

            # Create a mask with the red overlay only in the "special area"
            special_area_mask = np.stack([resized_mask] * 3, axis=-1)  # Convert to 3-channel mask
            special_area_mask = special_area_mask.astype(np.uint8)
            special_area_mask[:, :, :] = special_area_mask[:, :, :] * highlight_overlay

            # Blend the original image and the red overlay in the "special area"
            img = cv2.addWeighted(img, 1, special_area_mask, 0.5, 0)
        ################### zzz plot mask ####################

        # ############### zzz plot the box ###############
        # if isinstance(box, torch.Tensor):
        #     box = box.cpu().detach().numpy()
        # p1 = np.array([int(box[0] * 640), int(box[1] * 480)])
        # plot_x = np.copy((scaled_xylw[1] * 480 - 6) / self.mm2px)
        # plot_y = np.copy((scaled_xylw[0] * 640 - 320) / self.mm2px)
        # label1 = 'cls: %d, conf: %.5f' % (cls, conf)
        # label2 = 'index: %d, x: %.4f, y: %.4f' % (index, plot_x, plot_y)
        # box_center = np.array([plot_x, plot_y])
        # if label:
        #     w, h = cv2.getTextSize(label, 0, fontScale=zzz_lw / 3, thickness=tf)[0]  # text width, z_mm_center
        #     outside = p1[1] - h >= 3
        #     im = cv2.putText(im, label1, (p1[0] - 50, p1[1] + 22 if outside else p1[1] + h + 2),
        #                      0, zzz_lw / 3, (0, 0, 255), thickness=tf, lineType=cv2.LINE_AA)
        #     im = cv2.putText(im, label2, (p1[0] - 50, p1[1] + 32 if outside else p1[1] + h + 12),
        #                      0, zzz_lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
        #     im = cv2.circle(im, (int(scaled_xylw[0] * 640), int(scaled_xylw[1] * 480)), 1, (255, 0, 255), -1, lineType=cv2.LINE_AA)
        # ############### zzz plot the box ###############

        # result = np.concatenate((box_center, [z_mm_center]))
        result = masks.data.squeeze().cpu().detach().numpy()

        return img, result

    def yolo_seg_dataset(self, start_index, end_index):

        model = self.para_dict['yolo_model_path']
        mask_dir = '../../knolling_dataset/yolo_seg_sundry_205/masks/'
        os.makedirs(mask_dir, exist_ok=True)

        for i in range(int(end_index - start_index)):
            image_source = self.para_dict['dataset_path'] + '%012d.png' % (start_index + i)
            args = dict(model=model, source=image_source, conf=self.para_dict['yolo_conf'],
                        iou=self.para_dict['yolo_iou'], device=self.para_dict['device'])
            from ultralytics import YOLO
            pre_images = YOLO(model)(**args)
            result_one_img = pre_images[0]

            new_img, mask_results = self.plot_result(result_one_img)

            np.save(mask_dir + '%012d.npy' % (start_index + i), arr=mask_results)

            # test = np.load(mask_dir + '%012d.npy' % (start_index + i))

            # cv2.namedWindow('zzz', 0)
            # cv2.resizeWindow('zzz', 1280, 960)
            # cv2.imshow('zzz', new_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

if __name__ == '__main__':

    para_dict = {'device': 'cuda:0', 'yolo_conf': 0.5, 'yolo_iou': 0.6,
                 'yolo_model_path': '../ASSET/models/205_seg_sundry/weights/best.pt',
                 'dataset_path': '../../knolling_dataset/yolo_seg_sundry_205/images/train/',
                 'index_begin': 44000}


    model_threshold_start = 0.3
    model_threshold_end = 0.8
    check_point = 10
    valid_num = 20000
    model_threshold = np.linspace(model_threshold_start, model_threshold_end, check_point)

    zzz_yolo = Yolo_seg_model(para_dict=para_dict)
    zzz_yolo.yolo_seg_dataset(start_index=0, end_index=3200)