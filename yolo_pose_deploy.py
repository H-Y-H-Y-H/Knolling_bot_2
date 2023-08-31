from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import numpy as np
import torch
import cv2
from function import *
import pyrealsense2 as rs

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

class Yolo_pose_model():

    def __init__(self, para_dict):

        self.mm2px = 530 / 0.34
        self.px2mm = 0.34 / 530
        self.para_dict = para_dict

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
        z_mm_center = 0.006

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

        # randomly exchange the length and width to make the yolo result match the expectation of the knolling model
        if np.random.rand() < 0.5:
            temp = length
            length = width
            width = temp
            # if my_ori > np.pi / 2:
            #     my_ori = my_ori - np.pi / 2
            # else:
            #     my_ori = my_ori + np.pi / 2

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

        # cv2.rectangle(self.im, p1, p2, color, thickness=zzz_lw, lineType=cv2.LINE_AA)
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

    def yolo_pose_predict(self, cfg=DEFAULT_CFG, real_flag=False, img_path=None, img=None, target=None, boxes_num=None, height_data=None, test_pile_detection=None):

        if real_flag == True:
            model = self.para_dict['yolo_model_path']
            pipeline = rs.pipeline()
            config = rs.config()

            # Get device product line for setting a supporting resolution
            pipeline_wrapper = rs.pipeline_wrapper(pipeline)
            pipeline_profile = config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()
            device_product_line = str(device.get_info(rs.camera_info.product_line))

            found_rgb = False
            for s in device.sensors:
                if s.get_info(rs.camera_info.name) == 'RGB Camera':
                    found_rgb = True
                    break
            if not found_rgb:
                print("The demo requires Depth camera with Color sensor")
                exit(0)

            # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
            # Start streaming
            pipeline.start(config)

            mean_floor = (160, 160, 160)
            origin_point = np.array([0, -0.20])

            total_pred_result = []
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                color_image = np.asanyarray(color_frame.get_data())
                color_colormap_dim = color_image.shape
                resized_color_image = color_image

                cv2.imwrite(img_path + '.png', resized_color_image)
                img_path_input = img_path + '.png'
                args = dict(model=model, source=img_path_input, conf=self.para_dict['yolo_conf'], iou=self.para_dict['yolo_iou'])
                use_python = True
                if use_python:
                    from ultralytics import YOLO
                    images = YOLO(model)(**args)
                else:
                    predictor = PosePredictor(overrides=args)
                    predictor.predict_cli()

                origin_img = cv2.imread(img_path_input)
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
                # ############ fill the rest of result with zeros if the number of result is less than 10 #############
                # if len(pred_result) < boxes_num:
                #     pred_result = np.concatenate((pred_result, np.zeros((int(boxes_num - len(pred_result)), pred_result.shape[1]))), axis=0)
                # print('this is result\n', pred_result)
                # ############ fill the rest of result with zeros if the number of result is less than 10 #############
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
            use_python = True
            if use_python:
                from ultralytics import YOLO
                images = YOLO(model)(**args)
            else:
                predictor = PosePredictor(overrides=args)
                predictor.predict_cli()

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


            ######## order based on distance to draw it on the image while deploying the model ########
            if self.para_dict['data_collection'] == False:
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

            if test_pile_detection == True:
                for j in range(len(target)):
                    tar_xylw = np.copy(target[j, 1:5])
                    tar_keypoints = np.copy((target[j, 5:]).reshape(-1, 3)[:, :2])

                    # plot target
                    print('this is tar xylw', tar_xylw)
                    origin_img, _ = self.plot_and_transform(im=origin_img, box=tar_xylw, label='0: target', color=(255, 255, 0), txt_color=(255, 255, 255),
                                                            index=j, scaled_xylw=tar_xylw, keypoints=tar_keypoints,
                                                            cls=0, conf=1,
                                                            use_xylw=use_xylw, truth_flag=True)

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

    def plot_grasp(self, manipulator_before, prediction, model_output):

        x_px_center = manipulator_before[:, 0] * self.mm2px + 6
        y_px_center = manipulator_before[:, 1] * self.mm2px + 320
        zzz_lw = 1
        tf = 1
        img_path = self.para_dict['dataset_path'] + 'origin_images/%012d' % (0)
        for i in range(len(manipulator_before)):
            if prediction[i] == 0:
                label = 'False %.03f' % model_output[i, 1]
            else:
                label = 'True %.03f' % model_output[i, 1]
            self.img_output = cv2.putText(self.img_output, label, (int(y_px_center[i]) - 10, int(x_px_center[i])),
                             0, zzz_lw / 3, (0, 255, 0), thickness=tf, lineType=cv2.LINE_AA)

        if self.para_dict['save_img_flag'] == True:
            cv2.namedWindow('zzz', 0)
            cv2.resizeWindow('zzz', 1280, 960)
            cv2.imshow('zzz', self.img_output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            img_path_output = img_path + '_pred_grasp.png'
            cv2.imwrite(img_path_output, self.img_output)