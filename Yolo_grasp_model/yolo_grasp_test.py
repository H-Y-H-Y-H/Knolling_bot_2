from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import numpy as np
import torch
import cv2
from utils import *
import pyrealsense2 as rs
from models.grasp_model_deploy import *
import matplotlib.pyplot as plt

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

class Yolo_grasp_model():

    def __init__(self, para_dict):

        self.mm2px = 530 / 0.34
        self.px2mm = 0.34 / 530
        self.para_dict = para_dict

        self.pred_TP = 0
        self.pred_TN = 0
        self.pred_FP = 0
        self.pred_FN = 0

        self.yolo_device = self.para_dict['device']
        print('this is yolo device', self.yolo_device)

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

        if truth_flag == False:
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

        w, h = cv2.getTextSize(label, 0, fontScale=zzz_lw / 3, thickness=tf)[0]  # text width, z_mm_center
        outside = p1[1] - h >= 3
        if truth_flag == True:
            txt_color = (0, 0, 255)
            label = '%d' % cls
            im = cv2.putText(im, label, (int(y_px_center) - 5, int(x_px_center)),
                                     0, zzz_lw / 3, (0, 255, 0), thickness=tf, lineType=cv2.LINE_AA)
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
                x_coord, y_coord = k[0] * 640, k[1] * 480
                im = cv2.circle(im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)
        ############### zzz plot the keypoints ###############

        result = np.concatenate((box_center, [z_mm_center], [round(length, 4)], [round(width, 4)], [my_ori]))

        return im, result

    def calculate_confusion_matrix(self, pred_xylw, tar_xylw, pred_cls, tar_cls, pred_conf):

        # pair the tar and pred
        new_tar_xylw = []
        new_tar_cls = []
        for i in range(len(pred_xylw)):
            best_candidate = np.argmin(np.linalg.norm(pred_xylw[i] - tar_xylw, axis=1))
            new_tar_xylw.append(tar_xylw[best_candidate])
            new_tar_cls.append(tar_cls[best_candidate])
        new_tar_xylw = np.asarray(new_tar_xylw)
        new_tar_cls = np.asarray(new_tar_cls)

        for i in range(len(pred_xylw)):
            if new_tar_cls[i] == 1:
                if pred_cls[i] == 1:
                    self.pred_TP += 1
                elif pred_cls[i] == 0:
                    self.pred_TN += 1
            if new_tar_cls[i] == 0:
                if pred_cls[i] == 1:
                    self.pred_FP += 1
                elif pred_cls[i] == 0:
                    self.pred_FN += 1

    def yolo_grasp_predict(self, target=None, gt_boxes_num=None, test_pile_detection=None, epoch=0):

        self.pred_TP = 0
        self.pred_TN = 0
        self.pred_FP = 0
        self.pred_FN = 0

        self.epoch = epoch

        model = self.para_dict['yolo_model_path']
        source = self.para_dict['dataset_path'] + 'images/val_mini/'
        tar_path = self.para_dict['dataset_path'] + 'images/pred_mini/'
        os.makedirs(tar_path, exist_ok=True)
        args = dict(model=model, source=source, conf=self.para_dict['yolo_conf'], iou=self.para_dict['yolo_iou'], device=self.yolo_device)
        use_python = True
        if use_python:
            from ultralytics import YOLO
            images = YOLO(model)(**args)
        else:
            predictor = PosePredictor(overrides=args)
            predictor.predict_cli()

        use_xylw = False # use lw or keypoints to export length and width

        for i in range(len(images)):

            origin_img = cv2.imread(self.para_dict['dataset_path'] + 'images/val_mini/%012d.png' % int(i + self.para_dict['index_begin']))
            target = np.loadtxt(self.para_dict['dataset_path'] + 'labels/val_mini/%012d.txt' % int(i + self.para_dict['index_begin']))
            target_xylws = np.copy(target[:, 1:5])

            self.gt_num = len(target)
            one_img_result = images[i]

            pred_result = []
            pred_xylws = one_img_result.boxes.xywhn.cpu().detach().numpy()
            if len(pred_xylws) < self.gt_num:
                print('pred less that gt', int(self.gt_num - len(pred_xylws)))
                pred_cls = one_img_result.boxes.cls.cpu().detach().numpy()
                pred_conf = one_img_result.boxes.conf.cpu().detach().numpy()
                pred_keypoints = one_img_result.keypoints.cpu().detach().numpy()
                pred_keypoints[:, :, :2] = pred_keypoints[:, :, :2] / np.array([640, 480])
                pred_keypoints = pred_keypoints.reshape(len(pred_xylws), -1)
                pred = np.concatenate((np.zeros((len(pred_xylws), 1)), pred_xylws, pred_keypoints), axis=1)
            else:
                print('pred more that gt', int(len(pred_xylws) - self.gt_num))
                pred_xylws = pred_xylws[:self.gt_num]
                pred_cls = one_img_result.boxes.cls.cpu().detach().numpy()[:self.gt_num]
                pred_conf = one_img_result.boxes.conf.cpu().detach().numpy()[:self.gt_num]
                pred_keypoints = one_img_result.keypoints.cpu().detach().numpy()[:self.gt_num]
                pred_keypoints[:, :, :2] = pred_keypoints[:, :, :2] / np.array([640, 480])
                pred_keypoints = pred_keypoints.reshape(len(pred_xylws), -1)
                pred = np.concatenate((np.zeros((len(pred_xylws), 1)), pred_xylws, pred_keypoints), axis=1)

            ######## order based on distance to draw it on the image while deploying the model ########
            mm2px = 530 / 0.34
            x_px_center = pred_xylws[:, 1] * 480
            y_px_center = pred_xylws[:, 0] * 640
            mm_center = np.concatenate((((x_px_center - 6) / mm2px).reshape(-1, 1), ((y_px_center - 320) / mm2px).reshape(-1, 1)), axis=1)
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

                origin_img, result = self.plot_and_transform(im=origin_img, box=pred_xylw, label='0:, predic', color=(0, 0, 0), txt_color=(255, 255, 255),
                                                             index=j, scaled_xylw=pred_xylw, keypoints=pred_keypoint,
                                                             cls=pred_cls[j], conf=pred_conf[j],
                                                             use_xylw=use_xylw, truth_flag=False)
                pred_result.append(result)

            for j in range(len(target)):
                tar_xylw = np.copy(target[j, 1:5])
                tar_keypoints = np.copy((target[j, 5:]).reshape(-1, 3)[:, :2])
                tar_cls = target[j, 0]

                # plot target
                origin_img, _ = self.plot_and_transform(im=origin_img, box=tar_xylw, label='0: target', color=(255, 255, 0),
                                                        txt_color=(255, 255, 255),
                                                        index=j, scaled_xylw=tar_xylw, keypoints=tar_keypoints,
                                                        cls=tar_cls, conf=1,
                                                        use_xylw=use_xylw, truth_flag=True)

            cv2.imwrite(tar_path + '%012d.png' % int(i + self.para_dict['index_begin']), origin_img)
            pred_result = np.asarray(pred_result)

            self.calculate_confusion_matrix(pred_xylws, target_xylws, pred_cls, target[:, 0], pred_conf)


            manipulator_before = np.concatenate((pred_result[:, :3], np.zeros((len(pred_result), 2)), pred_result[:, 5].reshape(-1, 1)), axis=1)
            new_lwh_list = np.concatenate((pred_result[:, 3:5], np.ones((len(pred_result), 1)) * 0.016), axis=1)

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

        img_path_output = self.para_dict['data_tar_path'] + 'sim_images/%012d' % (output_epoch) + '_grasp.png'

        # img_path_output = self.para_dict['data_tar_path'] + 'unstack_images/%012d' % (output_epoch) + '_grasp.png'
        cv2.imwrite(img_path_output, output_img)
        pass

if __name__ == '__main__':

    para_dict = {'device': 'cuda:0', 'yolo_conf': 0.004, 'yolo_iou': 0.8,
                 'yolo_model_path': '../models/919_grasp/weights/best.pt',
                 'dataset_path': '../../knolling_dataset/yolo_grasp_dataset_919/',
                 'index_begin': 8000}

    log_save_path = '../models/919_grasp/'

    model_threshold_start = 0.5
    model_threshold_end = 0.6
    check_point = 2
    valid_num = 20000
    model_threshold = np.linspace(model_threshold_start, model_threshold_end, check_point)

    zzz_yolo = Yolo_grasp_model(para_dict= para_dict)
    total_recall = []
    total_precision = []
    total_accuracy = []
    max_precision = -np.inf
    max_accuracy = -np.inf
    for i in model_threshold:
        zzz_yolo.para_dict['yolo_conf'] = i
        zzz_yolo.yolo_grasp_predict()

        print('this is TP', zzz_yolo.pred_TP)
        print('this is TN', zzz_yolo.pred_TN)
        print('this is FP', zzz_yolo.pred_FP)
        print('this is FN', zzz_yolo.pred_FN)

        if zzz_yolo.pred_TP + zzz_yolo.pred_FN == 0:
            recall = 0
        else:
            recall = (zzz_yolo.pred_TP) / (zzz_yolo.pred_TP + zzz_yolo.pred_FN)
        total_recall.append(recall)
        if zzz_yolo.pred_TP + zzz_yolo.pred_FP == 0:
            precision = 0
        else:
            precision = (zzz_yolo.pred_TP) / (zzz_yolo.pred_TP + zzz_yolo.pred_FP)
        total_precision.append(precision)
        accuracy = (zzz_yolo.pred_TP + zzz_yolo.pred_FN) / (zzz_yolo.pred_TP + zzz_yolo.pred_TN + zzz_yolo.pred_FP + zzz_yolo.pred_FN)
        total_accuracy.append(accuracy)

        if precision > max_precision:
            max_precision_threshold = i
            max_precision = precision
        if accuracy > max_accuracy:
            max_accuracy_threshold = i
            max_accuracy = accuracy

    total_recall = np.asarray(total_recall)
    total_precision = np.asarray(total_precision)
    total_accuracy = np.asarray(total_accuracy)

    print(f'When the threshold is {max_accuracy_threshold}, the max accuracy is {max_accuracy}')
    print(f'When the threshold is {max_precision_threshold}, the max precision is {max_precision}')

    plt.plot(model_threshold, total_recall, label='model_pred_recall')
    plt.plot(model_threshold, total_precision, label='model_pred_precision')
    plt.plot(model_threshold, total_accuracy, label='model_pred_accuracy')
    plt.xlabel('model_threshold')
    plt.title('analysis of model prediction')
    plt.legend()
    plt.savefig(log_save_path + 'yolo_analysis.png')
    plt.show()

    total_evaluate_data = np.concatenate(([model_threshold], [total_recall], [total_precision], [total_accuracy]), axis=0).T
    np.savetxt(log_save_path + 'yolo_analysis.txt', total_evaluate_data)

    with open(log_save_path + "model_pred_anlysis_labels_4.txt", "w") as f:
        f.write('----------- Dataset -----------\n')
        f.write(f'threshold_start: {model_threshold_start}\n')
        f.write(f'threshold_end: {model_threshold_end}\n')
        f.write(f'threshold: {max_accuracy_threshold}, max accuracy: {max_accuracy}\n')
        f.write(f'threshold: {max_precision_threshold}, max precision: {max_precision}\n')
        f.write('----------- Dataset -----------\n')

        for i in range(len(model_threshold)):
            f.write(f'threshold: {model_threshold[i]:.6f}, recall: {total_recall[i]:.4f}, precision: {total_precision[i]:.4f}, accuracy: {total_accuracy[i]:.4f}\n')