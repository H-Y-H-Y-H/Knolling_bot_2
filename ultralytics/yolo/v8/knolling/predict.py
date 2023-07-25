# Ultralytics YOLO 🚀, GPL-3.0 license
import numpy as np

from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

import torch
import cv2

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

def plot_and_transform(im, box, label='', color=(0, 0, 0), txt_color=(255, 255, 255), index=None, scaled_xylw=None, keypoints=None, use_lw=True, truth_flag=None):
    # Add one xyxy box to image with label

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

    # this is the knolling sequence, not opencv!!!!
    keypoints_x = ((keypoints[:, 1] - 6) / mm2px).reshape(-1, 1)
    keypoints_y = ((keypoints[:, 0] - 320) / mm2px).reshape(-1, 1)
    keypoints_mm = np.concatenate((keypoints_x, keypoints_y), axis=1)
    keypoints_center = np.average(keypoints_mm, axis=0)
    if use_lw == True:
        length = scaled_xylw[2]
        width = scaled_xylw[3]
        c1 = np.array([length / (3 * 2), width / (3 * 2)])
        c2 = np.array([length / (3 * 2), -width / (3 * 2)])
        c3 = np.array([-length / (3 * 2), width / (3 * 2)])
        c4 = np.array([-length / (3 * 2), -width / (3 * 2)])
    else:
        length = max(np.linalg.norm(keypoints_mm[0] - keypoints_mm[-1]),
                   np.linalg.norm(keypoints_mm[1] - keypoints_mm[2]))
        width = min(np.linalg.norm(keypoints_mm[0] - keypoints_mm[-1]),
                   np.linalg.norm(keypoints_mm[1] - keypoints_mm[2]))
        c1 = np.array([length / (2), width / (2)])
        c2 = np.array([length / (2), -width / (2)])
        c3 = np.array([-length / (2), width / (2)])
        c4 = np.array([-length / (2), -width / (2)])

    all_distance = np.linalg.norm((keypoints_mm - keypoints_center), axis=1)
    k = 2
    max_index = all_distance.argsort()[-k:]
    lucky_keypoint_index = np.argmax([keypoints_mm[max_index[0], 1], keypoints_mm[max_index[1], 1]])
    lucky_keypoint = keypoints_mm[max_index[lucky_keypoint_index]]
    print('the ori keypoint is ', keypoints_mm[max_index[lucky_keypoint_index]])
    my_ori = np.arctan2(lucky_keypoint[1] - keypoints_center[1], lucky_keypoint[0] - keypoints_center[0])
    # if 0.75 < l / w <=1.25:
    #     my_ori = ori / 4
    # else:
    #     my_ori = ori / 2
    print('this is ori', my_ori)



    rot_z = [[np.cos(my_ori), -np.sin(my_ori)],
             [np.sin(my_ori), np.cos(my_ori)]]
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
    p1 = np.array([int(box[0]), int(box[1])])
    p2 = np.array([int(box[2]), int(box[3])])
    # print('this is p1 and p2', p1, p2)

    # cv2.rectangle(self.im, p1, p2, color, thickness=zzz_lw, lineType=cv2.LINE_AA)
    im = cv2.line(im, (int(corn1[1]), int(corn1[0])), (int(corn2[1]), int(corn2[0])), color, 1)
    im = cv2.line(im, (int(corn2[1]), int(corn2[0])), (int(corn4[1]), int(corn4[0])), color, 1)
    im = cv2.line(im, (int(corn4[1]), int(corn4[0])), (int(corn3[1]), int(corn3[0])), color, 1)
    im = cv2.line(im, (int(corn3[1]), int(corn3[0])), (int(corn1[1]), int(corn1[0])), color, 1)
    if label:
        w, h = cv2.getTextSize(label, 0, fontScale=zzz_lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        # cv2.rectangle(self.im, p1, p2, color, 0, cv2.LINE_AA)  # filled
        im = cv2.putText(im,
                    str(index), (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    zzz_lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)
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
        x_coord, y_coord = k[0], k[1]
        # if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
        #     if len(k) == 3:
        #         conf = k[2]
        #         if conf < 0.5:
        #             continue
        im = cv2.circle(im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)
    ############### zzz plot the keypoints ###############

    result = np.concatenate((keypoints_center, [length], [width], [my_ori]))

    return im, result

def predict(cfg=DEFAULT_CFG, use_python=False, data_path=None, model_path=None):
    data_path = '/home/zhizhuo/ADDdisk/Create Machine Lab/datasets/'
    model_path = '/home/zhizhuo/ADDdisk/Create Machine Lab/YOLOv8/runs/pose/train_standard_1000/weights/best.pt'
    model = model_path
    source_pth = data_path + 'real_image_collect/'
    # source_pth = data_path + 'knolling_data_small/images/train/'
    args = dict(model=model, source=source_pth, conf=0.8)
    use_python = True
    if use_python:
        from ultralytics import YOLO
        images = YOLO(model)(**args)
    else:
        predictor = PosePredictor(overrides=args)
        predictor.predict_cli()
    device = 'cuda:0'


    for i in range(len(images)):

        # origin_img = cv2.imread(source_pth + '%012d.png' % int(i))
        origin_img = cv2.imread(source_pth + 'img_%s.png' % int(i))

        use_lw = True
        # target = np.loadtxt(data_path + 'yolo_pose4keypoints/labels/train/%012d.txt' % int(i))
        # # target = np.loadtxt(data_path + 'knolling_data_small/labels/train/%012d.txt' % int(i))
        # target_order = np.lexsort((target[:, 2], target[:, 1]))
        # target = target[target_order]

        one_img = images[i]
        j = 0
        total_pred_xylw = []
        total_pred_keypoints = []
        pred_result = []
        for elements in one_img:

            pred_keypoints = elements.keypoints.cpu().detach().numpy()
            # pred_cos_sin = (keypoints[:, :2] / torch.tensor((640, 480)).to(device)) * 2 - 1
            # pred_cos_sin = pred_cos_sin.cpu().detach().numpy().reshape(-1, )
            box = elements.boxes
            pred_xylw = box.xywhn.cpu().detach().numpy().reshape(-1, )
            pred_name = elements.names
            pred_label = (f'{pred_name}')
            pred_keypoints_label = np.copy(pred_keypoints)
            pred_keypoints_label[:, :2] = pred_keypoints_label[:, :2] / np.array([640, 480])
            pred_keypoints_label = pred_keypoints.reshape(-1, )
            total_pred_xylw.append(pred_xylw)
            total_pred_keypoints.append(pred_keypoints_label)

            # plot pred
            print('this is pred xylw', pred_xylw)
            # print('this is pred cos sin', pred_cos_sin)
            origin_img, result = plot_and_transform(im=origin_img, box=box.xyxy.squeeze(), label=pred_label, color=(0, 0, 0), txt_color=(255, 255, 255), index=j,
                                            scaled_xylw=pred_xylw, keypoints=pred_keypoints[:, :2], use_lw=False, truth_flag=False)
            pred_result.append(result)
            print('this is j', j)
            print('this is i', i)
            if i == 4:
                print('aaaa')
            # tar_xylw = np.copy(target[j, 1:5])
            # tar_keypoints = np.copy((target[j, 5:]).reshape(-1, 3)[:, :2])
            # # tar_keypoints = (target[j, 5:])
            # tar_keypoints[:, 0] *= 640
            # tar_keypoints[:, 1] *= 480
            # tar_label = '0: "target"'
            #
            # # plot target
            # print('this is tar xylw', tar_xylw)
            # print('this is tar cos sin', tar_keypoints)
            # origin_img, _ = plot_and_transform(im=origin_img, box=box.xyxy.squeeze(), label=tar_label, color=(255, 255, 0), txt_color=(255, 255, 255), index=j,
            #                                 scaled_xylw=tar_xylw, keypoints=tar_keypoints, use_lw=use_lw, truth_flag=True)
            # origin_img = plot_box_label(im=origin_img, box=box.xyxy.squeeze(), label=tar_label, color=(255, 255, 0),
            #                             txt_color=(255, 255, 255), index=i,
            #                             scaled_xylw=tar_xylw, keypoints=tar_keypoints)

            j += 1

        total_pred_keypoints = np.asarray(total_pred_keypoints)
        total_pred_xylw = np.asarray(total_pred_xylw)
        pred = np.concatenate((np.zeros((j, 1)), total_pred_xylw, total_pred_keypoints), axis=1)
        pred_order = np.lexsort((pred[:, 2], pred[:, 1]))
        pred = pred[pred_order]

        # loss_mean = np.mean((target - pred) ** 2)
        # loss_std = np.std((target - pred), dtype=np.float64)
        # print('this is pred\n', pred)
        # print('this is target\n', target)
        # print('this is mean error', loss_mean)
        # print('this is std error', loss_std)

        # cv2.namedWindow('zzz', 0)
        # cv2.imshow('zzz', origin_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.imwrite(source_pth + 'img_%s_pred.png' % int(i), origin_img)

    print('this is key point')


if __name__ == '__main__':

    data_path = '/home/zhizhuo/ADDdisk/Create Machine Lab/datasets/'
    model_path = '/home/zhizhuo/ADDdisk/Create Machine Lab/YOLOv8/runs/pose/train_standard_1000/weights/best.pt'
    predict(data_path=data_path, model_path=model_path)
    # from ultralytics import YOLO
    #
    # model_path = "/home/ubuntu/Desktop/YOLOv8/runs/pose/train4/weights/"
    #
    # # Load a model
    # model = YOLO('yolov8n-knolling.yaml')  # build from YAML and transfer weights
    # model.load(model_path+'last.pt')
    #
    # source_pth = '/home/ubuntu/Desktop/datasets/knolling_data/images/val'
    #
    #
    # result = model(source=source_pth,conf = 0.5,save=True)
    # print(result)
