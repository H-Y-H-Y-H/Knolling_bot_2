import numpy as np
# from yolo_data_collection_env import *
import os
import cv2
import shutil
from itertools import combinations, permutations
from tqdm import tqdm

def yolo_box(img, label):
    # label = [0,x,y,l,w],[0,x,y,l,w],...
    # label = label[:,1:]
    for i in range(len(label)):
        # label = label[i]
        # print('1',label)
        x_lt = int(label[i][1] * 640 - label[i][3] * 640/2)
        y_lt = int(label[i][2] * 480 - label[i][4] * 480/2)

        x_rb = int(label[i][1] * 640 + label[i][3] * 640/2)
        y_rb = int(label[i][2] * 480 + label[i][4] * 480/2)

        # img = img/255
        img = cv2.rectangle(img,(x_lt,y_lt),(x_rb,y_rb), color = (0,0,0), thickness = 1)

    img = cv2.resize(img, (1280, 960), interpolation=cv2.INTER_AREA)
    # cv2.namedWindow("zzz", 0)
    cv2.imshow('zzz', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img

def yolo_points(img, label):

    for i in range(len(label)):
        kpts = label[i][1:].reshape(-1, 2)
        kpts_px = np.copy(kpts)
        kpts_px[:, 0] = kpts_px[:, 0] * 640
        kpts_px[:, 1] = kpts_px[:, 1] * 480
        for point in kpts_px:
            point_x = int(point[0])
            point_y = int(point[1])
            img = cv2.circle(img, (point_x, point_y), radius=2, color=(255, 0, 0))

    img = cv2.resize(img, (1280, 960), interpolation=cv2.INTER_AREA)
    # cv2.namedWindow("zzz", 0)
    cv2.imshow('zzz', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_corner(x, y, l, w, yaw):

    gamma = yaw
    rot_z = [[np.cos(gamma), -np.sin(gamma)],
             [np.sin(gamma), np.cos(gamma)]]
    rot_z = np.asarray(rot_z)

    c1 = [l / 2, w / 2]
    c2 = [l / 2, -w / 2]
    c3 = [-l / 2, w / 2]
    c4 = [-l / 2, -w / 2]

    c1,c2,c3,c4 = np.asarray(c1),np.asarray(c2),np.asarray(c3),np.asarray(c4)

    corn1 = np.dot(rot_z,c1)
    corn2 = np.dot(rot_z,c2)
    corn3 = np.dot(rot_z,c3)
    corn4 = np.dot(rot_z,c4)

    corn11 = [corn1[0] + x, corn1[1] + y]
    corn22 = [corn2[0] + x, corn2[1] + y]
    corn33 = [corn3[0] + x, corn3[1] + y]
    corn44 = [corn4[0] + x, corn4[1] + y]

    return corn11, corn22, corn33, corn44

total_1 = 0
total_2 = 0
def find_keypoints(xpos, ypos, l, w, ori, mm2px, total_1, total_2):

    gamma = ori
    rot_z = [[np.cos(gamma), -np.sin(gamma)],
             [np.sin(gamma), np.cos(gamma)]]
    rot_z = np.asarray(rot_z)

    kp1 = np.asarray([l / 2, 0])
    kp2 = np.asarray([0, w / 2])
    kp3 = np.asarray([-l / 2, 0])
    kp4 = np.asarray([0, -w / 2])

    # here is simulation xy sequence, not yolo lw sequence!
    keypoint1 = np.dot(rot_z, kp1)
    keypoint2 = np.dot(rot_z, kp2)
    keypoint3 = np.dot(rot_z, kp3)
    keypoint4 = np.dot(rot_z, kp4)
    keypoints = np.concatenate((keypoint1, keypoint2, keypoint3, keypoint4), axis=0).reshape(-1, 2)

    # top_left_corner =

    # change the sequence of keypoints based on xy
    keypoints_order = np.lexsort((keypoints[:, 1], keypoints[:, 0]))[::-1]
    keypoints = keypoints[keypoints_order]

    keypoints = np.concatenate(((((keypoints[:, 1] + ypos) * mm2px + 320) / 640).reshape(-1, 1),
                                (((keypoints[:, 0] + xpos) * mm2px + 6) / 480).reshape(-1, 1),
                                np.ones((4, 1))), axis=1).reshape(-1, 3)

    return keypoints, total_1, total_2

def pose4keypoints(data_root, target_path, start_num, end_num):
    os.makedirs(data_root + 'labels/', exist_ok=True)
    os.makedirs(target_path + 'images/', exist_ok=True)
    os.makedirs(target_path + 'preprocess_labels/', exist_ok=True)
    mm2px = 530 / 0.34  # (1558)

    import warnings
    with warnings.catch_warnings(record=True) as w:

        total_1 = 0
        total_2 = 0
        for i in range(start_num, end_num):
            real_world_data = np.loadtxt(os.path.join(data_root, "sim_labels/%012d.txt") % i)
            # real_world_img = cv2.imread(data_root + "origin_images/%012d.png" % i)
            corner_list = []
            label_plot = []
            label = []

            print('this is index of images', i)
            for j in range(len(real_world_data)):
                # print(real_world_data[j])
                # print('this is index if legos', j)
                grasp_flag = real_world_data[j][0]
                xpos1, ypos1 = real_world_data[j][1], real_world_data[j][2]
                l, w = real_world_data[j][4], real_world_data[j][5]
                yawori = real_world_data[j][9]
                if l < w:
                    l = real_world_data[j][5]
                    w = real_world_data[j][4]
                    if yawori > np.pi / 2:
                        yawori = yawori - np.pi / 2
                    else:
                        yawori = yawori + np.pi / 2

                # ensure the yolo sequence!
                label_y = (xpos1 * mm2px + 6) / 480
                label_x = (ypos1 * mm2px + 320) / 640
                length = l * 3
                width = w * 3
                # ensure the yolo sequence!
                keypoints, total_1, total_2 = find_keypoints(xpos1, ypos1, l, w, yawori, mm2px, total_1, total_2)

                element = np.concatenate(([grasp_flag], [label_x, label_y], [length, width], keypoints.reshape(-1)))
                # print(label)

                corn1, corn2, corn3, corn4 = find_corner(xpos1, ypos1, l, w, yawori)
                corner_list.append([corn1, corn2, corn3, corn4])
                corns = corner_list[j]

                col_offset = 320
                # row_offset = (0.154 - (0.3112 - 0.154)) * mm2px + 5
                row_offset = 0

                col_list = np.array([mm2px * corns[0][1] + col_offset, mm2px * corns[3][1] + col_offset,
                                     mm2px * corns[1][1] + col_offset, mm2px * corns[2][1] + col_offset])
                row_list = np.array([mm2px * corns[0][0] - row_offset, mm2px * corns[3][0] - row_offset,
                                     mm2px * corns[1][0] - row_offset, mm2px * corns[2][0] - row_offset])

                col_list = np.sort(col_list)
                row_list = np.sort(row_list)
                col_list[3] = col_list[3] + 10
                col_list[0] = col_list[0] - 10
                row_list[3] = row_list[3] + 10
                row_list[0] = row_list[0] - 10

                label_x_plot = ((col_list[0] + col_list[3]) / 2) / 640
                label_y_plot = (((row_list[0] + row_list[3]) / 2) + 6) / 480
                label_y = (xpos1 * mm2px + 6) / 480
                label_x = (ypos1 * mm2px + 320) / 640

                length_plot = (col_list[3] - col_list[0]) / 640
                width_plot = (row_list[3] - row_list[0]) / 480
                element_plot = []
                element_plot.append(0)
                element_plot.append(label_x_plot)
                element_plot.append(label_y_plot)
                element_plot.append(length_plot)
                element_plot.append(width_plot)
                element_plot = np.asarray(element_plot)
                label_plot.append(element_plot)

                # change the lw to yolo_lw in label!!!!!!
                element[3] = length_plot
                element[4] = width_plot
                label.append(element)

            label = np.asarray(label)
            # print('this is element\n', label)
            # print('this is plot element\n', label_plot)


            np.savetxt(os.path.join(data_root, "preprocess_labels/%012d.txt") % i, label, fmt='%.8s')
            # img = cv2.imread(os.path.join(data_root, "images/%012d.png") % i)
            # img = yolo_box(real_world_img, label_plot)
            # color_segmentation(real_world_img, 5, label_plot)
        print('this is total_1', total_1)
        print('this is total_2', total_2)
def segmentation(data_root, target_path, start_num, end_num, show_flag=False):

    os.makedirs(data_root + 'labels/', exist_ok=True)
    os.makedirs(target_path + 'preprocess_labels/', exist_ok=True)
    mm2px = 530 / 0.34  # (1558)
    # import warnings
    # with warnings.catch_warnings(record=True) as w:

    total_1 = 0
    total_2 = 0
    for i in range(start_num, end_num):
        # real_world_data = np.loadtxt(os.path.join(data_root, "origin_labels/%012d.txt") % i)

        with open(file=data_root + "origin_labels/%012d.txt" % i, mode="r") as f:
            data = []
            raw_data = f.readlines()
            for line in raw_data:
                data.append([float(d) for d in line.split()])
        corner_list = []
        label_plot = []
        label = []
        print('this is index of images', i)
        for j in range(len(data)):

            center = data[j][:2]
            center_px = [center[0] * mm2px + 6, center[1] * mm2px + 320]

            ori = data[j][2]
            rot_z = np.array([[np.cos(ori), -np.sin(ori)],
                            [np.sin(ori), np.cos(ori)]])
            flag = data[j][3]
            keypoints = np.asarray(data[j][4:]).reshape(-1, 2)
            keypoints_rotate = []
            for point in keypoints:
                keypoints_rotate.append(np.dot(rot_z, point).T)
            keypoints_rotate = np.asarray(keypoints_rotate) + center
            keypoints_rotate_px = np.copy(keypoints_rotate)
            keypoints_rotate_px[:, 0] = (keypoints_rotate_px[:, 0] * mm2px + 6) / 480
            keypoints_rotate_px[:, 1] = (keypoints_rotate_px[:, 1] * mm2px + 320) / 640

            ############## swap the x and y ###############
            keypoints_rotate_px[:, [0, 1]] = keypoints_rotate_px[:, [1, 0]]
            ############## swap the x and y ###############

            label_box = np.concatenate(([flag], keypoints_rotate_px.reshape(-1, )))
            label.append(label_box)
        # label = np.asarray(label)
        with open(file=target_path + "preprocess_labels/%.012d.txt" % i, mode="w") as f:
            for m in range(len(label)):
                output_data = list(label[m])
                output = ' '.join(str(item) for item in output_data)
                f.write(output)
                f.write('\n')
        if show_flag == True:
            img = cv2.imread(data_root + 'origin_images/%012d.png' % i)
            yolo_points(img, label)

    print('this is total_1', total_1)
    print('this is total_2', total_2)

def manual_pose4keypoints(data_root, target_path):
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(target_path, exist_ok=True)
    os.makedirs(target_path + 'images/', exist_ok=True)
    os.makedirs(target_path + 'labels/', exist_ok=True)
    mm2px = 530 / 0.34  # (1558)
    img_index = input('Enter the index of img:')
    img_index = int(img_index)

    cur_path = os.path.join(data_root, "images/%012d.png") % img_index
    tar_path = os.path.join(target_path, "images/%012d.png") % img_index
    shutil.copy(cur_path, tar_path)

    # num_item = input('Enter the num of boxes in one img:')
    #
    # data_total = []
    # for i in range(int(num_item)):
    #     data = input('Enter the xylwori of one img (five datas in total):').split(" ")
    #     data_total.append(np.array([float(j) for j in data]))
    # data_total = np.concatenate((np.zeros((len(data_total), 1)), np.asarray(data_total)), axis=1)
    # data_total[:, 5] = data_total[:, 5] / 180 * np.pi
    # print(data_total)
    # np.savetxt(os.path.join(data_root, "labels/%012d.txt") % img_index, data_total, fmt='%.8s')

    data_total = np.loadtxt(os.path.join(data_root, "labels/%012d.txt") % img_index)
    data_total[:, 5] = data_total[:, 5] / 180 * np.pi

    corner_list = []
    label_plot = []
    label = []
    total_1 = 0
    total_2 = 0
    for j in range(len(data_total)):
        print(data_total[j])
        # print('this is index if legos', j)
        xpos1, ypos1 = data_total[j][1], data_total[j][2]
        l, w = data_total[j][3], data_total[j][4]
        yawori = data_total[j][5]
        if l < w:
            l = data_total[j][4]
            w = data_total[j][3]
            if yawori > np.pi / 2:
                yawori = yawori - np.pi / 2
            else:
                yawori = yawori + np.pi / 2

        # ensure the yolo sequence!
        label_y = (xpos1 * mm2px + 6) / 480
        label_x = (ypos1 * mm2px + 320) / 640
        length = l * 3
        width = w * 3
        # ensure the yolo sequence!
        keypoints, total_1, total_2 = find_keypoints(xpos1, ypos1, l, w, yawori, mm2px, total_1, total_2)
        # keypoints_order = np.lexsort((keypoints[:, 0], keypoints[:, 1]))[::-1]
        # keypoints = keypoints[keypoints_order]

        element = np.concatenate(([0], [label_x, label_y], [length, width], keypoints.reshape(-1)))
        # print(label)

        corn1, corn2, corn3, corn4 = find_corner(xpos1, ypos1, l, w, yawori)
        corner_list.append([corn1, corn2, corn3, corn4])
        corns = corner_list[j]

        col_offset = 320
        # row_offset = (0.154 - (0.3112 - 0.154)) * mm2px + 5
        row_offset = 0

        col_list = np.array([mm2px * corns[0][1] + col_offset, mm2px * corns[3][1] + col_offset,
                             mm2px * corns[1][1] + col_offset, mm2px * corns[2][1] + col_offset])
        row_list = np.array([mm2px * corns[0][0] - row_offset, mm2px * corns[3][0] - row_offset,
                             mm2px * corns[1][0] - row_offset, mm2px * corns[2][0] - row_offset])

        col_list = np.sort(col_list)
        row_list = np.sort(row_list)
        col_list[3] = col_list[3]
        col_list[0] = col_list[0]
        row_list[3] = row_list[3]
        row_list[0] = row_list[0]

        label_x_plot = ((col_list[0] + col_list[3]) / 2) / 640
        label_y_plot = (((row_list[0] + row_list[3]) / 2) + 6) / 480
        label_y = (xpos1 * mm2px + 6) / 480
        label_x = (ypos1 * mm2px + 320) / 640

        length_plot = (col_list[3] - col_list[0]) / 640
        width_plot = (row_list[3] - row_list[0]) / 480
        element_plot = []
        element_plot.append(0)
        element_plot.append(label_x_plot)
        element_plot.append(label_y_plot)
        element_plot.append(length_plot)
        element_plot.append(width_plot)
        element_plot = np.asarray(element_plot)
        label_plot.append(element_plot)

        # change the lw to yolo_lw in label!!!!!!
        element[3] = length_plot
        element[4] = width_plot
        label.append(element)

    label = np.asarray(label)
    print('this is element\n', label)
    # print('this is plot element\n', label_plot)
    img = cv2.imread(os.path.join(data_root, "images/%012d.png") % img_index)
    img = yolo_box(img, label_plot)

    np.savetxt(os.path.join(target_path, "labels/%012d.txt") % img_index, label, fmt='%.8s')

def train_test_split(data_root, target_path, start_num, end_num):

    import shutil
    ratio = 0.8
    total_num = end_num - start_num
    train_num = int(total_num * ratio)
    test_num = int(total_num - train_num)
    print(train_num)
    print(test_num)

    os.makedirs(target_path + '/labels/train', exist_ok=True)
    os.makedirs(target_path + '/labels/val', exist_ok=True)
    os.makedirs(target_path + '/images/train', exist_ok=True)
    os.makedirs(target_path + '/images/val', exist_ok=True)

    for i in tqdm(range(start_num, train_num + start_num)):
        cur_path = os.path.join(data_root, 'sim_images/%012d.png') % (i)
        tar_path = os.path.join(target_path, 'images/train/%012d.png') % i
        shutil.copy(cur_path, tar_path)

        cur_path = os.path.join(data_root, 'preprocess_labels/%012d.txt') % (i)
        tar_path = os.path.join(target_path, 'labels/train/%012d.txt') % i
        shutil.copy(cur_path, tar_path)

    for i in tqdm(range(train_num + start_num, end_num)):
        cur_path = os.path.join(data_root, 'sim_images/%012d.png') % (i)
        tar_path = os.path.join(target_path, 'images/val/%012d.png') % i
        shutil.copy(cur_path, tar_path)

        cur_path = os.path.join(data_root, 'preprocess_labels/%012d.txt') % (i)
        tar_path = os.path.join(target_path, 'labels/val/%012d.txt') % i
        shutil.copy(cur_path, tar_path)

if __name__ == '__main__':

    start_num = 0
    end_num = 10000
    data_root = '../../../knolling_dataset/yolo_grasp_dataset_919/'
    target_path = '../../../knolling_dataset/yolo_grasp_dataset_919/'

    # pose4keypoints(data_root, target_path, start_num, end_num)

    # data_root = '../../../knolling_dataset/yolo_segmentation_820/'
    # target_path = '../../../knolling_dataset/yolo_segmentation_820/'
    # start_num = 0
    # end_num = 4000
    # segmentation(data_root, target_path, start_num, end_num, show_flag=False)

    # data_root = '../../../knolling_dataset/yolo_pile_830_real_box/'
    # target_path = '../../../knolling_dataset/yolo_pile_830_real_box/'
    #
    train_test_split(data_root, target_path, start_num, end_num)
