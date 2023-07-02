import numpy as np
# from yolo_data_collection_env import *
import os
import cv2
import shutil
from itertools import combinations, permutations

def yolo_box(img, label, kpts_label=[]):
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

    radius = 1
    for j in range(len(kpts_label)):
        keypoints = kpts_label[j, 5:].reshape(-1, 3)
        for i, k in enumerate(keypoints):
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
            img = cv2.circle(img, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)

    img = cv2.resize(img, (1280, 960), interpolation=cv2.INTER_AREA)
    cv2.namedWindow("zzz", 0)
    cv2.imshow('zzz', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img

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

def object_detection(data_root, target_path, total_num=None):

    os.makedirs(data_root, exist_ok=True)
    os.makedirs(data_root + 'labels/', exist_ok=True)
    os.makedirs(target_path, exist_ok=True)
    os.makedirs(target_path + 'images/', exist_ok=True)
    os.makedirs(target_path + 'labels/', exist_ok=True)
    mm2px = 530 / 0.34  # (1558)

    for i in range(total_num):
        real_world_data = np.loadtxt(os.path.join(data_root, "origin_labels/%012d.txt") % i)

        corner_list = []
        label_plot = []
        for j in range(len(real_world_data)):
            # print(real_world_data[j])
            xpos1, ypos1 = real_world_data[j][1], real_world_data[j][2]
            l, w, yawori = real_world_data[j][3], real_world_data[j][4], real_world_data[j][5]

            corn1, corn2, corn3, corn4 = find_corner(xpos1, ypos1, l, w, yawori)
            corner_list.append([corn1, corn2, corn3, corn4])
            corns = corner_list[j]

            col_offset = 320
            row_offset = 0

            col_list = [int(mm2px * corns[0][1] + col_offset), int(mm2px * corns[3][1] + col_offset),
                        int(mm2px * corns[1][1] + col_offset), int(mm2px * corns[2][1] + col_offset)]
            row_list = [int(mm2px * corns[0][0] - row_offset), int(mm2px * corns[3][0] - row_offset),
                        int(mm2px * corns[1][0] - row_offset), int(mm2px * corns[2][0] - row_offset)]

            col_list = np.sort(col_list)
            row_list = np.sort(row_list)
            col_list[3] = col_list[3] + 10
            col_list[0] = col_list[0] - 10
            row_list[3] = row_list[3] + 10
            row_list[0] = row_list[0] - 10

            label_x_plot = ((col_list[0] + col_list[3]) / 2) / 640
            label_y_plot = (((row_list[0] + row_list[3]) / 2) + 6) / 480

            length_plot = (col_list[3] - col_list[0]) / 640
            width_plot = (row_list[3] - row_list[0]) / 480
            element_plot = []
            element_plot.append(real_world_data[j][0])
            element_plot.append(label_x_plot)
            element_plot.append(label_y_plot)
            element_plot.append(length_plot)
            element_plot.append(width_plot)
            element_plot = np.asarray(element_plot)
            label_plot.append(element_plot)
        label_plot = np.asarray(label_plot)

        np.savetxt(os.path.join(data_root, "labels/%012d.txt") % i, label_plot, fmt='%.8s')
        img = cv2.imread(os.path.join(data_root, "origin_images/%012d.png") % i)
        if i == 3:
            print('here!')
        img = yolo_box(img, label_plot)
        if i == 3:
            print('here!')
    print('this is total_1', total_1)
    print('this is total_2', total_2)

def color_segmentation(image, num_clusters, label):

    # Reshape the image to a 2D array of pixels

    for i in range(len(label)):
        # label = label[i]
        # print('1',label)
        x_lt = int(label[i][1] * 640 - label[i][3] * 640/2)
        y_lt = int(label[i][2] * 480 - label[i][4] * 480/2)

        x_rb = int(label[i][1] * 640 + label[i][3] * 640/2)
        y_rb = int(label[i][2] * 480 + label[i][4] * 480/2)

        image_part = image[y_lt:y_rb, x_lt:x_rb, :]
        shape = image_part.shape[:2]

        pixels = image_part.reshape((-1, 3))

        # Convert the pixel values to floating point
        pixels = np.float32(pixels)

        # Define the criteria and apply k-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Reshape the labels to the original image shape
        labels = labels.reshape(image_part.shape[:2])
        center_label = labels[int(shape[0] / 2), int(shape[1] / 2)]
        center_mask = np.array(labels == center_label)

        result = cv2.bitwise_and(image_part, image_part, mask=center_mask)
        result[np.where(result != 0)] = 30

        cv2.namedWindow("zzz", 0)
        cv2.imshow('zzz', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        bg_mask = cv2.bitwise_not(center_mask)
        image_part_bg = cv2.bitwise_and(image_part, image_part, mask=bg_mask)

        cv2.namedWindow("zzz", 0)
        cv2.imshow('zzz', image_part_bg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        new_image_part = cv2.add(image_part_bg, result)

        cv2.namedWindow("zzz", 0)
        cv2.imshow('zzz', new_image_part)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Create masks for each cluster label
        masks = []
        for j in range(num_clusters):
            masks.append(np.uint8(labels == j))

        # Show the segmented regions
        for j, mask in enumerate(masks):
            result = cv2.bitwise_and(image_part, image_part, mask=mask)
            cv2.namedWindow("Segmented Region " + str(j + 1), 0)
            cv2.imshow("Segmented Region " + str(j + 1), result)

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
    # change the sequence of keypoints based on xy
    keypoints_order = np.lexsort((keypoints[:, 1], keypoints[:, 0]))[::-1]
    keypoints = keypoints[keypoints_order]

    keypoints = np.concatenate(((keypoints[:, 0] + xpos).reshape(-1, 1), (keypoints[:, 1] + ypos).reshape(-1, 1)), axis=1)

    return keypoints, total_1, total_2

def apply_rotation(lwh, ori):

    num_boxes = len(lwh)
    raw_l = np.concatenate((lwh[:, 0].reshape(-1, 1), np.zeros((num_boxes, 2))), axis=1)
    raw_w = np.concatenate((np.zeros((num_boxes, 1)).reshape(-1, 1), lwh[:, 1].reshape(-1, 1), np.zeros((num_boxes, 1)).reshape(-1, 1)), axis=1)
    raw_h = np.concatenate((np.zeros((num_boxes, 2)), lwh[:, 2].reshape(-1, 1)), axis=1)
    rotated_lwh = np.zeros((3, num_boxes, 3))
    rotated_l = []
    rotated_w = []
    rotated_h = []

    for i in range(num_boxes):
        # Calculate rotation matrix components
        cos_roll = np.cos(ori[i, 0])
        sin_roll = np.sin(ori[i, 0])
        cos_pitch = np.cos(ori[i, 1])
        sin_pitch = np.sin(ori[i, 1])
        cos_yaw = np.cos(ori[i, 2])
        sin_yaw = np.sin(ori[i, 2])

        # Create rotation matrix
        rotation_matrix = np.array([[cos_yaw * cos_pitch, cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll,
                                     cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll],
                                    [sin_yaw * cos_pitch, sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll,
                                     sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll],
                                    [-sin_pitch, cos_pitch * sin_roll, cos_pitch * cos_roll]])
        rotated_l.append(np.dot(rotation_matrix, raw_l[i])[:3])
        rotated_w.append(np.dot(rotation_matrix, raw_w[i])[:3])
        rotated_h.append(np.dot(rotation_matrix, raw_h[i])[:3])

    rotated_l = np.asarray(rotated_l)
    rotated_w = np.asarray(rotated_w)
    rotated_h = np.asarray(rotated_h)
    rotated_lwh[0] = rotated_l
    rotated_lwh[1] = rotated_w
    rotated_lwh[2] = rotated_h
    xy_projection_l = np.sqrt(np.power(rotated_l[:, 0], 2) + np.power(rotated_l[:, 1], 2))
    xy_projection_w = np.sqrt(np.power(rotated_w[:, 0], 2) + np.power(rotated_w[:, 1], 2))
    xy_projection_h = np.sqrt(np.power(rotated_h[:, 0], 2) + np.power(rotated_h[:, 1], 2))
    xy_projection_lwh = np.concatenate((xy_projection_l, xy_projection_w, xy_projection_h), axis=0).reshape(-1, num_boxes).T
    order = np.argsort(xy_projection_lwh, axis=1)
    l_data = xy_projection_lwh[np.arange(num_boxes), order[:, -1]]
    w_data = xy_projection_lwh[np.arange(num_boxes), order[:, -2]]
    # yaw_data = rotated_lwh[]
    longest_projection_xyz = rotated_lwh[np.argmax(xy_projection_lwh,axis=1)]
    angle_x = longest_projection_xyz[np.arange(num_boxes), np.arange(num_boxes), 0]
    angle_y = longest_projection_xyz[np.arange(num_boxes), np.arange(num_boxes), 1]
    yaw = np.arctan2(angle_y, angle_x)
    for i in range(num_boxes):
        if yaw[i] > np.pi:
            yaw[i] = yaw[i] - np.pi
        elif yaw[i] < 0:
            yaw[i] = yaw[i] + np.pi
    return l_data, w_data, yaw

def pose4keypoints(data_root, target_path, total_num=None, flag=None):

    os.makedirs(data_root, exist_ok=True)
    os.makedirs(data_root + 'labels/', exist_ok=True)
    os.makedirs(target_path, exist_ok=True)
    os.makedirs(target_path + 'images/', exist_ok=True)
    os.makedirs(target_path + 'labels/', exist_ok=True)
    mm2px = 530 / 0.34  # (1558)

    import warnings
    with warnings.catch_warnings(record=True) as w:

        total_1 = 0
        total_2 = 0
        for i in range(total_num):
            real_world_data = np.loadtxt(os.path.join(data_root, "origin_labels/%012d.txt") % i).reshape(-1, 11)
            # real_world_img = cv2.imread(data_root + "origin_images/%012d.png" % i)
            lwh = real_world_data[:, 4:7]
            pos = real_world_data[:, 1:4]
            ori = real_world_data[:, 7:]
            if flag == 'grasp':
                grasp_flag = real_world_data[:, 0].reshape(-1, 1)
            # l_data, w_data, yaw_data = apply_rotation(lwh, ori)
            l_data = lwh[:, 0]
            w_data = lwh[:, 1]
            yaw_data = ori[:, 2]
            conf_data = real_world_data[:, -1]

            corner_list = []
            label_plot = []
            label = []
            print('this is index of images', i)
            for j in range(len(real_world_data)):
                # xpos1, ypos1 = real_world_data[j][1], real_world_data[j][2]
                xpos1, ypos1 = pos[j, 0], pos[j, 1]
                l, w, yawori = l_data[j], w_data[j], yaw_data[j]
                conf = conf_data[j]
                if l < w:
                    l = real_world_data[j][4]
                    w = real_world_data[j][3]
                    if yawori > np.pi / 2:
                        yawori = yawori - np.pi / 2
                    #################################################### here!!!!!
                    elif yawori < -np.pi / 2:
                        yawori = yawori + np.pi / 2

                # ensure the yolo sequence!
                label_y = (xpos1 * mm2px + 6) / 480
                label_x = (ypos1 * mm2px + 320) / 640
                # ensure the yolo sequence!
                keypoints, total_1, total_2 = find_keypoints(xpos1, ypos1, l, w, yawori, mm2px, total_1, total_2)
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

                length_plot = (col_list[3] - col_list[0]) / 640
                width_plot = (row_list[3] - row_list[0]) / 480
                element_plot = []
                element_plot.append(grasp_flag[j])
                element_plot.append(label_x_plot)
                element_plot.append(label_y_plot)
                element_plot.append(length_plot)
                element_plot.append(width_plot)
                element_plot = np.asarray(element_plot)
                label_plot.append(element_plot)

                # change the lw to yolo_lw in label!!!!!!
                element = np.concatenate((grasp_flag[j], [conf, xpos1, ypos1], [l, w], keypoints.reshape(-1)))
                label.append(element)

            label = np.asarray(label)
            # print('this is element\n', label)
            # print('this is plot element\n', label_plot)


            np.savetxt(os.path.join(data_root, "labels/%012d.txt") % i, label, fmt='%.8s')
            img = cv2.imread(os.path.join(data_root, "origin_images/%012d.png") % i)
            if i == 3:
                print('here!')
            img = yolo_box(img, label_plot, label)
            if i == 3:
                print('here!')
            # color_segmentation(real_world_img, 5, label_plot)
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

def train_test_split(data_root, target_path, total_num=None):

    import shutil
    ratio = 0.8
    train_num = int(total_num * ratio)
    test_num = int(total_num - train_num)
    print(train_num)
    print(test_num)

    os.makedirs(target_path + '/labels/train', exist_ok=True)
    os.makedirs(target_path + '/labels/val', exist_ok=True)
    os.makedirs(target_path + '/images/train', exist_ok=True)
    os.makedirs(target_path + '/images/val', exist_ok=True)

    for i in range(0, train_num):
        cur_path = os.path.join(data_root, 'origin_images/%012d.png') % (i)
        tar_path = os.path.join(target_path, 'images/train/%012d.png') % i
        shutil.copy(cur_path, tar_path)

        cur_path = os.path.join(data_root, 'labels/%012d.txt') % (i)
        tar_path = os.path.join(target_path, 'labels/train/%012d.txt') % i
        shutil.copy(cur_path, tar_path)

    for i in range(train_num, total_num):
        cur_path = os.path.join(data_root, 'origin_images/%012d.png') % (i)
        tar_path = os.path.join(target_path, 'images/val/%012d.png') % i
        shutil.copy(cur_path, tar_path)

        cur_path = os.path.join(data_root, 'labels/%012d.txt') % (i)
        tar_path = os.path.join(target_path, 'labels/val/%012d.txt') % i
        shutil.copy(cur_path, tar_path)

if __name__ == '__main__':

    total_num = 6000
    flag = 'grasp'

    # data_root = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/yolo_pose4keypoints_510_tuning/'
    # target_path = '/home/zhizhuo/ADDdisk/Create Machine Lab/datasets/yolo_pose4keypoints_510_tuning/'
    # manual_pose4keypoints(data_root, target_path)

    data_root = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/grasp_pile_625_2/'
    target_path = '/home/zhizhuo/ADDdisk/Create Machine Lab/datasets/grasp_pile_625_2/'

    # data_root = 'C:/Users/24356/Desktop/knolling_dataset/yolo_pose4keypoints_5/'
    # target_path = 'C:/Users/24356/Desktop/datasets/yolo_pose4keypoints_519_gray/'
    pose4keypoints(data_root, target_path, total_num, flag)
    # object_detection(data_root, target_path, total_num)

    # data_root = 'C:/Users/24356/Desktop/knolling_dataset/yolo_pose4keypoints_5/'
    # target_path = 'C:/Users/24356/Desktop/datasets/yolo_pose4keypoints_519_gray/'
    # train_test_split(data_root, target_path, total_num)

    # data_root = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/yolo_pose4keypoints/labels/'
    # files = os.listdir(data_root)
    # print(files)
    #
    # l = [i for i in files if 'normal' in i]
    #
    # for m in l:
    #     os.remove(data_root + m)
