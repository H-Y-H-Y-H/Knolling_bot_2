import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

def change_sequence(pos_before):

    origin_point = np.array([0, -0.2])
    delete_index = np.where(pos_before == 0)[0]
    distance = np.linalg.norm(pos_before[:, 1:3] - origin_point, axis=1)
    order = np.argsort(distance)
    return order

def data_preprocess_csv(path, data_num, start_index):

    max_conf_1 = 0
    no_grasp = 0
    data = []
    i = 0
    origin_data = np.loadtxt(path + 'labels/%012d.txt' % i).reshape(-1, 7)
    data = origin_data
    # data = np.delete(origin_data, [6, 7, 8], axis=1)
    # load data and remove four axis: z, height, roll, pitch
    for i in range(1, data_num):
        origin_data = np.loadtxt(path + 'labels/%012d.txt' % i).reshape(-1, 7)
        # origin_data = np.delete(origin_data, [6, 7, 8], axis=1)
        if origin_data[0, 0] == 1:
            print(f'yolo dominated {i}')
            max_conf_1 += 1
        if np.all(origin_data[:, 0] == 0):
            print(f'no grasp {i}')
            no_grasp += 1
        data = np.concatenate((data, origin_data), axis=0)
    print('this is yolo dominated', max_conf_1)
    print('this is no grasp', no_grasp)

    conf_1_index = []
    for i in range(len(data)):
        if data[i, 0] == 1:
            conf_1_index.append(i)
    conf_1_index = np.asarray(conf_1_index)
    data_conf_1 = data[conf_1_index, -1]
    data_conf_0 = np.delete(data, conf_1_index, axis=0)[:, -1]
    print(f'mean conf of grasp 1 {np.mean(data_conf_1)}')
    print(f'std conf of grasp 1 {np.std(data_conf_1)}')
    print(f'max conf of grasp 1 {np.max(data_conf_1)}')
    print(f'min conf of grasp 1 {np.min(data_conf_1)}')
    print(f'mean conf of grasp 0 {np.mean(data_conf_0)}')
    print(f'std conf of grasp 0 {np.std(data_conf_0)}')
    print(f'max conf of grasp 0 {np.max(data_conf_0)}')
    print(f'min conf of grasp 0 {np.min(data_conf_0)}')

    print(data.shape)
    data_frame = pd.DataFrame({
        'grasp_flag': data[:, 0],
        'pos_x': data[:, 1],
        'pos_y': data[:, 2],
        'pos_z': data[:, 3],
        'box_length': data[:, 4],
        'box_width': data[:, 5],
        'ori_yaw': data[:, 6],
        'detect_conf': data[:, 7],
        })
    data_frame.to_csv(path + 'grasp_data.csv', index=False)

def data_preprocess_np_min_max(path, data_num, start_index=0, target_data_path=None, target_start_index=None):

    target_path = target_data_path + 'labels/'
    os.makedirs(target_path, exist_ok=True)
    scaler = MinMaxScaler()
    data_range = np.array([[0, 0, -0.14, 0, 0, 0, 0.5],
                           [1, 0.3, 0.14, 0.06, 0.06, np.pi, 1]])
    scaler.fit(data_range)

    # print(np.loadtxt(path + 'origin_labels/%012d.txt' % start_index).reshape(-1, 11)[0])
    # print(np.loadtxt(path + 'origin_labels/%012d.txt' % (start_index + 10000)).reshape(-1, 11)[0])

    # print(np.loadtxt(path + 'labels/%012d.txt' % 50000).reshape(-1, 7))
    # print(np.loadtxt(path + 'labels/%012d.txt' % 60000).reshape(-1, 7))

    max_length = 0
    for i in tqdm(range(start_index, data_num + start_index)):
        origin_data = np.loadtxt(path + 'origin_labels/%012d.txt' % i).reshape(-1, 11)
        data = np.delete(origin_data, [3, 6, 7, 8], axis=1)
        # data = origin_data
        tar_index = np.where(data[:, -2] < 0)[0]
        if len(tar_index) > 0:
            pass
            # print('this is tar index', tar_index)
            # print('this is ori', data[tar_index, :])
        data[tar_index, -2] += np.pi

        order = change_sequence(data)
        data = data[order]

        if len(data) > max_length:
            max_length = len(data)
        # print('this is origin data\n', data)
        # normal_data = scaler.transform(data)
        # print('this is normal data\n', normal_data)

        # print(i + target_start_index - start_index)
        np.savetxt(target_path + '%012d.txt' % (i + target_start_index - start_index), data, fmt='%.04f')

    print('this is the max length in the dataset', max_length)

def data_preprocess_np_standard(path, data_num, start_index, target_data_path=None, target_start_index=None):
    target_path = target_data_path + 'labels/'
    os.makedirs(target_path, exist_ok=True)
    scaler = StandardScaler()

    # print(np.loadtxt(path + 'labels/%012d.txt' % 50000).reshape(-1, 7))
    # print(np.loadtxt(path + 'labels/%012d.txt' % 60000).reshape(-1, 7))

    total_array = np.loadtxt(path + 'labels/%012d.txt' % start_index).reshape(-1, 7)
    # total_array = np.delete(total_array, [3, 6, 7, 8], axis=1)
    tar_index = np.where(total_array[:, -2] < 0)[0]
    total_array[tar_index, -2] += np.pi
    total_len = [total_array.shape[0]]
    for i in tqdm(range(start_index + 1, data_num + start_index)):
        origin_data = np.loadtxt(path + 'labels/%012d.txt' % i).reshape(-1, 7)
        # data = np.delete(origin_data, [3, 6, 7, 8], axis=1)
        tar_index = np.where(origin_data[:, -2] < 0)[0]
        origin_data[tar_index, -2] += np.pi

        total_array = np.concatenate((total_array, origin_data), axis=0)
        total_len.append(origin_data.shape[0])

    total_array[:, 1:] = scaler.fit_transform(total_array[:, 1:])

    seg_start_index = 0
    for i in tqdm(range(start_index, data_num + start_index)):
        # print('before\n', np.loadtxt(path + 'origin_labels/%012d.txt' % i).reshape(-1, 11))
        data_after = total_array[seg_start_index:seg_start_index + total_len[i - start_index], :]
        seg_start_index += total_len[i - start_index]
        # print('after\n', data_after)
        np.savetxt(target_path + '%012d.txt' % (i + target_start_index - start_index), data_after, fmt='%.04f')

    print(np.loadtxt(target_path + '%012d.txt' % 50000).reshape(-1, 7))
    print(np.loadtxt(target_path + '%012d.txt' % 60000).reshape(-1, 7))


def data_move(source_path, target_path, source_start_index, data_num, target_start_index):

    import shutil

    index_list = np.arange(target_start_index, data_num + target_start_index, 10000)
    for i in index_list:
        data = np.loadtxt(target_path + 'labels/%012d.txt' % i).reshape(-1, 7)
        print(np.round(data, 4))

    # print(np.loadtxt(target_path + '%012d.txt' % 50000).reshape(-1, 7))
    # print(np.loadtxt(target_path + '%012d.txt' % 60000).reshape(-1, 7))

    # for i in range(source_start_index, int(data_num + source_start_index)):
    #     cur_path = source_path + '%012d.txt' % (i)
    #     tar_path = target_path + '%012d.txt' % (i + target_start_index - source_start_index)
    #     shutil.copy(cur_path, tar_path)

def check_dataset():

    pass

if __name__ == '__main__':

    np.set_printoptions(suppress=True)

    # data_root = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/'
    data_root = '/home/ubuntu/Desktop/knolling_dataset/'
    # data_root = '/home/zhizhuo/Creative_Machines_Lab/knolling_dataset/'
    # data_path = data_root + 'grasp_dataset_03004/'
    data_path = data_root + 'grasp_pile_722_lab_test/'
    # data_path = data_root + 'origin_labels_713_lab/'

    target_data_path = data_root + 'grasp_pile_722_lab_test/'
    # target_data_path = data_root + 'origin_labels_713_lab/'

    data_num = 10000
    start_index = 0
    target_start_index = 0
    # data_preprocess_csv(data_path, data_num, start_index)
    # data_preprocess_np_standard(data_path, data_num, start_index, target_data_path, target_start_index)
    data_preprocess_np_min_max(data_path, data_num, start_index, target_data_path, target_start_index)

    # # source_path = '/home/zhizhuo/Creative_Machines_Lab/knolling_dataset/grasp_pile_715_lab_add/labels/'
    # source_path = '/home/ubuntu/Desktop/knolling_dataset/grasp_dataset_721_heavy/'
    # target_path = '/home/ubuntu/Desktop/knolling_dataset/grasp_dataset_721_heavy/'
    # os.makedirs(target_path, exist_ok=True)
    # source_start_index = 0
    # target_start_index = 480000
    # num = 300000
    # data_move(source_path, target_path, source_start_index, num, target_start_index)