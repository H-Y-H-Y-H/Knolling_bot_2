import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

def data_preprocess_csv(path, data_num, start_index):

    max_conf_1 = 0
    no_grasp = 0
    data = []
    i = 0
    origin_data = np.loadtxt(path + 'origin_labels/%012d.txt' % i).reshape(-1, 11)
    data = np.delete(origin_data, [6, 7, 8], axis=1)
    # load data and remove four axis: z, height, roll, pitch
    for i in range(1, data_num):
        origin_data = np.loadtxt(path + 'origin_labels/%012d.txt' % i).reshape(-1, 11)
        origin_data = np.delete(origin_data, [6, 7, 8], axis=1)
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
    scaler = StandardScaler()
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
        # delete the
        data = np.delete(origin_data, [3, 6, 7, 8], axis=1)
        tar_index = np.where(data[:, -2] < 0)[0]
        if len(tar_index) > 0:
            pass
            # print('this is tar index', tar_index)
            # print('this is ori', data[tar_index, :])
        data[tar_index, -2] += np.pi
        if len(data) > max_length:
            max_length = len(data)
        # print('this is origin data\n', data)
        # normal_data = scaler.transform(data)
        # print('this is normal data\n', normal_data)

        # print(i + target_start_index - start_index)
        np.savetxt(target_path + '%012d.txt' % (i + target_start_index - start_index), data, fmt='%.04f')

    print('this is the max length in the dataset', max_length)

def data_preprocess_np_standard(path, data_num, start_index):
    target_path = path + 'labels/'
    os.makedirs(target_path, exist_ok=True)
    scaler = StandardScaler()

    total_array = np.loadtxt(path + 'origin_labels/%012d.txt' % start_index).reshape(-1, 11)
    total_array = np.delete(total_array, [3, 6, 7, 8], axis=1)
    tar_index = np.where(total_array[:, -2] < 0)[0]
    total_array[tar_index, -2] += np.pi
    total_len = [total_array.shape[0]]
    for i in tqdm(range(start_index + 1, data_num + start_index)):
        origin_data = np.loadtxt(path + 'origin_labels/%012d.txt' % i).reshape(-1, 11)
        data = np.delete(origin_data, [3, 6, 7, 8], axis=1)
        tar_index = np.where(data[:, -2] < 0)[0]
        data[tar_index, -2] += np.pi

        total_array = np.concatenate((total_array, data), axis=0)
        total_len.append(data.shape[0])

    total_array[:, 1:] = scaler.fit_transform(total_array[:, 1:])

    seg_start_index = 0
    for i in tqdm(range(start_index, data_num + start_index)):
        # print('before\n', np.loadtxt(path + 'origin_labels/%012d.txt' % i).reshape(-1, 11))
        data_after = total_array[seg_start_index:seg_start_index + total_len[i - start_index], :]
        seg_start_index += total_len[i - start_index]
        # print('after\n', data_after)
        np.savetxt(target_path + '%012d.txt' % (i - 20000), data_after, fmt='%.04f')

    # print('here')


def data_move(source_path, target_path, source_start_index, data_num, target_start_index):
    import shutil

    print(np.loadtxt(source_path + '%012d.txt' % 50000).reshape(-1, 7))
    print(np.loadtxt(source_path + '%012d.txt' % 60000).reshape(-1, 7))

    for i in range(source_start_index, int(data_num + source_start_index)):
        cur_path = source_path + '%012d.txt' % (i)
        tar_path = target_path + '%012d.txt' % (i + target_start_index - source_start_index)
        shutil.copy(cur_path, tar_path)

        # os.remove(target_path + '%012d.txt' % (i + 350000))

def check_dataset():

    pass

if __name__ == '__main__':

    # data_root = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/'
    data_root = '/home/zhizhuo/Creative_Machines_Lab/knolling_dataset/'
    # data_path = data_root + 'grasp_dataset_03004/'
    data_path = data_root + 'grasp_pile_714_laptop/'
    # data_path = data_root + 'origin_labels_713_lab/'

    target_data_path = data_root + 'grasp_pile_714_laptop/'
    # target_data_path = data_root + 'origin_labels_713_lab/'

    data_num = 480000
    start_index = 0
    target_start_index = 0
    # data_preprocess_csv(data_path, data_num, start_index)
    # data_preprocess_np_standard(data_path, data_num, start_index)
    data_preprocess_np_min_max(data_path, data_num, start_index, target_data_path, target_start_index)

    # # source_path = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/origin_labels_710_lab/labels/'
    # # source_path = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/grasp_pile_710_laptop/labels/'
    # source_path = '/home/zhizhuo/Creative_Machines_Lab/knolling_dataset/origin_labels_713_lab/labels/'
    # # target_path = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/grasp_dataset_711/labels/'
    # target_path = '/home/zhizhuo/Creative_Machines_Lab/knolling_dataset/grasp_dataset_713/labels/'
    # os.makedirs(target_path, exist_ok=True)
    # source_start_index = 0
    # target_start_index = 450000
    # num = 700000
    # data_move(source_path, target_path, source_start_index, num, target_start_index)