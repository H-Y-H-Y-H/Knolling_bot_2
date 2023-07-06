import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

def data_preprocess_csv(path, data_num, start_index):

    max_conf_1 = 0
    data = []
    i = 0
    origin_data = np.loadtxt(path + 'origin_labels/%012d.txt' % i).reshape(-1, 11)
    data = np.delete(origin_data, [6, 7, 8], axis=1)
    # load data and remove four axis: z, height, roll, pitch
    for i in range(1, data_num):
        origin_data = np.loadtxt(path + 'origin_labels/%012d.txt' % i).reshape(-1, 11)
        origin_data = np.delete(origin_data, [6, 7, 8], axis=1)
        if origin_data[0, 0] == 1 and len(origin_data) != 1:
            print(f'here {i}')
            max_conf_1 += 1
        data = np.concatenate((data, origin_data), axis=0)
    print(max_conf_1)

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

def data_preprocess_np(path, data_num, start_index=0):

    target_path = path + 'labels/'
    os.makedirs(target_path, exist_ok=True)
    scaler = MinMaxScaler()
    data_range = np.array([[0, 0, -0.14, 0, 0, 0, 0.5],
                           [1, 0.3, 0.14, 0.06, 0.06, np.pi, 1]])
    scaler.fit(data_range)

    # print(np.loadtxt(path + 'origin_labels/%012d.txt' % start_index).reshape(-1, 11)[0])
    # print(np.loadtxt(path + 'origin_labels/%012d.txt' % (start_index + 10000)).reshape(-1, 11)[0])

    # print(np.loadtxt(path + 'labels/%012d.txt' % 150000).reshape(-1, 7))
    # print(np.loadtxt(path + 'labels/%012d.txt' % 160000).reshape(-1, 7))

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
        # print('this is origin data\n', data)
        normal_data = scaler.transform(data)
        # print('this is normal data\n', normal_data)

        np.savetxt(target_path + '%012d.txt' % (i), normal_data, fmt='%.04f')

def data_move(source_path, target_path, source_start_index, data_num):
    import shutil

    print(np.loadtxt(source_path + '%012d.txt' % 50000).reshape(-1, 7))
    print(np.loadtxt(source_path + '%012d.txt' % 60000).reshape(-1, 7))

    for i in range(source_start_index, int(data_num + source_start_index)):
        cur_path = source_path + '%012d.txt' % (i)
        tar_path = target_path + '%012d.txt' % (i + 350000)
        shutil.copy(cur_path, tar_path)

def check_dataset():

    pass

if __name__ == '__main__':

    data_root = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/'
    # data_path = data_root + 'grasp_dataset_03004/'
    data_path = data_root + 'grasp_pile_706_laptop/'
    # data_path = data_root + 'origin_labels_706_lab/'

    data_num = 5000
    start_index = 0
    data_preprocess_csv(data_path, data_num, start_index)
    # data_preprocess_np(data_path, data_num, start_index)

    # source_path = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/origin_labels_706_lab/labels/'
    # target_path = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/grasp_dataset_03004/labels/'
    # source_start_index = 0
    # num = 100000
    # data_move(source_path, target_path, source_start_index, num)