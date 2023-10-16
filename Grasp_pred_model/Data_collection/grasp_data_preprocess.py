import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

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
    origin_data = np.loadtxt(path + 'origin_labels/%012d.txt' % i).reshape(-1, 11)
    # data = origin_data
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

def data_preprocess_np_min_max(path, data_num, start_index=0, target_data_path=None, target_start_index=None, dropout_prob=None):

    target_path = target_data_path
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
    total_not_all_zero = 0
    tar_true = 0
    tar_false = 0
    output_index = target_start_index - 1
    for i in tqdm(range(start_index, data_num + start_index)):
        origin_data = np.loadtxt(path + '%012d.txt' % i).reshape(-1, 7)
        # data_x_offset = np.random.uniform(-0.05, 0.05)
        # data_y_offset = np.random.uniform(-0.10, 0.10)
        # data = np.delete(origin_data, [3, 6, 7, 8], axis=1)
        # data[:, 1] += data_x_offset
        # data[:, 2] += data_y_offset
        data = origin_data
        if np.any(data[:, 0] != 0):
            # print('this is index of not all zero data', i)
            total_not_all_zero += 1
            output_index += 1
            # print(total_not_all_zero)
        else:
            flag = np.random.rand()
            if flag < dropout_prob:
                continue
            else:
                # print('this is output index', output_index)
                output_index += 1
                pass

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
        tar_true += len(np.where(data[:, 0] == 1)[0])
        tar_false += len(np.where(data[:, 0] == 0)[0])
        # np.savetxt(target_path + '%012d.txt' % (output_index), data, fmt='%.04f')

    print('this is total not all zero', total_not_all_zero)
    print('this is the max length in the dataset', max_length)
    print('this is total tar true in the result dataset', tar_true)
    print('this is total tar false in the result dataset', tar_false)
    print('this is total num of images', output_index)

def data_analysis_preprocess(path, data_num, start_index=0, target_data_path=None, target_start_index=None, dropout_prob=None, set_conf=None):

    target_path = target_data_path
    os.makedirs(target_path, exist_ok=True)

    max_length = 0
    total_not_all_zero = 0
    tar_true = 0
    tar_false = 0
    output_index = target_start_index - 1
    conf_low_num = 0
    total_num = 0
    for i in tqdm(range(start_index, data_num + start_index)):
        origin_data = np.loadtxt(path + '%012d.txt' % i).reshape(-1, 7)
        # data_x_offset = np.random.uniform(-0.05, 0.05)
        # data_y_offset = np.random.uniform(-0.10, 0.10)
        # data = np.delete(origin_data, [3, 6, 7, 8], axis=1)
        # data[:, 1] += data_x_offset
        # data[:, 2] += data_y_offset
        data = origin_data

        conf_low_index = np.where(data[:, -1] < set_conf)[0]
        # manually set the conf
        data[conf_low_index, -1] = set_conf

        conf_low_num += len(conf_low_index)
        total_num += len(data)

        if np.any(data[:, 0] != 0):
            # print('this is index of not all zero data', i)
            total_not_all_zero += 1
            output_index += 1
            # print(total_not_all_zero)
        else:
            flag = np.random.rand()
            if flag < dropout_prob:
                continue
            else:
                # print('this is output index', output_index)
                output_index += 1
                pass

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
        tar_true += len(np.where(data[:, 0] == 1)[0])
        tar_false += len(np.where(data[:, 0] == 0)[0])
        # np.savetxt(target_path + '%012d.txt' % (output_index), data, fmt='%.04f')

    print('this is total not all zero', total_not_all_zero)
    print('this is the max length in the dataset', max_length)
    print('this is total tar true in the result dataset', tar_true)
    print('this is total tar false in the result dataset', tar_false)
    print('this is total num of images', output_index)

    print('this is total num', total_num)
    print('this is num with high conf', conf_low_num)


def data_preprocess_mix(source_path_1, data_num_1, start_index_1, source_path_2, data_num_2, start_index_2,
                        target_data_path=None, target_start_index=None, dropout_prob=None):

    target_path = target_data_path
    os.makedirs(target_path, exist_ok=True)

    ratio = data_num_1 / data_num_2

    index_2 = start_index_2
    output_index = start_index_1
    for i in tqdm(range(start_index_1, data_num_1 + start_index_1)):

        origin_data_1 = np.loadtxt(source_path_1 + '%012d.txt' % i).reshape(-1, 11)
        data_1 = np.delete(origin_data_1, [3, 6, 7, 8], axis=1)

        temp_index = np.where(data_1[:, -2] < 0)[0]
        data_1[temp_index, -2] += np.pi
        order_1 = change_sequence(data_1)
        data_1 = data_1[order_1]

        np.savetxt(target_path + '%012d.txt' % (output_index), data_1, fmt='%.04f')
        output_index += 1

        if i % ratio == 0:

            origin_data_2 = np.loadtxt(source_path_2 + '%012d.txt' % index_2).reshape(-1, 11)
            data_2 = np.delete(origin_data_2, [3, 6, 7, 8], axis=1)

            temp_index = np.where(data_2[:, -2] < 0)[0]
            data_2[temp_index, -2] += np.pi
            order_2 = change_sequence(data_2)
            data_2 = data_2[order_2]
            np.savetxt(target_path + '%012d.txt' % (output_index), data_2, fmt='%.04f')
            index_2 += 1
            output_index += 1

def data_move(source_path, target_path, source_start_index, data_num, target_start_index):

    import shutil

    index_list = np.arange(target_start_index, data_num + target_start_index, 10000)
    for i in index_list:
        data = np.loadtxt(target_path + 'labels/%012d.txt' % i).reshape(-1, 7)
        print(np.round(data, 4))

    print(np.loadtxt(target_path + '%012d.txt' % 50000).reshape(-1, 7))
    print(np.loadtxt(target_path + '%012d.txt' % 60000).reshape(-1, 7))

    # for i in tqdm(range(source_start_index, int(data_num + source_start_index))):
    #     cur_path = source_path + '%012d.txt' % (i)
    #     tar_path = target_path + '%012d.txt' % (i + target_start_index - source_start_index)
    #     shutil.copy(cur_path, tar_path)

def set_yolo_conf(source_path, total_num, target_path, start_index, set_conf=0.6):

    os.makedirs(target_path, exist_ok=True)

    print(np.loadtxt(target_path + '%012d.txt' % 1000).reshape(-1, 7))

    for i in tqdm(range(start_index, total_num + start_index)):
        origin_data = np.loadtxt(source_path + '%012d.txt' % i).reshape(-1, 7)
        origin_data[:, -1] = set_conf
        np.savetxt(target_path + '%012d.txt' % i, origin_data)

def yolo_accuracy_analysis(path, analysis_path, total_num, ratio, threshold_start, threshold_end, valid_num, check_point = 20):

    criterion = nn.CrossEntropyLoss()

    valid_start_index = int(total_num * ratio)
    data = np.loadtxt(path + '%012d.txt' % valid_start_index)
    for i in tqdm(range(valid_start_index + 1, valid_num + valid_start_index)):
        new_data = np.loadtxt(path + '%012d.txt' % i)
        data = np.concatenate((data, new_data), axis=0)

    yolo_threshold = np.linspace(threshold_start, threshold_end, check_point)
    print(yolo_threshold)

    yolo_pred_recall = []
    yolo_pred_precision = []
    yolo_pred_accuracy = []
    max_precision = -np.inf
    max_accuracy = -np.inf
    total_loss = []
    total_TP = []
    total_TN = []
    total_FP = []
    total_FN = []
    for i in range(len(yolo_threshold)):
        yolo_pred_P = np.where((data[:, -1] >= yolo_threshold[i]))[0]
        yolo_pred_N = np.where((data[:, -1] < yolo_threshold[i]))[0]
        tar_True = np.where(data[:, 0] == 1)[0]
        tar_False = np.where(data[:, 0] == 0)[0]

        yolo_pred_TP = len(np.intersect1d(yolo_pred_P, tar_True))
        yolo_pred_TN = len(np.intersect1d(yolo_pred_N, tar_True))
        yolo_pred_FP = len(np.intersect1d(yolo_pred_P, tar_False))
        yolo_pred_FN = len(np.intersect1d(yolo_pred_N, tar_False))
        if yolo_pred_TP + yolo_pred_FN == 0:
            recall = 0
            yolo_pred_recall.append(0)
        else:
            recall = (yolo_pred_TP) / (yolo_pred_TP + yolo_pred_FN)
            yolo_pred_recall.append(recall)
        if yolo_pred_TP + yolo_pred_FP == 0:
            precision = 0
            yolo_pred_precision.append(0)
        else:
            precision = (yolo_pred_TP) / (yolo_pred_TP + yolo_pred_FP)
            yolo_pred_precision.append(precision)
        accuracy = (yolo_pred_TP + yolo_pred_FN) / (yolo_pred_TP + yolo_pred_TN + yolo_pred_FP + yolo_pred_FN)
        yolo_pred_accuracy.append(accuracy)

        if precision > max_precision:
            max_precision_threshold = yolo_threshold[i]
            max_precision = precision
        if accuracy > max_accuracy:
            max_accuracy_threshold = yolo_threshold[i]
            max_accuracy = accuracy

        # calculate the loss mean and std
        yolo_output = np.zeros((len(data), 2))
        yolo_output[yolo_pred_P, 1] = 1
        yolo_output[yolo_pred_N, 0] = 1
        yolo_output = torch.from_numpy(yolo_output)
        yolo_label = torch.from_numpy(data[:, 0])
        total_loss.append(criterion(yolo_output, yolo_label.long()))

        print('this is yolo pred P', len(yolo_pred_P))
        print('this is yolo pred N', len(yolo_pred_N))
        print('this is yolo pred TP', yolo_pred_TP)
        print('this is yolo pred TN', yolo_pred_TN)
        print('this is yolo pred FP', yolo_pred_FP)
        print('this is yolo pred FN', yolo_pred_FN)
        total_TP.append(yolo_pred_TP)
        total_TN.append(yolo_pred_TN)
        total_FP.append(yolo_pred_FP)
        total_FN.append(yolo_pred_FN)
        print('Accuracy (TP + FN) / all: %.04f' % accuracy)
        if (yolo_pred_TP + yolo_pred_FN) == 0:
            print('Recall (TP / (TP + FN)) 0')
        else:
            print('Recall (TP / (TP + FN)) %.04f' % (yolo_pred_TP / (yolo_pred_TP + yolo_pred_FN)))
        if (yolo_pred_TP + yolo_pred_FP) == 0:
            print('Precision (TP / (TP + FP)) 0')
        else:
            print('Precision (TP / (TP + FP)) %.04f' % (yolo_pred_TP / (yolo_pred_TP + yolo_pred_FP)))
        print('(FN / (FP + FN)) %.04f' % (yolo_pred_FN / (yolo_pred_FP + yolo_pred_FN)))
        print('\n')

    yolo_pred_recall = np.asarray(yolo_pred_recall)
    yolo_pred_precision = np.asarray(yolo_pred_precision)
    yolo_pred_accuracy = np.asarray(yolo_pred_accuracy)
    yolo_loss = np.asarray(total_loss)
    yolo_loss_mean = np.mean(yolo_loss)
    yolo_loss_std = np.std(yolo_loss)
    total_TP = np.asarray(total_TP)
    total_TN = np.asarray(total_TN)
    total_FP = np.asarray(total_FP)
    total_FN = np.asarray(total_FN)

    print(f'When the threshold is {max_accuracy_threshold}, the max accuracy is {max_accuracy}')
    print(f'When the threshold is {max_precision_threshold}, the max precision is {max_precision}')

    plt.plot(yolo_threshold, yolo_pred_recall, label='yolo_pred_recall')
    plt.plot(yolo_threshold, yolo_pred_precision, label='yolo_pred_precision')
    plt.plot(yolo_threshold, yolo_pred_accuracy, label='yolo_pred_accuracy')
    plt.xlabel('yolo_threshold')
    plt.title('analysis of yolo prediction')
    plt.legend()
    plt.savefig(analysis_path + 'yolo_pred_analysis.png')
    plt.show()

    total_evaluate_data = np.concatenate(([yolo_threshold], [yolo_pred_recall], [yolo_pred_precision], [yolo_pred_accuracy],
                                                [total_TP], [total_TN], [total_FP], [total_FN]), axis=0).T
    np.savetxt(analysis_path + 'yolo_data.txt', total_evaluate_data)
    np.savetxt(analysis_path + 'yolo_loss.txt', yolo_loss)

    with open(analysis_path + "yolo_pred_anlysis.txt", "w") as f:
        f.write('----------- Dataset -----------\n')
        f.write(f'valid_num: {valid_num}\n')
        f.write(f'tar_true: {len(tar_True)}\n')
        f.write(f'tar_false: {len(tar_False)}\n')
        f.write(f'threshold_start: {threshold_start}\n')
        f.write(f'threshold_end: {threshold_end}\n')
        f.write(f'threshold: {max_accuracy_threshold}, max accuracy: {max_accuracy}\n')
        f.write(f'threshold: {max_precision_threshold}, max precision: {max_precision}\n')
        f.write('----------- Dataset -----------\n')

        f.write('----------- Statistics -----------\n')
        f.write(f'yolo_loss_mean: {yolo_loss_mean}\n')
        f.write(f'yolo_loss_std: {yolo_loss_std}\n')
        f.write('----------- Statistics -----------\n')

        for i in range(len(yolo_threshold)):
            f.write(f'threshold: {yolo_threshold[i]:.6f}, recall: {yolo_pred_recall[i]:.4f}, precision: {yolo_pred_precision[i]:.4f}, accuracy: {yolo_pred_accuracy[i]:.4f},'
                f'TP: {total_TP[i]}, TN: {total_TN[i]}, FP: {total_FP[i]}, FN: {total_FN[i]}\n')

if __name__ == '__main__':

    np.set_printoptions(suppress=True)

    # data_path = '../../../knolling_dataset/grasp_dataset_829/labels_1_high_conf/'
    # target_data_path = '../../../knolling_dataset/grasp_dataset_829/labels_1_high_conf/'
    # # target_data_path = data_root + 'origin_labels_713_lab/'
    #
    # data_num = 520000
    # start_index = 0
    # target_start_index = 0
    # dropout_prob = 0
    # set_conf = 0.95
    # data_analysis_preprocess(data_path, data_num, start_index, target_data_path, target_start_index, dropout_prob, set_conf)

    # source_path = '../../../knolling_dataset/grasp_dataset_914_sparse_labdesk/sim_labels/'
    # target_path = '../../../knolling_dataset/grasp_dataset_914/labels_1/'
    # os.makedirs(target_path, exist_ok=True)
    # source_start_index = 120000
    # target_start_index = 120000
    # num = 30000
    # data_move(source_path, target_path, source_start_index, num, target_start_index)

    # source_path = '../../../knolling_dataset/grasp_dataset_914/labels_1/'
    # target_path = '../../../knolling_dataset/grasp_dataset_914/labels_2/'
    # total_num = 450000
    # start_index = 0
    # set_yolo_conf(source_path=source_path, total_num=total_num, target_path=target_path, start_index=start_index, set_conf=0.97)

    data_path = '../../../knolling_dataset/grasp_dataset_914/labels_1/'
    analysis_path = '../../ASSET/models/LSTM_918_0/'
    valid_num = 20000
    yolo_accuracy_analysis(path=data_path, total_num=450000, ratio=0.8, threshold_start=0.0, threshold_end=1, check_point=51, valid_num=valid_num, analysis_path=analysis_path)

    # source_path_1 = '../../../knolling_dataset/grasp_dataset_914_crowded_lab/sim_labels/'
    # source_path_2 = '../../../knolling_dataset/grasp_dataset_914_sparse_lab/sim_labels/'
    # num_1 = 300000
    # num_2 = 150000
    # start_index_1 = 0
    # start_index_2 = 0
    # target_data_path = '../../../knolling_dataset/grasp_dataset_914/labels_1/'
    # target_start_index = 0
    # data_preprocess_mix(source_path_1, num_1, start_index_1, source_path_2, num_2, start_index_2,
    #                     target_data_path, target_start_index, dropout_prob=None)