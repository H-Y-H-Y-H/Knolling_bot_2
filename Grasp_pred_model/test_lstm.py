import sys
import numpy as np
import random
import os
sys.path.append('/home/zhizhuo/Creative_Machines_Lab/Knolling_bot_2/')
sys.path.append('/home/ubuntu/Desktop/Knolling_bot_2/')
# from Grasp_pred_model.Data_collection.grasp_or_yolo_collection import Arm_env
from network_lstm import LSTMRegressor
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from train_lstm import Generate_Dataset, data_split, collate_fn, para_dict
import matplotlib.pyplot as plt

if __name__ == '__main__':

    use_dataset = True

    if use_dataset == True:

        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        print("Device:", device)

        model_threshold_start = 0.0
        model_threshold_end = 1
        check_point = 51
        valid_num = 20000
        model_threshold = np.linspace(model_threshold_start, model_threshold_end, check_point)
        para_dict['wandb_flag'] = False
        para_dict['num_img'] = 450000
        para_dict['model_path'] = '../models/LSTM_918_0/'
        para_dict['data_path'] = '../../knolling_dataset/grasp_dataset_914/labels_1/'
        para_dict['run_name'] = para_dict['run_name'] + '_test'
        para_dict['hidden_size'] = 32
        para_dict['num_layers'] = 8
        para_dict['hidden_node_1'] = 32
        para_dict['hidden_node_2'] = 8
        para_dict['batch_size'] = 64
        para_dict['set_dropout'] = 0
        # test_file_para = '730_2_TPFN_'
        total_error = []

        num_img = para_dict['num_img']
        ratio = para_dict['ratio']
        box_one_img = para_dict['box_one_img']
        data_path = para_dict['data_path']
        box_test, grasp_test = data_split(data_path, num_img, ratio, box_one_img, test_model=True, valid_num=valid_num, set_conf=None)

        # create the train dataset and test dataset
        batch_size = para_dict['batch_size']
        test_dataset = Generate_Dataset(box_data=box_test, grasp_data=grasp_test)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_fn, drop_last=True)

        # initialize the parameters of the model
        input_size = para_dict['input_size']
        hidden_size = para_dict['hidden_size']
        num_layers = para_dict['num_layers']
        output_size = para_dict['output_size']
        learning_rate = para_dict['learning_rate']
        if para_dict['use_mse'] == True:
            model = LSTMRegressor(input_dim=input_size, hidden_dim=hidden_size, output_dim=output_size,
                                  num_layers=num_layers,
                                  batch_size=batch_size, device=device)
        else:
            model = LSTMRegressor(input_dim=input_size, hidden_dim=hidden_size, output_dim=output_size,
                                  num_layers=num_layers, hidden_node_1=para_dict['hidden_node_1'],
                                  hidden_node_2=para_dict['hidden_node_2'],
                                  batch_size=batch_size, device=device, criterion=nn.CrossEntropyLoss(),
                                  set_dropout=para_dict['set_dropout'])

        ###########################################################################
        model.load_state_dict(torch.load(para_dict['model_path'] + 'best_model.pt', map_location=device))
        ###########################################################################

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=para_dict['patience'], factor=para_dict['factor'])

        model.eval()

        model_pred_recall = []
        model_pred_precision = []
        model_pred_accuracy = []
        model_loss = []
        total_TP = []
        total_TN = []
        total_FP = []
        total_FN = []
        max_precision = -np.inf
        max_accuracy = -np.inf
        for i in range(len(model_threshold)):
            valid_loss = []
            test_loss = []
            with torch.no_grad():
                # print('eval')
                for batch_id, (box_data_batch, grasp_data_batch, num_box) in enumerate(test_loader):
                    # print(batch_id)
                    box_data_batch = box_data_batch.to(device, dtype=torch.float32)
                    grasp_data_batch = grasp_data_batch.to(device, dtype=torch.float32)

                    box_data_pack = pack_padded_sequence(box_data_batch, num_box, batch_first=True)
                    out = model.forward(box_data_pack)
                    if para_dict['use_mse'] == True:
                        loss = model.maskedMSELoss(predict=out, target=grasp_data_batch)
                        valid_loss.append(loss.item())
                    else:
                        # loss = model.maskedCrossEntropyLoss(predict=out, target=grasp_data_batch)
                        loss = model.maskedCrossEntropyLoss(predict=out, target=grasp_data_batch, boxes_data=box_data_batch)
                        valid_loss.append(loss.item())
                        not_one_result_per_img, tar_true, tar_false, TP, TN, FP, FN,\
                        yolo_dominated_TP, yolo_dominated_TN, yolo_dominated_FP, yolo_dominated_FN, \
                        grasp_dominated_TP, grasp_dominated_TN, grasp_dominated_FP, grasp_dominated_FN, loss = model.detect_accuracy(predict=out, target=grasp_data_batch, box_conf=box_data_batch, model_threshold=model_threshold[i])
                        test_loss.append(loss.item())
                        pass

            if TP + FN == 0:
                recall = 0
                model_pred_recall.append(0)
            else:
                recall = (TP) / (TP + FN)
                model_pred_recall.append(recall)
            if TP + FP == 0:
                precision = 0
                model_pred_precision.append(0)
            else:
                precision = (TP) / (TP + FP)
                model_pred_precision.append(precision)
            accuracy = (TP + FN) / (TP + TN + FP + FN)
            model_pred_accuracy.append(accuracy)

            if precision > max_precision:
                max_precision_threshold = model_threshold[i]
                max_precision = precision
            if accuracy > max_accuracy:
                max_accuracy_threshold = model_threshold[i]
                max_accuracy = accuracy

            avg_valid_loss = np.mean(valid_loss)
            avg_test_loss = np.mean(test_loss)
            model_loss.append(avg_test_loss)
            print('\navg valid loss', avg_valid_loss)
            print('avg test loss', avg_test_loss)

            print('total_img', valid_num)
            print('not one result per img', not_one_result_per_img)
            print('threshold', model_threshold[i])
            print('tar_true', tar_true)
            print('tar_false', tar_false)
            print('TP:', TP)
            print('TN:', TN)
            print('FP:', FP)
            print('FN:', FN)
            total_TP.append(TP)
            total_TN.append(TN)
            total_FP.append(FP)
            total_FN.append(FN)
            print('Accuracy (TP + FN) / all: %.04f' % ((TP + FN) / (tar_true + tar_false)))
            if (TP + FN) == 0:
                print('Recall (TP / (TP + FN)) 0')
            else:
                print('Recall (TP / (TP + FN)) %.04f' % (TP / (TP + FN)))
            if (TP + FP) == 0:
                print('Precision (TP / (TP + FP)) 0')
            else:
                print('Precision (TP / (TP + FP)) %.04f' % (TP / (TP + FP)))
            print('(FN / (FP + FN)) %.04f' % (FN / (FP + FN)))

            model.tar_true = 0
            model.tar_false = 0
            model.TP = 0
            model.TN = 0
            model.FP = 0
            model.FN = 0
            model.yolo_dominated_TP = 0
            model.yolo_dominated_TN = 0
            model.yolo_dominated_FP = 0
            model.yolo_dominated_FN = 0
            model.grasp_dominated_TP = 0
            model.grasp_dominated_TN = 0
            model.grasp_dominated_FP = 0
            model.grasp_dominated_FN = 0

            model.not_one_result_per_img = 0

        model_pred_recall = np.asarray(model_pred_recall)
        model_pred_precision = np.asarray(model_pred_precision)
        model_pred_accuracy = np.asarray(model_pred_accuracy)
        model_loss = np.asarray(model_loss)
        model_loss_mean = np.mean(model_loss)
        model_loss_std = np.std(model_loss)
        total_TP = np.asarray(total_TP)
        total_TN = np.asarray(total_TN)
        total_FP = np.asarray(total_FP)
        total_FN = np.asarray(total_FN)

        print(f'When the threshold is {max_accuracy_threshold}, the max accuracy is {max_accuracy}')
        print(f'When the threshold is {max_precision_threshold}, the max precision is {max_precision}')

        plt.plot(model_threshold, model_pred_recall, label='LSTM_pred_recall')
        plt.plot(model_threshold, model_pred_precision, label='LSTM_pred_precision')
        plt.plot(model_threshold, model_pred_accuracy, label='LSTM_pred_accuracy')
        plt.xlabel('model_threshold')
        plt.title('analysis of LSTM prediction')
        plt.legend()
        plt.savefig(para_dict['model_path'] + 'model_pred_analysis_labels_1.png')
        plt.show()

        np.savetxt(para_dict['model_path'] + 'model_loss_labels_1.txt', model_loss)

        total_evaluate_data = np.concatenate(([model_threshold], [model_pred_recall], [model_pred_precision], [model_pred_accuracy],
                                              [total_TP], [total_TN], [total_FP], [total_FN]), axis=0).T
        np.savetxt(para_dict['model_path'] + 'model_data_labels_1.txt', total_evaluate_data)

        with open(para_dict['model_path'] + "model_pred_anlysis_labels_1.txt", "w") as f:
            f.write('----------- Dataset -----------\n')
            f.write(f'valid_num: {valid_num}\n')
            f.write(f'tar_true: {tar_true}\n')
            f.write(f'tar_false: {tar_false}\n')
            f.write(f'threshold_start: {model_threshold_start}\n')
            f.write(f'threshold_end: {model_threshold_end}\n')
            f.write(f'threshold: {max_accuracy_threshold}, max accuracy: {max_accuracy}\n')
            f.write(f'threshold: {max_precision_threshold}, max precision: {max_precision}\n')
            f.write('----------- Dataset -----------\n')

            f.write('----------- Statistics -----------\n')
            f.write(f'model_loss_mean: {model_loss_mean}\n')
            f.write(f'model_loss_std: {model_loss_std}\n')
            f.write('----------- Statistics sundry_box_4-----------\n')

            for i in range(len(model_threshold)):
                f.write(f'threshold: {model_threshold[i]:.6f}, recall: {model_pred_recall[i]:.4f}, precision: {model_pred_precision[i]:.4f}, accuracy: {model_pred_accuracy[i]:.4f},'
                        f'TP: {total_TP[i]}, TN: {total_TN[i]}, FP: {total_FP[i]}, FN: {total_FN[i]}\n')


    else:
        startnum = 0
        endnum =   3
        thread = 0
        CLOSE_FLAG = False
        pile_flag = True
        try_grasp_flag = True
        test_pile_detection = False
        save_img_flag = True
        use_grasp_model = True

        max_box_num = 21
        min_box_num = 18
        mm2px = 530 / 0.34

        # np.random.seed(150)
        # random.seed(150)
        urdf_path = '../ASSET/urdf/'
        if use_grasp_model == True:

            if torch.cuda.is_available():
                device = 'cuda:0'
            else:
                device = 'cpu'
            print("Device:", device)
            total_error = []
            env = Arm_env(max_step=1, is_render=True, endnum=endnum, save_img_flag=save_img_flag, urdf_path=urdf_path,
                          use_grasp_model=use_grasp_model, para_dict=para_dict, total_error=total_error)
        else:
            env = Arm_env(max_step=1, is_render=True, endnum=endnum, save_img_flag=save_img_flag, urdf_path=urdf_path)
        data_root = '../temp_data/'
        os.makedirs(data_root, exist_ok=True)
        os.makedirs(data_root + 'origin_images/', exist_ok=True)
        os.makedirs(data_root + 'origin_labels/', exist_ok=True)

        conf_crowded_total = []
        conf_normal_total = []
        exist_img_num = startnum
        while True:
            num_item = int(np.random.uniform(min_box_num, max_box_num + 1))
            img_per_epoch = env.reset_table(close_flag=CLOSE_FLAG, data_root=data_root,
                                         num_item=num_item, thread=thread, epoch=exist_img_num,
                                         pile_flag=pile_flag,
                                         try_grasp_flag=try_grasp_flag)
            exist_img_num += img_per_epoch
