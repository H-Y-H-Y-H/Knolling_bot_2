import sys
import numpy as np
import random
import os
sys.path.append('/home/zhizhuo/Creative_Machines_Lab/Knolling_bot_2/')
sys.path.append('/home/ubuntu/Desktop/Knolling_bot_2/')
# from Grasp_pred_model.Data_collection.grasp_or_yolo_collection import Arm_env
from network import LSTMRegressor
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from train_model import Generate_Dataset, data_split, collate_fn, para_dict

if __name__ == '__main__':

    use_dataset = True

    if use_dataset == True:

        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        print("Device:", device)

        para_dict['wandb_flag'] = False
        para_dict['num_img'] = 296000
        para_dict['model_path'] = '../Grasp_pred_model/results/LSTM_727_2_heavy_multi_dropout0.5/'
        para_dict['data_path'] = '../../knolling_dataset/grasp_dataset_726_ratio_multi/labels_2/'
        para_dict['run_name'] = para_dict['run_name'] + '_test'
        para_dict['hidden_size'] = 32
        para_dict['num_layers'] = 8
        para_dict['hidden_node_1'] = 32
        para_dict['hidden_node_2'] = 8
        para_dict['batch_size'] = 64
        test_file_para = '727_2_TPFN_'
        total_error = []
        # '/home/zhizhuo/Creative_Machines_Lab/knolling_dataset/grasp_dataset_713/labels/'

        num_img = para_dict['num_img']
        ratio = para_dict['ratio']
        box_one_img = para_dict['box_one_img']
        data_path = para_dict['data_path']
        box_test, grasp_test, yolo_dominated_true, grasp_dominated, tar_true_grasp, tar_false_grasp = data_split(data_path, num_img, ratio, box_one_img, test_model=True)

        # create the train dataset and test dataset
        batch_size = para_dict['batch_size']
        test_dataset = Generate_Dataset(box_data=box_test, grasp_data=grasp_test)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True,
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
        model.load_state_dict(torch.load(para_dict['model_path'] + 'best_model.pt'))
        ###########################################################################

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=para_dict['patience'], factor=para_dict['factor'])

        valid_loss = []
        model.eval()
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
                    not_one_result, tar_true, tar_false, TP, TN, FP, FN,\
                    yolo_dominated_TP, yolo_dominated_TN, yolo_dominated_FP, yolo_dominated_FN, \
                    grasp_dominated_TP, grasp_dominated_TN, grasp_dominated_FP, grasp_dominated_FN = model.detect_accuracy(predict=out, target=grasp_data_batch, box_conf=box_data_batch)


        avg_valid_loss = np.mean(valid_loss)
        print('\n avg valid loss', avg_valid_loss)

        print('total_img', int(num_img - num_img * ratio))
        print('not one result', not_one_result)

        print('tar_true', tar_true)
        print('tar_false', tar_false)
        print('TP:', TP)
        print('TN:', TN)
        print('FP:', FP)
        print('FN:', FN)
        print('Accuracy (TP + FN) / all: %.04f' % ((TP + FN) / (tar_true + tar_false)))
        print('Recall (TP / (TP + FN)) %.04f' % (TP / (TP + FN)))
        print('Precision (TP / (TP + FP)) %.04f' % (TP / (TP + FP)))
        print('(FN / (FP + FN)) %.04f' % (FN / (FP + FN)))
        print('yolo_dominated_TP:', yolo_dominated_TP)
        print('yolo_dominated_TN:', yolo_dominated_TN)
        print('yolo_dominated_FP:', yolo_dominated_FP)
        print('yolo_dominated_FN:', yolo_dominated_FN)
        print('yolo_dominated_Accuracy: %.04f' % ((yolo_dominated_TP + yolo_dominated_FN) / (yolo_dominated_TP + yolo_dominated_TN + yolo_dominated_FP + yolo_dominated_FN)))
        print('yolo_dominated_Recall: %.04f' % (yolo_dominated_TP / (yolo_dominated_TP + yolo_dominated_FN)))
        print('yolo_dominated_Precision: %.04f' % (yolo_dominated_TP / (yolo_dominated_TP + yolo_dominated_FP)))
        print('grasp_dominated_TP:', grasp_dominated_TP)
        print('grasp_dominated_TN:', grasp_dominated_TN)
        print('grasp_dominated_FP:', grasp_dominated_FP)
        print('grasp_dominated_FN:', grasp_dominated_FN)
        print('grasp_dominated_Accuracy: %.04f' % ((grasp_dominated_TP + grasp_dominated_FN) / (grasp_dominated_TP + grasp_dominated_TN + grasp_dominated_FP + grasp_dominated_FN)))
        print('grasp_dominated_Recall: %.04f' % (grasp_dominated_TP / (grasp_dominated_TP + grasp_dominated_FN)))
        print('grasp_dominated_Precision: %.04f' % (grasp_dominated_TP / (grasp_dominated_TP + grasp_dominated_FP)))

        with open(para_dict['model_path'] + test_file_para + "test.txt", "w") as f:
            f.write('----------- Dataset -----------\n')
            f.write(f'total img: {int(num_img - num_img * ratio)}\n')
            f.write(f'yolo dominated: {yolo_dominated_true}\n')
            f.write(f'yolo dominated: {yolo_dominated_true}\n')
            f.write(f'grasp dominated: {grasp_dominated}\n')
            f.write(f'not one result: {not_one_result}\n')
            f.write(f'tar_true {tar_true}\n')
            f.write(f'tar_false {tar_false}\n')
            f.write('----------- Dataset -----------\n')
            f.write('----------- Prediction -----------\n')
            f.write(f'TP: {TP}\n')
            f.write(f'TN: {TN}\n')
            f.write(f'FP: {FP}\n')
            f.write(f'FN: {FN}\n')
            f.write('Accuracy (TP + FN) / all: %.04f\n' % ((TP + FN) / (tar_true + tar_false)))
            f.write('Recall (TP / (TP + FN)) %.04f\n' % (TP / (TP + FN)))
            f.write('Precision (TP / (TP + FP)) %.04f\n' % (TP / (TP + FP)))
            f.write(f'FN / (FN + FP): {(FN / (FP + FN))}\n')
            f.write(f'yolo_dominated_TP: {yolo_dominated_TP}\n')
            f.write(f'yolo_dominated_TN: {yolo_dominated_TN}\n')
            f.write(f'yolo_dominated_FP: {yolo_dominated_FP}\n')
            f.write(f'yolo_dominated_FN: {yolo_dominated_FN}\n')
            f.write('yolo_dominated_Accuracy: %.04f\n' % ((yolo_dominated_TP + yolo_dominated_FN) / (yolo_dominated_TP + yolo_dominated_TN + yolo_dominated_FP + yolo_dominated_FN)))
            f.write('yolo_dominated_Recall: %.04f\n' % (yolo_dominated_TP / (yolo_dominated_TP + yolo_dominated_FN)))
            f.write('yolo_dominated_Precision: %.04f\n' % (yolo_dominated_TP / (yolo_dominated_TP + yolo_dominated_FP)))
            f.write(f'grasp_dominated_TP: {grasp_dominated_TP}\n')
            f.write(f'grasp_dominated_TN: {grasp_dominated_TN}\n')
            f.write(f'grasp_dominated_FP: {grasp_dominated_FP}\n')
            f.write(f'grasp_dominated_FN: {grasp_dominated_FN}\n')
            f.write('grasp_dominated_Accuracy: %.04f\n' % ((grasp_dominated_TP + grasp_dominated_FN) / (grasp_dominated_TP + grasp_dominated_TN + grasp_dominated_FP + grasp_dominated_FN)))
            f.write('grasp_dominated_Recall: %.04f\n' % (grasp_dominated_TP / (grasp_dominated_TP + grasp_dominated_FN)))
            f.write('grasp_dominated_Precision: %.04f\n' % (grasp_dominated_TP / (grasp_dominated_TP + grasp_dominated_FP)))
            f.write('----------- Prediction -----------')
        print('over!')


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
        urdf_path = '../urdf/'
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
