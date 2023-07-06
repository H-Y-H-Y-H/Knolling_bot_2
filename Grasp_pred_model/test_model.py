import sys
import numpy as np
import random
import os
sys.path.append('/home/zhizhuo/ADDdisk/Create Machine Lab/Knolling_bot_2/')
from Grasp_pred_model.Data_collection.grasp_or_yolo_collection import Arm_env
from network import LSTMRegressor
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from train_model import Generate_Dataset, data_split, collate_fn

if __name__ == '__main__':

    use_dataset = True

    if use_dataset == True:

        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        print("Device:", device)

        para_dict = {'num_img': 50000,
                     'ratio': 0.8,
                     'epoch': 200,
                     'model_path': '../Grasp_pred_model/results/LSTM_705_3_cross/',
                     'data_path': '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/grasp_dataset_03004/labels/',
                     'learning_rate': 0.01, 'stepLR': 30, 'gamma': 0.5,
                     'network': 'binary',
                     'batch_size': 32,
                     'input_size': 6,
                     'hidden_size': 16,
                     'box_one_img': 21,
                     'num_layers': 2,
                     'output_size': 2,
                     'abort_learning': 40,
                     'run_name': '705_2',
                     'project_name': 'zzz_LSTM_softmax',
                     'wandb_flag': True,
                     'use_mse': False}
        total_error = []

        num_img = para_dict['num_img']
        ratio = para_dict['ratio']
        box_one_img = para_dict['box_one_img']
        data_root = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/'
        data_path = para_dict['data_path']
        box_train, box_test, grasp_train, grasp_test = data_split(data_path, num_img, ratio, box_one_img)

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
                                  num_layers=num_layers,
                                  batch_size=batch_size, device=device, criterion=nn.CrossEntropyLoss())

        ###########################################################################
        model.load_state_dict(torch.load(para_dict['model_path'] + 'best_model.pt'))
        ###########################################################################

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=para_dict['stepLR'], gamma=para_dict['gamma'])

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
                else:
                    loss = model.maskedCrossEntropyLoss(predict=out, target=grasp_data_batch)
                valid_loss.append(loss.item())

        avg_valid_loss = np.mean(valid_loss)



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
            para_dict = {'input_size': 6, 'hidden_size': 64, 'num_layers': 2,
                         'output_size': 1, 'batch_size': 1, 'model_path': '../Grasp_pred_model/results/LSTM_702/', 'device': device}
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
