import os
import numpy as np
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from network_lstm import LSTMRegressor
from tqdm import tqdm
import sys
sys.path.append('../')
# from pos_encoder import *
import torch.nn.functional as F
# from model import *
from sklearn.preprocessing import StandardScaler


def data_split(path, total_num, ratio, max_box, valid_num, test_model=False, use_scaler=False):

    num_train = int(total_num * ratio)

    box_data_train = []
    box_data_test = []
    grasp_data_train = []
    grasp_data_test = []
    data_total = []

    if use_scaler == True:
        scaler = StandardScaler()
        print('load the data ...')
        for i in tqdm(range(total_num)):
            data_total.append(np.loadtxt(path + '%012d.txt' % i).reshape(-1, 7))
        print('\ntotal data:', i)
        data_total = scaler.fit_transform(np.asarray(data_total))

        data_train = data_total[:num_train, :]
        data_test = data_total[num_train:, :]
        box_data_train = data_train[:, 1:]
        box_data_test = data_test[:, 1:]
        grasp_data_train = data_train[:, 0].reshape(-1, 1)
        grasp_data_test = data_test[:, 0].reshape(-1, 1)

        print('total train data', len(box_data_train))
        print('total test data', len(box_data_test))

        return box_data_train, box_data_test, grasp_data_train, grasp_data_test

    else:
        if test_model == False:
            print('load the train data ...')
            for i in tqdm(range(num_train)):
                data_train = np.loadtxt(path + '%012d.txt' % i).reshape(-1, 7)
                # if data_train[0, 0] == 1:
                #     continue
                box_data_train.append(data_train[:, 1:])
                grasp_data_train.append(data_train[:, 0].reshape(-1, 1))
            print('\ntotal train data:', len(grasp_data_train))

            print('load the valid data ...')
            for i in tqdm(range(num_train, total_num)):
                data_test = np.loadtxt(path + '%012d.txt' % i).reshape(-1, 7)
                # if data_test[0, 0] == 1:
                #     continue
                box_data_test.append(data_test[:, 1:])
                grasp_data_test.append(data_test[:, 0].reshape(-1, 1))
            print('total valid data:', len(grasp_data_test))

            return box_data_train, box_data_test, grasp_data_train, grasp_data_test
        else:
            print('load the valid data ...')
            yolo_dominated_true = 0
            no_grasp_img = 0
            at_least_one_grasp_img = 0
            grasp_dominated_true = 0
            tar_true_grasp = 0
            tar_false_grasp = 0
            for i in tqdm(range(num_train, valid_num + num_train)):
                data_test = np.loadtxt(path + '%012d.txt' % i).reshape(-1, 7)
                yolo_dominated_index = np.argmax(data_test[:, -1])

                if np.all(data_test[:, 0] == 0):
                    no_grasp_img += 1
                else:
                    at_least_one_grasp_img += 1
                    conf_sequence = np.argsort(data_test[:, -1])[::-1]
                    data_conf_sequence = data_test[conf_sequence]
                    if np.where(data_conf_sequence[:, 0] == 1)[0][0] == yolo_dominated_index:
                        # print(f'yolo dominated {i}')
                        yolo_dominated_true += 1
                    else:
                        pass

                for j in range(len(data_test)):
                    if data_test[j, 0] == 1:
                        tar_true_grasp += 1
                    else:
                        tar_false_grasp += 1

                box_data_test.append(data_test[:, 1:])
                grasp_data_test.append(data_test[:, 0].reshape(-1, 1))
            print('total valid data:', int(total_num - num_train))

            print('this is yolo dominated true:', yolo_dominated_true)
            print('this is grasp dominated true:', tar_true_grasp - yolo_dominated_true)
            print('this is no grasp per img:', no_grasp_img)
            print('this is at least one grasp per img:', at_least_one_grasp_img)
            print('this is tar success grasp:', tar_true_grasp)
            print('this is tar false grasp:', tar_false_grasp)
            print('this is yolo dominated true rate in dataset', yolo_dominated_true / tar_true_grasp)
            # print('this is grasp dominated:', grasp_dominated_true)

            return box_data_test, grasp_data_test

def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    box_data = [sq[0] for sq in data]
    grasp_data = [sq[1] for sq in data]
    data_length = [len(sq) for sq in box_data]

    # box_data_demo = torch.ones(21, 6)
    # grasp_data_demo = torch.ones(21, 1)
    # box_data.append(box_data_demo)
    # grasp_data.append(grasp_data_demo)

    # box_data_pad = pad_sequence(box_data, batch_first=True, padding_value=0.0)[:batch_size, :, :]
    # grasp_data_pad = pad_sequence(grasp_data, batch_first=True, padding_value=-100.0)[:batch_size, :, :]
    box_data_pad = pad_sequence(box_data, batch_first=True, padding_value=0.0)
    grasp_data_pad = pad_sequence(grasp_data, batch_first=True, padding_value=-100.0)
    return box_data_pad, grasp_data_pad, data_length

class Generate_Dataset(Dataset):
    def __init__(self, box_data, grasp_data):
        self.box_data = box_data
        self.grasp_data = grasp_data

    def __getitem__(self, idx):
        box_sample = self.box_data[idx]
        grasp_sample = self.grasp_data[idx]

        box_sample = torch.from_numpy(box_sample)
        grasp_sample = torch.from_numpy(grasp_sample)

        # sample = {'box': box_sample, 'grasp': grasp_sample}
        sample = (box_sample, grasp_sample)

        return sample

    def __len__(self):
        return len(self.box_data)

# para_dict = {'decive': 'cuda:0',
#              'num_img': 520000,
#              'ratio': 0.8,
#              'epoch': 300,
#              'model_path': '../Grasp_pred_model/results/LSTM_829_3_heavy_dropout0/',
#              'data_path': '../../knolling_dataset/grasp_dataset_726_ratio_multi/labels_4/',
#              'learning_rate': 0.001, 'patience': 10, 'factor': 0.1,
#              'network': 'binary',
#              'batch_size': 64,
#              'input_size': 6,
#              'hidden_size': 16,
#              'box_one_img': 10,
#              'num_layers': 8,
#              'output_size': 2,
#              'abort_learning': 20,
#              'set_dropout': 0.1,
#              'run_name': '829_3_distance_dropout0',
#              'project_name': 'zzz_LSTM_cross_no_scaler_heavy',
#              'wandb_flag': True,
#              'use_mse': False,
#              'use_scaler': False,
#              'fine-tuning': False,
#              'hidden_node_1': 16, 'hidden_node_2': 8}

para_dict = {'decive': 'cuda:1',
             'num_img': 300000,
             'ratio': 0.8,
             'epoch': 300,
             'model_path': '../Grasp_pred_model/results/LSTM_913_tuning_829_0/',
             'data_path': '../../knolling_dataset/grasp_dataset_730/labels_1/',
             'learning_rate': 0.0001, 'patience': 10, 'factor': 0.1,
             'network': 'binary',
             'batch_size': 64,
             'input_size': 6,
             'hidden_size': 32,
             'box_one_img': 10,
             'num_layers': 8,
             'output_size': 2,
             'abort_learning': 20,
             'set_dropout': 0.1,
             'run_name': '913_0',
             'project_name': 'zzz_LSTM_cross_no_scaler_heavy',
             'wandb_flag': False,
             'use_mse': False,
             'use_scaler': False,
             'fine-tuning': True,
             'load_path': '../Grasp_pred_model/results/LSTM_829_1_heavy_dropout0/',
             'hidden_node_1': 32, 'hidden_node_2': 8}

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = para_dict['decive']

    else:
        device = 'cpu'
    print("Device:", device)

    import wandb
    if para_dict['wandb_flag'] == True:
        wandb.config = para_dict
        wandb.init(project=para_dict['project_name'],
                 notes='knolling_bot_2',
                 tags=['baseline', 'paper1'],
                 name=para_dict['run_name'])
        wandb.config.update(para_dict)
        print('this is para_dict\n', para_dict)


    # define the basic parameters
    model_save_path = para_dict['model_path']
    os.makedirs(model_save_path, exist_ok=True)
    epoch = para_dict['epoch']
    abort_learning = 0

    # target_batch = torch.tensor([[1,2,3,4],
    #                              [1,2,3,4],
    #                              [1,2,3,4]])

    # split the raw data into box and grasp flag
    num_img = para_dict['num_img']
    ratio = para_dict['ratio']
    box_one_img = para_dict['box_one_img']
    data_path = para_dict['data_path']
    box_train, box_test, grasp_train, grasp_test = data_split(data_path, num_img, ratio, box_one_img, para_dict['use_scaler'])

    # create the train dataset and test dataset
    batch_size = para_dict['batch_size']
    train_dataset = Generate_Dataset(box_data=box_train, grasp_data=grasp_train)
    test_dataset = Generate_Dataset(box_data=box_test, grasp_data=grasp_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, drop_last=True)

    # initialize the parameters of the model
    input_size = para_dict['input_size']
    hidden_size = para_dict['hidden_size']
    num_layers = para_dict['num_layers']
    output_size = para_dict['output_size']
    learning_rate = para_dict['learning_rate']
    if para_dict['use_mse'] == True:
        model = LSTMRegressor(input_dim=input_size, hidden_dim=hidden_size, output_dim=output_size, num_layers=num_layers,
                          batch_size=batch_size, device=device)
    else:
        model = LSTMRegressor(input_dim=input_size, hidden_dim=hidden_size, output_dim=output_size,
                              num_layers=num_layers, hidden_node_1=para_dict['hidden_node_1'], hidden_node_2=para_dict['hidden_node_2'],
                              batch_size=batch_size, device=device, criterion=nn.CrossEntropyLoss(), set_dropout=para_dict['set_dropout'])

    ##########################################################################
    if para_dict['fine-tuning'] == True:
        model_load_path = para_dict['load_path']
        model.load_state_dict(torch.load(model_load_path + 'best_model.pt'))
    else:
        print('not fine-tuning')
    ##########################################################################

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=para_dict['stepLR'], gamma=para_dict['gamma'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=para_dict['patience'], factor=para_dict['factor'])
    min_loss = np.inf

    all_train_loss = []
    all_valid_loss = []
    current_epoch = 0
    for i in range(epoch):
        # print(i)
        t0 = time.time()
        train_loss = []
        valid_loss = []

        model.train()
        # print('train')
        for batch_id, (box_data_batch, grasp_data_batch, num_box) in enumerate(train_loader):
            # print('this is batch id', batch_id)
            # print('this is box data\n', box_data_batch)
            # print('this is grasp data\n', grasp_data_batch)
            box_data_batch = box_data_batch.to(device, dtype=torch.float32)
            grasp_data_batch = grasp_data_batch.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            box_data_pack = pack_padded_sequence(box_data_batch, num_box, batch_first=True)
            out = model.forward(box_data_pack)
            if para_dict['use_mse'] == True:
                loss = model.maskedMSELoss(predict=out, target=grasp_data_batch)
            else:
                loss = model.maskedCrossEntropyLoss(predict=out, target=grasp_data_batch, boxes_data=box_data_batch)
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        avg_train_loss = np.mean(train_loss)
        all_train_loss.append(avg_train_loss)

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
                    loss = model.maskedCrossEntropyLoss(predict=out, target=grasp_data_batch, boxes_data=box_data_batch)
                valid_loss.append(loss.item())

        avg_valid_loss = np.mean(valid_loss)
        all_valid_loss.append(avg_valid_loss)

        if avg_valid_loss < min_loss:
            print('Training_Loss At Epoch ' + str(i) + ':\t', np.around(avg_train_loss, 6))
            print('Testing_Loss At Epoch ' + str(i) + ':\t', np.around(avg_valid_loss, 6))
            min_loss = avg_valid_loss
            PATH = model_save_path + 'best_model.pt'
            torch.save(model.state_dict(), PATH)
            abort_learning = 0
        else:
            abort_learning += 1
        np.savetxt(model_save_path + "train_loss_LSTM.txt", np.asarray(all_train_loss), fmt='%.06f')
        np.savetxt(model_save_path + "valid_loss_LSTM.txt", np.asarray(all_valid_loss), fmt='%.06f')
        t1 = time.time()
        # print(f"epoch{i}, time used: {round((t1 - t0), 2)}, lr: {scheduler.get_last_lr()}")
        print(f"epoch{i}, time used: {round((t1 - t0), 2)}, lr: {optimizer.param_groups[0]['lr']}")


        if abort_learning > para_dict['abort_learning']:
            break
        else:
            scheduler.step(avg_valid_loss)
        current_epoch += 1

        if para_dict['wandb_flag'] == True:
            wandb.log({'Train_loss': avg_train_loss})
            wandb.log({'Valid_loss': avg_valid_loss})

    # if para_dict['wandb_flag'] == True:
    #     wandb.init()
    #     for step in range(current_epoch):
    #         wandb.log({'Train_loss': all_train_loss[step]}, step=step)
    #         wandb.log({'Valid_loss': all_valid_loss[step]}, step=step)
    #     wandb.finish()
