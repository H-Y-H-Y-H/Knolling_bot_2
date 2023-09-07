import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import time
from network_mlp import MLP

def data_preprocess(box_path, unstack_path, total_num, ratio, valid_num=None, test_model=False, use_scaler=False):

    num_train = int(total_num * ratio)

    total_box_data_train = []
    total_box_data_test = []
    total_unstack_data_train = []
    total_unstack_data_test = []
    data_total = []

    if valid_num is None:
        valid_num = total_num

    if use_scaler == True:
        scaler = StandardScaler()
        print('load the data ...')
        for i in tqdm(range(total_num)):
            data_total.append(np.loadtxt(path + '%012d.txt' % i).reshape(-1, 7))
        print('\ntotal data:', i)
        data_total = scaler.fit_transform(np.asarray(data_total))

        data_train = data_total[:num_train, :]
        data_test = data_total[num_train:, :]
        total_box_data_train = data_train[:, 1:]
        box_data_test = data_test[:, 1:]
        total_unstack_data_train = data_train[:, 0].reshape(-1, 1)
        grasp_data_test = data_test[:, 0].reshape(-1, 1)

        print('total train data', len(total_box_data_train))
        print('total test data', len(box_data_test))

        return total_box_data_train, box_data_test, total_unstack_data_train, grasp_data_test

    else:
        if test_model == False:
            print('load the train data ...')
            for i in tqdm(range(num_train)):
                box_data_train = np.loadtxt(box_path + '%012d.txt' % i)
                unstack_data_train = np.loadtxt(unstack_path + '%012d.txt' % i)
                # box_data_train = box_data_train[:, [0, 1, 5, 6, 7, 9, 11]].reshape(-1, )
                # unstack_data_train = unstack_data_train[:, [0, 1, 5]].reshape(-1, )
                total_box_data_train.append(box_data_train.reshape(-1, ))
                total_unstack_data_train.append(unstack_data_train.reshape(-1, ))
            print('\ntotal train data:', len(total_unstack_data_train))

            print('load the valid data ...')
            for i in tqdm(range(num_train, total_num)):
                box_data_test = np.loadtxt(box_path + '%012d.txt' % i)
                unstack_data_test = np.loadtxt(unstack_path + '%012d.txt' % i)
                # box_data_test = box_data_test[:, [0, 1, 5, 6, 7, 9, 11]].reshape(-1, )
                # unstack_data_test = unstack_data_test[:, [0, 1, 5]].reshape(-1, )
                total_box_data_test.append(box_data_test.reshape(-1, ))
                total_unstack_data_test.append(unstack_data_test.reshape(-1, ))
            print('total valid data:', len(total_unstack_data_test))

            return total_box_data_train, total_box_data_test, total_unstack_data_train, total_unstack_data_test

        else:
            print('load the valid data ...')
            for i in tqdm(range(num_train, valid_num + num_train)):
                box_data_test = np.loadtxt(box_path + '%012d.txt' % i)
                unstack_data_test = np.loadtxt(unstack_path + '%012d.txt' % i)
                total_box_data_test.append(box_data_test.reshape(-1, ))
                total_unstack_data_test.append(unstack_data_test.reshape(-1, ))
            print('total valid data:', len(total_unstack_data_test))

            return total_box_data_test, total_unstack_data_test


def data_padding(box_train, box_test):

    max_box = 5
    for i in range(len(box_train)):
        if len(box_train[i]) < max_box * 7:
            box_train[i] = np.append(box_train[i], np.zeros(int(max_box * 7 - len(box_train[i]))))
    for i in range(len(box_test)):
        if len(box_test[i]) < max_box * 7:
            box_test[i] = np.append(box_test[i], np.zeros(int(max_box * 7 - len(box_test[i]))))

    return box_train, box_test

class Generate_Dataset(Dataset):
    def __init__(self, box_data, unstack_data):
        self.box_data = box_data
        self.unstack_data = unstack_data

    def __getitem__(self, idx):
        box_sample = self.box_data[idx]
        unstack_sample = self.unstack_data[idx]

        box_sample = torch.from_numpy(box_sample)
        unstack_sample = torch.from_numpy(unstack_sample)

        # sample = {'box': box_sample, 'grasp': grasp_sample}
        sample = (box_sample, unstack_sample)

        return sample

    def __len__(self):
        return len(self.box_data)

# use conf
para_dict = {'device': 'cuda:0',
             'num_img': 160000,
             'ratio': 0.8,
             'epoch': 300,
             'model_path': './results/MLP_905_3/',
             'input_data_path': '../../knolling_dataset/MLP_unstack_905/labels_box/',
             'output_data_path': '../../knolling_dataset/MLP_unstack_905/labels_unstack/',
             'learning_rate': 0.001, 'patience': 10, 'factor': 0.1,
             'batch_size': 64,
             'output_size': 6,
             'abort_learning': 20,
             'set_dropout': 0.05,
             'num_boxes': 5,
             'run_name': 'MLP_905_3',
             'project_name': 'zzz_MLP_unstack',
             'wandb_flag': True,
             'use_mse': True,
             'use_scaler': False,
             'fine-tuning': False,
             'node_1': 128,
             'node_2': 32,
             'node_3': 8}

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = para_dict['device']

    else:
        device = 'cpu'
    print("Device:", device)

    import wandb
    if para_dict['wandb_flag'] == True:
        wandb.config = para_dict
        wandb.init(project=para_dict['project_name'],
                 name=para_dict['run_name'])
        wandb.config.update(para_dict)
        print('this is para_dict\n', para_dict)


    # define the basic parameters
    model_save_path = para_dict['model_path']
    os.makedirs(model_save_path, exist_ok=True)
    epoch = para_dict['epoch']
    abort_learning = 0

    # split the raw data into box and grasp flag
    num_img = para_dict['num_img']
    ratio = para_dict['ratio']
    input_data_path = para_dict['input_data_path']
    output_data_path = para_dict['output_data_path']
    box_train, box_test, unstack_train, unstack_test = data_preprocess(input_data_path, output_data_path, num_img, ratio, para_dict['use_scaler'])
    box_train_padding, box_test_padding = data_padding(box_train, box_test)
    # create the train dataset and test dataset
    batch_size = para_dict['batch_size']
    train_dataset = Generate_Dataset(box_data=box_train, unstack_data=unstack_train)
    test_dataset = Generate_Dataset(box_data=box_test, unstack_data=unstack_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # initialize the parameters of the model
    learning_rate = para_dict['learning_rate']
    model = MLP(num_boxes=para_dict['num_boxes'], output_size=para_dict['output_size'],
                node_1=para_dict['node_1'], node_2=para_dict['node_2'], node_3=para_dict['node_3'], device=para_dict['device'])

    ##########################################################################
    if para_dict['fine-tuning'] == True:
        model.load_state_dict(torch.load(model_save_path + 'best_model.pt'))
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
        for batch_id, (box_data, unstack_data) in enumerate(train_loader):
            box_data = box_data.to(device, dtype=torch.float32)
            box_data_test = box_data.cpu().detach().numpy()
            unstack_data = unstack_data.to(device, dtype=torch.float32)
            unstack_data_test = unstack_data.cpu().detach().numpy()

            optimizer.zero_grad()
            out = model.forward(box_data)
            if para_dict['use_mse'] == True:
                loss = model.maskedMSELoss(predict=out, target=unstack_data)
            else:
                loss = model.maskedCrossEntropyLoss(predict=out, target=unstack_data, boxes_data=box_data)
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        avg_train_loss = np.mean(train_loss)
        all_train_loss.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            # print('eval')
            for batch_id, (box_data, unstack_data) in enumerate(test_loader):
                # print(batch_id)
                box_data = box_data.to(device, dtype=torch.float32)
                unstack_data = unstack_data.to(device, dtype=torch.float32)

                out = model.forward(box_data)
                if para_dict['use_mse'] == True:
                    loss = model.maskedMSELoss(predict=out, target=unstack_data)
                else:
                    loss = model.maskedCrossEntropyLoss(predict=out, target=unstack_data, boxes_data=box_data)
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