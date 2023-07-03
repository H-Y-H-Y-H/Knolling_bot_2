import os
import numpy as np
import time
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from network import LSTMRegressor
import sys
sys.path.append('../')
# from pos_encoder import *
import torch.nn.functional as F
# from model import *


def data_split(path, total_num, ratio, max_box):

    num_train = int(total_num * ratio)

    box_data_train = []
    box_data_test = []
    grasp_data_train = []
    grasp_data_test = []

    for i in range(num_train):
        data_train = np.loadtxt(path + '%012d.txt' % i).reshape(-1, 7)
        box_data_train.append(data_train[:, 1:])
        grasp_data_train.append(data_train[:, 0].reshape(-1, 1))
    for i in range(num_train, total_num):
        data_test = np.loadtxt(path + '%012d.txt' % i).reshape(-1, 7)
        box_data_test.append(data_test[:, 1:])
        grasp_data_test.append(data_test[:, 0].reshape(-1, 1))

    return box_data_train, box_data_test, grasp_data_train, grasp_data_test

def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    box_data = [sq[0] for sq in data]
    grasp_data = [sq[1] for sq in data]
    data_length = [len(sq) for sq in box_data]
    box_data = pad_sequence(box_data, batch_first=True, padding_value=0.0)
    grasp_data = pad_sequence(grasp_data, batch_first=True, padding_value=-100.0)
    return box_data, grasp_data, data_length
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


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print("Device:", device)

    # define the basic parameters
    model_save_path = '../Grasp_pred_model/results/LSTM_702/'
    os.makedirs(model_save_path, exist_ok=True)
    epoch = 100
    abort_learning = 0

    target_batch = torch.tensor([[1,2,3,4],
                                 [1,2,3,4],
                                 [1,2,3,4]])

    # split the raw data into box and grasp flag
    num_img = 10000
    ratio = 0.8
    box_one_img = 21
    data_root = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/'
    data_path = data_root + 'grasp_pile_628_no_img/labels/'
    box_train, box_test, grasp_train, grasp_test = data_split(data_path, num_img, ratio, box_one_img)

    # create the train dataset and test dataset
    batch_size = 16
    train_dataset = Generate_Dataset(box_data=box_train, grasp_data=grasp_train)
    test_dataset = Generate_Dataset(box_data=box_test, grasp_data=grasp_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # initialize the parameters of the model
    input_size = 6
    hidden_size = 64
    num_layers = 2
    output_size = 1
    learning_rate = 0.001
    model = LSTMRegressor(input_dim=input_size, hidden_dim=hidden_size, output_dim=output_size, num_layers=num_layers,
                          batch_size=batch_size, device=device)
    model.load_state_dict(torch.load(model_save_path + 'best_model.pt'))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    min_loss = np.inf

    all_train_loss = []
    all_valid_loss = []
    for i in range(epoch):
        t0 = time.time()
        train_loss = []
        valid_loss = []

        model.train()
        for batch_id, (box_data_batch, grasp_data_batch, num_box) in enumerate(train_loader):
            # print('this is box data\n', box_data_batch)
            # print('this is grasp data\n', grasp_data_batch)
            box_data_batch = box_data_batch.to(device, dtype=torch.float32)
            grasp_data_batch = grasp_data_batch.to(device, dtype=torch.float32)

            box_data_pack = pack_padded_sequence(box_data_batch, num_box, batch_first=True)
            out = model.forward(box_data_pack)
            loss = model.maskedMSELoss(predict=out, target=grasp_data_batch)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = np.mean(train_loss)
        all_train_loss.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            for batch_id, (box_data_batch, grasp_data_batch, num_box) in enumerate(test_loader):
                box_data_batch = box_data_batch.to(device, dtype=torch.float32)
                grasp_data_batch = grasp_data_batch.to(device, dtype=torch.float32)

                box_data_pack = pack_padded_sequence(box_data_batch, num_box, batch_first=True)
                out = model.forward(box_data_pack)
                loss = model.maskedMSELoss(predict=out, target=grasp_data_batch)
                valid_loss.append(loss.item())

        avg_valid_loss = np.mean(valid_loss)
        all_valid_loss.append(avg_valid_loss)

        if avg_valid_loss < min_loss:
            print('Training_Loss At Epoch ' + str(epoch) + ':\t', np.around(avg_train_loss, 6))
            print('Testing_Loss At Epoch ' + str(epoch) + ':\t', np.around(avg_valid_loss, 6))
            min_loss = avg_valid_loss
            PATH = model_save_path + 'best_model.pt'
            torch.save(model.state_dict(), PATH)
            abort_learning = 0
        else:
            abort_learning += 1
        np.savetxt(model_save_path + "train_loss_LSTM.txt", np.asarray(all_train_loss), fmt='%.06f')
        np.savetxt(model_save_path + "valid_loss_LSTM.txt", np.asarray(all_valid_loss), fmt='%.06f')
        t1 = time.time()
        print(f"epoch{i}, time used: {round((t1 - t0), 2)}, lr: {scheduler.get_last_lr()}")
        if abort_learning > 20:
            break
        else:
            scheduler.step()

    # mask = torch.ones_like(target_batch, dtype=torch.bool)
    #
    # target_batch_atten_mask = (target_batch == 0).bool()
    # target_batch.masked_fill_(label_mask, -100)
    #
    # torch.nn.utils.rnn.pack_padded_sequence
