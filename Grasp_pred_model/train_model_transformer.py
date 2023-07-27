import os
import numpy as np
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from network import LSTMRegressor
from tqdm import tqdm
import sys
sys.path.append('../')
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

def data_split(path, total_num, ratio, test_model=False, use_scaler=False):

    num_train = int(total_num * ratio)

    box_data_train = []
    box_data_test = []
    grasp_data_train = []
    grasp_data_test = []
    data_total = []

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
        yolo_dominated = 0
        no_grasp = 0
        grasp_dominated = 0
        for i in tqdm(range(num_train, total_num)):
            data_test = np.loadtxt(path + '%012d.txt' % i).reshape(-1, 7)
            yolo_dominated_index = np.argmax(data_test[:, -1])
            if np.all(data_test[:, 0] == 0):
                # print(f'no grasp {i}')
                no_grasp += 1
            elif np.where(data_test[:, 0] == 1)[0][0] == yolo_dominated_index:
                # print(f'yolo dominated {i}')
                yolo_dominated += 1
            # elif data_test[0, 0] != 1:
            else:
                grasp_dominated += 1
            box_data_test.append(data_test[:, 1:])
            grasp_data_test.append(data_test[:, 0].reshape(-1, 1))
        print('this is yolo dominated', yolo_dominated)
        print('this is no grasp', no_grasp)
        print('this is grasp dominated', grasp_dominated)
        print('total valid data:', int(total_num - num_train))

        return box_data_test, grasp_data_test, yolo_dominated

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

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, n_layers):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Fully connected layer
        self.fc = nn.Linear(input_dim, output_dim)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, src):
        # src expected shape: (seq_len, batch_size, input_dim)
        output = self.transformer_encoder(src)
        # Take the encoding of the last time step
        output = self.fc(output[-1])
        return output

    def maskedCrossEntropyLoss(self, predict, target, boxes_data, ignore_index = -100):

        mask = target.view(-1, ).ne(ignore_index)
        target_merge = target.view(-1, )
        predict_merge = predict.view(self.batch_size * predict.size(1), 2)
        pred_mask = predict_merge[mask]
        tar_mask = target_merge[mask]

        pred_mask_soft = F.softmax(pred_mask)
        pred_mask_soft_numpy = pred_mask_soft.cpu().detach().numpy()
        tar_mask_numpy = tar_mask.cpu().detach().numpy().reshape(-1, 1)
        boxes_data_mask_numpy = boxes_data.view(-1, 6)[mask].cpu().detach().numpy()
        loss = self.criterion(pred_mask, tar_mask.long())

        return loss

para_dict = {'decive': 'cuda:0',
             'num_img': 10000,
             'ratio': 0.8,
             'epoch': 300,
             'model_path': '../Grasp_pred_model/results/TF_727_1_multi/',
             'data_path': '../../knolling_dataset/grasp_dataset_726_multi/labels/',
             'learning_rate': 0.001, 'patience': 10, 'factor': 0.1,
             'network': 'binary',
             'batch_size': 64,
             'input_size': 6,
             'hidden_size': 32,
             'num_heads': 4,
             'num_layers': 2,
             'output_size': 2,
             'abort_learning': 20,
             'set_dropout': 0.1,
             'run_name': '727_1_multi',
             'project_name': 'zzz_TF_heavy',
             'wandb_flag': True,
             'use_mse': False,
             'use_scaler': False,
             'fine-tuning': False,
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

    # Assume input_dim=12, hidden_dim=50, output_dim=12, n_heads=4, n_layers=2
    model = TransformerModel(para_dict['input_size'], para_dict['hidden_size'], para_dict['output_size'], para_dict['num_heads'], para_dict['num_layers'])

    # Define loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=para_dict['patience'], factor=para_dict['factor'])

    # Generate some mock data
    num_batches = 10
    batch_size = 32
    seq_len = 12
    input_dim = 12

    # Training data and labels
    box_train, box_test, grasp_train, grasp_test = data_split(para_dict['data_path'], para_dict['num_img'], para_dict['ratio'], para_dict['use_scaler'])
    train_dataset = Generate_Dataset(box_data=box_train, grasp_data=grasp_train)
    test_dataset = Generate_Dataset(box_data=box_test, grasp_data=grasp_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=para_dict['batch_size'], shuffle=True,
                              collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=para_dict['batch_size'], shuffle=False,
                             collate_fn=collate_fn, drop_last=True)

    X_train = torch.randn(num_batches, batch_size, seq_len, input_dim)  # random data

    # Labels (one-hot encoded, with random indices set to 1)
    y_train = torch.zeros(num_batches, batch_size, seq_len, input_dim)
    for i in range(num_batches):
        for j in range(batch_size):
            idx = np.random.choice(seq_len)
            y_train[i, j, idx] = 1

    # Sample training loop
    all_train_loss = []
    all_valid_loss = []
    min_loss = np.inf
    current_epoch = 0
    for epoch in range(para_dict['epoch']):  # number of epochs

        t0 = time.time()
        train_loss = []
        valid_loss = []
        model.train()

        for batch_id, (box_data_batch, grasp_data_batch, num_box) in enumerate(train_loader):

            box_data_batch = box_data_batch.to(device, dtype=torch.float32)
            grasp_data_batch = grasp_data_batch.to(device, dtype=torch.float32)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            box_data_pack = pack_padded_sequence(box_data_batch, num_box, batch_first=True)
            outputs = model.forward(box_data_pack)  # rearrange batch to be compatible with Transformer

            # Compute loss
            loss = model.maskedCrossEntropyLoss(predict=outputs, target=grasp_data_batch, boxes_data=box_data_batch)
            train_loss.append(loss.item())

            # Backward pass and optimization
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
                loss_valid = model.maskedCrossEntropyLoss(predict=out, target=grasp_data_batch, boxes_data=box_data_batch)
                valid_loss.append(loss_valid.item())

        avg_valid_loss = np.mean(valid_loss)
        all_valid_loss.append(avg_valid_loss)

        if avg_valid_loss < min_loss:
            print('Training_Loss At Epoch ' + str(i) + ':\t', np.around(avg_train_loss, 6))
            print('Testing_Loss At Epoch ' + str(i) + ':\t', np.around(avg_valid_loss, 6))
            min_loss = avg_valid_loss
            PATH = para_dict['model_path'] + 'best_model.pt'
            torch.save(model.state_dict(), PATH)
            abort_learning = 0
        else:
            abort_learning += 1
        np.savetxt(para_dict['model_path'] + "train_loss_LSTM.txt", np.asarray(all_train_loss), fmt='%.06f')
        np.savetxt(para_dict['model_path'] + "valid_loss_LSTM.txt", np.asarray(all_valid_loss), fmt='%.06f')
        t1 = time.time()
        # print(f"epoch{i}, time used: {round((t1 - t0), 2)}, lr: {scheduler.get_last_lr()}")
        print(f"epoch{i}, time used: {round((t1 - t0), 2)}, lr: {optimizer.param_groups[0]['lr']}")

        if abort_learning > para_dict['abort_learning']:
            break
        else:
            scheduler.step(avg_valid_loss)
        current_epoch += 1

    if para_dict['wandb_flag'] == True:
        wandb.init()
        for step in range(current_epoch):
            wandb.log({'Train_loss': all_train_loss[step]}, step=step)
            wandb.log({'Valid_loss': all_valid_loss[step]}, step=step)
        wandb.finish()
