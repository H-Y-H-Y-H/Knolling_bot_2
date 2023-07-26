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

def data_split(path, total_num, ratio, max_box, test_model=False, use_scaler=False):

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

class MyDataSet(Dataset):

    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, n_layers):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Fully connected layer
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, src):
        # src expected shape: (seq_len, batch_size, input_dim)
        output = self.transformer_encoder(src)
        # Take the encoding of the last time step
        output = self.fc(output[-1])
        return output


# Assume input_dim=12, hidden_dim=50, output_dim=12, n_heads=4, n_layers=2
model = TransformerModel(12, 50, 12, 4, 2)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Generate some mock data
num_batches = 10
batch_size = 32
seq_len = 12
input_dim = 12

# Training data
X_train = torch.randn(num_batches, batch_size, seq_len, input_dim)  # random data

# Labels (one-hot encoded, with random indices set to 1)
y_train = torch.zeros(num_batches, batch_size, seq_len, input_dim)
for i in range(num_batches):
    for j in range(batch_size):
        idx = np.random.choice(seq_len)
        y_train[i, j, idx] = 1

# Sample training loop
for epoch in range(100):  # number of epochs
    for i in range(num_batches):  # iterate over batches
        batch = X_train[i]
        labels = y_train[i]

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch.permute(2, 0, 1))  # rearrange batch to be compatible with Transformer

        # Compute loss
        labels = torch.argmax(labels, dim=2)  # assuming one-hot encoding
        labels = labels.float()
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print('Epoch: %d, Loss: %.4f' % (epoch, loss.item()))
