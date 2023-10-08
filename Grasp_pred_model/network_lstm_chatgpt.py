import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2,
                 device=None, batch_size=None, criterion=None, set_dropout=0.05,
                 hidden_node_1=None, hidden_node_2=None):
        super(LSTMRegressor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.batch_size = batch_size
        self.criterion = criterion
        self.nllloss = nn.NLLLoss

        # Define the LSTM layer
        binary = True
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers,
                            bidirectional=binary, batch_first=True, dropout=set_dropout).to(device)
        if binary == False:
            self.num_directions = 1
        else:
            self.num_directions = 2

        # Define the output layer
        self.linear1 = nn.Linear(self.hidden_dim * self.num_directions, hidden_node_1).to(self.device)
        self.linear2 = nn.Linear(hidden_node_1, hidden_node_2).to(self.device)
        self.linear3 = nn.Linear(hidden_node_2, output_dim).to(self.device)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input_data, deploy_flag=False):

        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_dim).to(self.device, dtype=torch.float32)
        c0 = torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_dim).to(self.device, dtype=torch.float32)

        output, (hn, cn) = self.lstm(input_data, (h0.detach(), c0.detach()))
        if deploy_flag == True:
            unpacked_out = output
        else:
            unpacked_out, out_length = pad_packed_sequence(output, batch_first=True)
            
        if self.criterion == None:
            out = self.softmax(self.linear(unpacked_out))
        else:
            out = self.relu(self.linear1(unpacked_out))
            out = self.relu(self.linear2(out))
            out = self.linear3(out)
        return out

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

if __name__ == '__main__':

    para_dict = {'device': 'cuda:0',
                 'num_img': 450000,
                 'ratio': 0.8,
                 'epoch': 300,
                 'model_path': '../Grasp_pred_model/results/LSTM_918_0/',
                 'data_path': '../../knolling_dataset/grasp_dataset_914/labels_1/',
                 'learning_rate': 0.001, 'patience': 10, 'factor': 0.1,
                 'network': 'binary',
                 'batch_size': 64,
                 'input_size': 6,
                 'hidden_size': 32,
                 'box_one_img': 10,
                 'num_layers': 8,
                 'output_size': 2,
                 'abort_learning': 20,
                 'set_dropout': 0.1,
                 'run_name': '918_0',
                 'project_name': 'zzz_LSTM_cross_no_scaler_heavy',
                 'wandb_flag': False,
                 'use_mse': False,
                 'use_scaler': False,
                 'fine-tuning': False,
                 'load_path': '../Grasp_pred_model/results/LSTM_829_1_heavy_dropout0/',
                 'hidden_node_1': 32, 'hidden_node_2': 8}

    model = LSTMRegressor(input_dim=para_dict['input_size'], hidden_dim=para_dict['hidden_size'], output_dim=para_dict['output_size'],
                          num_layers=para_dict['num_layers'], hidden_node_1=para_dict['hidden_node_1'],
                          hidden_node_2=para_dict['hidden_node_2'],
                          batch_size=para_dict['batch_size'], device=para_dict['device'], criterion=nn.CrossEntropyLoss(),
                          set_dropout=para_dict['set_dropout'])
    print(model)