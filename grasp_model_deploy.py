import numpy as np
from Grasp_pred_model.network import LSTMRegressor
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class Grasp_model():

    def __init__(self, para_dict, lstm_dict):

        self.para_dict = para_dict
        self.lstm_dict = lstm_dict
        self.model = LSTMRegressor(input_dim=self.lstm_dict['input_size'], hidden_dim=self.lstm_dict['hidden_size'], output_dim=self.lstm_dict['output_size'],
                                  num_layers=self.lstm_dict['num_layers'], hidden_node_1=self.lstm_dict['hidden_node_1'],
                                  hidden_node_2=self.lstm_dict['hidden_node_2'],
                                  batch_size=self.lstm_dict['batch_size'], device=self.lstm_dict['device'], criterion=nn.CrossEntropyLoss(),
                                  set_dropout=self.lstm_dict['set_dropout'])
        self.model.load_state_dict(torch.load(self.lstm_dict['grasp_model_path']))
        self.softmax = nn.Softmax(dim=2)

    def pred(self, manipulator_before, lwh_list, conf_list):

        knolling_flag = False
        num_item = len(conf_list)
        input_data = np.concatenate((manipulator_before[:, :2],
                                     lwh_list[:, :2],
                                     manipulator_before[:, -1].reshape(-1, 1),
                                     conf_list.reshape(-1, 1)), axis=1)
        input_data = torch.unsqueeze(torch.from_numpy(input_data), 0)

        with torch.no_grad():
            # print('eval')
            input_data = input_data.to(self.para_dict['device'], dtype=torch.float32)
            # box_data_batch = box_data_batch.to(device, dtype=torch.float32)
            # grasp_data_batch = grasp_data_batch.to(device, dtype=torch.float32)

            # box_data_pack = pack_padded_sequence(box_data_batch, num_box, batch_first=True)
            out = self.model.forward(input_data, deploy_flag=True)
            output = self.softmax(out).cpu().detach().numpy().squeeze()

        pred_N = np.where(output[:, 1] < self.lstm_dict['threshold'])[0]
        # move_list = np.arange(int(num_item / 2), num_item)
        move_list = pred_N

        return move_list, knolling_flag