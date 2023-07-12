import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2, device=None, batch_size=None, criterion=None):
        super(LSTMRegressor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.batch_size = batch_size
        self.criterion = criterion
        self.nllloss = nn.NLLLoss
        self.tar_success = 0
        self.grasp_dominated_tar_success = 0
        self.grasp_dominated_pred_success = 0
        self.pred_sucess = 0
        self.pred_positive = 0
        self.true_positive = 0
        self.not_one_result = 0

        # Define the LSTM layer
        binary = True
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers,
                            bidirectional=binary, batch_first=True, dropout=0.05).to(device)
        if binary == False:
            self.num_directions = 1
        else:
            self.num_directions = 2

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim * self.num_directions, output_dim).to(self.device)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input):

        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_dim).to(self.device, dtype=torch.float32)
        c0 = torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_dim).to(self.device, dtype=torch.float32)

        output, (hn, cn) = self.lstm(input, (h0.detach(), c0.detach()))
        unpacked_out, out_length = pad_packed_sequence(output, batch_first=True)
        # print(unpacked_out.shape)
        # Index hidden state of last time step
        # out = self.relu(self.linear(unpacked_out))
        # out = self.linear(unpacked_out)
        if self.criterion == None:
            out = self.softmax(self.linear(unpacked_out))
        else:
            out = self.linear(unpacked_out)
        return out

    def maskedMSELoss(self, predict, target, ignore_index = -100):

        mask = target.ne(ignore_index)
        mse_loss = (predict - target).pow(2) * mask
        mse_loss_mean = mse_loss.sum() / mask.sum()
        return mse_loss_mean

    def maskedCrossEntropyLoss(self, predict, target, ignore_index = -100):

        mask = target.view(-1, ).ne(ignore_index)
        target = target.view(-1, )
        predict = predict.view(self.batch_size * predict.size(1), 2)
        pred_mask = predict[mask]
        tar_mask = target[mask]
        loss = self.criterion(predict[mask], target[mask].long())

        return loss

    def detect_accuracy(self, predict, target, ignore_index = -100):

        mask_tar = target.ne(ignore_index)
        mask_pred = mask_tar.repeat(1, 1, 2)
        # tar = target[mask_tar].cpu().detach().numpy().reshape(-1, 1)
        # pred = predict[mask_pred].cpu().detach().numpy().reshape(-1, 2)

        tar = target.cpu().detach().numpy()
        pred = predict.cpu().detach().numpy()
        pred_soft = self.softmax(predict).cpu().detach().numpy()

        for i in range(pred_soft.shape[0]):
            if len(np.where(pred_soft[i, :, 1] > pred_soft[i, :, 0])[0]) > 1:
                # print('pred not only one result!')
                # print(pred_soft[i])
                self.not_one_result += 1
            for j in range(pred_soft.shape[1]):
                criterion = pred_soft[i, j, 1] > pred_soft[i, j, 0] and pred_soft[i, j, 1] - pred_soft[i, j, 0] > 0.1
                if tar[i, j, 0] == 1: # test the recall
                    self.tar_success += 1
                    if criterion:
                        self.pred_sucess += 1
                    if j != 0:
                        # print('grasp_dominated')
                        self.grasp_dominated_tar_success += 1
                        if criterion:
                            self.grasp_dominated_pred_success += 1
                    elif criterion:
                        pass
                elif tar[i, j, 0] == -100:
                    continue

                if criterion and tar[i, j, 0] != -100: # test the precision
                    self.pred_positive += 1
                    if tar[i, j, 0] == 1:
                        self.true_positive += 1

            if np.all(tar[i] < 0.5):
                # print('here')
                pass

        return self.tar_success, self.pred_sucess, self.grasp_dominated_tar_success, self.grasp_dominated_pred_success, \
               self.pred_positive, self.true_positive, self.not_one_result

    # f.write(f'total_img: {int(num_img - num_img * ratio)}\n')
    # f.write(f'model_path: {para_dict["model_path"]}\n')
    # f.write(f'data_path: {para_dict["data_path"]}\n')