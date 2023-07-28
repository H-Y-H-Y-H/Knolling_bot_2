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

        self.tar_true = 0
        self.tar_false = 0
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.yolo_dominated_TP = 0
        self.yolo_dominated_TN = 0
        self.yolo_dominated_FP = 0
        self.yolo_dominated_FN = 0
        self.grasp_dominated_TP = 0
        self.grasp_dominated_TN = 0
        self.grasp_dominated_FP = 0
        self.grasp_dominated_FN = 0

        self.not_one_result_per_img = 0

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

    def forward(self, input_data):

        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_dim).to(self.device, dtype=torch.float32)
        c0 = torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_dim).to(self.device, dtype=torch.float32)

        output, (hn, cn) = self.lstm(input_data, (h0.detach(), c0.detach()))
        unpacked_out, out_length = pad_packed_sequence(output, batch_first=True)
        # print(unpacked_out.shape)
        # Index hidden state of last time step
        # out = self.relu(self.linear(unpacked_out))
        # out = self.linear(unpacked_out)
        if self.criterion == None:
            out = self.softmax(self.linear(unpacked_out))
        else:
            out = self.relu(self.linear1(unpacked_out))
            out = self.relu(self.linear2(out))
            out = self.linear3(out)
        return out

    def maskedMSELoss(self, predict, target, ignore_index = -100):

        mask = target.ne(ignore_index)
        mse_loss = (predict - target).pow(2) * mask
        mse_loss_mean = mse_loss.sum() / mask.sum()
        return mse_loss_mean

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

    def detect_accuracy(self, predict, target, box_conf, model_threshold, ignore_index = -100):

        mask_tar = target.ne(ignore_index)
        mask_pred = mask_tar.repeat(1, 1, 2)
        # tar = target[mask_tar].cpu().detach().numpy().reshape(-1, 1)
        # pred = predict[mask_pred].cpu().detach().numpy().reshape(-1, 2)

        box_conf = box_conf[:, :, -1].cpu().detach().numpy()
        tar = target.cpu().detach().numpy()
        pred = predict.cpu().detach().numpy()
        pred_soft = self.softmax(predict).cpu().detach().numpy()

        for i in range(pred_soft.shape[0]): # every image
            pred_1_index = np.where(pred_soft[i, :, 1] > model_threshold)[0]
            if len(pred_1_index) > 1:
                # print('pred not only one result!')
                # print(pred_soft[i])
                temp_num = 0
                for j in range(len(pred_1_index)):
                    if tar[i, pred_1_index[j], 0] != -100:
                        temp_num += 1
                if temp_num > 1:
                    self.not_one_result_per_img += 1
            for j in range(pred_soft.shape[1]): # every boxes
                criterion = pred_soft[i, j, 1] > model_threshold
                yolo_dominated_index = np.argmax(box_conf[i])

                if tar[i, j, 0] == -100:
                    continue
                else:

                    # if tar[i, j, 0] == 1:
                    #     self.tar_true += 1
                    #     if criterion:
                    #         self.TP += 1
                    #         if j == yolo_dominated_index:
                    #             self.yolo_dominated_TP += 1
                    #         else:
                    #             self.grasp_dominated_TP += 1
                    #     else:
                    #         self.TN += 1
                    # if tar[i, j, 0] == 0:
                    #     self.tar_false += 1
                    #     if criterion:
                    #         self.FP += 1
                    #     else:
                    #         self.FN += 1

                    if tar[i, j, 0] == 1:
                        self.tar_true += 1
                        if criterion:
                            self.TP += 1
                            if j == yolo_dominated_index:
                                self.yolo_dominated_TP += 1
                            else:
                                self.grasp_dominated_TP += 1
                        else:
                            self.TN += 1
                            if j == yolo_dominated_index:
                                self.yolo_dominated_TN += 1
                            else:
                                self.grasp_dominated_TN += 1
                    if tar[i, j, 0] == 0:
                        self.tar_false += 1
                        if criterion:
                            self.FP += 1
                            if j == yolo_dominated_index:
                                self.yolo_dominated_FP += 1
                            else:
                                self.grasp_dominated_FP += 1
                        else:
                            self.FN += 1
                            if j == yolo_dominated_index:
                                self.yolo_dominated_FN += 1
                            else:
                                self.grasp_dominated_FN += 1

            if np.all(tar[i] < 0.5):
                # print('here')
                pass

        return self.not_one_result_per_img, self.tar_true, self.tar_false, self.TP, self.TN, self.FP, self.FN, \
               self.yolo_dominated_TP, self.yolo_dominated_TN, self.yolo_dominated_FP, self.yolo_dominated_FN, \
               self.grasp_dominated_TP, self.grasp_dominated_TN, self.grasp_dominated_FP, self.grasp_dominated_FN,

    # f.write(f'total_img: {int(num_img - num_img * ratio)}\n')
    # f.write(f'model_path: {para_dict["model_path"]}\n')
    # f.write(f'data_path: {para_dict["data_path"]}\n')