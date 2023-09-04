import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np

class MLP(nn.Module):
    def __init__(self, num_boxes, output_size, node_1, node_2, node_3, device):
        super(MLP, self).__init__()

        self.input_dim = num_boxes * 7
        self.output_dim = output_size


        # Define the output layer
        self.linear1 = nn.Linear(self.input_dim, node_1).to(device)
        self.linear2 = nn.Linear(node_1, node_2).to(device)
        self.linear3 = nn.Linear(node_2, node_3).to(device)
        self.linear4 = nn.Linear(node_3, self.output_dim).to(device)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input_data, deploy_flag=False):

        out = self.relu(self.linear1(input_data))
        out = self.relu(self.linear2(out))
        out = self.relu(self.linear3(out))
        out = self.linear4(out)
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
