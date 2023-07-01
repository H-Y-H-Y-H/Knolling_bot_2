import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import Transformer
from torch.utils.data import Dataset, DataLoader
import math
import torch.optim as optim
import sys
sys.path.append('../')
from pos_encoder import *
import torch.nn.functional as F
from model import *


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
        super(LSTMRegressor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_dim).to(x.device)

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_dim).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        out = self.linear(out)
        return out

# CrossEntropy
    def maskedMSELoss(self, predictions, target, ignore_index = -100):
        mask = target.ne(ignore_index)
        mse_loss = (predictions - target).pow(2) * mask
        mse_loss = mse_loss.sum() / mask.sum()

        return mse_loss


if __name__ == '__main__':

    target_batch = torch.tensor([[1,2,3,4],
                                 [1,2,3,4],
                                 [1,2,3,4]])

    mask = torch.ones_like(target_batch, dtype=torch.bool)

    target_batch_atten_mask = (target_batch == 0).bool()
    target_batch.masked_fill_(label_mask, -100)

    torch.nn.utils.rnn.pack_padded_sequence
