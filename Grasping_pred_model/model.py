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

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.linear(out)
        return out
