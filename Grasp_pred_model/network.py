import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

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
        self.softmax = nn.Softmax(dim=1)

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

        loss = self.criterion(predict[mask], target[mask].long())

        return loss