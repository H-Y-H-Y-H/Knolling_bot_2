import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
data = np.array([[[random.random() for _ in range(7)] for _ in range(random.randint(4, 15))] for _ in range(1000)])
print(data.shape)

# Custom dataset class
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Assuming you have a list of 3D sequences as your data
data = [
    [[1, 2, 3], [4, 5], [6, 7, 8, 9]],
    [[10, 11], [12, 13, 14, 15, 16], [17, 18, 19]],
    [[20, 21, 22], [23, 24], [25, 26, 27, 28, 29, 30]]
]

# Create an instance of your custom dataset
dataset = MyDataset(data)

# Specify batch size and number of workers for loading the data
batch_size = 2
num_workers = 2

# Create a data loader for your dataset
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


# Define your LSTM model
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        # Sort sequences by length in descending order
        sorted_lengths, sorted_indices = lengths.sort(descending=True)
        sorted_inputs = x[sorted_indices]

        # Pack sequences
        packed_inputs = pack_padded_sequence(sorted_inputs, sorted_lengths, batch_first=True)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        packed_out, _ = self.lstm(packed_inputs, (h0, c0))

        # Unpack sequences
        unpacked_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # Revert the original sequence order
        _, original_indices = sorted_indices.sort()
        unpacked_out = unpacked_out[original_indices]

        out = self.fc(unpacked_out[:, -1, :])
        return out


# Define hyperparameters
input_size = 1
hidden_size = 64
num_layers = 2

# Create an instance of your LSTM model
model = LSTMModel(input_size, hidden_size, num_layers)

# Loop through the data using the data loader
for batch in dataloader:
    inputs = batch

    # Compute the lengths of each sequence in the batch
    lengths = torch.tensor([len(seq) for seq in inputs])

    # Pad the sequences in the batch along the second dimension
    padded_inputs = pad_sequence([torch.tensor(seq) for seq in inputs], batch_first=True)

    # Forward pass
    outputs = model(padded_inputs, lengths)

    # Compute loss (assuming regression)
    targets = torch.zeros(batch_size)  # Assuming your model is for regression, change this as needed
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(outputs.squeeze(), targets)

    # Backward pass and optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
