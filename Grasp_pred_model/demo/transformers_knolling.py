import torch
from torch import nn
import numpy as np


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
