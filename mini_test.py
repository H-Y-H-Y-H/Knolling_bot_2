import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'


np.random.seed(1)
def generate_synthetic_data(num_samples, input_size, output_size, num_components):
    X = np.random.randn(num_samples, input_size).astype(np.float32)  # Input features
    Y = np.zeros((num_samples, output_size), dtype=np.float32)  # Placeholder for labels

    # Randomly generate parameters for GMM components for simplicity
    means = np.random.randn(num_components, output_size).astype(np.float32) * 5
    variances = np.abs(np.random.randn(num_components, output_size).astype(np.float32)) + 1
    weights = np.abs(np.random.randn(num_samples, num_components).astype(np.float32))
    weights /= weights.sum(axis=1, keepdims=True)

    for i in range(num_samples):
        component = np.random.choice(num_components, p=weights[i])
        Y[i] = means[component] + np.random.randn(output_size) * np.sqrt(variances[component])

    return torch.tensor(X,device=device), torch.tensor(Y,device=device)

# Generate synthetic data
num_samples = 1200
train_data = 1000
input_size = 2
output_size = 2
num_components = 3
X, Y = generate_synthetic_data(num_samples, input_size, output_size, num_components)
X_train = X[:train_data]
Y_train = Y[:train_data]
X_test = X[train_data:]
Y_test = Y[train_data:]
print(X.shape,Y.shape)
import matplotlib.pyplot as plt

# for i in range(1000):
#     plt.scatter(X[i][0],X[i][1],c='r')
#     plt.scatter(Y[i][0], Y[i][1],c='g')
# plt.show()

class GMM_MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_components):
        super(GMM_MLP, self).__init__()
        self.num_components = num_components
        self.output_size = 2  # For 2D data

        # Calculate the output layer size based on GMM parameters
        # Each component needs 2 means, 2 variances, and 1 weight
        gmm_param_size = num_components * (2 * self.output_size + 1)

        self.network = nn.Sequential()
        layer_sizes = [input_size] + hidden_sizes + [gmm_param_size]
        for h1, h2 in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.network.add_module(f"layer{h1}_to_{h2}", nn.Linear(h1, h2))
            if h2 != gmm_param_size:  # No activation after the last layer
                self.network.add_module(f"relu_{h2}", nn.ReLU())

    def forward(self, x):
        x = self.network(x)
        # Splitting the output into means, variances, and weights
        means = x[:, :self.num_components * self.output_size].view(-1,num_components,output_size)
        variances = x[:, self.num_components * self.output_size:2 * self.num_components * self.output_size].view(-1,num_components,output_size)
        weights = x[:, -self.num_components:].view(-1,num_components)

        # Ensure variances are positive and weights sum to 1
        variances = F.softplus(variances)
        weights = F.softmax(weights, dim=1)

        return means, variances, weights


# def gmm_loss(means, variances, weights, Y):
#     """
#     Calculate the negative log-likelihood loss for a batch of data Y given
#     the parameters of a Gaussian Mixture Model.
#
#     Parameters:
#     - means: Tensor of shape [batch_size, num_components, output_size].
#     - variances: Tensor of shape [batch_size, num_components, output_size], positive values.
#     - weights: Tensor of shape [batch_size, num_components], should sum to 1 across components.
#     - Y: Tensor of shape [batch_size, output_size], the target data.
#
#     Returns:
#     - loss: Scalar tensor, the negative log-likelihood loss.
#     """
#     batch_size, num_components, output_size = means.shape
#     Y = Y.unsqueeze(1).expand(-1, num_components, -1)  # Reshape for broadcasting
#
#     # Calculate the Gaussian PDF component-wise
#     inv_variances = 1.0 / variances
#     exp_term = torch.exp(-0.5 * ((Y - means) ** 2) * inv_variances)
#     norm_term = 1.0 / torch.sqrt(2 * torch.pi * variances)
#     pdf_vals = exp_term * norm_term  # Shape: [batch_size, num_components, output_size]
#
#     # Product of probabilities across dimensions (assuming independence)
#     pdf_vals_prod = torch.prod(pdf_vals, dim=2)  # Shape: [batch_size, num_components]
#
#     # Weighted sum across components and log
#     weighted_pdf_vals = weights * pdf_vals_prod  # Shape: [batch_size, num_components]
#     log_likelihood = torch.log(torch.sum(weighted_pdf_vals, dim=1))  # Shape: [batch_size]
#
#     # Negative log-likelihood loss
#     loss = -torch.mean(log_likelihood)
#
#     return loss

def mse_loss(means, variances, weights, target_value, num_samples=10):
    std = torch.sqrt(variances)
    sample_v = std*torch.rand(1000,3,2,device=device)
    samples = sample_v + means
    samples_errors = abs(samples - target_value.unsqueeze(1)).sum(2)
    # Calculate the values between
    min_samples_errors_id = torch.argmin(samples_errors, dim=1)
    unique_indices, counts = min_samples_errors_id.unique(return_counts=True)
    # Calculate the probability of each index
    probabilities = counts.float() / min_samples_errors_id.size(0)
    samples_prob = probabilities[min_samples_errors_id]

    samples_errors = samples_errors[np.arange(len(min_samples_errors_id)),min_samples_errors_id]

    weighted_loss = samples_errors*samples_prob

    loss_all = weighted_loss.mean()

    return loss_all



def min_sampling_error_loss(samples, targets):
    """
    Compute the loss as the minimum squared error among all samples from the GMM.
    Args:
    - samples: Tensor of shape [batch_size, num_samples, output_dim], sampled from the GMM.
    - targets: Tensor of shape [batch_size, output_dim], true labels.

    Returns:
    - loss: Scalar tensor, the mean minimum squared error across the batch.
    """
    batch_size, num_samples, output_dim = samples.size()
    targets = targets.unsqueeze(1).expand(-1, num_samples, -1)  # Match samples shape
    mse = (samples - targets).pow(2).mean(dim=2)  # Squared error for each sample
    min_mse = mse.min(dim=1)[0]  # Minimum error across samples
    return min_mse.mean()  # Mean minimum error across batch


def sample_gmm(means, variances, weights, num_samples=1):
    """
    Sample from the Gaussian mixture model defined by means, variances, and weights.
    """
    N, C, D = means.shape  # Batch size, Number of components, Dimension of output
    samples = torch.zeros((N, num_samples, D), device=means.device)

    # Choose components based on weights
    cumulative_weights = weights.cumsum(dim=1)
    rand_vals = torch.rand(N, num_samples, device=means.device).unsqueeze(-1)
    component_indices = (rand_vals > cumulative_weights.unsqueeze(1)).sum(dim=2)  # Find which component each sample belongs to

    for i in range(N):  # Loop over batch
        for j in range(num_samples):  # Loop over samples
            component = component_indices[i, j]
            mean = means[i, component]
            variance = variances[i, component]
            samples[i, j] = torch.normal(mean, torch.sqrt(variance))

    return samples.squeeze()


def gmm_nll_loss(means, variances, weights, targets):
    """
    Compute the negative log-likelihood of targets under a Gaussian Mixture Model.

    Parameters:
    - means: Predicted means of the GMM components, shape [batch_size, num_components, output_dim].
    - variances: Predicted variances of the GMM components, shape [batch_size, num_components, output_dim].
    - weights: Predicted mixture weights of the GMM components, shape [batch_size, num_components].
    - targets: True target values, shape [batch_size, output_dim].

    Returns:
    - nll_loss: The negative log-likelihood loss, a scalar tensor.
    """
    batch_size, num_components, output_dim = means.size()
    targets = targets.unsqueeze(1).expand(-1, num_components, -1)  # [batch_size, num_components, output_dim]

    # Compute the Gaussian probability density for each component
    variances = variances.clamp(min=1e-6)  # Ensure variance is not too close to zero
    inv_variances = 1.0 / variances
    exp_term = ((targets - means) ** 2) * inv_variances
    exp_term = torch.sum(exp_term, dim=2)  # Sum over output dimensions
    norm_term = torch.log(2 * torch.pi * variances).sum(dim=2)  # Log normalization term
    log_prob = -0.5 * (norm_term + exp_term)

    # Log-sum-exp trick for numerical stability
    log_weights = torch.log(weights.clamp(min=1e-6))  # Log mixture weights
    logsumexp_term = torch.logsumexp(log_prob + log_weights, dim=1)

    # Negative log-likelihood loss
    nll_loss = -torch.mean(logsumexp_term)

    return nll_loss

from torch.utils.data import DataLoader, TensorDataset

# Assuming X and Y are your input and target tensors
dataset = TensorDataset(X_train, Y_train)
batch_size = 1024  # You can adjust the batch size as needed

# Create a DataLoader to handle batching and shuffling
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_model(model, X_train, Y_train, epochs=3000, lr=0.01,alpha= 0.5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, Y_batch in data_loader:
            optimizer.zero_grad()
            means, variances, weights = model(X_batch)
            nll_loss = gmm_nll_loss(means, variances, weights, Y_batch)
            # Sample from the GMM and compute loss
            mse_loss_value = mse_loss(means, variances, weights, Y_batch, num_samples)
            # sampling_loss  = min_sampling_error_loss(samples, Y_batch)
            # Combined Loss
            loss = alpha * nll_loss + (1 - alpha) * mse_loss_value
            # loss = mse_loss_value
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(data_loader)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {average_loss}',"MSE",mse_loss_value.item())

    # Example of generating predictions for a single input
    x_new = X_train[:120]  # Replace with your new input
    model.eval()
    with torch.no_grad():
        means, variances, weights = model(x_new)
        predicted_samples = sample_gmm(means, variances, weights, num_samples=10)  # Generate 10 samples

    x_new = x_new.detach().cpu().numpy()
    X, Y = X_train.detach().cpu().numpy(), Y_train.detach().cpu().numpy()
    predicted_samples = predicted_samples.detach().cpu().numpy()

    for i in range(120):
        plt.scatter(x_new[i][0],x_new[i][1],c='r')
        plt.scatter(predicted_samples[i][0], predicted_samples[i][1],c='g')
        plt.scatter(X[i][0],X[i][1],c='y')
        plt.scatter(Y[i][0], Y[i][1],c='b')
    plt.show()


model = GMM_MLP(input_size, [32,64,64], num_components).to(device)
train_model(model, X, Y)

