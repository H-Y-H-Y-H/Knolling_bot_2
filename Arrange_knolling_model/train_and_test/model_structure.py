import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import Transformer
from torch.utils.data import Dataset, DataLoader
import math
import torch.optim as optim
import torch.nn.functional as F
import cv2
DATAROOT = "../../../knolling_dataset/learning_data_205_10/"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
SHIFT_DATA = 100
SCALE_DATA = 100

def pad_sequences(sequences, max_seq_length=10, pad_value=0):
    padded_sequences = []
    for i in tqdm(range(len(sequences))):
        seq = sequences[i]
        if np.sum(np.any(seq != 0, axis=1)) < max_seq_length:
            padding_length = max_seq_length - len(seq)
            padded_seq = list(seq) + [[pad_value] * 2 for _ in range(padding_length)]
            padded_sequences.append(padded_seq)
        else:
            padded_sequences.append(seq)

    padded_sequences = np.asarray(padded_sequences)
    return padded_sequences


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_data, output_data):
        self.input_data = input_data    # length and width
        self.output_data = output_data  # gt position

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.output_data[idx]


class PositionalEncoder(nn.Module):
    def __init__(
            self,
            d_input: int,
            n_freqs: int,
            log_space: bool = False
    ):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [lambda x: x]

        if self.log_space:
            freq_bands = 2. ** torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** (self.n_freqs - 1), self.n_freqs)

        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))
    def forward(
            self,
            x
    ) -> torch.Tensor:
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()

        self.attention = nn.MultiheadAttention(embed_dim=embed_size,
                                               num_heads=heads, batch_first=False)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        atten_outputs, atten_output_weights = self.attention(value, key, query)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(atten_outputs + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
            self,
            input_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size

        self.layers = nn.ModuleList(
            [TransformerBlock(
                embed_size,
                heads,
                dropout=dropout,
                forward_expansion=forward_expansion)
            ] * num_layers
        )

        self.dropout = nn.Dropout(dropout)
        self.fc_in = nn.Linear(input_size,embed_size)

    def forward(self, x):
        x = self.fc_in(x)
        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            x = layer(x, x, x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout,device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        # self.attention = SelfAttention(embed_size, heads=heads)
        self.attention = nn.MultiheadAttention(embed_dim=embed_size,
                                               num_heads=heads, batch_first=False)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key):
        atten_outputs, atten_output_weights = self.attention(x, x, x)
        query = self.dropout(self.norm(atten_outputs + x))
        out = self.transformer_block(value, key, query)
        return out


class Decoder(nn.Module):
    def __init__(
            self,
            input_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            embed_size,
    ):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout,device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.activate_F = nn.ReLU()
        self.fc_in = nn.Linear(input_size, embed_size)

    def forward(self, enc_out,x):
        x = self.fc_in(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_out, enc_out)
        out = self.fc_out(x)
        return out



def calculate_collision_loss(pred_pos, obj_length_width, overlap_loss_weight=1):
    # Calculate half dimensions for easier overlap checking
    half_sizes = obj_length_width / 2.0

    # Expand dimensions to calculate pairwise differences between all objects
    pred_pos_expanded = pred_pos.unsqueeze(1)  # Shape: [Batchsize, 1, 10, 2]
    half_sizes_expanded = half_sizes.unsqueeze(1)  # Shape: [Batchsize, 1, 10, 2]

    # Compute differences in positions and sum of half sizes for all pairs
    pos_diff = torch.abs(pred_pos_expanded - pred_pos_expanded.transpose(1, 2))  # Shape: [Batchsize, 10, 10, 2]
    size_sum = half_sizes_expanded + half_sizes_expanded.transpose(1, 2)  # Shape: [Batchsize, 10, 10, 2]

    # Calculate overlap in each dimension
    overlap = size_sum - pos_diff  # Shape: [Batchsize, 10, 10, 2]
    overlap = torch.clamp(overlap, min=0)  # Remove negative values, no overlap

    # Calculate area of overlap for each pair of objects
    overlap_indicator = (overlap[...,0]>0)&(overlap[...,1]>0)
    overlap_area = overlap[..., 0] * overlap[..., 1] *overlap_indicator.float() # Shape: [Batchsize, 10, 10]

    # Multiply overlap area by 10 to heavily penalize collisions
    overlap_area *= overlap_loss_weight

    batch_size, num_objects, _ = pred_pos.shape
    collision_mask = ~torch.eye(num_objects, dtype=torch.bool,device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    overlap_area *= collision_mask

    # Calculate collision loss as the sum of all overlap areas
    collision_loss = overlap_area.sum(dim=[1, 2]).float()  # Sum over all pairs

    # Optionally, normalize the collision loss by the number of pairs to stabilize training
    num_pairs = num_objects * (num_objects - 1) / 2
    collision_loss = collision_loss / num_pairs

    return collision_loss.mean()

class Knolling_Transformer(nn.Module):
    def __init__(
            self,
            input_length=10,
            input_size=2,
            output_size = 2,
            map_embed_d_dim=128,
            num_layers=6,
            forward_expansion=4,
            heads=2,
            dropout=0.,
            all_zero_target=0,
            forwardtype = 0,
            pos_encoding_Flag=True,
            high_dim_encoder=True,
            all_steps = False,
            in_obj_num = 10,
            num_gaussians=4,
            overlap_loss_factor=10
    ):

        super(Knolling_Transformer, self).__init__()
        self.pos_encoding_Flag = pos_encoding_Flag
        self.all_zero_target = all_zero_target
        self.forwardtype = forwardtype
        self.losstype = 1
        self.high_dim_encoder = high_dim_encoder
        self.all_steps = all_steps
        self.min_std_dev = 3e-3  # Define a minimum standard deviation

        self.best_gap = 0.015
        self.padding_value = 1
        # self.min_overlap_area = np.inf
        self.min_overlap_num = np.inf
        self.overlap_loss_weight = overlap_loss_factor

        self.in_obj_num = in_obj_num # maximun 10
        self.num_gaussians = num_gaussians
        self.positional_encoding = PositionalEncoding(d_model = input_size, max_len=input_length)

        n_freqs = 5
        self.position_encoder = PositionalEncoder(d_input = 2, n_freqs =n_freqs)

        if high_dim_encoder:
            input_size = input_size * (1 + 2 * n_freqs)

        self.encoder = Encoder(
            input_size,
            embed_size = map_embed_d_dim,
            num_layers = num_layers,
            heads = heads,
            forward_expansion = forward_expansion,
            dropout=dropout
        )

        self.decoder = Decoder(
            input_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            embed_size=map_embed_d_dim,
        )

        self.l1 = nn.Linear(map_embed_d_dim, map_embed_d_dim*2)
        self.l2 = nn.Linear(map_embed_d_dim*2, map_embed_d_dim)

        # self.bn1 = nn.BatchNorm1d(map_embed_d_dim)
        self.l0_out = nn.Linear(map_embed_d_dim, map_embed_d_dim//4)
        self.l1_out = nn.Linear(map_embed_d_dim//4, num_gaussians*(2+2+1))

        self.acti = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, tart_x_gt=None,temperature=1):
        # This section defines the forward pass of the model.

        # If positional encoding is needed, apply it
        if self.pos_encoding_Flag == True:
            x = self.positional_encoding(x)

        # If high dimensional encoder is set, apply it
        if self.high_dim_encoder:
            x = self.position_encoder(x)
            tart_x_gt_high = self.position_encoder(tart_x_gt)
        else:
            tart_x_gt_high = tart_x_gt


        # Pass input through the encoder
        enc_x = self.encoder(x)

        # Depending on the forwardtype, pass through a specific set of layers
        if self.forwardtype == 0:
            x = self.l1(enc_x)
            x = self.acti(x)
            x = self.l2(x)
            x = self.acti(x)
            out = self.l_out(x)
            return out


        elif self.forwardtype == 2:
            out = self.decoder(enc_x, tart_x_gt_high)
            out = self.l_out(out)
            return out

        else:
            # Autoregressive decoding
            tart_x = torch.clone(tart_x_gt)
            # out = torch.zeros(enc_x.size(-1)).to(
            #     enc_x.device)  # Initialize with the correct shape and move to the same device as enc_x
            outputs = []
            pis = []  # Collect pi for all timesteps
            sigmas = []  # Collect sigma for all timesteps
            mus = []  # Collect mu for all timesteps

            results = 0
            for t in range(self.in_obj_num):
                tart_x_gt_high = self.position_encoder(tart_x)
                dec_output = self.decoder(enc_x, tart_x_gt_high)

                x = self.l0_out(dec_output)
                x = self.l1_out(x)
                x = x.view(x.shape[0], x.shape[1], self.num_gaussians, 5)
                # Split along the last dimension to extract means, std_devs, and weights
                means = x[:, :, :, :2]  # First two are means for x and y
                std_devs = torch.nn.functional.softplus(x[:, :, :, 2:4]) + self.min_std_dev
                weights = F.softmax(x[:, :, :, 4],dim=-1)  # Last one is the weight, apply softmax across gaussians for each object

                pis.append(weights[t].unsqueeze(0))
                sigmas.append(std_devs[t].unsqueeze(0))
                mus.append(means[t].unsqueeze(0))

                if temperature != 0:
                    # Apply temperature to weights
                    weights = torch.exp(weights[t] / temperature)
                    weights /= weights.sum(dim=-1, keepdim=True)

                # return the idx of selecting Gaussion
                indices = torch.multinomial(weights, 1)

                # Gather the chosen means and std_devs based on sampled indices
                # Batch is the second dimension, we adjust gathering for means and std_devs
                chosen_means = torch.gather(means[t], 1, indices.unsqueeze(-1).expand(-1, -1, 2))
                chosen_std_devs = torch.gather(std_devs[t], 1, indices.unsqueeze(-1).expand(-1, -1, 2))

                # Generate output for the current step
                out = chosen_means + chosen_std_devs * torch.randn_like(chosen_means)
                out = out.squeeze()

                outputs.append(out.unsqueeze(0))


                if t == 0:
                    tart_x = torch.cat((out.unsqueeze(0), tart_x[1:]), dim=0)

                elif t < tart_x.size(0):
                    tart_x = torch.cat((torch.cat(outputs), tart_x[t + 1:]), dim=0)

            # Concatenate collected pi, sigma, mu for all timesteps
            pi = torch.cat(pis, dim=0)
            sigma = torch.cat(sigmas, dim=0)
            mu = torch.cat(mus, dim=0)

            if self.all_steps:
                return results, pi, sigma, mu

            elif self.in_obj_num < tart_x.size(0):
                pad_data_shape = outputs[0].shape
                outputs = outputs + [torch.zeros(pad_data_shape, device=device) for _ in range(tart_x.size(0) - self.in_obj_num)]
            out = torch.cat(outputs, dim=0)
            return out, pi, sigma, mu

    def calculate_loss(self, pred_pos, tar_pos, obj_length_width):

        MSE_loss = self.masked_MSE_loss(pred_pos, tar_pos, ignore_index=-100)

        # The length and width of objects:
        obj_length_width = ((obj_length_width - SHIFT_DATA) / SCALE_DATA).transpose(1, 0)

        # The predicted position of objects:

        pred_pos_raw = ((pred_pos - SHIFT_DATA) / SCALE_DATA).transpose(1, 0)

        overlap_loss = calculate_collision_loss(pred_pos_raw,obj_length_width,self.overlap_loss_weight).mean()
        total_loss = MSE_loss + overlap_loss


        return total_loss, overlap_loss

    def mdn_loss_function(self, weights, variances, means, targets):
        """
        Compute the negative log-likelihood of targets under a Gaussian Mixture Model.

        Parameters:
        - means: Predicted means of the GMM components, shape [step, batch_size, num_components, output_dim].
        - variances: Predicted variances of the GMM components, shape [step, batch_size, num_components, output_dim].
        - weights: Predicted mixture weights of the GMM components, shape [step, batch_size, num_components].
        - targets: True target values, shape [step, batch_size, output_dim].

        Returns:
        - nll_loss: The negative log-likelihood loss, a scalar tensor.
        """

        num_step, batch_size, num_components, output_dim = means.size()
        targets = targets.unsqueeze(2).expand(-1,-1, num_components, -1)  # [step, batch_size, num_components, output_dim]

        # Compute the Gaussian probability density for each component
        variances = variances.clamp(min=1e-6)  # Ensure variance is not too close to zero
        inv_variances = 1.0 / variances
        exp_term = ((targets - means) ** 2) * inv_variances
        exp_term = torch.sum(exp_term, dim=3)  # Sum over output dimensions
        norm_term = torch.log(2 * torch.pi * variances).sum(dim=3)  # Log normalization term
        log_prob = -0.5 * (norm_term + exp_term)

        # Log-sum-exp trick for numerical stability
        log_weights = torch.log(weights.clamp(min=1e-6))  # Log mixture weights
        logsumexp_term = torch.logsumexp(log_prob + log_weights, dim=2)

        # Negative log-likelihood loss
        nll_loss = -torch.mean(logsumexp_term)

        return nll_loss


        # # Reshape y to match the Gaussian components (Num_of_objects, Batch_size, 1, 2)
        # y = y.unsqueeze(2)  # Assuming y has shape (Num_of_objects, Batch_size, 2)
        #
        # # Calculate the mixture of Gaussian probabilities
        # # Assuming sigma is positive and represents standard deviation
        # norm = torch.distributions.Normal(mu, sigma)
        #
        # prob = norm.log_prob(y)  # This computes the log probability of y for each Gaussian
        # prob = torch.clamp(prob,-float('inf'),-1e-9)
        # # prob shape is (Num_of_objects, Batch_size, Num_of_Gaussian,2), need to sum log probs over dimensions
        # log_prob = prob.sum(dim=-1)  # New shape: [Num_of_objects, Batch_size, Num_of_Gaussian]
        #
        # # Weighting log probabilities by the mixing coefficients (pi)
        # # Adding a small value to pi to avoid log(0)
        # weighted_log_prob = log_prob + torch.log(pi + 1e-9)
        #
        # # Using logsumexp to sum across Gaussians for numerical stability
        # # This combines the weighted log probabilities for all Gaussians for each object in the batch
        # loss = torch.logsumexp(weighted_log_prob, dim=2)  # Shape: [Num_of_objects, Batch_size]
        #
        # # Taking the negative and averaging over all objects and all batch items
        # loss = -loss.mean()
        # return loss

    def masked_MSE_loss(self, pred_pos, tar_pos, ignore_index=-100):

        mask = tar_pos.ne(ignore_index)
        mse_loss = (pred_pos - tar_pos).pow(2) * mask
        mse_loss = mse_loss.sum() / mask.sum()

        return mse_loss

    def calcualte_overlap_loss(self, pred_pos, tar_pos, tar_lw, tar_cls):

        # Assuming tar_pos, tar_lw, pred_pos are PyTorch tensors, and self.mm2px is a scalar tensor.
        self.mm2px = 530 / (0.34 * self.canvas_factor)
        tar_pos_px = tar_pos * self.mm2px + 10
        tar_lw_px = tar_lw * self.mm2px
        pred_pos_px = pred_pos * self.mm2px + 10

        data_pred = torch.cat(((pred_pos_px[:, :, 0] - tar_lw_px[:, :, 0] / 2).unsqueeze(2),
                               (pred_pos_px[:, :, 1] - tar_lw_px[:, :, 1] / 2).unsqueeze(2),
                               (pred_pos_px[:, :, 0] + tar_lw_px[:, :, 0] / 2).unsqueeze(2),
                               (pred_pos_px[:, :, 1] + tar_lw_px[:, :, 1] / 2).unsqueeze(2)), dim=2).type(torch.int32).to(device)

        # Iterate through each rectangle and draw them on the canvas
        penalty_list = []
        avg_overlap_area = []
        avg_overlap_num = []
        for i in range(data_pred.shape[0]):
            x_offset_px = torch.min(data_pred[i, :, 0])
            y_offset_px = torch.min(data_pred[i, :, 1])
            data_pred[i, :, [0, 2]] -= x_offset_px
            data_pred[i, :, [1, 3]] -= y_offset_px
            x_max_px = torch.max(data_pred[i, :, 2])
            y_max_px = torch.max(data_pred[i, :, 3])
            canvas_pred = torch.zeros(int(x_max_px), int(y_max_px))

            for j in range(data_pred.shape[1]):
                corner_data_pred = data_pred[i, j]
                canvas_pred[corner_data_pred[0]:corner_data_pred[2],
                corner_data_pred[1]:corner_data_pred[3]] += self.padding_value

            overlap_num = torch.clamp(torch.max(canvas_pred) / self.padding_value, 1)
            avg_overlap_num.append(overlap_num)
            penalty_list.append(overlap_num)

        avg_overlap_num = torch.mean(torch.stack(avg_overlap_num))
        penalty = torch.mean(torch.stack(penalty_list)).to(device).requires_grad_()

        if avg_overlap_num < self.min_overlap_num:
            self.min_overlap_num = avg_overlap_num
            print('this is min overlap num:', self.min_overlap_num)

        return penalty


def min_smaple_loss(weights, variances, means, target_value):
    std = torch.sqrt(variances)
    sample_v = std*torch.randn_like(std,device=device)
    samples = sample_v + means
    samples_errors = abs(samples - target_value.unsqueeze(2)).sum(3)
    # Calculate the values between
    min_samples_errors_id = torch.argmin(samples_errors, dim=2)
    unique_indices, counts = min_samples_errors_id.unique(return_counts=True)
    # Calculate the probability of each index
    probabilities = counts.float() / min_samples_errors_id.size(0)
    samples_prob = probabilities[min_samples_errors_id]
    min_samples_errors_id = torch.clamp(min_samples_errors_id, 0, 3 - 1)  # Ensure indices are in the range [0, 2]

    selected_values = torch.gather(samples_errors, 2, min_samples_errors_id.unsqueeze(-1))

    # Squeeze the last dimension to get the shape (3, 512)
    selected_values = selected_values.squeeze(-1)


    weighted_loss = selected_values*samples_prob

    loss_all = weighted_loss.mean()

    return loss_all


if __name__ == "__main__":
    from datetime import datetime

    max_seq_length = 10
    in_obj_num =3

    d_dim = 32
    layers_num = 4
    heads_num = 4
    num_gaussian = 4
    print(d_dim,layers_num,heads_num)
    model = Knolling_Transformer(
        input_length=max_seq_length,
        input_size=2,
        map_embed_d_dim=d_dim,
        num_layers=layers_num,
        forward_expansion=4,
        heads=heads_num,
        dropout=0.0,
        all_zero_target=0,
        pos_encoding_Flag = True,
        forwardtype= 1,
        high_dim_encoder = True,
        all_steps=False,
        in_obj_num=in_obj_num,
        num_gaussians=num_gaussian,
        overlap_loss_factor=0  # 1
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20, verbose=True) #tainloss

    DATAROOT = "../../../knolling_dataset/learning_data_0126_10/"

    EPOCHS = 10000
    noise_std = 0.1
    batch_size =512
    lr=1e-3
    scheduler_factor=0.1
    patience=20

    train_input_data = []
    train_output_data = []
    train_cls_data = []
    input_data = []
    output_data = []
    valid_input_data = []
    valid_output_data = []
    valid_cls_data = []

    DATA_CUT = 1000 #1 000 000 data

    SHIFT_DATASET_ID = 3
    policy_num = 1
    configuration_num = 1
    solu_num = int(policy_num * configuration_num)
    info_per_object = 7

    for f in range(SHIFT_DATASET_ID, SHIFT_DATASET_ID+solu_num):
        raw_data =np.loadtxt(DATAROOT+'test_data.txt')[:1000]

        raw_data = raw_data * SCALE_DATA + SHIFT_DATA

        train_data = raw_data[:int(len(raw_data) * 0.8)]
        test_data = raw_data[int(len(raw_data) * 0.8):]

        train_lw = []
        valid_lw = []
        train_pos = []
        valid_pos = []
        for i in range(in_obj_num):
            train_lw.append(train_data[:, i * info_per_object + 2:i * info_per_object + 4])
            valid_lw.append(test_data[:, i * info_per_object + 2:i * info_per_object + 4])
            train_pos.append(train_data[:, i * info_per_object:i * info_per_object + 2])
            valid_pos.append(test_data[:, i * info_per_object:i * info_per_object + 2])

        train_lw = np.asarray(train_lw).transpose(1, 0, 2)
        valid_lw = np.asarray(valid_lw).transpose(1, 0, 2)
        train_pos = np.asarray(train_pos).transpose(1, 0, 2)
        valid_pos = np.asarray(valid_pos).transpose(1, 0, 2)

        train_input_data += list(train_lw)
        train_output_data += list(train_pos)
        valid_input_data += list(valid_lw)
        valid_output_data += list(valid_pos)

    train_input_padded = pad_sequences(train_input_data,  max_seq_length=max_seq_length)
    train_label_padded = pad_sequences(train_output_data, max_seq_length=max_seq_length)
    test_input_padded = pad_sequences(valid_input_data,   max_seq_length=max_seq_length)
    test_label_padded = pad_sequences(valid_output_data,  max_seq_length=max_seq_length)

    train_dataset = CustomDataset(train_input_padded, train_label_padded)
    test_dataset = CustomDataset(test_input_padded, test_label_padded)

    num_data = (len(train_dataset), len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=scheduler_factor,
                                                           patience=patience, verbose=True)

    num_epochs = 1000
    train_loss_list = []
    valid_loss_list = []
    train_loss2_list = []
    valid_loss2_list = []

    model.to(device)
    abort_learning = 0
    min_loss = np.inf
    for epoch in range(num_epochs):
        print_flag = True
        model.train()
        train_loss = 0
        train_loss_overlap = 0

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()

            input_batch = torch.from_numpy(np.asarray(input_batch, dtype=np.float32)).to(device)
            target_batch = torch.from_numpy(np.asarray(target_batch, dtype=np.float32)).to(device)
            input_batch = input_batch.transpose(1, 0)
            target_batch = target_batch.transpose(1, 0)

            target_batch_atten_mask = (target_batch == 0).bool()
            target_batch.masked_fill_(target_batch_atten_mask, -100)

            # create all -100 input for decoder
            mask = torch.ones_like(target_batch, dtype=torch.bool)
            input_target_batch = torch.clone(target_batch)

            # input_target_batch = torch.normal(input_target_batch, noise_std)
            input_target_batch.masked_fill_(mask, -100)

            # Forward pass
            # object number > masked number
            output_batch, pi, sigma, mu = model(input_batch,
                                 tart_x_gt=input_target_batch)

            # Calculate log-likelihood loss
            ll_loss = model.mdn_loss_function(pi, sigma, mu, target_batch[:model.in_obj_num])
            # Calculate min sample loss
            ms_min_smaple_loss = min_smaple_loss(pi, sigma, mu, target_batch[:model.in_obj_num])
            # Calculate collision loss
            overlap_loss = calculate_collision_loss(output_batch[:model.in_obj_num].transpose(0,1),target_batch[:model.in_obj_num].transpose(0,1))

            loss = ll_loss+ms_min_smaple_loss+overlap_loss
            if epoch % 10 == 0 and print_flag:
                print('output', output_batch[:, 0].flatten())
                print('target', target_batch[:, 0].flatten())
                print('loss and overlap loss:',loss.item())
                print_flag = False

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loss_overlap += overlap_loss.item()
        train_loss = train_loss / len(train_loader)
        train_loss_overlap = train_loss_overlap/len(train_loader)
        train_loss_list.append(train_loss)
        train_loss2_list.append(train_loss_overlap)
        # Validate
        model.eval()
        print_flag = True
        with torch.no_grad():
            total_loss = 0
            valid_overlap_loss = 0
            for input_batch, target_batch in val_loader:
                input_batch = torch.from_numpy(np.asarray(input_batch, dtype=np.float32)).to(device)
                target_batch = torch.from_numpy(np.asarray(target_batch, dtype=np.float32)).to(device)
                input_batch = input_batch.transpose(1, 0)
                target_batch = target_batch.transpose(1, 0)

                # # zero to False
                # input_batch_atten_mask = (input_batch == 0).bool()
                # input_batch = torch.normal(input_batch, noise_std)  ## Add noise
                # input_batch.masked_fill_(input_batch_atten_mask, -100)

                target_batch_atten_mask = (target_batch == 0).bool()
                target_batch.masked_fill_(target_batch_atten_mask, -100)

                # create all -100 input for decoder
                mask = torch.ones_like(target_batch, dtype=torch.bool)
                input_target_batch = torch.clone(target_batch)

                # input_target_batch = torch.normal(input_target_batch, noise_std)
                input_target_batch.masked_fill_(mask, -100)

                # label_mask = torch.ones_like(target_batch, dtype=torch.bool)
                # label_mask[:object_num] = False
                # target_batch.masked_fill_(label_mask, -100)

                # Forward pass
                # object number > masked number
                output_batch, pi, sigma, mu = model(input_batch,
                                                    tart_x_gt=input_target_batch)

                # Calculate log-likelihood loss
                ll_loss = model.mdn_loss_function(pi, sigma, mu, target_batch[:model.in_obj_num])
                # Calculate min sample loss
                ms_min_smaple_loss = min_smaple_loss(pi, sigma, mu, target_batch[:model.in_obj_num])
                # Calculate collision loss
                overlap_loss = calculate_collision_loss(output_batch[:model.in_obj_num].transpose(0, 1),
                                                        target_batch[:model.in_obj_num].transpose(0, 1))
                # loss all
                loss = ll_loss + ms_min_smaple_loss + overlap_loss

                if epoch % 10 == 0 and print_flag:
                    print('val_output', output_batch[:, 0].flatten())
                    print('val_target', target_batch[:, 0].flatten())
                    print('loss and overlap loss:', loss.item(), overlap_loss.item())

                    print_flag = False

                total_loss += loss.item()
                valid_overlap_loss += overlap_loss.item()
            avg_loss = total_loss / len(val_loader)
            valid_overlap_loss = valid_overlap_loss/len(val_loader)
            scheduler.step(valid_overlap_loss)

            valid_loss_list.append(avg_loss)
            valid_loss2_list.append(valid_overlap_loss)

            if avg_loss < min_loss:
                min_loss = avg_loss
                abort_learning = 0
            if avg_loss>10e8:
                abort_learning=100
            else:
                abort_learning += 1

            print(f"{datetime.now()}Epoch {epoch},train_loss: {train_loss}, validation_loss: {avg_loss},"
                  f" no_improvements: {abort_learning}")

