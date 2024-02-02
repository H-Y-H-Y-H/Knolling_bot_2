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

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

# input min&max: [0.016, 0.048]
# label min&max: [-0.14599999962002042, 0.294500007390976]
# input_min,input_max = 0.016,0.048
# label_min,label_max = -0.14599999962002042, 0.294500007390976
SCALE_DATA = 100
SHIFT_DATA = 50
# DATAROOT = "C:/Users/yuhan/Downloads/learning_data_804_20w/"
# DATAROOT = "../../../knolling_dataset/learning_data_826/"
# DATAROOT = "../../knolling_dataset/learning_data_910/"
DATAROOT = "../../../knolling_dataset/learning_data_1019_5w/"

def pad_sequences(sequences, max_seq_length=10, pad_value=0):
    padded_sequences = []
    # for i in tqdm(range(len(sequences))):
    #     seq = sequences[i]
    #     if len(seq) < max_seq_length:
    #         padding_length = max_seq_length - len(seq)
    #         padded_seq = list(seq) + [[pad_value] * 2 for _ in range(padding_length)]
    #         padded_sequences.append(padded_seq)
    #     else:
    #         padded_sequences.append(seq)
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
    def __init__(self, input_data, output_data, cls_data):
        self.input_data = input_data    # length and width
        self.output_data = output_data  # gt position
        self.cls_data = cls_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.output_data[idx], self.cls_data[idx]


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


class ProximityAwareLoss(nn.Module):
    def __init__(self, primary_loss_fn, min_distance):
        super(ProximityAwareLoss, self).__init__()
        self.primary_loss_fn = primary_loss_fn
        self.min_distance = min_distance

    def forward(self, predictions, targets):
        # Primary loss (e.g., mean squared error)
        primary_loss = self.primary_loss_fn(predictions, targets)

        # Proximity penalty
        n_objects = predictions.shape[0]
        proximity_penalty = 0

        for i in range(n_objects):
            for j in range(i+1, n_objects):
                distance = torch.norm(predictions[i] - predictions[j])
                if distance < self.min_distance:
                    proximity_penalty += (self.min_distance - distance) ** 2

        # Total loss
        total_loss = primary_loss + proximity_penalty
        return total_loss


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
            pos_encoding_Flag = False,
            forwardtype = 0,
            high_dim_encoder=False,
            all_steps = False,
            max_obj_num = 10,
            num_gaussians=3,
            canvas_factor=1,
            use_overlap_loss=True,
            mse_loss_factor=1,
            overlap_loss_factor=1
    ):

        super(Knolling_Transformer, self).__init__()
        self.pos_encoding_Flag = pos_encoding_Flag
        self.all_zero_target = all_zero_target
        self.forwardtype = forwardtype
        self.losstype = 1
        self.high_dim_encoder = high_dim_encoder
        self.all_steps = all_steps

        self.best_gap = 0.015
        self.canvas_factor = canvas_factor
        self.padding_value = 1
        # self.min_overlap_area = np.inf
        self.min_overlap_num = np.inf
        self.use_overlap_loss = use_overlap_loss
        self.mse_loss_factor = mse_loss_factor
        self.overlap_loss_factor = overlap_loss_factor

        self.max_obj_num = max_obj_num# maximun 10
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
        self.l1_out = nn.Linear(map_embed_d_dim//4, output_size*num_gaussians*3)

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

        # x_mask = torch.ones_like(x, dtype=torch.bool)
        # x_mask[:obj_num] = False
        # x.masked_fill_(x_mask, 0)

        # Pass input through the encoder
        enc_x = self.encoder(x)

        # Depending on the forwardtype, pass through a specific set of layers
        if self.forwardtype == 0:
            x = self.l1(enc_x)
            x = self.acti(x)
            x = self.l2(x)
            x = self.acti(x)
            out = self.l_out(x)

        elif self.forwardtype == 2:
            out = self.decoder(enc_x, tart_x_gt_high)
            out = self.l_out(out)

        else:
            # Autoregressive decoding
            tart_x = torch.clone(tart_x_gt)
            # out = torch.zeros(enc_x.size(-1)).to(
            #     enc_x.device)  # Initialize with the correct shape and move to the same device as enc_x
            outputs = []
            results = 0
            for t in range(self.max_obj_num):
                tart_x_gt_high = self.position_encoder(tart_x)
                dec_output = self.decoder(enc_x, tart_x_gt_high)

                x = self.l0_out(dec_output)
                x = self.l1_out(x)
                x = x.view(x.shape[0],x.shape[1],x.shape[2]//(self.num_gaussians*3),self.num_gaussians*3)
                means, std_devs, weights = torch.split(x, split_size_or_sections=x.shape[-1] // 3, dim=-1)

                std_devs = torch.exp(std_devs)
                weights = F.softmax(weights, dim=-1)

                if temperature != 0:
                    # Apply temperature to weights
                    weights = torch.exp(weights / temperature)
                    weights /= weights.sum(dim=-1, keepdim=True)

                # Reshape the weights to a 2D tensor to apply torch.multinomial
                weights = weights.view(-1, weights.shape[-1])
                indices = torch.multinomial(weights, 1)

                # Reshape indices back to the original dimensions
                indices = indices.view(*weights.shape[:-1])

                # Reshape the means and std_devs to match the indices dimensions
                means = means.view(-1, means.shape[-1])
                std_devs = std_devs.view(-1, std_devs.shape[-1])

                # Use the indices to gather the means and std_devs
                chosen_means = torch.gather(means, 1, indices.view(-1, 1))
                chosen_std_devs = torch.gather(std_devs, 1, indices.view(-1, 1))

                # Reshape the means and std_devs back to the original dimensions
                chosen_means = chosen_means.view(*x.shape[:-1])
                chosen_std_devs = chosen_std_devs.view(*x.shape[:-1])

                out = chosen_means + chosen_std_devs * torch.randn_like(chosen_means)

                if t == self.max_obj_num-1:
                    results = out
                out = out[t]
                outputs.append(out.unsqueeze(0))

                if t == 0:
                    tart_x = torch.cat((out.unsqueeze(0), tart_x[1:]), dim=0)

                elif t < tart_x.size(0):
                    tart_x = torch.cat((torch.cat(outputs), tart_x[t + 1:]), dim=0)

            if self.all_steps:
                return results

            elif self.max_obj_num < tart_x.size(0):
                pad_data_shape = outputs[0].shape
                outputs = outputs + [torch.zeros(pad_data_shape, device=device) for _ in range(tart_x.size(0) - self.max_obj_num)]
            out = torch.cat(outputs, dim=0)
            return out


    def maskedMSELoss(self, predictions, target, ignore_index = -100):
        mask = target.ne(ignore_index)
        mse_loss = (predictions - target).pow(2) * mask
        mse_loss = mse_loss.sum() / mask.sum()

        return mse_loss

    def calculate_loss(self, pred_pos, tar_pos, tar_lw, tar_cls=None):

        MSE_loss = self.masked_MSE_loss(pred_pos, tar_pos, ignore_index=-100)

        tar_lw_raw = ((tar_lw - SHIFT_DATA) / SCALE_DATA).transpose(1, 0)
        tar_cls_raw = ((tar_cls - SHIFT_DATA) / SCALE_DATA).transpose(1, 0)
        tar_pos_raw = ((tar_pos - SHIFT_DATA) / SCALE_DATA).transpose(1, 0)
        pred_pos_raw = ((pred_pos - SHIFT_DATA) / SCALE_DATA).transpose(1, 0)

        if self.use_overlap_loss == True:
            Overlap_loss = self.calcualte_overlap_loss(pred_pos=pred_pos_raw, tar_pos=tar_pos_raw,
                                                     tar_lw=tar_lw_raw, tar_cls=tar_cls_raw)
            total_loss = MSE_loss * self.mse_loss_factor + self.overlap_loss_factor * Overlap_loss
        else:
            Overlap_loss = 0
            total_loss = MSE_loss

        # scaled_overlap_loss = 1 + 1 / self.max_obj_num * Overlap_loss
        # total_loss = MSE_loss + scaled_overlap_loss

        return total_loss

    def masked_MSE_loss(self, pred_pos, tar_pos, ignore_index=-100):

        mask = tar_pos.ne(ignore_index)
        mse_loss = (pred_pos - tar_pos).pow(2) * mask
        mse_loss = mse_loss.sum() / mask.sum()

        return mse_loss

    def calculate_distance_loss(self, pred_pos, tar_pos, tar_lw, tar_cls):

        tar_pos_x_low = (tar_pos[:, :, 0] - tar_lw[:, :, 0] / 2).unsqueeze(2)
        tar_pos_x_high = (tar_pos[:, :, 0] + tar_lw[:, :, 0] / 2).unsqueeze(2)
        tar_pos_y_low = (tar_pos[:, :, 1] - tar_lw[:, :, 1] / 2).unsqueeze(2)
        tar_pos_y_high = (tar_pos[:, :, 1] + tar_lw[:, :, 1] / 2).unsqueeze(2)

        pred_pos_x_low = (pred_pos[:, :, 0] - tar_lw[:, :, 0] / 2).unsqueeze(2)
        pred_pos_x_high = (pred_pos[:, :, 0] + tar_lw[:, :, 0] / 2).unsqueeze(2)
        pred_pos_y_low = (pred_pos[:, :, 1] - tar_lw[:, :, 1] / 2).unsqueeze(2)
        pred_pos_y_high = (pred_pos[:, :, 1] + tar_lw[:, :, 1] / 2).unsqueeze(2)

        num_cls = torch.unique(tar_cls.flatten())

        tar_total_min = []
        pred_total_min = []
        for i in num_cls:
            cls_index = tar_cls == i
            cls_mask = cls_index & cls_index.transpose(1, 2)

            tar_x_distance = tar_pos_x_low - tar_pos_x_high.transpose(1, 2)
            tar_y_distance = tar_pos_y_low - tar_pos_y_high.transpose(1, 2)
            tar_x_mask = tar_x_distance <= 0
            tar_y_mask = tar_y_distance <= 0
            tar_x_distance.masked_fill_(tar_x_mask, 100)
            tar_y_distance.masked_fill_(tar_y_mask, 100)

            tar_x_temp = tar_x_distance[cls_mask]
            tar_x_gap = tar_x_temp[tar_x_temp < 100]
            tar_y_temp = tar_y_distance[cls_mask]
            tar_y_gap = tar_y_temp[tar_y_temp < 100]
            if len(tar_x_gap) == 0 and len(tar_y_gap) == 0:
                print('x none and y none')
                tar_x_gap = torch.tensor(0.0001)
                tar_y_gap = torch.tensor(0.0001)
            elif len(tar_x_gap) == 0 and len(tar_y_gap) != 0:
                print('x none')
                tar_x_gap = torch.clone(tar_y_gap)
            elif len(tar_x_gap) != 0 and len(tar_y_gap) == 0:
                print('y none')
                tar_y_gap = torch.clone(tar_x_gap)

            # if len(tar_x_temp[tar_x_temp < 100]) == 0 or len(tar_y_temp[tar_y_temp < 100]) == 0:
            #     print('here')
            tar_total_min.append(torch.min(tar_x_gap))
            tar_total_min.append(torch.min(tar_y_gap))

            pred_x_distance = pred_pos_x_low - pred_pos_x_high.transpose(1, 2)
            pred_y_distance = pred_pos_y_low - pred_pos_y_high.transpose(1, 2)
            pred_x_mask = pred_x_distance <= 0
            pred_y_mask = pred_y_distance <= 0
            pred_x_distance.masked_fill_(pred_x_mask, 100)
            pred_y_distance.masked_fill_(pred_y_mask, 100)

            pred_x_temp = pred_x_distance[cls_mask]
            pred_y_temp = pred_y_distance[cls_mask]
            pred_x_gap = pred_x_temp[pred_x_temp < 100]
            pred_y_gap = pred_y_temp[pred_y_temp < 100]
            if len(pred_x_gap) == 0 and len(pred_y_gap) == 0:
                print('x none and y none')
                pred_x_gap = torch.tensor(0.0001)
                pred_y_gap = torch.tensor(0.0001)
            elif len(pred_x_gap) == 0 and len(pred_y_gap) != 0:
                print('x none')
                pred_x_gap = torch.clone(pred_y_gap)
            elif len(pred_x_gap) != 0 and len(pred_y_gap) == 0:
                print('y none')
                pred_y_gap = torch.clone(pred_x_gap)

            try:
                pred_total_min.append(torch.min(pred_x_gap))
                pred_total_min.append(torch.min(pred_y_gap))
            except:
                print('this is pred_x_gap', pred_x_gap)
                print('this is pred_y_gap', pred_y_gap)
                print('this is len x', len(pred_x_temp))
                print('this is len y', len(pred_y_temp))

        total_pred_min = torch.mean(torch.tensor(pred_total_min))

        return self.scale_distance_loss(total_pred_min)

    def scale_distance_loss(self, raw_loss):

        if raw_loss > 0.015:
            return 0 + (raw_loss * 200 / 3 - 1) * 0.2
        else:
            log_loss = torch.abs(torch.log(raw_loss * 200 / 3)) + 1
            if log_loss > 15:
                print('this is log loss', log_loss)
            # return torch.abs(torch.log(raw_loss * 200 / 3)) + 1
            return log_loss

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

        # tar_pos_px = tar_pos.detach().cpu().numpy() * self.mm2px + 10
        # tar_lw_px = tar_lw.detach().cpu().numpy() * self.mm2px
        # pred_pos_px = pred_pos.detach().cpu().numpy() * self.mm2px + 10
        #
        # data_pred = np.concatenate(((pred_pos_px[:, :, 0] - tar_lw_px[:, :, 0] / 2)[:, :, np.newaxis],
        #                            (pred_pos_px[:, :, 1] - tar_lw_px[:, :, 1] / 2)[:, :, np.newaxis],
        #                            (pred_pos_px[:, :, 0] + tar_lw_px[:, :, 0] / 2)[:, :, np.newaxis],
        #                            (pred_pos_px[:, :, 1] + tar_lw_px[:, :, 1] / 2)[:, :, np.newaxis],
        #                            ), axis=2).astype(np.int32)
        #
        # # Iterate through each rectangle and draw them on the canvas
        # penalty_list = []
        # avg_overlap_area = []
        # avg_overlap_num = []
        # for i in range(data_pred.shape[0]):
        #
        #     x_offset_px = np.min(data_pred[i, :, 0])
        #     y_offset_px = np.min(data_pred[i, :, 1])
        #     data_pred[i, :, [0, 2]] -= x_offset_px
        #     data_pred[i, :, [1, 3]] -= y_offset_px
        #     x_max_px = np.max(data_pred[i, :, 2])
        #     y_max_px = np.max(data_pred[i, :, 3])
        #     # canvas_tar = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
        #     canvas_pred = np.zeros((x_max_px, y_max_px), dtype=np.uint8)
        #
        #     for j in range(data_pred.shape[1]):
        #         # corner_data_tar = data_tar[i, j]
        #         corner_data_pred = data_pred[i, j]
        #         # canvas_tar[corner_data_tar[0]:corner_data_tar[2], corner_data_tar[1]:corner_data_tar[3]] += padding_value
        #         canvas_pred[corner_data_pred[0]:corner_data_pred[2], corner_data_pred[1]:corner_data_pred[3]] += self.padding_value
        #         # canvas = cv2.rectangle(canvas, (corner_data[0], corner_data[1]), (corner_data[2], corner_data[3]), 255, -1)
        #
        #     # if np.max(canvas_pred) <= 0 or np.any(data_pred[i, :, 0] <= 0) or np.any(data_pred[i, :, 1] <= 0):
        #     #     avg_overlap_area.append(100)
        #     #     avg_overlap_num.append(100)
        #     #     penalty_list.append(100)
        #     # else:
        #     #     overlap_area = np.clip(len(np.where(canvas_pred > self.padding_value)[0]) / self.overlap_area_factor, 1, None)
        #     #     if overlap_area < self.min_overlap_area:
        #     #         self.min_overlap_area = overlap_area
        #     #         print('this is max_overlap_area', self.min_overlap_area)
        #     #     overlap_num = np.clip(int(np.max(canvas_pred) / self.padding_value), 1, None)
        #     #     avg_overlap_area.append(overlap_area)
        #     #     avg_overlap_num.append(overlap_num)
        #     #     penalty_list.append(overlap_num * overlap_area)
        #
        #     # overlap_area = np.clip(len(np.where(canvas_pred > self.padding_value)[0]) / self.overlap_area_factor, 1, None)
        #     # if overlap_area < self.min_overlap_area:
        #     #     self.min_overlap_area = overlap_area
        #     #     print('this is max_overlap_area', self.min_overlap_area)
        #     overlap_num = np.clip(int(np.max(canvas_pred) / self.padding_value), 1, None)
        #     # avg_overlap_area.append(overlap_area)
        #     avg_overlap_num.append(overlap_num)
        #     penalty_list.append(overlap_num)
        # # avg_overlap_area = np.mean(np.asarray(avg_overlap_area))
        # avg_overlap_num = np.mean(np.asarray(avg_overlap_num))
        # penalty = np.mean(np.asarray(penalty_list))
        # if avg_overlap_num < self.min_overlap_num:
        #     self.min_overlap_num = avg_overlap_num
        #     print('this is min overlap num:', self.min_overlap_num)


        return penalty

if __name__ == "__main__":

    max_length = 10
    d_dim = 32
    layers_num = 4
    heads_num = 4
    print(d_dim,layers_num,heads_num)
    model = Knolling_Transformer(
        input_length=max_length,
        input_size=2,
        map_embed_d_dim=d_dim,
        num_layers=layers_num,
        forward_expansion=4,
        heads=heads_num,
        dropout=0.0,
        all_zero_target=0,
        pos_encoding_Flag = True,
        forwardtype= 1,
        high_dim_encoder = True
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20, verbose=True) #tainloss


    BATCH_SIZE = 64
    EPOCHS = 10000
    NUM_objects = 10
    noise_std = 0.1
    partial_dataset = 1000
    dataset_path = DATAROOT + 'cfg_1/'
    raw_data = np.loadtxt(dataset_path + 'labels_after/num_%d.txt' % NUM_objects)[:partial_dataset]*SCALE_DATA+SHIFT_DATA
    # raw_data +=1
    train_data = raw_data[:int(len(raw_data) * 0.8)]
    test_data = raw_data[int(len(raw_data) * 0.8):]
    print('NUM OBJECT: ', NUM_objects)
    print('BATCH: ',    BATCH_SIZE)
    print('noise_std ', noise_std)
    print('partial data ', partial_dataset)


    train_input = []
    test_input = []
    train_label = []
    test_label = []
    for i in range(NUM_objects):
        train_input.append(train_data[:, i * 5 + 2:i * 5 + 4])
        test_input.append(test_data[:, i * 5 + 2:i * 5 + 4])
        train_label.append(train_data[:, i * 5:i * 5 + 2])
        test_label.append(test_data[:, i * 5:i * 5 + 2])


    train_input = np.asarray(train_input).transpose(1, 0, 2)
    test_input = np.asarray(test_input).transpose(1, 0, 2)
    train_label = np.asarray(train_label).transpose(1, 0, 2)
    test_label = np.asarray(test_label).transpose(1, 0, 2)

    train_input = pad_sequences(train_input, max_seq_length=max_length)
    test_input  = pad_sequences(test_input , max_seq_length=max_length)
    train_label = pad_sequences(train_label, max_seq_length=max_length)
    test_label  = pad_sequences(test_label , max_seq_length=max_length)

    train_dataset = CustomDataset(train_input, train_label)
    test_dataset = CustomDataset(test_input, test_label)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.to(device)
    # Training loop

    noise_std = torch.tensor(noise_std).to(device)

    for epoch in range(EPOCHS):
        prin_flag = True
        model.train()
        for input_batch, target_batch in train_loader:
            # Zero the gradients
            optimizer.zero_grad()

            input_batch = torch.from_numpy(np.asarray(input_batch, dtype=np.float32)).to(device)
            target_batch = torch.from_numpy(np.asarray(target_batch, dtype=np.float32)).to(device)
            input_batch = input_batch.transpose(1,0)
            target_batch = target_batch.transpose(1,0)

            # zero to False
            target_batch_atten_mask = (target_batch == 0).bool()
            target_batch.masked_fill_(target_batch_atten_mask, -100)

            # Mask the target:
            mask_point = np.random.randint(0, target_batch.size(0)+1)
            # mask_point = 0
            mask = torch.ones_like(target_batch, dtype=torch.bool)
            mask[:mask_point] = False
            input_target_batch = torch.clone(target_batch)

            # Add noise
            input_target_batch = torch.normal(input_target_batch, noise_std)
            input_batch = torch.normal(input_batch, noise_std)

            input_target_batch.masked_fill_(mask, -100)

            label_mask = torch.ones_like(target_batch, dtype=torch.bool)
            label_mask[:(mask_point+1)] = False
            target_batch.masked_fill_(label_mask, -100)

            # Forward pass

            output_batch = model(input_batch, input_target_batch)

            # Calculate loss
            loss = model.maskedMSELoss(output_batch, target_batch)

            if epoch %10 ==0 and prin_flag:
                # print('-')
                # print('output',output_batch[:,0].flatten())
                # print('target',target_batch[:,0].flatten())
                prin_flag = False

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()


        # Validation
        # model.eval()
        # validation_loss = 0.0
        # with torch.no_grad():
        #     for input_batch, target_batch in test_loader:
        #         input_batch = torch.from_numpy(np.asarray(input_batch, dtype=np.float32)).to(device)
        #         target_batch = torch.from_numpy(np.asarray(target_batch, dtype=np.float32)).to(device)
        #
        #         # Forward pass
        #         input_batch = input_batch.transpose(1, 0)
        #         target_batch = target_batch.transpose(1, 0)
        #         output_batch = model(input_batch)
        #
        #         # Calculate loss
        #         loss = criterion(output_batch, target_batch)
        #         validation_loss += loss.item()
        #
        # validation_loss /= len(test_loader)
        # print(f"Epoch {epoch + 1}/{EPOCHS}, Training Loss: {loss.item()}, Validation Loss: {validation_loss}")

        print(f"Epoch {epoch + 1}/{EPOCHS}, Training Loss: {loss.item()}")
        scheduler.step(loss.item())
