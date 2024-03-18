import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os
import cv2
from tqdm import tqdm
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, input_dir, output_dir, num_img, num_total, start_idx, transform=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.num_img = num_img
        self.num_total=num_total
        self.start_idx = start_idx
        self.transform = transform

        self.img_input = []
        self.img_output = []
        for i in tqdm(range(num_img)):

            img_input = Image.open(input_dir + '%d.png' % (i+ self.start_idx))
            img_input = self.transform(img_input)
            self.img_input.append(img_input)

            # img_check = (img_input.numpy() * 255).astype(np.uint8)
            # img_check = np.transpose(img_check, [1, 2, 0])
            # cv2.namedWindow('zzz', 0)
            # cv2.resizeWindow('zzz', 1280, 960)
            # cv2.imshow("zzz", img_check)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            img_output = Image.open(output_dir + '%d.png' % (i+ self.start_idx))
            img_output = self.transform(img_output)
            self.img_output.append(img_output)

            # img_check = (img_output.numpy() * 255).astype(np.uint8)
            # img_check = np.transpose(img_check, [1, 2, 0])
            # cv2.namedWindow('zzz', 0)
            # cv2.resizeWindow('zzz', 1280, 960)
            # cv2.imshow("zzz", img_check)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

    def __len__(self):
        return self.num_img

    def __getitem__(self, idx):

        return self.img_input[idx], self.img_output[idx]

# VAE Definition
class Encoder(nn.Module):
    def __init__(self, conv_hiddens, latent_dim, img_length_width, prev_channels=3):
        super(Encoder, self).__init__()

        temp_linear_dim = 512
        self.img_length_width = img_length_width
        self.prev_channels = prev_channels

        modules = []
        for cur_channels in conv_hiddens:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(self.prev_channels,
                              cur_channels,
                              kernel_size=4,
                              stride=2,
                              padding=1), nn.BatchNorm2d(cur_channels),
                    nn.ReLU()))
            self.prev_channels = cur_channels
            self.img_length_width //= 2
        self.encoder = nn.Sequential(*modules)

        self.mean_linear = nn.Sequential(nn.Linear(self.prev_channels * self.img_length_width * self.img_length_width, temp_linear_dim),
                                         nn.Linear(temp_linear_dim, latent_dim))
        self.var_linear = nn.Sequential(nn.Linear(self.prev_channels * self.img_length_width * self.img_length_width, temp_linear_dim),
                                        nn.Linear(temp_linear_dim, latent_dim))

    def forward(self, x):

        x = self.encoder(x)
        x = torch.flatten(x, 1)
        mean = self.mean_linear(x)
        logvar = self.var_linear(x)

        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, conv_hiddens, latent_dim, img_length_width, prev_channels):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(latent_dim, prev_channels * img_length_width * img_length_width)
        self.decoder_input_chw = (prev_channels, img_length_width, img_length_width)

        modules = []
        conv_hiddens.reverse()
        conv_hiddens.append(3)
        for cur_channels in conv_hiddens[1:]:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(prev_channels,
                              cur_channels,
                              kernel_size=4,
                              stride=2,
                              padding=1), nn.BatchNorm2d(cur_channels),
                    nn.ReLU()))
            prev_channels = cur_channels
        self.decoder = nn.Sequential(*modules)

    def forward(self, x):
        # x = F.relu(self.fc(x))
        x = self.fc(x)
        x = x.view(-1, *self.decoder_input_chw)
        x = self.decoder(x)
        return x

class VAE(nn.Module):
    def __init__(self, conv_hiddens, latent_dim, img_length_width):
        super(VAE, self).__init__()

        self.encoder = Encoder(conv_hiddens, latent_dim, img_length_width)
        pre_channels = self.encoder.prev_channels
        img_length_width = self.encoder.img_length_width
        self.decoder = Decoder(conv_hiddens, latent_dim, img_length_width, pre_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD