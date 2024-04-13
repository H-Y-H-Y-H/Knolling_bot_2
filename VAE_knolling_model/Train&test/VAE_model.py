import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.nn.functional as F

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

class VAE(nn.Module):
    """VAE for 64x64 face generation.

    The hidden dimensions can be tuned.
    """

    def __init__(self, conv_hiddens=[16, 32, 64, 128, 256], latent_dim=128, img_length_width=128) -> None:
        super().__init__()

        # encoder
        prev_channels = 3
        modules = []
        img_length = 128
        self.kl_weight = 1

        for cur_channels in conv_hiddens:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels,
                              cur_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1), nn.BatchNorm2d(cur_channels),
                    nn.ReLU()))
            prev_channels = cur_channels
            img_length //= 2
        self.encoder = nn.Sequential(*modules)
        self.mean_linear = nn.Linear(prev_channels * img_length * img_length,
                                     latent_dim)
        self.var_linear = nn.Linear(prev_channels * img_length * img_length,
                                    latent_dim)
        self.latent_dim = latent_dim
        # decoder
        modules = []
        self.decoder_projection = nn.Linear(
            latent_dim, prev_channels * img_length * img_length)
        self.decoder_input_chw = (prev_channels, img_length, img_length)
        for i in range(len(conv_hiddens) - 1, 0, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(conv_hiddens[i],
                                       conv_hiddens[i - 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(conv_hiddens[i - 1]), nn.ReLU()))
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(conv_hiddens[0],
                                   conv_hiddens[0],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(conv_hiddens[0]), nn.ReLU(),
                nn.Conv2d(conv_hiddens[0], 3, kernel_size=3, stride=1, padding=1),
                nn.ReLU()))
        self.decoder = nn.Sequential(*modules)

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = torch.flatten(encoded, 1)
        mean = self.mean_linear(encoded)
        logvar = self.var_linear(encoded)
        eps = torch.randn_like(logvar)
        std = torch.exp(logvar / 2)
        z = eps * std + mean
        x = self.decoder_projection(z)
        x = torch.reshape(x, (-1, *self.decoder_input_chw))
        decoded = self.decoder(x)

        return decoded, mean, logvar

    def sample(self, device='cuda'):
        z = torch.randn(1, self.latent_dim).to(device)
        x = self.decoder_projection(z)
        x = torch.reshape(x, (-1, *self.decoder_input_chw))
        decoded = self.decoder(x)
        return decoded

    def loss_function(self, y, y_hat, mean, logvar):
        recons_loss = F.mse_loss(y_hat, y)
        kl_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mean ** 2 - torch.exp(logvar), 1), 0)
        loss = recons_loss + kl_loss * self.kl_weight
        return loss
