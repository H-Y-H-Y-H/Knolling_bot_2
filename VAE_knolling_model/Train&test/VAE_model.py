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
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1) # Output: 240x320
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # Output: 120x160
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # Output: 60x80
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # Output: 30x40
        self.fc1 = nn.Linear(256*15*20, 1024)
        self.fc21 = nn.Linear(1024, 256)  # mean vector
        self.fc22 = nn.Linear(1024, 256)  # log variance vector

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc21(x), self.fc22(x)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(256, 256*15*20)
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.view(-1, 256, 15, 20)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        return x

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

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