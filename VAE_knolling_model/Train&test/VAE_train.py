import cv2
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from VAE_model import VAE, CustomImageDataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import wandb
import yaml
import os
import argparse

def main(epochs):

    config = {}
    config = argparse.Namespace(**config)
    config.pre_trained = False

    if wandb_flag == True:

        wandb.init(project=proj_name)
        running_name = wandb.run.name

        if config.pre_trained == True:
            # Load the YAML file
            pretrained_model = 'tough-sweep-1'
            with open(f'data/{pretrained_model}/config.yaml', 'r') as yaml_file:
                config_dict = yaml.safe_load(yaml_file)
            config = {k: v for k, v in config_dict.items() if not k.startswith('_')}
            config = argparse.Namespace(**config)

            pretrain_model_path = f'data/{pretrained_model}/best_model.pt'
    else:
        running_name = 'zzz_test'

    config.log_pth = f'data/{running_name}/'
    config.patience = 80
    config.dataset_path = dataset_path
    config.scheduler_factor = 0.1

    os.makedirs(config.log_pth, exist_ok=True)

    # Assuming 'config' is your W&B configuration object
    try:
        config_dict = dict(config)  # Convert to dictionary if necessary
    except:
        config_dict = vars(config)
    # Save as YAML
    with open(config.log_pth + 'config.yaml', 'w') as yaml_file:
        yaml.dump(config_dict, yaml_file, default_flow_style=False)
    print(config)

    # Mapping device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Use: ', device)

    # Model instantiation
    model = VAE(
        conv_hiddens=[16, 32, 64, 128], latent_dim=128, img_length_width=128
    ).to(device)
    if config.pre_trained:
        checkpoint = torch.load(pretrain_model_path, map_location=device)
        model.load_state_dict(checkpoint)

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # criterion = nn.BCELoss(reduction='sum')

    min_loss = np.inf
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (img_rdm, img_neat) in enumerate(train_loader):
            img_rdm = img_rdm.to(device)
            img_neat = img_neat.to(device)
            optimizer.zero_grad()
            img_recon, mu, log_var = model(img_rdm)

            # img_check = (img_recon.squeeze().cpu().detach().numpy().transpose([1, 2, 0]) * 255).astype(np.uint8)
            # cv2.namedWindow('zzz', 0)
            # cv2.resizeWindow('zzz', 1280, 960)
            # cv2.imshow("zzz", img_check)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            loss = model.loss_function(img_recon, img_neat, mu, log_var)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        train_loss /= len(train_loader.dataset)
        print(f'Epoch {epoch}, Train Loss: {train_loss}')

        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for batch_idx, (img_rdm, img_neat) in test_loader:
                img_rdm = img_rdm.to(device)
                img_neat = img_neat.to(device)
                img_recon, mu, log_var = model(img_rdm)
                loss = model.loss_function(img_recon, img_neat, mu, log_var)
                validation_loss += loss.item()

        validation_loss /= len(test_loader.dataset)
        print(f'Epoch {epoch}, Validation Loss: {validation_loss}')

        if validation_loss < min_loss:
            min_loss = validation_loss
            PATH = config.log_pth + '/best_model.pt'
            torch.save(model.state_dict(), PATH)
            abort_learning = 0
        if validation_loss > 10e8:
            abort_learning = 10000
        else:
            abort_learning += 1

        if epoch % 20 == 0:
            torch.save(model.state_dict(), config.log_pth + '/latest_model.pt')

        wandb.log({"train_loss": train_loss,
                   "valid_loss": validation_loss,
                   "learning_rate": optimizer.param_groups[0]['lr'],
                   })

        if abort_learning > config.patience:
            print('abort training!')
            break



if __name__ == "__main__":

    num_epochs = 10
    num_data = 100
    dataset_path = '../../../knolling_dataset/VAE_317_obj4/'
    wandb_flag = False
    proj_name = "VAE_knolling"

    train_input = []
    train_output = []
    test_input = []
    test_output = []
    num_train = int(num_data * 0.8)
    num_test = int(num_data - num_train)

    batch_size = 1
    transform = Compose([
                        ToTensor()  # Normalize the image
                        ])
    train_dataset = CustomImageDataset(input_dir=dataset_path + 'images_before/',
                                       output_dir=dataset_path + 'images_before/',
                                       num_img=num_train, num_total=num_data, start_idx=0,
                                       transform=transform)
    test_dataset = CustomImageDataset(input_dir=dataset_path + 'images_before/',
                                       output_dir=dataset_path + 'images_before/',
                                       num_img=num_test, num_total=num_data, start_idx=num_train,
                                        transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Training the VAE for a few epochs to see the initial behavior
    main(num_epochs)
