from torch.utils.data import DataLoader
from VAE_model import VAE, CustomImageDataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, ToPILImage
import torch

def reconstruct(device, dataloader, model):
    model.eval()

    for img_rdm, img_neat in dataloader:
        img_rdm = img_rdm.to(device)
        output, mu, log_var = model(img_rdm)
        loss = model.loss_function(output, img_rdm, mu, log_var)
        output = output[0].detach().cpu()
        input = img_rdm[0].detach().cpu()
        combined = torch.cat((output, input), 1)
        img = ToPILImage()(combined)
        img.save(f'data/{running_name}/tmp.jpg')

    # batch = next(iter(dataloader))
    # x = batch[0:1, ...].to(device)
    # output = model(x)[0]
    # output = output[0].detach().cpu()
    # input = batch[0].detach().cpu()
    # combined = torch.cat((output, input), 1)
    # img = ToPILImage()(combined)
    # img.save(f'data/{running_name}/tmp.jpg')

def main():
    num_epochs = 100
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

    batch_size = 4
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

    device = 'cuda:0'

    model = torch.load(f'data/{running_name}/best_model.pt', 'cuda:0').to(device)

    reconstruct(device, dataloader=test_loader, model=model)

def show_structure():

    device = 'cuda:0'
    model = torch.load(f'data/{running_name}/best_model.pt', 'cuda:0').to(device)

    print(model)

if __name__ == '__main__':
    running_name = 'zzz_test'
    # main()
    show_structure()
