import numpy as np
import wandb
import argparse
from train_mlp import *

if __name__ == '__main__':

    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs("zzz_MLP_unstack")

    name = "MLP_902_4"
    # model_name = 'latest_model.pt'
    model_name = "best_model.pt"

    summary_list, config_list, name_list = [], [], []
    config = None
    for run in runs:
        if run.name == name:
            print("found: ", name)
            config = {k: v for k, v in run.config.items() if not k.startswith('_')}
    print(config)
    config = argparse.Namespace(**config)

    num_img = config.num_img
    valid_num = 1000
    ratio = config.ratio
    input_data_path = config.input_data_path
    output_data_path = config.output_data_path
    box_test, unstack_test = data_preprocess(input_data_path, output_data_path, num_img,
                                           ratio, use_scaler=config.use_scaler, test_model=True, valid_num=valid_num)

    _, box_test_padding = data_padding([], box_test)

    batch_size = config.batch_size
    test_dataset = Generate_Dataset(box_data=box_test, unstack_data=unstack_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    model = MLP(num_boxes=config.num_boxes, output_size=config.output_size,
                node_1=config.node_1, node_2=config.node_2, node_3=config.node_3, device=config.device)
    model.load_state_dict(torch.load(config.model_path + 'best_model.pt'))

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=para_dict['stepLR'], gamma=para_dict['gamma'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config.patience,
                                                           factor=config.factor)
    min_loss = np.inf

    all_train_loss = []
    all_valid_loss = []
    current_epoch = 0
    abort_learning = 0
    device = 'cuda:0'

    model.eval()
    with torch.no_grad():
        valid_loss = []
        # print('eval')
        for batch_id, (box_data, unstack_data) in enumerate(test_loader):
            # print(batch_id)
            box_data = box_data.to(device, dtype=torch.float32)
            unstack_data = unstack_data.to(device, dtype=torch.float32)

            out = model.forward(box_data)
            if config.use_mse == True:
                loss = model.maskedMSELoss(predict=out, target=unstack_data)
            else:
                loss = model.maskedCrossEntropyLoss(predict=out, target=unstack_data, boxes_data=box_data)
            valid_loss.append(loss.item())

        avg_valid_loss = np.mean(valid_loss)
        all_valid_loss.append(avg_valid_loss)

        print('Testing_Loss At Epoch ', np.around(avg_valid_loss, 6))
