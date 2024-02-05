import yaml
from datetime import datetime
import os
from model_structure import *
import wandb

DATAROOT = "../../../knolling_dataset/learning_data_0126_10/"


def main():
    # if sweep_train_flag == True:
    #     wandb.init(project='knolling_tuning')  # ,mode = 'disabled'
    # else:

    wandb.init(project=proj_name)
    config = wandb.config
    running_name = wandb.run.name

    if not sweep_train_flag:
        config.map_embed_d_dim = 32
        config.num_attention_heads = 4
        config.num_layers = 10
        config.lr = 1e-3
        config.num_gaussian = 3
        config.SCALE_DATA = 100
        config.SHIFT_DATA = 100

    SCALE_DATA = config.SCALE_DATA
    SHIFT_DATA = config.SHIFT_DATA
    config.forwardtype = 1
    config.dropout_prob = 0.0
    config.max_seq_length = 10
    config.inputouput_size = 10
    config.log_pth = 'data/%s/' % running_name
    config.pos_encoding_Flag = True
    config.all_zero_target = 0  # 1 tart_x = zero like, 0: tart_x = tart_x
    config.forward_expansion = 4
    config.pre_trained = False
    config.high_dim_encoder = True
    config.all_steps = False
    config.canvas_factor = None
    config.use_overlap_loss = False
    config.patience = 20
    config.batch_size = 128

    config.dataset_path = DATAROOT
    config.scheduler_factor = 0.1
    os.makedirs(config.log_pth, exist_ok=True)

    # import json
    #
    # # Save config to a JSON file
    # config_file_path = 'config.json'
    # with open(config_file_path, 'w') as json_file:
    #     json.dump(config, json_file)
    #
    # # Log the config file to wandb
    # wandb.init(project='your_project_name', config=config)
    # wandb.save(config_file_path)


    # if config.pre_trained:
    #     pre_name = 'devoted-terrain-29'
    #     if pre_name == 'devoted-terrain-29':
    #         with open('data/%s/config-%s.yaml' % (pre_name, pre_name)) as f:
    #             read_data = yaml.safe_load(f)
    #         config.all_zero_target = read_data['all_zero_target']
    #         # config.all_steps = read_data['all_steps']
    #         config.batch_size = read_data['batch_size']
    #         config.dropout_prob = read_data['dropout_prob']
    #         config.forward_expansion = read_data['forward_expansion']
    #         config.forwardtype = read_data['forwardtype']
    #         config.high_dim_encoder = read_data['high_dim_encoder']
    #         config.map_embed_d_dim = read_data['map_embed_d_dim']
    #         config.max_seq_length = read_data['max_seq_length']
    #         config.num_attention_heads = read_data['num_attention_heads']
    #         # config.model_params = read_data['model_params']
    #         # config.num_data = read_data['num_data']
    #         config.num_layers = read_data['num_layers']
    #         config.pos_encoding_Flag = read_data['pos_encoding_Flag']

    pre_name = 'devoted-terrain-29'

    model = Knolling_Transformer(
        input_length=config.inputouput_size,
        input_size=2,
        map_embed_d_dim=config.map_embed_d_dim,
        num_layers=config.num_layers,
        forward_expansion=config.forward_expansion,
        heads=config.num_attention_heads,
        dropout=config.dropout_prob,
        all_zero_target=config.all_zero_target,
        pos_encoding_Flag=config.pos_encoding_Flag,
        forwardtype=config.forwardtype,
        high_dim_encoder=config.high_dim_encoder,
        all_steps=config.all_steps,
        max_obj_num=config.max_seq_length,
        num_gaussians=config.num_gaussian,
        canvas_factor=config.canvas_factor,
        use_overlap_loss=config.use_overlap_loss,
        mse_loss_factor=None, # 1
        overlap_loss_factor=10000 # 1
    )



    if config.pre_trained:
        PATH = 'data/%s/best_model.pt' % pre_name
        checkpoint = torch.load(PATH, map_location=device)
        model.load_state_dict(checkpoint)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config.model_params = num_params

    # load dataset
    train_input_data = []
    train_output_data = []
    train_cls_data = []
    input_data = []
    output_data = []
    valid_input_data = []
    valid_output_data = []
    valid_cls_data = []

    config.DATA_CUT = 1000 #1 000 000 data

    SHIFT_DATASET_ID = 3
    policy_num = 1
    configuration_num = 1
    config.solu_num = int(policy_num * configuration_num)
    info_per_object = 7

    for f in range(SHIFT_DATASET_ID, SHIFT_DATASET_ID+config.solu_num):
        dataset_path = DATAROOT + 'num_%d_after_%d.txt' % (config.inputouput_size, f)
        print('load data:', dataset_path)
        raw_data = np.loadtxt(dataset_path)[:config.DATA_CUT, :config.max_seq_length * info_per_object]
        # num_list = np.random.choice(np.arange(4, 11), len(raw_data))
        # mask = ~(np.arange(config.max_seq_length * info_per_object) < (num_list * info_per_object)[:, None])
        raw_data = raw_data * SCALE_DATA + SHIFT_DATA
        # raw_data[mask] = 0 # this is the customized padding process

        train_data = raw_data[:int(len(raw_data) * 0.8)]
        test_data = raw_data[int(len(raw_data) * 0.8):]

        train_lw = []
        valid_lw = []
        train_pos = []
        valid_pos = []
        train_cls = []
        valid_cls = []
        for i in range(config.max_seq_length):
            train_lw.append(train_data[:, i * info_per_object + 2:i * info_per_object + 4])
            valid_lw.append(test_data[:, i * info_per_object + 2:i * info_per_object + 4])
            train_cls.append(train_data[:, [i * info_per_object + 5]])
            valid_cls.append(test_data[:, [i * info_per_object + 5]])
            train_pos.append(train_data[:, i * info_per_object:i * info_per_object + 2])
            valid_pos.append(test_data[:, i * info_per_object:i * info_per_object + 2])

        train_lw = np.asarray(train_lw).transpose(1, 0, 2)
        valid_lw = np.asarray(valid_lw).transpose(1, 0, 2)
        train_pos = np.asarray(train_pos).transpose(1, 0, 2)
        valid_pos = np.asarray(valid_pos).transpose(1, 0, 2)
        train_cls = np.asarray(train_cls).transpose(1, 0, 2)
        valid_cls = np.asarray(valid_cls).transpose(1, 0, 2)

        train_input_data += list(train_lw)
        train_output_data += list(train_pos)
        train_cls_data += list(train_cls)
        valid_input_data += list(valid_lw)
        valid_output_data += list(valid_pos)
        valid_cls_data += list(valid_cls)

    train_input_padded = pad_sequences(train_input_data,  max_seq_length=config.inputouput_size)
    train_label_padded = pad_sequences(train_output_data, max_seq_length=config.inputouput_size)
    test_input_padded = pad_sequences(valid_input_data,   max_seq_length=config.inputouput_size)
    test_label_padded = pad_sequences(valid_output_data,  max_seq_length=config.inputouput_size)

    train_dataset = CustomDataset(train_input_padded, train_label_padded)
    test_dataset = CustomDataset(test_input_padded, test_label_padded)

    config.num_data = (len(train_dataset), len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.scheduler_factor,
                                                           patience=config.patience, verbose=True)

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
            output_batch = model(input_batch,
                                 tart_x_gt=input_target_batch)

            # Calculate loss
            loss,overlap_loss = model.calculate_loss(output_batch, target_batch, input_batch)

            if epoch % 10 == 0 and print_flag:
                print('output', output_batch[:, 0].flatten())
                print('target', target_batch[:, 0].flatten())
                print('loss and overlap loss:',loss.item(),overlap_loss.item())
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
                output_batch = model(input_batch,
                                     # obj_num=object_num,
                                     tart_x_gt=input_target_batch)

                # Calculate loss
                v_loss,overlap_loss = model.calculate_loss(output_batch, target_batch, input_batch)

                if epoch % 10 == 0 and print_flag:
                    print('val_output', output_batch[:, 0].flatten())
                    print('val_target', target_batch[:, 0].flatten())
                    print('loss and overlap loss:', v_loss.item(), overlap_loss.item())

                    print_flag = False

                total_loss += v_loss.item()
                valid_overlap_loss += overlap_loss.item()
            avg_loss = total_loss / len(val_loader)
            valid_overlap_loss = valid_overlap_loss/len(val_loader)
            scheduler.step(avg_loss)

            valid_loss_list.append(avg_loss)
            valid_loss2_list.append(valid_overlap_loss)

            if avg_loss < min_loss:
                min_loss = avg_loss
                PATH = config.log_pth + '/best_model.pt'
                torch.save(model.state_dict(), PATH)
                abort_learning = 0
            if avg_loss>10e8:
                abort_learning=100
            else:
                abort_learning += 1

            if epoch % 100 == 0:
                torch.save(model.state_dict(), config.log_pth + '/latest_model.pt')

            print(f"{datetime.now()}Epoch {epoch},train_loss: {train_loss}, validation_loss: {avg_loss},"
                  f" no_improvements: {abort_learning}")

            wandb.log({"train_loss": train_loss,
                       "train_overlap_loss":train_loss_overlap,
                       "valid_loss": avg_loss,
                       "valid_overlap_loss": valid_overlap_loss,
                       "learning_rate": optimizer.param_groups[0]['lr'],
                       "scheduler_factor": config.scheduler_factor,
                       "min_loss": min_loss,
                       "norm_loss": min_loss/(config.SCALE_DATA**2)})
            if abort_learning > config.patience:
                print('abort training!')
                break

if __name__ == '__main__':

    sweep_train_flag = True
    proj_name = "knolling0204_10_overlap"
    if sweep_train_flag:
        sweep_configuration = {
            "method": "random",
            "metric": {"goal": "minimize", "name": "min_loss"},
            "parameters": {
                # "mse_loss_factor": {"max": 2.0, "min": 0.1},
                # "overlap_loss_factor": {"max": 2.0, "min": 0.1},
                "lr": {"values": [1e-3]},
                "map_embed_d_dim": {"values": [32,64,128]},
                "num_attention_heads": {"values": [4,8,16,32]},
                "num_layers":{"values":[2,4,8,16]},
                # "batch_size":{"values":[512]},
                "SCALE_DATA": {"values": [100]},
                "SHIFT_DATA": {"values": [100]},
                "num_gaussian":{"values":[8,16,32,64]}
            },
        }

        sweep_id = wandb.sweep(sweep=sweep_configuration, project=proj_name)

        wandb.agent(sweep_id, function=main, count=100)
    else:
        main()