import yaml
from datetime import datetime
import os
from model_structure import *
import wandb
import argparse



def main():
    # if sweep_train_flag == True:
    #     wandb.init(project='knolling_tuning')  # ,mode = 'disabled'
    # else:
    if sweep_train_flag:
        wandb.init(project=proj_name)
        running_name = wandb.run.name

        config = wandb.config

        # config.map_embed_d_dim = 32
        # config.num_attention_heads = 16
        # config.num_layers = 10
        # config.lr = 1e-3
        # config.num_gaussian = 32
        # config.SCALE_DATA = 100
        # config.SHIFT_DATA = 100
        # config.overlap_loss_factor = 10000
        # config.batch_size = 128

        config.forwardtype = 1
        config.dropout_prob = 0.0
        config.max_seq_length = max_seq_length
        config.inputouput_size = inputouput_size
        config.log_pth = 'data/%s/' % running_name
        config.all_zero_target = 0  # 1 tart_x = zero like, 0: tart_x = tart_x
        config.forward_expansion = 4
        config.pre_trained = False
        config.all_steps = False
        config.patience = 300
        loss_d_epoch = 200

        model_path = None
    else:
        pretrained_model = 'stilted-sweep-1'
        wandb.init(project=proj_name)
        running_name = wandb.run.name
        # Load the YAML file
        with open(f'data/{pretrained_model}/config.yaml', 'r') as yaml_file:
            config_dict = yaml.safe_load(yaml_file)

        model_path = f'data/{pretrained_model}/best_model.pt'


        config = {k: v for k, v in config_dict.items() if not k.startswith('_')}
        config = argparse.Namespace(**config)

        config.log_pth = f'data/{running_name}/'
        os.makedirs(config.log_pth,exist_ok=True)
        config.pre_trained = True
        config.inputouput_size = inputouput_size
        config.k_ll = 0.01
        config.k_op = 1
        config.k_pos= 0.2

        config.patience = 300
        loss_d_epoch = 200


        print(config)

    config.dataset_path = DATAROOT
    config.scheduler_factor = 0.1
    os.makedirs(config.log_pth, exist_ok=True)
    k_ll = config.k_ll
    k_op = config.k_op
    k_pos = config.k_pos
    # Assuming 'config' is your W&B configuration object
    try:
        config_dict = dict(config)  # Convert to dictionary if necessary
    except:
        config_dict = vars(config)
    # Save as YAML
    with open(config.log_pth + 'config.yaml', 'w') as yaml_file:
        yaml.dump(config_dict, yaml_file, default_flow_style=False)



    model = Knolling_Transformer(
        input_length=config.max_seq_length,
        input_size=2,
        map_embed_d_dim=config.map_embed_d_dim,
        num_layers=config.num_layers,
        forward_expansion=config.forward_expansion,
        heads=config.num_attention_heads,
        dropout=config.dropout_prob,
        all_zero_target=config.all_zero_target,
        forwardtype=config.forwardtype,
        all_steps=config.all_steps,
        in_obj_num=config.inputouput_size,
        num_gaussians=config.num_gaussian,
    )



    if config.pre_trained:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config.model_params = num_params


    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.scheduler_factor,
                                                           patience=loss_d_epoch, verbose=True)

    num_epochs = 100000
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
            # Calcluate position loss
            pos_loss = model.masked_MSE_loss(output_batch, target_batch)
            tloss = k_ll * ll_loss + ms_min_smaple_loss + k_op * overlap_loss + k_pos * pos_loss

            if epoch % 10 == 0 and print_flag:
                print('output', output_batch[:, 0].flatten())
                print('target', target_batch[:, 0].flatten())
                print('loss and overlap loss:', tloss.item(), ll_loss.item(), ms_min_smaple_loss.item(),overlap_loss.item(),pos_loss.item())

                print_flag = False

            tloss.backward()
            optimizer.step()
            train_loss += tloss.item()
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
                # Calcluate position loss
                pos_loss = model.masked_MSE_loss(output_batch, target_batch)
                vloss = k_ll * ll_loss + ms_min_smaple_loss + k_op * overlap_loss + k_pos * pos_loss
                if epoch % 10 == 0 and print_flag:
                    print('val_output', output_batch[:, 0].flatten())
                    print('val_target', target_batch[:, 0].flatten())
                    print('loss and overlap loss:', vloss.item(), ll_loss.item(), ms_min_smaple_loss.item(),
                          overlap_loss.item(), pos_loss.item())

                    print_flag = False

                total_loss += vloss.item()
                valid_overlap_loss += overlap_loss.item()
            avg_loss = total_loss / len(val_loader)
            valid_overlap_loss = valid_overlap_loss/len(val_loader)
            scheduler.step(vloss)

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

    # load dataset
    train_input_data = []
    train_output_data = []
    train_cls_data = []
    input_data = []
    output_data = []
    valid_input_data = []
    valid_output_data = []
    valid_cls_data = []

    DATA_CUT = 10000 #1 000 000 data

    SHIFT_DATASET_ID = 0
    policy_num = 1
    configuration_num = 1
    solu_num = int(policy_num * configuration_num)
    info_per_object = 7

    inputouput_size = 2

    # how many data used during training.
    max_seq_length = 10
    batch_size = 512

    for f in range(SHIFT_DATASET_ID, SHIFT_DATASET_ID+solu_num):
        dataset_path = DATAROOT + 'num_%d_after_%d.txt' % (max_seq_length, f)
        print('load data:', dataset_path)
        raw_data = np.loadtxt(dataset_path)[:DATA_CUT, :inputouput_size * info_per_object]
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
        for i in range(inputouput_size):
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

    train_input_padded = pad_sequences(train_input_data,  max_seq_length=max_seq_length)
    train_label_padded = pad_sequences(train_output_data, max_seq_length=max_seq_length)
    test_input_padded = pad_sequences(valid_input_data,   max_seq_length=max_seq_length)
    test_label_padded = pad_sequences(valid_output_data,  max_seq_length=max_seq_length)

    train_dataset = CustomDataset(train_input_padded, train_label_padded)
    test_dataset = CustomDataset(test_input_padded, test_label_padded)

    num_data = (len(train_dataset), len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    sweep_train_flag = False

    proj_name = "knolling0205_2_overlap"
    if sweep_train_flag:
        sweep_configuration = {
            "method": "random",
            "metric": {"goal": "minimize", "name": "min_loss"},
            "parameters": {
                # "mse_loss_factor": {"max": 2.0, "min": 0.1},
                # "overlap_loss_factor": {"max": 2.0, "min": 0.1},
                "lr": {"values": [1e-3]},
                "map_embed_d_dim": {"values": [32]},
                "num_attention_heads": {"values": [4]},
                "num_layers":{"values":[4]},
                # "batch_size":{"values":[512]},
                "SCALE_DATA": {"values": [100]},
                "SHIFT_DATA": {"values": [100]},
                "num_gaussian":{"values":[4]},
                "batch_size":{"values":[512]},
                "k_ll":{"values":[0.01]},
                "k_op":{'values':[1]},
                'k_pos':{'values':[0.2]}
            },
        }

        sweep_id = wandb.sweep(sweep=sweep_configuration, project=proj_name)

        wandb.agent(sweep_id, function=main, count=100)
    else:
        main()