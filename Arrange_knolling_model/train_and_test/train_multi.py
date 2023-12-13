from datetime import datetime
import os
from new_model import *
import wandb

if __name__ == '__main__':

    def main():
        if sweep_train_flag == True:
            wandb.init(project='knolling_tuning')  # ,mode = 'disabled'
        else:
            wandb.init(project='knolling_multi')
            DATAROOT = "../../../knolling_dataset/learning_data_826/"
        config = wandb.config
        running_name = wandb.run.name

        config.forwardtype = 1
        config.map_embed_d_dim = 32
        config.num_attention_heads = 4
        config.num_layers = 4
        config.dropout_prob = 0.0
        config.max_seq_length = 5
        if sweep_train_flag == False:
            config.lr = 1e-4
        config.batch_size = 512
        config.log_pth = 'data/%s/' % running_name
        config.pos_encoding_Flag = True
        config.all_zero_target = 0  # 1 tart_x = zero like, 0: tart_x = tart_x
        config.forward_expansion = 4
        config.pre_trained = False
        config.high_dim_encoder = True
        config.all_steps = False
        config.object_num = -1
        config.canvas_factor = 2
        config.use_overlap_loss = False
        config.patience = 30
        config.num_gaussian = 4
        config.dataset_path = DATAROOT
        os.makedirs(config.log_pth, exist_ok=True)

        model = Knolling_Transformer(
            input_length=config.max_seq_length,
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
            all_steps = config.all_steps,
            max_obj_num = config.max_seq_length,
            num_gaussians = config.num_gaussian,
            canvas_factor=config.canvas_factor,
            use_overlap_loss=config.use_overlap_loss,
            mse_loss_factor=1,
            overlap_loss_factor=1
        )

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        config.model_params = num_params

        if config.pre_trained:
            pre_name = 'fallen-morning-156'
            PATH = 'data/%s/best_model.pt' % pre_name
            checkpoint = torch.load(PATH, map_location=device)
            model.load_state_dict(checkpoint)

        # load dataset
        train_input_data = []
        train_output_data = []
        train_cls_data = []
        input_data = []
        output_data = []
        valid_input_data = []
        valid_output_data = []
        valid_cls_data = []

        DATA_CUT = 50000

        solution_num = 4
        configuration_num = 1
        config.solu_num = int(solution_num * configuration_num)
        object_num = 30
        for f in range(config.solu_num):
            dataset_path = DATAROOT + 'num_%d_after_%d.txt' % (5, f)
            print('load data:', dataset_path)
            raw_data = np.loadtxt(dataset_path)[:DATA_CUT, :config.max_seq_length * 5]
            raw_data = raw_data * SCALE_DATA + SHIFT_DATA

            train_data = raw_data[:int(len(raw_data) * 0.8)]
            test_data = raw_data[int(len(raw_data) * 0.8):]

            train_input = []
            valid_input = []
            train_label = []
            valid_label = []
            train_cls = []
            valid_cls = []
            for i in range(config.max_seq_length):
                train_input.append(train_data[:, i * 5 + 2:i * 5 + 4])
                valid_input.append(test_data[:, i * 5 + 2:i * 5 + 4])
                train_cls.append(train_data[:, [i * 5 + 0]])
                valid_cls.append(test_data[:, [i * 5 + 0]])
                train_label.append(train_data[:, i * 5:i * 5 + 2])
                valid_label.append(test_data [:, i * 5:i * 5 + 2])

            train_input = np.asarray(train_input).transpose(1, 0, 2)
            valid_input = np.asarray(valid_input).transpose(1, 0, 2)
            train_label = np.asarray(train_label).transpose(1, 0, 2)
            valid_label = np.asarray(valid_label).transpose(1, 0, 2)
            train_cls = np.asarray(train_cls).transpose(1, 0, 2)
            valid_cls = np.asarray(valid_cls).transpose(1, 0, 2)

            train_input_data += list(train_input)
            train_output_data += list(train_label)
            train_cls_data += list(train_cls)
            valid_input_data += list(valid_input)
            valid_output_data += list(valid_label)
            valid_cls_data += list(valid_cls)

        train_input_padded = pad_sequences(train_input_data, max_seq_length=config.max_seq_length)
        train_label_padded = pad_sequences(train_output_data, max_seq_length=config.max_seq_length)
        train_cls_padded = pad_sequences(train_cls_data, max_seq_length=config.max_seq_length)
        test_input_padded = pad_sequences(valid_input_data, max_seq_length=config.max_seq_length)
        test_label_padded = pad_sequences(valid_output_data, max_seq_length=config.max_seq_length)
        test_cls_padded = pad_sequences(valid_cls_data, max_seq_length=config.max_seq_length)

        train_dataset = CustomDataset(train_input_padded, train_label_padded, train_cls_padded)
        test_dataset = CustomDataset(test_input_padded, test_label_padded, test_cls_padded)

        config.num_data = (len(train_dataset), len(test_dataset))

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=config.patience, verbose=True)

        num_epochs = 1000
        train_loss_list = []
        valid_loss_list = []
        model.to(device)
        abort_learning = 0
        config.max_loss_patience = 30
        min_loss = np.inf
        for epoch in range(num_epochs):
            print_flag = True
            model.train()
            train_loss = 0

            for input_batch, target_batch, input_cls in train_loader:
                optimizer.zero_grad()

                input_batch = torch.from_numpy(np.asarray(input_batch, dtype=np.float32)).to(device)
                target_batch = torch.from_numpy(np.asarray(target_batch, dtype=np.float32)).to(device)
                input_cls = torch.from_numpy(np.asarray(input_cls, dtype=np.float32)).to(device)
                input_batch = input_batch.transpose(1, 0)
                target_batch = target_batch.transpose(1, 0)
                input_cls = input_cls.transpose(1, 0)

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
                                     # obj_num=object_num,
                                     tart_x_gt=input_target_batch)

                # Calculate loss
                loss = model.calculate_loss(output_batch, target_batch, input_batch, input_cls)

                if epoch % 10 == 0 and print_flag:
                    print('output', output_batch[:, 0].flatten())
                    print('target', target_batch[:, 0].flatten())
                    print_flag = False

                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss = train_loss / len(train_loader)
            train_loss_list.append(train_loss)
            # Validate
            model.eval()
            print_flag = True
            with torch.no_grad():
                total_loss = 0
                for input_batch, target_batch, input_cls in val_loader:
                    input_batch = torch.from_numpy(np.asarray(input_batch, dtype=np.float32)).to(device)
                    target_batch = torch.from_numpy(np.asarray(target_batch, dtype=np.float32)).to(device)
                    input_cls = torch.from_numpy(np.asarray(input_cls, dtype=np.float32)).to(device)
                    input_batch = input_batch.transpose(1, 0)
                    target_batch = target_batch.transpose(1, 0)
                    input_cls = input_cls.transpose(1, 0)

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
                    loss = model.calculate_loss(output_batch, target_batch, input_batch, input_cls)

                    if epoch % 10 == 0 and print_flag:
                        print('val_output', output_batch[:, 0].flatten())
                        print('val_target', target_batch[:, 0].flatten())
                        print_flag = False

                    total_loss += loss.item()
                avg_loss = total_loss / len(val_loader)
                scheduler.step(avg_loss)

                # train_loss_list.append(avg_loss)
                valid_loss_list.append(avg_loss)

                if avg_loss < min_loss:
                    min_loss = avg_loss
                    PATH = config.log_pth + '/best_model.pt'
                    torch.save(model.state_dict(), PATH)
                    abort_learning = 0
                else:
                    abort_learning += 1

                if epoch % 100 == 0:
                    torch.save(model.state_dict(), config.log_pth + '/latest_model.pt')

                print(f"{datetime.now()}Epoch {epoch},train_loss: {train_loss}, validation_loss: {avg_loss},"
                      f" no_improvements: {abort_learning}")

                wandb.log({"train_loss": train_loss,
                           "valid_loss": avg_loss,
                           "learning_rate": optimizer.param_groups[0]['lr'],
                           "scheduler_factor": 0.5,
                           "min_loss": min_loss})
                if abort_learning > config.max_loss_patience:
                    print('abort training!')
                    break

    sweep_train_flag = False

    if sweep_train_flag == True:
        sweep_configuration = {
            "method": "random",
            "metric": {"goal": "minimize", "name": "min_loss"},
            "parameters": {
                "mse_loss_factor": {"max": 2.0, "min": 0.1},
                "overlap_loss_factor": {"max": 2.0, "min": 0.1},
                "lr": {"values": [1e-3, 1e-4]}
            },
        }

        sweep_id = wandb.sweep(sweep=sweep_configuration, project="knolling_tuning")

        wandb.agent(sweep_id, function=main, count=3)
    else:
        main()