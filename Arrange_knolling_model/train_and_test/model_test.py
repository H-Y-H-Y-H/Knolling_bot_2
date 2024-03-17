import os
import yaml
import numpy as np
from model_structure import *
from model_structure import *
from torch.utils.data import DataLoader, random_split

best_success_rate = 0
worst_success_rate = 100
best_index = 0
worst_index = 0
success_list = []


def test_model_batch(val_loader, model, log_path, selec_list, num_obj=10):

    model.to(device)
    model.eval()

    test_loss_list = []
    outputs = []
    ll_loss_list= []
    ms_min_sample_loss_list= []
    overlap_loss_list= []
    pos_loss_list= []
    v_entropy_loss_list= []

    with torch.no_grad():
        total_loss = 0

        for input_batch, target_batch in val_loader:
            input_batch = torch.from_numpy(np.asarray(input_batch, dtype=np.float32)).to(device)
            target_batch = torch.from_numpy(np.asarray(target_batch, dtype=np.float32)).to(device)
            # input_cls = torch.from_numpy(np.asarray(input_cls, dtype=np.float32)).to(device)
            input_batch = input_batch.transpose(1, 0)
            target_batch = target_batch.transpose(1, 0)


            # # zero to False
            # input_batch_atten_mask = (input_batch == 0).bool()
            # input_batch.masked_fill_(input_batch_atten_mask, MASK_VALUE)

            # zero to False
            target_batch_atten_mask = (target_batch == 0).bool()
            target_batch.masked_fill_(target_batch_atten_mask, MASK_VALUE)

            # create all MASK_VALUE input for decoder
            mask = torch.ones_like(target_batch, dtype=torch.bool)
            input_target_batch = torch.clone(target_batch)
            input_target_batch.masked_fill_(mask, MASK_VALUE)

            # Forward pass.
            if MIN_PRED:
                output_batch, pi, sigma, mu, ms_min_sample_loss = model.forward_min(input_batch,
                                                                                tart_x_gt=input_target_batch,
                                                                                gt_decoder=target_batch)


            else:
                output_batch, pi, sigma, mu = model.forward(input_batch,tart_x_gt=input_target_batch,given_idx = selec_list)

                ms_min_sample_loss, ms_id, output_batch_min = min_sample_loss(pi, sigma, mu,
                                                     target_batch[:model.in_obj_num],
                                                     Output_scaler=True,
                                                     contain_id_and_values = True)
            output_batch = output_batch[: model.in_obj_num]
            # Calculate log-likelihood loss
            ll_loss = model.mdn_loss_function(pi, sigma, mu, target_batch[:model.in_obj_num],
                                              Output_scaler=False)

            # Calculate collision loss
            overlap_loss = calculate_collision_loss(output_batch.transpose(0, 1),
                                                    input_batch[:model.in_obj_num].transpose(0, 1),
                                                    scale=False,
                                                    Output_scaler=False)
            # Calcluate position loss
            pos_loss = model.masked_MSE_loss(output_batch, target_batch[:model.in_obj_num],Output_scaler=False)
            # Calucluate Entropy loss:
            v_entropy_loss = entropy_loss(pi, Output_scaler=False)

            ll_loss_list.append(ll_loss.transpose(1, 0).detach().cpu().numpy())
            ms_min_sample_loss_list.append(ms_min_sample_loss.detach().cpu().numpy())
            overlap_loss_list.append(overlap_loss.detach().cpu().numpy())
            pos_loss_list.append(pos_loss.transpose(1, 0).detach().cpu().numpy())
            v_entropy_loss_list.append(v_entropy_loss.transpose(1, 0).detach().cpu().numpy())

            output_batch = output_batch.transpose(1, 0)
            target_batch = target_batch[:model.in_obj_num].transpose(1, 0)

            outputs.append(output_batch.detach().cpu().numpy())

            numpy_pred = (output_batch.detach().cpu().numpy() - config.SHIFT_DATA) / config.SCALE_DATA
            numpy_label = (target_batch.detach().cpu().numpy() - config.SHIFT_DATA) / config.SCALE_DATA

            numpy_loss = (numpy_pred-numpy_label)**2
            numpy_loss = numpy_loss.reshape(len(numpy_loss),-1)
            numpy_loss[:, num_obj*2:] = 0

            test_loss_list.append(numpy_loss)

    test_loss_list = np.concatenate(test_loss_list)
    outputs = np.concatenate(outputs)

    ll_loss_list= np.concatenate(ll_loss_list)
    # ms_min_sample_loss_list= np.concatenate(ms_min_sample_loss_list)
    overlap_loss_list= np.concatenate(overlap_loss_list)
    pos_loss_list= np.concatenate(pos_loss_list)
    v_entropy_loss_list= np.concatenate(v_entropy_loss_list)

    outputs = (outputs.reshape(-1, len(outputs[0]) * 2) - config.SHIFT_DATA) / config.SCALE_DATA
    np.savetxt(log_path + '/test_loss_list%d.csv' % num_obj, np.asarray(test_loss_list))
    np.savetxt(log_path + '/ll_loss%d.csv' % num_obj, ll_loss_list)
    np.savetxt(log_path + '/ms_min_sample_loss%d.csv' % num_obj, ms_min_sample_loss_list)
    np.savetxt(log_path + '/overlap_loss%d.csv' % num_obj, overlap_loss_list)
    np.savetxt(log_path + '/pos_loss%d.csv' % num_obj, pos_loss_list)
    np.savetxt(log_path + '/v_entropy_loss%d.csv' % num_obj, v_entropy_loss_list)

    return outputs, test_loss_list


if __name__ == '__main__':
    import wandb
    import argparse

    test_sweep_flag = False
    use_yaml = True

    # api = wandb.Api()
    # Project is specified by <entity/project-name>
    # runs = api.runs("knolling0205_2_overlap")

    name = 'daily-sweep-1'

    model_name = "best_model.pt"

    with open(f'data/{name}/config.yaml', 'r') as yaml_file:
        config_dict = yaml.safe_load(yaml_file)
    config = {k: v for k, v in config_dict.items() if not k.startswith('_')}

    config = argparse.Namespace(**config)
    MIN_PRED = False
    object_num = 10
    valid_lw_data = []
    valid_pos_data = []
    total_raw_data = []

    # load the test dataset
    file_num = 10
    test_num_scenario = 10

    solu_num = 1 #12
    info_per_object = 7
    SHIFT_DATASET_ID = 3 # color 3,4,5
    obj_name_list = []
    for s in range(SHIFT_DATASET_ID,solu_num+SHIFT_DATASET_ID):
        print('load data:', object_num)
        raw_data = np.loadtxt(DATAROOT + 'num_%d_after_%d.txt' % (file_num, s))[:,:object_num*info_per_object]

        # raw_data = np.loadtxt('test_batch.txt')
        raw_data = raw_data[:test_num_scenario]
        # np.savetxt('test_batch.txt',raw_data,fmt='%s')

        obj_name_data = np.loadtxt(DATAROOT + 'num_%d_after_name_%d.txt' % (file_num, s), dtype=str)[:,:object_num]
        obj_name_data = obj_name_data[:test_num_scenario]

        total_raw_data.append(raw_data)
        obj_name_list.append(obj_name_data)

        test_data = raw_data * config.SCALE_DATA + config.SHIFT_DATA
        valid_lw = []
        valid_pos = []

        for i in range(object_num):
            valid_lw.append(test_data[:, i * info_per_object + 2:i * info_per_object + 4])
            valid_pos.append(test_data[:, i * info_per_object:i * info_per_object + 2])

        valid_lw = np.asarray(valid_lw).transpose(1, 0, 2)
        valid_pos = np.asarray(valid_pos).transpose(1, 0, 2)

        valid_lw_data += list(valid_lw)
        valid_pos_data += list(valid_pos)

    test_input_padded = pad_sequences(valid_lw_data, max_seq_length=config.max_seq_length)
    test_label_padded = pad_sequences(valid_pos_data, max_seq_length=config.max_seq_length)

    test_dataset = CustomDataset(test_input_padded, test_label_padded)
    val_loader = DataLoader(test_dataset, batch_size=512, shuffle=False) # 不能用shuffle True，不然evaluate面积时对不上号


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
            in_obj_num=object_num,
            num_gaussians=config.num_gaussian
        )

    # Number of parameters: 87458
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    PATH = 'data/%s/%s' % (name, model_name)
    checkpoint = torch.load(PATH, map_location=device)
    model.load_state_dict(checkpoint)

    log_path = f'./results/{name}'
    os.makedirs(log_path, exist_ok=True)

    raw_data = np.concatenate(total_raw_data)
    obj_name_list = np.concatenate(obj_name_list)


    np.savetxt(log_path + '/num_%d_gt.txt' % object_num, raw_data)
    np.savetxt(log_path+'/obj_name_%d.txt' % object_num, obj_name_list,fmt="%s")
    n_solu = 20 #config.num_gaussian**object_num
    m = config.num_gaussian
    n = object_num
    import random
    for id_solutions in range(n_solu):
        # selec_list = to_base_4(id_solutions,object_num,n_gaussian=config.num_gaussian)
        selec_list = random.choices(range(m), k=n)
        outputs, loss_list = test_model_batch(val_loader, model, log_path, num_obj=object_num,selec_list=selec_list)

        for i in range(object_num):
            raw_data[:, i * info_per_object:i * info_per_object + 2] = outputs[:, i * 2:i * 2 + 2]

        np.savetxt(log_path + f'/num_{object_num}_pred_{id_solutions}.txt', raw_data)


