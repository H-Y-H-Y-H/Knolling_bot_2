import os
import yaml
import numpy as np

from new_model import *
from torch.utils.data import DataLoader, random_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_success_rate = 0
worst_success_rate = 100
best_index = 0
worst_index = 0
success_list = []

def evaluate_success_rate(raw_data, Num_objects, config_transformer, test_sweep_flag=True, count=1):

    global best_success_rate
    global worst_success_rate
    global best_index
    global worst_index
    global success_list
    mm2px = 530 / (0.34 * config_transformer.canvas_factor)
    padding_value = 1

    if DATAROOT == "../../../knolling_dataset/learning_data_1019_42w/":
        data = raw_data[:, :Num_objects * 6].reshape(-1, Num_objects, 6)
    if DATAROOT == "../../../knolling_dataset/learning_data_804/":
        data = raw_data[:, :Num_objects * 5].reshape(-1, Num_objects, 5)

    # if test_sweep_flag == True:
    #     data = raw_data[:, :Num_objects * 6].reshape(-1, Num_objects, 6)
    # else:
    #     data = raw_data[:, :Num_objects * 5].reshape(-1, Num_objects, 5)
    tar_lw = data[:, :, 2:4]
    pred_pos = data[:, :, :2]
    tar_lw_px = tar_lw * mm2px
    pred_pos_px = pred_pos * mm2px + 10

    data_pred = np.concatenate(((pred_pos_px[:, :, 0] - tar_lw_px[:, :, 0] / 2)[:, :, np.newaxis],
                                (pred_pos_px[:, :, 1] - tar_lw_px[:, :, 1] / 2)[:, :, np.newaxis],
                                (pred_pos_px[:, :, 0] + tar_lw_px[:, :, 0] / 2)[:, :, np.newaxis],
                                (pred_pos_px[:, :, 1] + tar_lw_px[:, :, 1] / 2)[:, :, np.newaxis],
                                ), axis=2).astype(np.int32)

    # Iterate through each rectangle and draw them on the canvas
    avg_overlap_num = []
    success_flag = []
    for i in range(data_pred.shape[0]):

        x_offset_px = np.min(data_pred[i, :, 0])
        y_offset_px = np.min(data_pred[i, :, 1])
        data_pred[i, :, [0, 2]] -= x_offset_px
        data_pred[i, :, [1, 3]] -= y_offset_px
        x_max_px = np.max(data_pred[i, :, 2])
        y_max_px = np.max(data_pred[i, :, 3])
        canvas_pred = np.zeros((x_max_px, y_max_px), dtype=np.uint8)

        for j in range(data_pred.shape[1]):
            corner_data_pred = data_pred[i, j]
            canvas_pred[corner_data_pred[0]:corner_data_pred[2],
            corner_data_pred[1]:corner_data_pred[3]] += padding_value

        overlap_num = np.clip(int(np.max(canvas_pred) / padding_value), 1, None)
        avg_overlap_num.append(overlap_num)
        if overlap_num > 1:
            # print(f'fail {i}')
            success_flag.append(0)
        else:
            success_flag.append(1)
    avg_overlap_num = np.asarray(avg_overlap_num)
    success_flag = np.asarray(success_flag)

    evaluate_file = log_path + '/evaluate_result.txt'
    success_rate = len(np.where(success_flag == 1)[0]) / len(success_flag)
    success_list.append(success_rate)

    if success_rate > best_success_rate:
        best_success_rate = success_rate
        best_index = count
        print('best success rate is:', best_success_rate)
    if success_rate < worst_success_rate:
        worst_index = count
        worst_success_rate = success_rate
        print('worst success rate is:', worst_success_rate)
    with open(evaluate_file, "w") as f:
        f.write(f'test_num {test_num_scenario}\n')
        f.write(f'success rate: {success_rate}\n')
        for i in range(data_pred.shape[0]):
            f.write(f'index: {i}, overlap_num: {avg_overlap_num[i]:.4f}, success: {success_flag[i]}\n')

    if test_sweep_flag == True:
        summary_file = summary_path + '/summary.txt'
        with open(summary_file, "a") as f:
            f.write(f'name: {name}, success rate: {success_rate}\n')
        if count == 19:
            success_list = np.asarray(success_list)
            with open(summary_file, "a") as f:
                f.write(f'--------------------- summary ----------------------\n')
                f.write(f'worst success rate: {np.min(success_list)}, index: {np.argmin(success_list) + 1}\n')
                f.write(f'best success rate: {np.max(success_list)}, index: {np.argmax(success_list) + 1}\n')
                f.write(f'average success rate: {np.mean(success_list)}\n')

def test_model_batch(val_loader, model, log_path, num_obj=10):

    model.to(device)
    model.eval()

    test_loss_list = []
    outputs = []

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
            # input_batch.masked_fill_(input_batch_atten_mask, -100)

            # zero to False
            target_batch_atten_mask = (target_batch == 0).bool()
            target_batch.masked_fill_(target_batch_atten_mask, -100)

            # create all -100 input for decoder
            mask = torch.ones_like(target_batch, dtype=torch.bool)
            input_target_batch = torch.clone(target_batch)
            input_target_batch.masked_fill_(mask, -100)

            # Forward pass
            predictions = model(input_batch, tart_x_gt=input_target_batch, temperature=0)

            target_batch[num_obj:] = -100
            loss = model.calculate_loss(predictions, target_batch, input_batch, input_cls)
            # target_batch_demo = target_batch.cpu().detach().numpy().reshape(5, 2)
            # predictions_demo = predictions.cpu().detach().numpy().reshape(5, 2)
            # input_demo = input_batch.cpu().detach().numpy().reshape(5, 2)

            print('output', predictions[:, 0].flatten())
            print('target', target_batch[:, 0].flatten())
            total_loss += loss.item()

            print('test_loss', loss)

            predictions = predictions.transpose(1, 0)
            target_batch = target_batch.transpose(1, 0)

            outputs.append(predictions.detach().cpu().numpy())

            numpy_pred = (predictions.detach().cpu().numpy() - SHIFT_DATA) / SCALE_DATA
            numpy_label = (target_batch.detach().cpu().numpy() - SHIFT_DATA) / SCALE_DATA

            numpy_loss = (numpy_pred-numpy_label)**2
            numpy_loss = numpy_loss.reshape(len(numpy_loss),-1)
            numpy_loss[:, num_obj*2:] = 0

            # print('numpy_loss',numpy_loss)
            test_loss_list.append(numpy_loss)

    test_loss_list = np.concatenate(test_loss_list)
    outputs = np.concatenate(outputs)
    outputs = (outputs.reshape(-1, len(outputs[0]) * 2) - SHIFT_DATA) / SCALE_DATA
    np.savetxt(log_path + '/test_loss_list_num_%d.csv' % num_obj, np.asarray(test_loss_list))
    np.savetxt(log_path + '/outputs.csv', outputs)

    return outputs, test_loss_list


if __name__ == '__main__':
    import wandb
    import argparse

    test_sweep_flag = False
    use_yaml = False
    api = wandb.Api()
    # Project is specified by <entity/project-name>

    DATAROOT = "../../../knolling_dataset/learning_data_0126/"

    if test_sweep_flag == False:
        runs = api.runs("knolling_sundry")
        # name = 'charmed-sweep-1'
        name = 'rosy-morning-8'
        # name = 'floral-bush-179'
    else:
        sweep = api.sweep('knolling_tuning/qtgswbjw')
        sweep_name = 'sweep_1204'
        runs = sweep.runs
        name = 'charmed-sweep-1'

    model_name = "best_model.pt"

    if use_yaml == False:
        summary_list, config_list, name_list = [], [], []
        config = None
        for run in runs:
            if run.name == name:
                print("found: ", name)
                config = {k: v for k, v in run.config.items() if not k.startswith('_')}
        print('using model: ', name)
        print(config)
    else:
        name = 'devoted-terrain-29'
        config = {}
        with open('data/' + name + '/config-' + name + '.yaml', 'r') as file:
            read_data = yaml.safe_load(file)
            config['max_seq_length'] = read_data['max_seq_length']
            config['map_embed_d_dim'] = read_data['map_embed_d_dim']
            config['num_layers'] = read_data['num_layers']
            config['forward_expansion'] = read_data['forward_expansion']
            config['num_attention_heads'] = read_data['num_attention_heads']
            config['dropout_prob'] = read_data['dropout_prob']
            config['all_zero_target'] = read_data['all_zero_target']
            config['pos_encoding_Flag'] = read_data['pos_encoding_Flag']
            config['forwardtype'] = read_data['forwardtype']
            config['high_dim_encoder'] = read_data['high_dim_encoder']
            config['all_steps'] = read_data['all_steps']
            config['num_gaussian'] = 3
            config['canvas_factor'] = 1
            config['use_overlap_loss'] = False
        print('using model: ', name)
        print(config)

    config = argparse.Namespace(**config)

    # load dataset
    train_input_data = []
    train_output_data = []
    input_data = []
    output_data = []
    valid_lw_data = []
    valid_pos_data = []
    valid_cls_data = []
    total_raw_data = []


    # load the test dataset
    file_num = 10
    test_num_scenario = 1000
    NUM_objects = config.max_seq_length
    solu_num = 12
    info_per_object = 7
    for s in range(solu_num):
        print('load data:', NUM_objects)

        if DATAROOT == "../../../knolling_dataset/learning_data_0126/":
            raw_data = np.loadtxt(DATAROOT + 'num_%d_after_%d.txt' % (file_num, s))

            raw_data = raw_data[int(len(raw_data) * 0.8):int(len(raw_data) * 0.8) + test_num_scenario]
            total_raw_data = np.append(total_raw_data, raw_data)
            test_data = raw_data * SCALE_DATA + SHIFT_DATA
            valid_lw = []
            valid_pos = []
            valid_cls = []
            for i in range(NUM_objects):
                valid_lw.append(test_data[:, i * info_per_object + 2:i * info_per_object + 4])
                valid_pos.append(test_data[:, i * info_per_object:i * info_per_object + 2])
                valid_cls.append(test_data[:, [i * info_per_object + 5]])

            valid_lw = np.asarray(valid_lw).transpose(1, 0, 2)
            valid_pos = np.asarray(valid_pos).transpose(1, 0, 2)
            valid_cls = np.asarray(valid_cls).transpose(1, 0, 2)

            valid_lw_data += list(valid_lw)
            valid_pos_data += list(valid_pos)
            valid_cls_data += list(valid_cls)
        # else:
        if DATAROOT == "../../../knolling_dataset/learning_data_1019_42w/":
            raw_data = np.loadtxt(DATAROOT + 'num_%d_after_%d.txt' % (file_num, s))

            raw_data = raw_data[int(len(raw_data) * 0.8):int(len(raw_data) * 0.8) + test_num_scenario]
            total_raw_data = np.append(total_raw_data, raw_data)
            test_data = raw_data * SCALE_DATA + SHIFT_DATA
            valid_lw = []
            valid_pos = []
            valid_cls = []
            for i in range(NUM_objects):
                valid_lw.append(test_data[:, i * 6 + 2:i * 6 + 4])
                valid_pos.append(test_data[:, i * 6:i * 6 + 2])
                valid_cls.append(test_data[:, [i * 6 + 5]])

            valid_lw = np.asarray(valid_lw).transpose(1, 0, 2)
            valid_pos = np.asarray(valid_pos).transpose(1, 0, 2)
            valid_cls = np.asarray(valid_cls).transpose(1, 0, 2)

            valid_lw_data += list(valid_lw)
            valid_pos_data += list(valid_pos)
            valid_cls_data += list(valid_cls)

    test_input_padded = pad_sequences(valid_lw_data, max_seq_length=config.max_seq_length)
    test_label_padded = pad_sequences(valid_pos_data, max_seq_length=config.max_seq_length)
    test_cls_padded = pad_sequences(valid_cls_data, max_seq_length=config.max_seq_length)

    test_dataset = CustomDataset(test_input_padded, test_label_padded, test_cls_padded)
    val_loader = DataLoader(test_dataset, batch_size=512, shuffle=False) # 不能用shuffle True，不然evaluate面积时对不上号

    if test_sweep_flag == False:

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
                all_steps=config.all_steps,
                max_obj_num=config.max_seq_length,
                num_gaussians=config.num_gaussian,
                canvas_factor=config.canvas_factor,
                use_overlap_loss=config.use_overlap_loss
            )

        # Number of parameters: 87458
        print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
        PATH = 'data/%s/%s' % (name, model_name)
        checkpoint = torch.load(PATH, map_location=device)
        model.load_state_dict(checkpoint)

        log_path = './results/%s' % (name)
        os.makedirs(log_path, exist_ok=True)

        raw_data = total_raw_data.reshape(test_num_scenario * solu_num, -1)
        outputs, loss_list = test_model_batch(val_loader, model, log_path, num_obj=NUM_objects)

        if DATAROOT == "../../../knolling_dataset/learning_data_1019_42w/":
            for i in range(NUM_objects):
                raw_data[:, i * 6:i * 6 + 2] = outputs[:, i * 2:i * 2 + 2]
                raw_data[:, i * 6 + 6] = 0
        if DATAROOT == "../../../knolling_dataset/learning_data_0126/":
            for i in range(NUM_objects):
                raw_data[:, i * info_per_object:i * info_per_object + 2] = outputs[:, i * 2:i * 2 + 2]

        # evaluate_success_rate(raw_data, NUM_objects, config, test_sweep_flag=test_sweep_flag)
        log_folder = './results/%s/pred_after' % (name)
        os.makedirs(log_folder, exist_ok=True)
        print(log_folder)
        np.savetxt(log_folder + '/num_%d_new.txt' % file_num, raw_data)

    else:
        # test the success rate in every sweep!
        block_list = ['breezy-sweep-22', 'dulcet-sweep-21', 'super-sweep-20']
        summary_path = './results/%s' % (sweep_name)

        with open(summary_path + '/summary.txt', "w") as f:
            f.truncate(0)

        count = 0
        for run in runs:
            name = run.name
            if name in block_list:
                continue
            else:
                count += 1
            config = {k: v for k, v in run.config.items() if not k.startswith('_')}
            print(config)
            config = argparse.Namespace(**config)

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
                num_gaussians=config.num_gaussian,
                canvas_factor=config.canvas_factor,
                use_overlap_loss=config.use_overlap_loss,
                mse_loss_factor=config.mse_loss_factor,
                overlap_loss_factor=config.overlap_loss_factor
            )

            # Number of parameters: 87458
            print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
            PATH = 'data/%s/%s/%s' % (sweep_name, name, model_name)
            checkpoint = torch.load(PATH, map_location=device)
            model.load_state_dict(checkpoint)

            log_path = './results/%s/%s' % (sweep_name, name)
            os.makedirs(log_path, exist_ok=True)
            # for NUM_objects in range(config.max_seq_length, config.max_seq_length + 1):
            raw_data = total_raw_data.reshape(test_num_scenario * solu_num, -1)
            outputs, loss_list = test_model_batch(val_loader, model, log_path, num_obj=NUM_objects)
            for i in range(NUM_objects):
                raw_data[:, i * 6:i * 6 + 2] = outputs[:, i * 2:i * 2 + 2]
                raw_data[:, i * 6 + 6] = 0
            evaluate_success_rate(raw_data, NUM_objects, config, count=count)
            log_folder = './results/%s/pred_after' % (name)
            os.makedirs(log_folder, exist_ok=True)
            print(log_folder)
            np.savetxt(log_folder + '/num_%d_new.txt' % file_num, raw_data)
