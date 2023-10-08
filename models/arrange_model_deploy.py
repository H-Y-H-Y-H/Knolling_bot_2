from Arrange_knolling_model.transformer_network import *
import os
import wandb
import argparse
from utils import *
import yaml

class configuration_zzz():

    def __init__(self, xyz_list, all_index, transform_flag, knolling_para):

        self.xyz_list = xyz_list
        self.all_index = all_index
        self.transform_flag = transform_flag
        self.gap_item = knolling_para['gap_item']
        self.gap_block = knolling_para['gap_block']
        self.item_odd_prevent = knolling_para['item_odd_prevent']
        self.block_odd_prevent = knolling_para['block_odd_prevent']
        self.upper_left_max = knolling_para['upper_left_max']
        self.forced_rotate_box = knolling_para['forced_rotate_box']

    def calculate_items(self, item_num, item_xyz):

        min_xy = np.ones(2) * 100
        best_item_config = []
        best_item_sequence = []
        item_iteration = 100
        item_odd_flag = False
        all_item_x = 100
        all_item_y = 100

        for i in range(item_iteration):

            fac = []  # 定义一个列表存放因子
            for i in range(1, item_num + 1):
                if item_num % i == 0:
                    fac.append(i)
                    continue
            # fac = fac[::-1]

            if self.item_odd_prevent == True:
                if item_num % 2 != 0 and len(fac) == 2 and item_num >=5:  # its odd! we should generate the factor again!
                    item_num += 1
                    item_odd_flag = True
                    fac = []  # 定义一个列表存放因子
                    for i in range(1, item_num + 1):
                        if item_num % i == 0:
                            fac.append(i)
                            continue

            item_sequence = np.random.choice(len(item_xyz), len(item_xyz), replace=False)
            if item_odd_flag == True:
                item_sequence = np.append(item_sequence, item_sequence[-1])

            for j in range(len(fac)):
                # if item_num == 3:
                #     item_num_row = 1
                #     item_num_column = 3
                # else:
                item_num_row = int(fac[j])
                item_num_column = int(item_num / item_num_row)
                item_sequence = item_sequence.reshape(item_num_row, item_num_column)
                item_min_x = 0
                item_min_y = 0

                for r in range(item_num_row):
                    new_row = item_xyz[item_sequence[r, :]]
                    item_min_x = item_min_x + np.max(new_row, axis=0)[0]



                for c in range(item_num_column):
                    new_column = item_xyz[item_sequence[:, c]]
                    item_min_y = item_min_y + np.max(new_column, axis=0)[1]

                item_min_x = item_min_x + (item_num_row - 1) * self.gap_item
                item_min_y = item_min_y + (item_num_column - 1) * self.gap_item

                if item_min_x + item_min_y < all_item_x + all_item_y:
                    best_item_config = [item_num_row, item_num_column]
                    best_item_sequence = item_sequence
                    all_item_x = item_min_x
                    all_item_y = item_min_y
                    min_xy = np.array([all_item_x, all_item_y])

        return min_xy, best_item_config, item_odd_flag, best_item_sequence

    def calculate_block(self):  # first: calculate, second: reorder!

        min_result = []
        best_config = []
        item_odd_list = []

        ################## zzz add sequence ###################
        item_sequence_list = []
        ################## zzz add sequence ###################

        for i in range(len(self.all_index)):
            item_index = self.all_index[i]
            item_xyz = self.xyz_list[item_index, :]
            item_num = len(item_index)
            xy, config, odd, item_sequence = self.calculate_items(item_num, item_xyz)
            # print(f'this is min xy {xy}')
            min_result.append(list(xy))
            # print(f'this is the best item config\n {config}')
            best_config.append(list(config))
            item_odd_list.append(odd)
            item_sequence_list.append(item_sequence)
        min_result = np.asarray(min_result).reshape(-1, 2)
        best_config = np.asarray(best_config).reshape(-1, 2)
        item_odd_list = np.asarray(item_odd_list)
        # print('this is item sequence list', item_sequence_list)
        # item_sequence_list = np.asarray(item_sequence_list, dtype=object)

        # print(best_config)

        if self.upper_left_max == True:
            # reorder the block based on the min_xy 哪个block面积大哪个在前
            s_block_sequence = np.argsort(min_result[:, 0] * min_result[:, 1])[::-1]
            new_all_index = []
            for i in s_block_sequence:
                new_all_index.append(self.all_index[i])
            self.all_index = new_all_index.copy()
            min_result = min_result[s_block_sequence]
            best_config = best_config[s_block_sequence]
            item_odd_list = item_odd_list[s_block_sequence]
            item_sequence_list = [item_sequence_list[i] for i in s_block_sequence]
            # item_sequence_list = item_sequence_list[s_block_sequence]
            # reorder the block based on the min_xy 哪个block面积大哪个在前

        # 安排总的摆放
        iteration = 100
        all_num = best_config.shape[0]
        all_x = 100
        all_y = 100
        odd_flag = False

        fac = []  # 定义一个列表存放因子
        for i in range(1, all_num + 1):
            if all_num % i == 0:
                fac.append(i)
                continue
        # fac = fac[::-1]

        if self.block_odd_prevent == True:
            if all_num % 2 != 0 and len(fac) == 2:  # its odd! we should generate the factor again!
                all_num += 1
                odd_flag = True
                fac = []  # 定义一个列表存放因子
                for i in range(1, all_num + 1):
                    if all_num % i == 0:
                        fac.append(i)
                        continue

        for i in range(iteration):

            if self.upper_left_max == True:
                sequence = np.concatenate((np.array([0]), np.random.choice(best_config.shape[0] - 1, size=len(self.all_index) - 1, replace=False) + 1))
            else:
                sequence = np.random.choice(best_config.shape[0], size=len(self.all_index), replace=False)
            # sequence = np.arange(len(self.all_index))

            if odd_flag == True:
                sequence = np.append(sequence, sequence[-1])
            else:
                pass
            zero_or_90 = np.random.choice(np.array([0, 90]))

            for j in range(len(fac)):

                min_xy = np.copy(min_result)
                # print(f'this is the min_xy before rotation\n {min_xy}')

                num_row = int(fac[j])
                num_column = int(all_num / num_row)
                sequence = sequence.reshape(num_row, num_column)
                min_x = 0
                min_y = 0
                rotate_flag = np.full((num_row, num_column), False, dtype=bool)
                # print(f'this is {sequence}')

                # the zero or 90 should permanently be 0
                for r in range(num_row):
                    for c in range(num_column):
                        new_row = min_xy[sequence[r][c]]
                        if self.forced_rotate_box == True:
                            if new_row[0] > new_row[1]:
                                zero_or_90 = 90
                        else:
                            zero_or_90 = np.random.choice(np.array([0, 90]))
                        if zero_or_90 == 90:
                            rotate_flag[r][c] = True
                            temp = new_row[0]
                            new_row[0] = new_row[1]
                            new_row[1] = temp

                    # insert 'whether to rotate' here
                for r in range(num_row):
                    new_row = min_xy[sequence[r, :]]
                    min_x = min_x + np.max(new_row, axis=0)[0]

                for c in range(num_column):
                    new_column = min_xy[sequence[:, c]]
                    min_y = min_y + np.max(new_column, axis=0)[1]

                if min_x + min_y < all_x + all_y:
                    best_all_config = sequence
                    all_x = min_x
                    all_y = min_y
                    best_rotate_flag = rotate_flag
                    best_min_xy = np.copy(min_xy)

        # print(f'in iteration{i}, the min all_x and all_y are {all_x} {all_y}')
        # print('this is best all sequence', best_all_config)

        return self.reorder_block(best_config, best_all_config, best_rotate_flag, best_min_xy, odd_flag, item_odd_list, item_sequence_list)

    def reorder_item(self, best_config, start_pos, index_block, item_index, item_xyz, rotate_flag, item_odd_list, item_sequence):

        # initiate the pos and ori
        # we don't analysis these imported oris
        # we directly define the ori is 0 or 90 degree, depending on the algorithm.

        item_row = item_sequence.shape[0]
        item_column = item_sequence.shape[1]
        item_odd_flag = item_odd_list[index_block]
        if item_odd_flag == True:
            item_pos = np.zeros([len(item_index) + 1, 3])
            item_ori = np.zeros([len(item_index) + 1, 3])
        else:
            item_pos = np.zeros([len(item_index), 3])
            item_ori = np.zeros([len(item_index), 3])

        # the initial position of the first items

        if rotate_flag == True:

            temp = np.copy(item_xyz[:, 0])
            item_xyz[:, 0] = item_xyz[:, 1]
            item_xyz[:, 1] = temp
            # 如果用的乐高块，这里是pi / 2, 否则是0
            item_ori[:, 2] = 0
            # print(item_ori)
            temp = item_row
            item_row = item_column
            item_column = temp
            # index_temp = index_temp.transpose()
            item_sequence = item_sequence.transpose()
        else:
            item_ori[:, 2] = 0

        start_item_x = np.array([start_pos[0]])
        start_item_y = np.array([start_pos[1]])
        previous_start_item_x = start_item_x
        previous_start_item_y = start_item_y

        print('this is item_xyz', item_xyz)
        print('this is item_sequence', item_sequence)
        for m in range(item_row):
            new_row = item_xyz[item_sequence[m, :]]
            start_item_x = np.append(start_item_x,
                                     (previous_start_item_x + np.max(new_row, axis=0)[0] + self.gap_item))
            previous_start_item_x = (previous_start_item_x + np.max(new_row, axis=0)[0] + self.gap_item)
        start_item_x = np.delete(start_item_x, -1)

        for n in range(item_column):
            new_column = item_xyz[item_sequence[:, n]]
            start_item_y = np.append(start_item_y,
                                     (previous_start_item_y + np.max(new_column, axis=0)[1] + self.gap_item))
            previous_start_item_y = (previous_start_item_y + np.max(new_column, axis=0)[1] + self.gap_item)
        start_item_y = np.delete(start_item_y, -1)

        x_pos, y_pos = np.copy(start_pos)[0], np.copy(start_pos)[1]

        for j in range(item_row):
            for k in range(item_column):
                if item_odd_flag == True and j == item_row - 1 and k == item_column - 1:
                    break
                ################### check whether to transform for each item in each block!################
                if self.transform_flag[item_index[item_sequence[j][k]]] == 1:
                    # print(f'the index {item_index[index_temp[j][k]]} should be rotated because of transformation')
                    item_ori[item_sequence[j][k], 2] -= np.pi / 2
                ################### check whether to transform for each item in each block!################
                x_pos = start_item_x[j] + (item_xyz[item_sequence[j][k]][0]) / 2
                y_pos = start_item_y[k] + (item_xyz[item_sequence[j][k]][1]) / 2
                item_pos[item_sequence[j][k]][0] = x_pos
                item_pos[item_sequence[j][k]][1] = y_pos
        if item_odd_flag == True:
            item_pos = np.delete(item_pos, -1, axis=0)
            item_ori = np.delete(item_ori, -1, axis=0)
        else:
            pass
        # print('this is the shape of item pos', item_pos.shape)
        return item_ori, item_pos

    def reorder_block(self, best_config, best_all_config, best_rotate_flag, min_xy, odd_flag, item_odd_list, item_sequence_list):

        num_all_row = best_all_config.shape[0]
        num_all_column = best_all_config.shape[1]

        start_x = [0]
        start_y = [0]
        previous_start_x = 0
        previous_start_y = 0

        for m in range(num_all_row):
            new_row = min_xy[best_all_config[m, :]]
            start_x.append((previous_start_x + np.max(new_row, axis=0)[0] + self.gap_block))
            previous_start_x = (previous_start_x + np.max(new_row, axis=0)[0] + self.gap_block)
        start_x = np.delete(start_x, -1)

        for n in range(num_all_column):
            new_column = min_xy[best_all_config[:, n]]
            start_y.append((previous_start_y + np.max(new_column, axis=0)[1] + self.gap_block))
            previous_start_y = (previous_start_y + np.max(new_column, axis=0)[1] + self.gap_block)
        start_y = np.delete(start_y, -1)

        # determine the start position per item
        item_pos = np.zeros([len(self.xyz_list), 3])
        item_ori = np.zeros([len(self.xyz_list), 3])
        for m in range(num_all_row):
            for n in range(num_all_column):
                if odd_flag == True and m == num_all_row - 1 and n == num_all_column - 1:
                    break  # this is the redundancy block
                item_index = self.all_index[best_all_config[m][n]]  # determine the index of blocks
                item_xyz = self.xyz_list[item_index, :]
                start_pos = np.asarray([start_x[m], start_y[n]])
                index_block = best_all_config[m][n]
                item_sequence = item_sequence_list[index_block]
                rotate_flag = best_rotate_flag[m][n]

                ori, pos = self.reorder_item(best_config, start_pos, index_block, item_index, item_xyz, rotate_flag,
                                        item_odd_list, item_sequence)
                if rotate_flag == True:
                    temp = self.xyz_list[item_index, 0]
                    self.xyz_list[item_index, 0] = self.xyz_list[item_index, 1]
                    self.xyz_list[item_index, 1] = temp
                item_pos[item_index] = pos
                item_ori[item_index] = ori

        return item_pos, item_ori  # pos_list, ori_list

class Arrange_model():

    def __init__(self, para_dict, arrange_dict):

        self.para_dict = para_dict
        self.arrange_dict = arrange_dict
        self.shift_data = 50
        self.scale_data = 100

        if self.arrange_dict['use_yaml'] == True:
            with open(self.arrange_dict['transformer_model_path'] + '/config-' + self.arrange_dict['running_name'] + '.yaml', 'r') as file:
                read_data = yaml.safe_load(file)
                self.arrange_dict['max_seq_length'] = read_data['max_seq_length']
                self.arrange_dict['map_embed_d_dim'] = read_data['map_embed_d_dim']
                self.arrange_dict['num_layers'] = read_data['num_layers']
                self.arrange_dict['forward_expansion'] = read_data['forward_expansion']
                self.arrange_dict['num_attention_heads'] = read_data['num_attention_heads']
                self.arrange_dict['dropout_prob'] = read_data['dropout_prob']
                self.arrange_dict['all_zero_target'] = read_data['all_zero_target']
                self.arrange_dict['pos_encoding_Flag'] = read_data['pos_encoding_Flag']
                self.arrange_dict['forwardtype'] = read_data['forwardtype']
                self.arrange_dict['high_dim_encoder'] = read_data['high_dim_encoder']
                self.arrange_dict['all_steps'] = read_data['all_steps']

            self.model = Knolling_Transformer(input_length=self.arrange_dict['max_seq_length'],
                                              input_size=2,
                                              map_embed_d_dim=self.arrange_dict['map_embed_d_dim'],
                                              num_layers=self.arrange_dict['num_layers'],
                                              forward_expansion=self.arrange_dict['forward_expansion'],
                                              heads=self.arrange_dict['num_attention_heads'],
                                              dropout=self.arrange_dict['dropout_prob'],
                                              all_zero_target=self.arrange_dict['all_zero_target'],
                                              pos_encoding_Flag=self.arrange_dict['pos_encoding_Flag'],
                                              forwardtype=self.arrange_dict['forwardtype'],
                                              high_dim_encoder=self.arrange_dict['high_dim_encoder'],
                                              all_steps=self.arrange_dict['all_steps'],
                                              max_obj_num=5,
                                              num_gaussians=3)

            print("Number of parameters:", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
            PATH = './models/%s/%s' % (self.arrange_dict['running_name'], 'best_model.pt')
            checkpoint = torch.load(PATH, map_location=self.para_dict['device'])
            self.model.load_state_dict(checkpoint)
            pass

        else:
            api = wandb.Api()
            # Project is specified by <entity/project-name>
            runs = api.runs("knolling_multi")

            name = self.arrange_dict['running_name']
            model_name = "best_model.pt"
            summary_list, config_list, name_list = [], [], []
            config = None
            for run in runs:
                if run.name == name:
                    print("found: ", name)
                    config = {k: v for k, v in run.config.items() if not k.startswith('_')}
            print(config)
            config = argparse.Namespace(**config)

            self.model = Knolling_Transformer(input_length=config.max_seq_length,
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
                                              max_obj_num=5,
                                              num_gaussians=5)

            print("Number of parameters:", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
            PATH = './models/%s/%s' % (name, model_name)
            checkpoint = torch.load(PATH, map_location=self.para_dict['device'])
            self.model.load_state_dict(checkpoint)

    def pred(self, pos_before_input, ori_before_input, lwh_before_input, grasp_index):

        self.model.to(self.para_dict['device'])
        self.model.eval()
        with torch.no_grad():

            # lwh_before_input = np.array([[0.021, 0.025],
            #                              [0.017, 0.018],
            #                              [0.031, 0.036],
            #                              [0.031, 0.036],
            #                              [0.04, 0.021]])
            # pos_before_input = np.array([[0.0265, -0.1415],
            #                              [0.0705, -0.1450],
            #                              [0.0315, -0.0960],
            #                              [0.0775, -0.0960],
            #                              [0.0360, -0.0525]])
            # pos_before_grasp = pos_before_input.reshape(1, 5, 2) * self.scale_data + self.shift_data
            # lwh_before_grasp = lwh_before_input.reshape(1, 5, 2) * self.scale_data + self.shift_data
            crowded_index = np.setdiff1d(np.arange(len(pos_before_input)), grasp_index)
            new_index = np.concatenate((grasp_index, crowded_index))

            pos_before_grasp = pos_before_input[new_index, :2].reshape(1, len(pos_before_input), 2) * self.scale_data + self.shift_data
            lwh_before_grasp = lwh_before_input[new_index, :2].reshape(1, len(lwh_before_input), 2) * self.scale_data + self.shift_data

            pos_before_grasp = pad_sequences(pos_before_grasp, max_seq_length=self.para_dict['boxes_num'])
            lwh_before_grasp = pad_sequences(lwh_before_grasp, max_seq_length=self.para_dict['boxes_num'])

            input_data = torch.from_numpy(lwh_before_grasp[:, :, :2]).to(device).float()
            target_data = torch.from_numpy(pos_before_grasp[:, :, :2]).to(device).float()
            input_data = input_data.transpose(1, 0)
            target_data = target_data.transpose(1, 0)

            # zero to False
            target_data_atten_mask = (target_data == 0).bool()
            target_data.masked_fill_(target_data_atten_mask, -100)

            # create all -100 input for decoder
            mask = torch.ones_like(target_data, dtype=torch.bool)
            input_target_data = torch.clone(target_data)
            input_target_data.masked_fill_(mask, -100)

            # self.model.max_obj_num = len(grasp_index)
            # self.model.all_steps = False
            predictions = self.model(input_data, tart_x_gt=input_target_data, temperature=0)
            loss = self.model.maskedMSELoss(predictions, target_data)
            print('test_loss', loss)
            print('output', predictions[:, 0].flatten())
            print('target', target_data[:, 0].flatten())
            outputs = predictions.cpu().detach().numpy()

        outputs = (outputs.reshape(-1, len(outputs[0]) * 2) - self.shift_data) / self.scale_data
        pos_after = np.concatenate((outputs, pos_before_input[0, 2].repeat(len(outputs)).reshape(-1, 1)), axis=1)

        return pos_after

def test_model_batch(val_loader, model, log_path, num_obj=10):
    model.to(device)
    model.eval()

    test_loss_list = []
    outputs = []

    with torch.no_grad():
        total_loss = 0
        for input_batch, target_batch in val_loader:
            input_batch = torch.from_numpy(np.asarray(input_batch, dtype=np.float32)).to(device)
            target_batch = torch.from_numpy(np.asarray(target_batch, dtype=np.float32)).to(device)
            input_batch = input_batch.transpose(1, 0)
            target_batch = target_batch.transpose(1, 0)

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
            loss = model.maskedMSELoss(predictions, target_batch)

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

    # load dataset
    train_input_data = []
    train_output_data = []
    input_data = []
    output_data = []
    valid_input_data = []
    valid_output_data = []
    cfg = 0
    dataset_path = DATAROOT + '/labels_after_%d/' % cfg
    raw_data = 0

    for NUM_objects in range(5, 6):
        print('load data:', NUM_objects)
        raw_data = np.loadtxt(dataset_path + 'num_%d.txt' % NUM_objects)

        # raw_data = np.loadtxt(dataset_path + 'real_before/num_%d_d9.txt' % NUM_objects)
        # if len(raw_data[0]) != 50:
        #     raw_data = np.hstack((raw_data,np.zeros((len(raw_data),50-len(raw_data[0])))))
        # raw_data = raw_data[int(len(raw_data) * 0.8):int(len(raw_data) * 0.81)]

        raw_data = raw_data[int(len(raw_data) * 0.8):]
        test_data = raw_data * SCALE_DATA + SHIFT_DATA
        valid_input = []
        valid_label = []
        for i in range(NUM_objects):
            valid_input.append(test_data[:, i * 5 + 2:i * 5 + 4])
            valid_label.append(test_data[:, i * 5:i * 5 + 2])

        valid_input = np.asarray(valid_input).transpose(1, 0, 2)
        valid_label = np.asarray(valid_label).transpose(1, 0, 2)

        valid_input_data += list(valid_input)
        valid_output_data += list(valid_label)

    test_input_padded = pad_sequences(valid_input_data, max_seq_length=config.max_seq_length)
    test_label_padded = pad_sequences(valid_output_data, max_seq_length=config.max_seq_length)

    test_dataset = CustomDataset(test_input_padded, test_label_padded)
    val_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

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
        max_obj_num = 5,
        num_gaussians=5
    )

    # Number of parameters: 87458
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    PATH = 'data/%s/%s' % (name, model_name)
    checkpoint = torch.load(PATH, map_location=device)
    model.load_state_dict(checkpoint)

    log_path = 'results/%s/cfg_%d' % (name, cfg)
    os.makedirs(log_path, exist_ok=True)
    for NUM_objects in range(5,6):
        outputs, loss_list = test_model_batch(val_loader, model, log_path, num_obj=NUM_objects)
        for i in range(NUM_objects):
            raw_data[:, i * 5:i * 5 + 2] = outputs[:, i * 2:i * 2 + 2]
            raw_data[:, i * 5 + 4] = 0
        log_folder = 'results/%s/cfg_%d/pred_after/' % (name, cfg)
        os.makedirs(log_folder, exist_ok=True)
        print(log_folder)
        np.savetxt(log_folder + '/num_%d.txt' % NUM_objects, raw_data)