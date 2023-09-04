from Arrange_knolling_model.transformer_network import *
import os
import wandb
import argparse

class Arrange_model():

    def __init__(self, para_dict, arrange_dict):

        self.para_dict = para_dict
        self.arrange_dict = arrange_dict
        self.shift_data = 50
        self.scale_data = 100

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
        print('here')

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