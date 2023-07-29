import numpy as np
from Grasp_pred_model.network import LSTMRegressor
import torch.nn as nn
import torch

class Grasp_model():

    def __init__(self, para_dict, lstm_dict):

        self.para_dict = para_dict
        self.lstm_dict = lstm_dict
        self.model = LSTMRegressor(input_dim=self.lstm_dict['input_size'], hidden_dim=self.lstm_dict['hidden_size'], output_dim=self.lstm_dict['output_size'],
                                  num_layers=self.lstm_dict['num_layers'], hidden_node_1=self.lstm_dict['hidden_node_1'],
                                  hidden_node_2=self.lstm_dict['hidden_node_2'],
                                  batch_size=self.lstm_dict['batch_size'], device=self.lstm_dict['device'], criterion=nn.CrossEntropyLoss(),
                                  set_dropout=self.lstm_dict['set_dropout'])
        self.model.load_state_dict(torch.load(self.lstm_dict['grasp_model_path']))

    def pred(self, manipulator_before, lwh_list, conf_list):

        knolling_flag = False
        num_item = len(conf_list)
        input_data = np.concatenate((manipulator_before[:, :2],
                                     lwh_list[:, :2],
                                     manipulator_before[:, -1].reshape(-1, 1),
                                     conf_list.reshape(-1, 1)), axis=1)

        with torch.no_grad():
            # print('eval')
            box_data_batch = box_data_batch.to(device, dtype=torch.float32)
            grasp_data_batch = grasp_data_batch.to(device, dtype=torch.float32)

            box_data_pack = pack_padded_sequence(box_data_batch, num_box, batch_first=True)
            out = model.forward(box_data_pack)
            if para_dict['use_mse'] == True:
                loss = model.maskedMSELoss(predict=out, target=grasp_data_batch)
                valid_loss.append(loss.item())
            else:
                # loss = model.maskedCrossEntropyLoss(predict=out, target=grasp_data_batch)
                loss = model.maskedCrossEntropyLoss(predict=out, target=grasp_data_batch, boxes_data=box_data_batch)
                valid_loss.append(loss.item())
                not_one_result_per_img, tar_true, tar_false, TP, TN, FP, FN, \
                yolo_dominated_TP, yolo_dominated_TN, yolo_dominated_FP, yolo_dominated_FN, \
                grasp_dominated_TP, grasp_dominated_TN, grasp_dominated_FP, grasp_dominated_FN = model.detect_accuracy(
                    predict=out, target=grasp_data_batch, box_conf=box_data_batch,
                    model_threshold=model_threshold[i])

        move_list = np.arange(int(num_item / 2), num_item)

        return move_list, knolling_flag