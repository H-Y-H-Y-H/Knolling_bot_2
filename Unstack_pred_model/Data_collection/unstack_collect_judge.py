import cv2

from models.visual_perception_config import *

class Judge_push():

    def __init__(self, para_dict, lstm_dict):
        self.para_dict = para_dict
        self.lstm_dict = lstm_dict
        if use_yolo == True:
            self.yolo_pose_model = Yolo_pose_model(para_dict=para_dict, lstm_dict=lstm_dict, use_lstm=self.para_dict['use_lstm_model'])
        else:
            self.grasp_pred_model = Grasp_model(para_dict=para_dict, lstm_dict=lstm_dict)
        self.num_ray = self.para_dict['num_rays']
        self.data_root = self.para_dict['dataset_path']

        self.output_index = self.para_dict['unstack_start_num']

    # def pred_yolo(self, epoch_index=0):
    #
    #     eval_index = np.arange(epoch_index * self.num_ray, (epoch_index + 1) * self.num_ray, dtype=np.int32)
    #     origin_img = cv2.imread(self.data_root + 'sim_images/%012d.png' % epoch_index)
    #     max_grasp_num = 1
    #     unstack_rays = []
    #     origin_info = np.loadtxt(self.data_root + 'sim_info/%012d.txt' % epoch_index)
    #     for i in range(len(eval_index)):
    #
    #         img_candidate = cv2.imread(self.data_root + 'unstack_images/%012d.png' % eval_index[i])
    #         ray_candidate = np.loadtxt(self.data_root + 'unstack_rays/%012d.txt' % eval_index[i])
    #         unstack_rays.append(ray_candidate)
    #
    #         manipulator_before, new_lwh_list, pred_conf, crowded_index, prediction, model_output \
    #         = self.yolo_pose_model.yolo_pose_predict(img=img_candidate, epoch=None)
    #         self.yolo_pose_model.plot_grasp(manipulator_before, prediction, model_output, img=img_candidate, epoch=eval_index[i])
    #
    #         test_grasp_num = len(manipulator_before) - len(crowded_index)
    #         if test_grasp_num > max_grasp_num:
    #             max_grasp_num = test_grasp_num
    #             max_grasp_index = i
    #
    #     if max_grasp_num == 1:
    #         print('no good push in this epoch, ignore it!')
    #         # os.remove(self.data_root + 'sim_images/%012d.png' % epoch_index)
    #     else:
    #         print(f'save the {max_grasp_index} as the best ray!')
    #         cv2.imwrite(self.data_root + 'input_images/%012d.png' % self.output_index, origin_img)
    #         np.savetxt(self.data_root + 'input_boxs/%012d.txt' % self.output_index, origin_info, fmt='%.04f')
    #         np.savetxt(self.data_root + 'output_rays/%012d.txt' % self.output_index, unstack_rays[max_grasp_index].reshape(2, -1), fmt='%.04f')
    #         self.output_index += 1

    def pred_yolo(self, epoch_index=0):

        # eval_index = np.arange(epoch_index * self.num_ray, (epoch_index + 1) * self.num_ray, dtype=np.int32)
        # origin_img = cv2.imread(self.data_root + 'sim_images/%012d.png' % epoch_index)
        max_grasp_num = 1
        unstack_rays = []
        # origin_info = np.loadtxt(self.data_root + 'sim_info/%012d.txt' % epoch_index)

        img_candidate = cv2.imread(self.data_root + 'unstack_images/%012d.png' % epoch_index)
        ray_candidate = np.loadtxt(self.data_root + 'unstack_rays/%012d.txt' % epoch_index)
        unstack_rays.append(ray_candidate)

        manipulator_before, new_lwh_list, pred_conf = self.yolo_pose_model.yolo_pose_predict(img=img_candidate, epoch=None)
        yolo_temp_info = np.concatenate((manipulator_before, new_lwh_list, pred_conf.reshape(-1, 1)), axis=1)

        cv2.imwrite(self.data_root + 'yolo_temp_info/%012d.png' % epoch_index, img_candidate)
        np.savetxt(self.data_root + 'yolo_temp_info/%012d.txt' % epoch_index, yolo_temp_info, fmt='%.05f')

    def pred_lstm(self, epoch_index=0):

        eval_index = np.arange(epoch_index * self.num_ray, (epoch_index + 1) * self.num_ray, dtype=np.int32)
        max_grasp_num = 1
        unstack_rays = []
        origin_info = np.loadtxt(self.data_root + 'sim_info/%012d.txt' % epoch_index)
        for i in range(len(eval_index)):

            yolo_info_candidate = np.loadtxt(self.data_root + 'yolo_temp_info/%012d.txt' % eval_index[i])
            manipulator_before = yolo_info_candidate[:, 6]
            new_lwh_list = yolo_info_candidate[:, 6:9]
            pred_conf = yolo_info_candidate[:, 9:]
            ray_candidate = np.loadtxt(self.data_root + 'unstack_rays/%012d.txt' % eval_index[i])
            unstack_rays.append(ray_candidate)

            crowded_index, prediction, model_output = self.grasp_pred_model.pred(manipulator_before=manipulator_before, lwh_list=new_lwh_list, conf_list=pred_conf)

            test_grasp_num = len(manipulator_before) - len(crowded_index)
            if test_grasp_num > max_grasp_num:
                max_grasp_num = test_grasp_num
                max_grasp_index = i

        if max_grasp_num == 1:
            print('no good push in this epoch, ignore it!')
            # os.remove(self.data_root + 'sim_images/%012d.png' % epoch_index)
        else:
            print(f'save the {max_grasp_index} as the best ray!')
            np.savetxt(self.data_root + 'input_boxs/%012d.txt' % self.output_index, origin_info, fmt='%.04f')
            np.savetxt(self.data_root + 'output_rays/%012d.txt' % self.output_index, unstack_rays[max_grasp_index].reshape(2, -1), fmt='%.04f')
            self.output_index += 1

if __name__ == '__main__':

    # np.random.seed(111)
    # random.seed(111)

    # simulation: iou 0.8
    # real world: iou=0.5

    use_yolo = True

    para_dict = {'unstack_start_num': 100000, 'unstack_end_num': 100005, 'thread': 0, 'input_output_label_offset': 100000, 'num_rays': 5,
                 'yolo_conf': 0.6, 'yolo_iou': 0.8, 'device': 'cuda:1',
                 'reset_pos': np.array([0.0, 0, 0.10]), 'reset_ori': np.array([0, np.pi / 2, 0]),
                 'save_img_flag': True,
                 'init_pos_range': [[0.13, 0.17], [-0.03, 0.03], [0.01, 0.02]], 'init_offset_range': [[-0.05, 0.05], [-0.1, 0.1]],
                 'init_ori_range': [[-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4]],
                 'boxes_num': np.random.randint(5, 6),
                 'is_render': False,
                 'box_range': [[0.016, 0.048], [0.016], [0.01, 0.016]],
                 'box_mass': 0.1,
                 'gripper_threshold': 0.002, 'gripper_sim_step': 10, 'gripper_force': 3,
                 'move_threshold': 0.001, 'move_force': 30,
                 'gripper_lateral_friction': 1, 'gripper_contact_damping': 1, 'gripper_contact_stiffness': 50000,
                 'box_lateral_friction': 1, 'box_contact_damping': 1, 'box_contact_stiffness': 50000,
                 'base_lateral_friction': 1, 'base_contact_damping': 1, 'base_contact_stiffness': 50000,
                 'dataset_path': '../../../knolling_dataset/MLP_unstack_908_intensive/',
                 'urdf_path': '../../urdf/',
                 'yolo_model_path': '../../models/627_pile_pose/weights/best.pt',
                 'real_operate': False, 'obs_order': 'sim_image_obj', 'data_collection': True, 'rl_configuration': True,
                 'use_knolling_model': False, 'use_lstm_model': False}

    lstm_dict = {'input_size': 6,
                 'hidden_size': 32,
                 'num_layers': 8,
                 'output_size': 2,
                 'hidden_node_1': 32, 'hidden_node_2': 8,
                 'batch_size': 1,
                 'set_dropout': 0.0,
                 'threshold': 0.6,
                 'device': 'cuda:0',
                 'grasp_model_path': '../../models/LSTM_829_1_heavy_dropout0/best_model.pt', }

    os.makedirs(para_dict['dataset_path'] + 'input_images/', exist_ok=True)
    os.makedirs(para_dict['dataset_path'] + 'input_boxs/', exist_ok=True)
    os.makedirs(para_dict['dataset_path'] + 'output_rays/', exist_ok=True)

    os.makedirs(para_dict['dataset_path'] + 'yolo_temp_info/', exist_ok=True)

    zzz_judge = Judge_push(para_dict=para_dict, lstm_dict=lstm_dict)

    epoch_index_start = para_dict['unstack_start_num']
    epoch_index_end = para_dict['unstack_end_num']
    for i in range(epoch_index_start, epoch_index_end):
        zzz_judge.pred_yolo(epoch_index=i)