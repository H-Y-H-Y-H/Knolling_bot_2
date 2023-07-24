from environment import Arm_env
import numpy as np
import random
import os

if __name__ == '__main__':

    np.random.seed(183)
    random.seed(183)

    para_dict = {'start_num': 00, 'end_num': 10000, 'thread': 0,
                 'yolo_conf': 0.6, 'yolo_iou': 0.8, 'device': 'cuda:0',
                 'save_img_flag': True,
                 'init_pos_range': [[0.13, 0.17], [-0.03, 0.03], [0.01, 0.02]],
                 'init_ori_range': [[-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4], [3 * np.pi / 16, np.pi / 4]],
                 'max_box_num': 5, 'min_box_num': 4,
                 'is_render': True,
                 'box_range': [[0.016, 0.048], [0.016], [0.01, 0.02]],
                 'box_mass': 0.1,
                 'gripper_threshold': 0.002, 'gripper_sim_step': 10, 'gripper_force': 3,
                 'move_threshold': 0.001, 'move_force': 3,
                 'box_lateral_friction': 1, 'box_spinning_friction': 1,
                 'box_restitution': 0, 'box_contact_damping': 100, 'box_contact_stiffness': 1000000,
                 'gripper_lateral_friction': 1, 'gripper_spinning_friction': 1,
                 'gripper_restitution': 0, 'gripper_contact_damping': 100, 'gripper_contact_stiffness': 1000000,
                 'base_lateral_friction': 1, 'base_spinning_friction': 1,
                 'base_restitution': 0, 'base_contact_damping': 10, 'base_contact_stiffness': 1000000,
                 'dataset_path': '/home/zhizhuo/Creative_Machines_Lab/knolling_dataset/',
                 'urdf_path': '/home/zhizhuo/Creative_Machines_Lab/Knolling_bot_2/urdf/',
                 'yolo_model_path': '/home/zhizhuo/Creative_Machines_Lab/Knolling_bot_2/train_pile_overlap_627/weights/best.pt'}

    startnum = para_dict['start_num']
    endnum = para_dict['end_num']
    thread = para_dict['thread']
    save_img_flag = para_dict['save_img_flag']
    init_pos_range = para_dict['init_pos_range']
    init_ori_range = para_dict['init_ori_range']

    data_root = para_dict['dataset_path'] + 'grasp_dataset_721_heavy_test/'
    # with open(para_dict['dataset_path'] + 'grasp_dataset_721_heavy_test_readme.txt', "w") as f:
    #     for key, value in para_dict.items():
    #         f.write(key + ': ')
    #         f.write(str(value) + '\n')

    os.makedirs(data_root, exist_ok=True)

    max_box_num = para_dict['max_box_num']
    min_box_num = para_dict['min_box_num']
    mm2px = 530 / 0.34

    env = Arm_env(max_step=1, endnum=endnum, save_img_flag=save_img_flag,
                  init_pos_range=init_pos_range, init_ori_range=init_ori_range, para_dict=para_dict)
    os.makedirs(data_root + 'origin_images/', exist_ok=True)
    os.makedirs(data_root + 'origin_labels/', exist_ok=True)

    exist_img_num = startnum
    while True:
        num_item = int(np.random.uniform(min_box_num, max_box_num + 1))
        env.reset(data_root=data_root, num_item=num_item, thread=thread, epoch=exist_img_num)
        img_per_epoch = env.try_grasp(data_root=data_root, img_index_start=exist_img_num)
        exist_img_num += img_per_epoch