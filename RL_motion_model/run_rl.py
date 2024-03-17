import os
import shutil

from environment_rl import *
import wandb
import argparse


np.random.seed(0)
random.seed(0)

def eval_model(num_episodes = 40, index_start=0):

    total_rewards = []

    shutil.rmtree(para_dict['data_source_path'] + 'sim_obs/')
    os.mkdir(para_dict['data_source_path'] + 'sim_obs/')
    shutil.rmtree(para_dict['data_source_path'] + 'sim_images/')
    os.mkdir(para_dict['data_source_path'] + 'sim_images/')

    for episode in range(index_start, index_start + num_episodes):
        obs, _ = env.reset(epoch=episode)
        done = False
        total_reward = 0
        total_step = 0

        while not done:
            total_step += 1
            action, _states = model.predict(obs, deterministic=True)
            # action[2] -= 0.003  # wrap the action space to make the model output 0.002
            # print(action)

            obs, reward, done, _, info = env.step(action)
            # print(reward)

            total_reward += reward

        print('this is the num of step per episode', total_step)

        # print(total_reward)
        total_rewards.append(total_reward)

    average_reward = np.mean(total_rewards)
    print("Average Reward:", average_reward)
    return average_reward

para_dict = {'reset_pos': np.array([-0.9, 0, 0.005]), 'reset_ori': np.array([0, np.pi / 2, 0]),
             'save_img_flag': True,
             'yolo_conf': 0.3, 'yolo_iou': 0.6, 'device': 'cuda:0', 'lstm_enable_flag': True,
             'init_pos_range': [[0.13, 0.17], [-0.03, 0.03], [0.01, 0.02]],
             'init_offset_range': [[-0.05, 0.05], [-0.1, 0.1]],
             'init_ori_range': [[-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4]],
             'boxes_num': 2,
             'boxes_num_max': 5,
             'is_render': False,
             'box_range': [[0.016, 0.048], [0.016], [0.01, 0.02]],
             'box_mass': 0.1,
             'gripper_force': 3,
             'move_force': 3,
             'real_operate': False,
             'gripper_lateral_friction': 1, 'gripper_contact_damping': 1, 'gripper_contact_stiffness': 50000,
             'box_lateral_friction': 1, 'box_contact_damping': 1, 'box_contact_stiffness': 50000,
             'base_lateral_friction': 1, 'base_contact_damping': 1, 'base_contact_stiffness': 50000,
             'urdf_path': '../ASSET/urdf/', }

para_dict['yolo_model_path'] = '../ASSET/models/627_pile_pose/weights/best.pt'
para_dict['data_source_path'] = '../../knolling_dataset/rl_quantatitive_evaluation/'
os.makedirs(para_dict['data_source_path'], exist_ok=True)

train_RL = False

lstm_dict = {'input_size': 6,
             'hidden_size': 32,
             'num_layers': 8,
             'output_size': 2,
             'hidden_node_1': 32, 'hidden_node_2': 8,
             'batch_size': 1,
             'device': 'cuda:0',
             'set_dropout': 0.1,
             'threshold': 0.35, # 0.55 real, 0.35 sim
             'grasp_model_path': '../ASSET/models/LSTM_918_0/best_model.pt',}

run_id = '16'
max_num_obj = para_dict['boxes_num_max']
loggerID = 16

RLmode = "SAC"
num_scence = 200
Two_obs_Flag = False
log_path = f"logger/{RLmode}_{max_num_obj}objobs/log{loggerID}/"

env = Arm_env(para_dict=para_dict, init_scene=num_scence,offline_data=False, two_obj_obs=Two_obs_Flag, lstm_dict=lstm_dict)

model = SAC.load(log_path + "/ppo_model_best.zip")

# Evaluate the model

np.random.seed(0)
random.seed(0)
eval_model()