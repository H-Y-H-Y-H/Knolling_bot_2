import numpy as np

from Robot.SundryRobot import Sundry_robot
from Env.SundryEnv import Sundry_env
from utils import *

class sundry_grasp_main():

    def __init__(self, para_dict=None, lstm_dict=None, knolling_para=None):

        self.para_dict = para_dict
        self.lstm_dict = lstm_dict
        self.knolling_para = knolling_para

        self.task = Sundry_env(para_dict=para_dict, lstm_dict=lstm_dict)
        self.robot = Sundry_robot(para_dict=para_dict, knolling_para=knolling_para)

    def reset(self):

        p.resetSimulation()
        self.task.create_scene()
        self.arm_id = self.robot.create_arm()
        self.robot.calculate_gripper()
        self.conn, self.real_table_height, self.sim_table_height = self.robot.arm_setup()

        self.task.create_objects()

        self.img_per_epoch = 0

        self.state_id = p.saveState()

    def try_grasp(self):

        obj_pos, obj_ori = self.task.get_gt_pos_ori()
        manipulator_before = np.concatenate((obj_pos, np.zeros((len(obj_ori), 2)), obj_ori[:, 2].reshape(len(obj_ori), -1)), axis=1)

        pos_ori_after = np.concatenate((self.para_dict['reset_pos'], np.zeros(3)), axis=0).reshape(-1, 6)
        manipulator_after = np.repeat(pos_ori_after, len(obj_ori), axis=0)

        start_end = np.concatenate((manipulator_before, manipulator_after), axis=1)
        grasp_width = np.min(self.task.obj_gt_lwh[:, :2], axis=1)
        offset_low = np.array([0, 0, 0.0])
        offset_high = np.array([0, 0, 0.04])
        
        for i in range(len(start_end)):
            trajectory_pos_list = [self.para_dict['reset_pos'],  # the origin position
                                   [0, grasp_width[i]],  # gripper open!
                                   offset_high + start_end[i][:3],  # move directly to the above of the target
                                   offset_low + start_end[i][:3],  # decline slowly
                                   [1, grasp_width[i]],  # gripper close
                                   offset_high + start_end[i][:3],  # lift the target up
                                   start_end[i][6:9]]  # move to the destination
            trajectory_ori_list = [self.para_dict['reset_ori'],
                                   self.para_dict['reset_ori'] + start_end[i][3:6],
                                   self.para_dict['reset_ori'] + start_end[i][3:6],
                                   self.para_dict['reset_ori'] + start_end[i][3:6],
                                   [1, grasp_width[i]],
                                   self.para_dict['reset_ori'] + start_end[i][3:6],
                                   self.para_dict['reset_ori'] + start_end[i][9:12]]

            last_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
            last_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))
            left_pos = np.asarray(p.getLinkState(self.arm_id, 7)[0])
            right_pos = np.asarray(p.getLinkState(self.arm_id, 8)[0])

            for j in range(len(trajectory_pos_list)):
                if len(trajectory_pos_list[j]) == 3:
                    if j == 2:
                        last_pos, left_pos, right_pos, _ = self.robot.move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j], index=j)
                    elif j == 3:
                        ####################### Detect whether the gripper is disturbed by other objects during moving the gripper ####################
                        last_pos, _, _, success_grasp_flag = self.robot.move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j],
                                                                       origin_left_pos=left_pos, origin_right_pos=right_pos, index=j)
                        if success_grasp_flag == False:
                            break
                        ####################### Detect whether the gripper is disturbed by other objects during moving the gripper ####################
                    else: # 0, 4, 5, 6
                        last_pos, _, _, _ = self.robot.move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j], index=j)
                    last_ori = np.copy(trajectory_ori_list[j])
                elif len(trajectory_pos_list[j]) == 2:
                    ####################### Dtect whether the gripper is disturbed by other objects during closing the gripper ####################
                    success_grasp_flag = self.robot.gripper(trajectory_pos_list[j][0], trajectory_pos_list[j][1], left_pos, right_pos, index=j)
                    ####################### Detect whether the gripper is disturbed by other objects during closing the gripper ####################
            p.restoreState(self.state_id)

if __name__ == '__main__':
    # np.random.seed(4)
    # random.seed(4)
    para_dict = {
        'reset_pos': np.array([0, 0, 0.12]), 'reset_ori': np.array([0, np.pi / 2, 0]),
        'save_img_flag': True,
        'init_pos_range': [[0.03, 0.27], [-0.15, 0.15], [0.01, 0.01]], 'init_offset_range': [[-0, 0], [-0, 0]],
        'init_ori_range': [[0, 0], [0, 0], [0, 0]],
        'objects_num': 3,
        'is_render': True,
        'gripper_lateral_friction': 1, 'gripper_contact_damping': 1, 'gripper_contact_stiffness': 50000,
        'obj_lateral_friction': 1, 'obj_contact_damping': 1, 'obj_contact_stiffness': 50000,
        'base_lateral_friction': 1, 'base_contact_damping': 1, 'base_contact_stiffness': 50000,
        'object_path': '../../knolling_dataset/sundry_301/',
        'urdf_path': '../ASSET/urdf/',
        'real_operate': False,
        'gripper_threshold': 0.002, 'gripper_sim_step': 10, 'gripper_force': 3,
        'move_threshold': 0.001, 'move_force': 3,
    }

    env = sundry_grasp_main(para_dict=para_dict)
    env.reset()
    env.try_grasp()

