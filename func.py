import pybullet as p
import time
import pybullet_data
import random
import os
import numpy as np
import csv

filename = "robot_arm1"

def reset(robotID):
    p.setJointMotorControlArray(robotID, [0,1,2,3,4,7,8], p.POSITION_CONTROL, targetPositions=[0,-np.pi/2,np.pi/2,0,0,0,0])


def sim_cmd2tarpos(tar_pos_cmds):
    tar_pos_cmds = np.asarray(tar_pos_cmds)
    motion_limit = np.asarray([1/180, 1/135, 1/135, 1/90, 1/180])
    reset_rad = [0, -np.pi/2, np.pi/2, 0, 0]
    reset_cmds = np.asarray([0.5, 0, 1, 0, 0.5])
    cmds_gap = np.subtract(tar_pos_cmds, reset_cmds)
    rad_gap = np.divide(cmds_gap, motion_limit) * np.pi/180
    tar_pos = np.add(reset_rad, rad_gap)
    return tar_pos

# real robot:motor limit:0-4095(0 to 360 degrees)

def real_cmd2tarpos(tar_pos_cmds): # sim to real!!!!!!!!!!!!!!!!

    # input: scaled angle of motors in pybullet, basically (0, 1)
    # output: angle of motors in real world, (0, 4096)

    tar_pos_cmds = np.asarray(tar_pos_cmds)
    pos2deg = 4095/360
    motion_limit = np.asarray([1/180, 1/135, 1/135, 1/90, 1/180])
    motion_limit2 = np.divide(motion_limit, pos2deg)
    reset_pos = [3075, 1025, 1050, 2050, 2050]
    reset_cmds = np.asarray([0.5, 0, 1, 0, 0.5])
    cmds_gap = np.subtract(tar_pos_cmds, reset_cmds)
    cmds_gap[2] = -cmds_gap[2]
    cmds_gap[3] = -cmds_gap[3]
    pos_gap = np.divide(cmds_gap, motion_limit2)
    tar_pos = np.add(reset_pos, pos_gap)
    tar_pos2 = np.insert(tar_pos, 2, tar_pos[1])
    # tar_pos2 = tar_pos2.astype(int)
    return tar_pos2

def real_tarpos2cmd(tar_pos): # real to sim!!!!!!!!!!!

    # input: angle of motors in real world, (0, 4096)
    # output: scaled angle of motors in pybullet, basically (0, 1)

    tar_pos = np.delete(tar_pos, 2)
    tar_pos = np.asarray(tar_pos)
    print('this is tar from calculation', tar_pos)
    pos2deg = 4095/360
    motion_limit = np.asarray([1/180, 1/135, 1/135, 1/90, 1/180])
    reset_pos = [3075, 1025, 1050, 2050, 2050]
    reset_cmds = np.asarray([0.5, 0, 1, 0, 0.5])
    motion_limit2 = np.divide(motion_limit, pos2deg)
    pos_gap = np.subtract(tar_pos, reset_pos)
    cmds_gap = np.multiply(motion_limit2, pos_gap)
    cmds_gap[2] = -cmds_gap[2]
    cmds_gap[3] = -cmds_gap[3]
    cmds = np.add(cmds_gap, reset_cmds)

    return cmds


def rad2cmd(cur_rad): # sim to real

    # input: angle of motors in pybullet, (-180, 180)
    # output: scaled angle of motors in pybullet, basically (0, 1)
    
    cur_rad = np.asarray(cur_rad)
    reset_rad = np.asarray([0, -np.pi/2, np.pi/2, 0, 0])
    rad_gap = np.subtract(cur_rad, reset_rad)
    motion_limit = np.asarray([1/180, 1/135, 1/135, 1/90, 1/180])
    cmds_gap = np.multiply(rad_gap, motion_limit) * 180/np.pi
    reset_cmds = np.asarray([0.5, 0, 1, 0, 0.5])
    tar_cmds = np.add(reset_cmds, cmds_gap)
    return tar_cmds

def cmd2rad(cur_cmd): # real to sim

    # input: scaled angle of motors in pybullet, basically (0, 1)
    # output: angle of motors in pybullet, (-180, 180)

    cur_cmd = np.asarray(cur_cmd)
    reset_cmds = np.asarray([0.5, 0, 1, 0, 0.5])
    cmds_gap = np.subtract(cur_cmd, reset_cmds)
    motion_limit = np.asarray([1/180, 1/135, 1/135, 1/90, 1/180])
    rad_gap = np.true_divide(cmds_gap, motion_limit) * (np.pi/180)
    reset_rad = np.asarray([0, -np.pi/2, np.pi/2, 0, 0])
    tar_rads = np.add(reset_rad, rad_gap)

    return tar_rads


def rad2pos(cur_rad): 
    tar_cmds = rad2cmd(cur_rad)
    # print("cmd", tar_cmds)
    pos = real_cmd2tarpos(tar_cmds)
    return pos

def change_sequence(pos_before):

    origin_point = np.array([0, -0.2])
    delete_index = np.where(pos_before == 0)[0]
    distance = np.linalg.norm(pos_before[:, :2] - origin_point, axis=1)
    order = np.argsort(distance)
    return order

class grasp_model():

    def __init__(self):

        pass

    def pred(self, manipulator_before, lwh_list, conf_list):

        knolling_flag = False
        num_item = len(conf_list)
        move_list = np.arange(int(num_item / 2), num_item)

        return move_list, knolling_flag

class configuration_zzz():

    def __init__(self, xyz_list, all_index, transform_flag, manual_knolling_parameters):

        self.xyz_list = xyz_list
        self.all_index = all_index
        self.transform_flag = transform_flag
        self.gap_item = manual_knolling_parameters['gap_item']
        self.gap_block = manual_knolling_parameters['gap_block']
        self.item_odd_prevent = manual_knolling_parameters['item_odd_prevent']
        self.block_odd_prevent = manual_knolling_parameters['block_odd_prevent']
        self.upper_left_max = manual_knolling_parameters['upper_left_max']
        self.forced_rotate_box = manual_knolling_parameters['forced_rotate_box']

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


if __name__ == "__main__":
    filename = "robot_arm1"

    # or p.DIRECT for non-graphical version
    physicsClient = p.connect(1)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    p.setGravity(0, 0, -9.8)
    planeId = p.loadURDF("plane.urdf")
    table_scale = 0.7
    table_surface_height = 0.625*table_scale
    print(table_surface_height)
    startPos = [0, 0, table_surface_height]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])


    boxId = p.loadURDF(filename + ".urdf", startPos, startOrientation, useFixedBase=1, flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
    # boxId2 = p.loadURDF("cube_small.urdf", [0.2, 0.2, table_surface_height+1], startOrientation, useFixedBase=1, flags=p.URDF_USE_SELF_COLLISION)
    boxId3 = p.loadURDF("table/table.urdf", [(0.5-0.16)*table_scale, 0, 0], p.getQuaternionFromEuler([0, 0, np.pi/2]), useFixedBase=1,
                        flags=p.URDF_USE_SELF_COLLISION, globalScaling=table_scale)
    boxId4 = p.loadURDF("urdf/cra.urdf",[0.2, 0.2, table_surface_height+0.003], startOrientation, useFixedBase=1, flags=p.URDF_USE_SELF_COLLISION)
    # boxId4 = p.loadURDF("samurai.urdf", [0.2, 0.5, table_surface_height], startOrientation, useFixedBase=1, flags=p.URDF_USE_SELF_COLLISION)

    # boxId4 = p.loadURDF("cube_small.urdf", [0, 0.2, table_surface_height], startOrientation, useFixedBase=0, flags=p.URDF_USE_SELF_COLLISION)
    p.changeDynamics(boxId, 7, lateralFriction=0.99,spinningFriction=0.02,rollingFriction=0.002)
    p.changeDynamics(boxId, 8, lateralFriction=0.99,spinningFriction=0.02,rollingFriction=0.002)
    # p.changeDynamics(boxId2, -1, lateralFriction=0.99,spinningFriction=0.02,rollingFriction=0.002)
    # num_joints = p.getNumJoints(boxId)
    # print(num_joints)
    # print(p.getLinkState(boxId,12))

    reset(boxId)
    # for i in range(600):
    #     p.stepSimulation()
    #     time.sleep(1/240)
    for i in range(200):
        p.stepSimulation()
        # p.setJointMotorControlArray(boxId, [7, 8], p.POSITION_CONTROL, targetPositions=[0.032*sin(t), 0.032*sin(t)])
        # t+=0.01
        time.sleep(1/240)
    for i in range(20000):
        p.stepSimulation()
        # p.setJointMotorControlArray(boxId, [7, 8], p.POSITION_CONTROL, targetPositions=[0.032*sin(t), 0.032*sin(t)])
        # t+=0.01
        time.sleep(1/240)

    p.disconnect()
