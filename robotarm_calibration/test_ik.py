import os
import numpy as np
import random

m1 = 0.20615
m2 = 0.200
len_wrist = 0.174
height_base = 0.105
theta_1_offset = np.arctan2(0.05, 0.2)
origin_offset = 0.084
height_offset = 0.005

def inverse_kinematic(pos, ori, parameters=None):
    if len(pos.shape) == 1:
        pos = pos.reshape(1, 3)
        ori = ori.reshape(1, 3)

    pos[:, 0] += origin_offset
    pos[:, 2] += height_offset
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    yaw = ori[:, 2]
    distance = np.sqrt(x ** 2 + y ** 2)
    z_cal = z + len_wrist - height_base

    theta_2 = np.arccos(((m1 ** 2 - m2 ** 2 - z_cal ** 2 - distance ** 2) / (2 * m2)) / np.sqrt(z_cal ** 2 + distance ** 2)) - np.arccos(z_cal / np.sqrt(z_cal ** 2 + distance ** 2))
    theta_1 = np.arccos((distance - m2 * np.sin(theta_2)) / m1)
    motor_0 = np.arctan2(y, x) + np.pi
    motor_1 = np.pi * 2 - (theta_1 + theta_1_offset + np.pi / 2)
    motor_2 = theta_2 + np.pi / 2 - theta_1 - theta_1_offset + np.pi / 2
    motor_3 = np.pi - theta_2
    motor_4 = np.pi - (yaw - (motor_0 - np.pi))

    motor = np.concatenate((motor_0.reshape(-1, 1), motor_1.reshape(-1, 1), motor_2.reshape(-1, 1), motor_3.reshape(-1, 1), motor_4.reshape(-1, 1)), axis=1)
    motor = motor / (np.pi * 2) * 4096
    motor[:, 1] = np.floor(motor[:, 1])
    motor[:, 2] = np.ceil(motor[:, 2])
    motor = np.insert(motor, 1, motor[:, 1], axis=1)
    return motor

def forward_kinematic(motor):

    motor = np.delete(motor, 2, axis=1)
    # print('this is motor', motor)
    motor = motor / 4096 * (np.pi * 2)
    theta_0 = motor[:, 0] - np.pi
    theta_1 = np.pi * 2 - motor[:, 1] - theta_1_offset - np.pi / 2
    theta_2 = motor[:, 2] - np.pi / 2 + theta_1 + theta_1_offset - np.pi / 2
    theta_3 = np.pi - motor[:, 3]
    yaw = motor[:, 0] - np.pi + np.pi - motor[:, 4]
    distance = m1 * np.cos(theta_1) + m2 * np.sin(theta_2)
    x = distance * np.cos(theta_0)
    y = distance * np.sin(theta_0)
    z = height_base + (m1 * np.sin(theta_1) - m2 * np.cos(theta_2)) - len_wrist

    pos = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis=1)
    pos[:, 0] -= origin_offset
    pos[:, 2] -= height_offset
    ori = np.concatenate((np.zeros(len(motor)).reshape(-1, 1), np.ones(len(motor)).reshape(-1, 1) * np.pi / 2, yaw.reshape(-1, 1)), axis=1)

    return pos, ori

def cartesian_slice(cur_pos, cur_ori, tar_pos, tar_ori):
    new_tar_ori = np.copy(tar_ori)
    new_cur_ori = np.copy(cur_ori)
    if tar_ori[2] > np.pi / 2:
        new_tar_ori[2] = tar_ori[2] - np.pi
        print('tar ori reduced!')
    elif tar_ori[2] < -np.pi / 2:
        new_tar_ori[2] = tar_ori[2] + np.pi
        print('tar ori increased!')
    if cur_ori[2] > np.pi / 2:
        new_cur_ori[2] = cur_ori[2] - np.pi
        print('cur ori reduced!')
    elif cur_ori[2] < -np.pi / 2:
        new_cur_ori[2] = cur_ori[2] + np.pi
        print('cur ori increased!')
    print('this is tar ori', tar_ori)
    print('this is tar ori after', new_tar_ori)
    print('this is cur ori', cur_ori)
    print('this is cur ori after', new_cur_ori)

    if abs(cur_pos[0] - tar_pos[0]) < 0.001 and abs(cur_pos[1] - tar_pos[1]) < 0.001:
        # vertical, choose a small slice
        move_slice = 0.004
    else:
        # horizontal, choose a large slice
        move_slice = 0.008

    target_pos = np.copy(tar_pos)
    target_ori = np.copy(new_tar_ori)

    vertical_flag = False
    print('this is tar pos', target_pos)
    print('this is cur pos', cur_pos)
    if np.abs(target_pos[2] - cur_pos[2]) > 0.01 \
            and np.abs(target_pos[0] - cur_pos[0]) < 0.01 \
            and np.abs(target_pos[1] - cur_pos[1]) < 0.01:
        mark_ratio = 0.8
        vertical_flag = True
        seg_time = 0
    else:
        mark_ratio = 1
        seg_time = 0

    cmd_motor = []
    cmd_xyz = []
    cmd_ori = []
    real_xyz = []

    # divide the whole trajectory into several segment
    seg_pos = mark_ratio * (target_pos - cur_pos) + cur_pos
    seg_ori = mark_ratio * (target_ori - new_cur_ori) + new_cur_ori
    distance = np.linalg.norm(seg_pos - cur_pos)
    num_step = np.ceil(distance / move_slice)
    step_pos = (seg_pos - cur_pos) / num_step
    step_ori = (seg_ori - new_cur_ori) / num_step
    print('this is seg pos', seg_pos)

    while True:
        tar_pos = cur_pos + step_pos
        new_tar_ori = new_cur_ori + step_ori
        cmd_xyz.append(tar_pos)
        cmd_ori.append(new_tar_ori)
        break_flag = abs(seg_pos[0] - tar_pos[0]) < 0.001 and abs(
            seg_pos[1] - tar_pos[1]) < 0.001 and abs(seg_pos[2] - tar_pos[2]) < 0.001 and \
                        abs(seg_ori[0] - new_tar_ori[0]) < 0.001 and abs(
            seg_ori[1] - new_tar_ori[1]) < 0.001 and abs(seg_ori[2] - new_tar_ori[2]) < 0.001
        if break_flag:
            break
        cur_pos = tar_pos
        new_cur_ori = new_tar_ori

    cmd_xyz = np.asarray(cmd_xyz)
    cmd_ori = np.asarray(cmd_ori)

    return cmd_xyz, cmd_ori

if __name__ == '__main__':

    tar_ori = np.array([ 0, 1.5707964, -0.5046517])
    cur_ori = np.array([0, 1.5707964, 0])
    tar_pos = np.array([0.21703127, 0.03508795, 0.04823438])
    cur_pos = np.array([0.08858008, -0.01423064,  0.05059762])
    cmd_xyz, cmd_ori = cartesian_slice(cur_pos, cur_ori, tar_pos, tar_ori)
    cmd_motor = np.asarray(inverse_kinematic(np.copy(cmd_xyz), np.copy(cmd_ori)), dtype=np.float32)
    pass