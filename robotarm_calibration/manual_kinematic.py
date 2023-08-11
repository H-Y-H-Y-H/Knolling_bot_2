import numpy as np
from sympy import *
from tqdm import tqdm

num_motor = 5
m1 = 0.20615
m2 = 0.200
len_wrist = 0.174
height_base = 0.105
origin_offset = 0.08
theta_1_offset = np.arctan2(0.05, 0.2)

def inverse_kinematic(pos, ori):

    pos[:, 0] += origin_offset
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    yaw = ori[:, 2]
    distance = np.sqrt(x ** 2 + y ** 2)
    z_cal = z + len_wrist - height_base

    theta_2 = np.arccos(((m1 ** 2 - m2 ** 2 - z_cal ** 2 - distance ** 2) / (2 * m2)) / np.sqrt(z_cal ** 2 + distance ** 2)) - np.arccos(z_cal / np.sqrt(z_cal ** 2 + distance ** 2))
    theta_1 = np.arccos((distance - m2 * np.sin(theta_2)) / m1)
    motor_0 = np.arctan2(y, x) + np.pi * 3 / 2
    motor_1 = np.pi * 2 - (theta_1 + theta_1_offset + np.pi / 2)
    motor_2 = theta_2 + np.pi / 2 - theta_1 - theta_1_offset + np.pi / 2
    motor_3 = np.pi - theta_2
    motor_4 = np.pi - (yaw - (motor_0 - np.pi * 3 / 2))

    motor = np.concatenate((motor_0.reshape(-1, 1), motor_1.reshape(-1, 1), motor_2.reshape(-1, 1), motor_3.reshape(-1, 1), motor_4.reshape(-1, 1)), axis=1)

    motor = motor / (np.pi * 2) * 4096
    motor = np.insert(motor, 1, motor[:, 1], axis=1)
    # print('motor_0', motor_0[0])
    # print('motor_1', motor_1[0])
    # print('motor_2', motor_2[0])
    # print('motor_3', motor_3[0])
    # print('motor_4', motor_4[0])
    # print('motor', motor[0])
    return motor

def forward_kinematic(motor):

    motor = motor / 4096 * (np.pi * 2)
    motor = np.delete(motor, 1, axis=1)
    # print(motor)
    theta_0 = motor[:, 0] - np.pi * 3 / 2
    theta_1 = np.pi * 2 - motor[:, 1] - theta_1_offset - np.pi / 2
    theta_2 = motor[:, 2] - np.pi / 2 + theta_1 + theta_1_offset - np.pi / 2
    theta_3 = np.pi - motor[:, 3]
    yaw = motor[:, 0] - np.pi * 3 / 2 + np.pi - motor[:, 4]
    distance = m1 * np.cos(theta_1) + m2 * np.sin(theta_2)
    x = distance * np.cos(theta_0)
    y = distance * np.sin(theta_0)
    z = height_base + (m1 * np.sin(theta_1) - m2 * np.cos(theta_2)) - len_wrist

    pos = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis=1)
    ori = np.concatenate((np.zeros(len(motor)).reshape(-1, 1), np.ones(len(motor)).reshape(-1, 1) * np.pi / 2, yaw.reshape(-1, 1)), axis=1)
    pos[:, 0] -= origin_offset
    return pos, ori

pos = np.array([[0.25, 0.14, 0.05],
                [0.25, 0.14, 0.05],
                [0.25, 0.14, 0.05],
                [0.25, 0.14, 0.05],
                [0.25, 0.14, 0.05],
                [0.25, 0.14, 0.05],
                [0.25, 0.14, 0.05],
                [0.25, 0.14, 0.05]])
ori = np.array([[0, np.pi / 2, 0],
                [0, np.pi / 2, 0],
                [0, np.pi / 2, 0],
                [0, np.pi / 2, 0],
                [0, np.pi / 2, 0],
                [0, np.pi / 2, 0],
                [0, np.pi / 2, 0],
                [0, np.pi / 2, 0]])

for i in tqdm(range(1)):
    print('pos\n', pos[0])
    motor_angle = inverse_kinematic(pos, ori)
    print('motor angle\n', motor_angle[0])
    pos, _ = forward_kinematic(motor_angle)
    print('forward pos\n', pos[0])
