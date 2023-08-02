from function import *
import numpy as np

data_path = '../../knolling_dataset/nn_data_731/'

cmd_motor = np.loadtxt(data_path + 'cmd_nn.txt')
real_motor = np.loadtxt(data_path + 'real_nn.txt')

for i in range(len(cmd_motor)):
    real_xyz_test = np.asarray(cmd2rad(real_tarpos2cmd(real_motor[i])), dtype=np.float32)