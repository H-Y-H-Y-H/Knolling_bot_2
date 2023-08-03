import matplotlib.pyplot as plt
import numpy as np

data_path = '../../knolling_dataset/nn_data_802/'

show = 'xyz'

if show == 'xyz':

    cmd_xyz = np.loadtxt(data_path + 'cmd_xyz_nn.txt')[:, :]
    real_xyz = np.loadtxt(data_path + 'real_xyz_nn.txt')[:, :]
    x = np.arange(len(cmd_xyz))

    print(x)

    plt.figure(figsize=(14, 8))

    plt.subplot(1, 3, 1)
    plt.title("X axis")
    plt.grid(True)
    plt.ylabel('Distance')
    plt.plot(x, cmd_xyz[:, 0], label='cmd')
    plt.plot(x, real_xyz[:, 0], label='real')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.title("Y axis")
    plt.grid(True)
    plt.ylabel('Distance')
    plt.plot(x, cmd_xyz[:, 1], label='cmd')
    plt.plot(x, real_xyz[:, 1], label='real')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.title("Z axis")
    plt.grid(True)
    plt.ylabel('Distance')
    plt.plot(x, cmd_xyz[:, 2], label='cmd')
    plt.plot(x, real_xyz[:, 2], label='real')
    plt.legend()

    plt.savefig(data_path + 'xyz_analysis.png')
    plt.show()

if show == 'motor':
    cmd_motor = np.loadtxt(data_path + 'cmd_nn.txt')[40:80, :]
    real_motor = np.loadtxt(data_path + 'real_nn.txt')[40:80, :]
    x = np.arange(len(cmd_motor))

    print(x)

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 3, 1)
    plt.title("Motor 1")
    plt.grid(True)
    plt.ylabel('Distance')
    plt.plot(x, cmd_motor[:, 0], label='cmd')
    plt.plot(x, real_motor[:, 0], label='real')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.title("Motor 2")
    plt.grid(True)
    plt.ylabel('Distance')
    plt.plot(x, cmd_motor[:, 1], label='cmd')
    plt.plot(x, real_motor[:, 1], label='real')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.title("Motor 3")
    plt.grid(True)
    plt.ylabel('Distance')
    plt.plot(x, cmd_motor[:, 2], label='cmd')
    plt.plot(x, real_motor[:, 2], label='real')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.title("Motor 4")
    plt.grid(True)
    plt.ylabel('Distance')
    plt.plot(x, cmd_motor[:, 3], label='cmd')
    plt.plot(x, real_motor[:, 3], label='real')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.title("Motor 5")
    plt.grid(True)
    plt.ylabel('Distance')
    plt.plot(x, cmd_motor[:, 4], label='cmd')
    plt.plot(x, real_motor[:, 4], label='real')
    plt.legend()

    plt.savefig(data_path + 'motor_analysis.png')
    plt.show()