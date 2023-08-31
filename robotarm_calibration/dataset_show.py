import matplotlib.pyplot as plt
import numpy as np

data_path = '../../knolling_dataset/data_805/'

show = 'xyz'

if show == 'xyz':

    cmd_xyz = np.loadtxt(data_path + 'cmd_xyz_nn.txt')[35:, :]
    real_xyz = np.loadtxt(data_path + 'real_xyz_nn.txt')[35:, :]
    tar_xyz = np.loadtxt(data_path + 'tar_xyz_nn.txt')[35:, :]
    error_xyz = np.loadtxt(data_path + 'error_xyz_nn.txt')[35:, :]
    x = np.arange(len(cmd_xyz))

    print(x)

    plt.figure(figsize=(14, 8))

    plt.subplot(1, 3, 1)
    plt.title("X axis")
    plt.grid(True)
    plt.ylabel('Distance')
    plt.plot(x, cmd_xyz[:, 0], label='cmd')
    plt.plot(x, real_xyz[:, 0], label='real')
    plt.plot(x, tar_xyz[:, 0], label='tar')
    plt.plot(x, error_xyz[:, 0], label='error')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.title("Y axis")
    plt.grid(True)
    plt.ylabel('Distance')
    plt.plot(x, cmd_xyz[:, 1], label='cmd')
    plt.plot(x, real_xyz[:, 1], label='real')
    plt.plot(x, tar_xyz[:, 1], label='tar')
    plt.plot(x, error_xyz[:, 1], label='error')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.title("Z axis")
    plt.grid(True)
    plt.ylabel('Distance')
    plt.plot(x, cmd_xyz[:, 2], label='cmd')
    plt.plot(x, real_xyz[:, 2], label='real')
    plt.plot(x, tar_xyz[:, 2], label='tar')
    plt.plot(x, error_xyz[:, 2], label='error')
    plt.legend()

    # plt.savefig(data_path + 'xyz_analysis.png')
    plt.show()

if show == 'motor':
    cmd_motor = np.loadtxt(data_path + 'cmd_nn.txt')[35:, :]
    real_motor = np.loadtxt(data_path + 'real_nn.txt')[35:, :]
    tar_motor = np.loadtxt(data_path + 'tar_nn.txt')[35:, :]
    x = np.arange(len(cmd_motor))

    print(x)

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 3, 1)
    plt.title("Motor 1")
    plt.grid(True)
    plt.ylabel('Distance')
    plt.plot(x, cmd_motor[:, 0], label='cmd')
    plt.plot(x, real_motor[:, 0], label='real')
    plt.plot(x, tar_motor[:, 0], label='tar')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.title("Motor 2")
    plt.grid(True)
    plt.ylabel('Distance')
    plt.plot(x, cmd_motor[:, 1], label='cmd')
    plt.plot(x, real_motor[:, 1], label='real')
    plt.plot(x, tar_motor[:, 1], label='tar')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.title("Motor 3")
    plt.grid(True)
    plt.ylabel('Distance')
    plt.plot(x, cmd_motor[:, 1], label='cmd')
    plt.plot(x, real_motor[:, 1], label='real')
    plt.plot(x, tar_motor[:, 1], label='tar')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.title("Motor 4")
    plt.grid(True)
    plt.ylabel('Distance')
    plt.plot(x, cmd_motor[:, 3], label='cmd')
    plt.plot(x, real_motor[:, 3], label='real')
    plt.plot(x, tar_motor[:, 3], label='tar')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.title("Motor 5")
    plt.grid(True)
    plt.ylabel('Distance')
    plt.plot(x, cmd_motor[:, 4], label='cmd')
    plt.plot(x, real_motor[:, 4], label='real')
    plt.plot(x, tar_motor[:, 4], label='tar')
    plt.legend()

    plt.subplot(2, 3, 6)
    plt.title("Motor 6")
    plt.grid(True)
    plt.ylabel('Distance')
    plt.plot(x, cmd_motor[:, 5], label='cmd')
    plt.plot(x, real_motor[:, 5], label='real')
    plt.plot(x, tar_motor[:, 5], label='tar')
    plt.legend()

#     plt.savefig(data_path + 'motor_analysis.png')
    plt.show()

if show == 'error_motor':
    error_motor = np.loadtxt(data_path + 'error_nn.txt')[:, :]
    x = np.arange(len(error_motor))

    print(x)

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 3, 1)
    plt.title("Motor 1")
    plt.grid(True)
    plt.ylabel('Distance')
    plt.plot(x, error_motor[:, 0], label='error')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.title("Motor 2")
    plt.grid(True)
    plt.ylabel('Distance')
    plt.plot(x, error_motor[:, 1], label='cmd')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.title("Motor 3")
    plt.grid(True)
    plt.ylabel('Distance')
    plt.plot(x, error_motor[:, 1], label='cmd')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.title("Motor 4")
    plt.grid(True)
    plt.ylabel('Distance')
    plt.plot(x, error_motor[:, 3], label='cmd')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.title("Motor 5")
    plt.grid(True)
    plt.ylabel('Distance')
    plt.plot(x, error_motor[:, 4], label='cmd')
    plt.legend()

    plt.subplot(2, 3, 6)
    plt.title("Motor 6")
    plt.grid(True)
    plt.ylabel('Distance')
    plt.plot(x, error_motor[:, 5], label='cmd')
    plt.legend()

#     plt.savefig(data_path + 'motor_analysis.png')
    plt.show()

if show == 'error_xyz':

    error_xyz = np.loadtxt(data_path + 'error_xyz_nn.txt')[:, :]
    x = np.arange(len(error_xyz))

    print(x)

    plt.figure(figsize=(14, 8))

    plt.subplot(1, 3, 1)
    plt.title("X axis")
    plt.grid(True)
    plt.ylabel('Distance')
    plt.plot(x, error_xyz[:, 0], label='cmd')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.title("Y axis")
    plt.grid(True)
    plt.ylabel('Distance')
    plt.plot(x, error_xyz[:, 1], label='cmd')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.title("Z axis")
    plt.grid(True)
    plt.ylabel('Distance')
    plt.plot(x, error_xyz[:, 2], label='cmd')
    plt.legend()

#     plt.savefig(data_path + 'xyz_analysis.png')
    plt.show()