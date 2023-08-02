import matplotlib.pyplot as plt
import numpy as np

data_path = '../../knolling_dataset/nn_data_801/'

cmd_xyz = np.loadtxt(data_path + 'cmd_xyz_nn.txt')[:100, :]
real_xyz = np.loadtxt(data_path + 'real_xyz_nn.txt')[:100, :]
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

