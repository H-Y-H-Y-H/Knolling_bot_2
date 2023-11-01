import numpy as np
import cv2

mm2px = 530 / 0.34

target_path = '../../../knolling_dataset/learning_data_1019/'
out_num_scene = 0

for i in range(12):
    data = np.loadtxt(target_path + 'num_30_after_%d.txt' % i).reshape(50000, 30, 6)
    data_pos = data[:, :, :2]
    data_lw = data[:, :, 2:4]
    data_y_far = (data_pos[:, :, 1] + data_lw[:, :, 1] / 2) * mm2px
    data_x_far = (data_pos[:, :, 0] + data_lw[:, :, 0] / 2) * mm2px
    x_temp = data_x_far > 480
    y_temp = data_y_far > 640
    x_mask = np.any(x_temp, axis=1)
    y_mask = np.any(y_temp, axis=1)

    out_num_scene += len(np.union1d(np.where(x_mask)[0], np.where(y_mask)[0]))

    print('here')