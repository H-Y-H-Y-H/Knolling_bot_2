import numpy as np
import cv2

# mm2px = 530 / 0.34
#
# target_path = '../../../knolling_dataset/learning_data_1019/'
# out_num_scene = 0
#
# for i in range(12):
#     data = np.loadtxt(target_path + 'num_30_after_%d.txt' % i).reshape(50000, 30, 6)
#     data_pos = data[:, :, :2]
#     data_lw = data[:, :, 2:4]
#     data_y_far = (data_pos[:, :, 1] + data_lw[:, :, 1] / 2) * mm2px
#     data_x_far = (data_pos[:, :, 0] + data_lw[:, :, 0] / 2) * mm2px
#     x_temp = data_x_far > 480
#     y_temp = data_y_far > 640
#     x_mask = np.any(x_temp, axis=1)
#     y_mask = np.any(y_temp, axis=1)
#
#     out_num_scene += len(np.union1d(np.where(x_mask)[0], np.where(y_mask)[0]))
#
#     print('here')

# def classify_by_value_indices(arr):
#     unique_elements = np.unique(arr)
#     classification_indices = []
#     for element in unique_elements:
#         indices = np.where(arr == element)[0]  # Get indices of each element
#         classification_indices.append(list(indices))
#     return classification_indices
#
# # Example usage
# arr = np.array([1, 2, 2, 3, 1, 4, 4, 4, 5])
# classified_arr = classify_by_value_indices(arr)
# print(classified_arr)

import numpy as np

# Given NumPy array
arr = np.array([0, 2, 1, 1, 3])

# Given dictionary
dict_map = {'0': 'bad', '1': 'well', '2': 'good', '3': 'perfect'}

# Converting dictionary keys to integers
dict_map = {int(k): v for k, v in dict_map.items()}

# Mapping array values to dictionary values
mapped_values = [dict_map[value] for value in arr]

print(mapped_values)

