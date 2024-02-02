import numpy as np
import torch
import cv2

# data_root = "../../../knolling_dataset/learning_data_1019/"
# data_raw = np.loadtxt(data_root + 'num_30_after_0.txt')[:, :60].reshape(-1, 10, 6)
#
# tar_lw = torch.tensor(data_raw[:, :, 2:4])
# tar_pos = torch.tensor(data_raw[:, :, :2])
# tar_cls = torch.tensor(data_raw[:, :, 5]).unsqueeze(2)
# num_cls = torch.unique(tar_cls.flatten()).numpy().astype(np.int32)
# mm2px = 530 / (0.34 * 3)
#
# points_topleft = 0
# points_downright = 0
#
# tensor = torch.cat(((tar_pos[:, :, 0] - tar_lw[:, :, 0] / 2),
#                      (tar_pos[:, :, 1] - tar_lw[:, :, 1] / 2),
#                      (tar_pos[:, :, 0] + tar_lw[:, :, 0] / 2),
#                      (tar_pos[:, :, 1] - tar_lw[:, :, 1] / 2),
#                      (tar_pos[:, :, 0] + tar_lw[:, :, 0] / 2),
#                      (tar_pos[:, :, 1] + tar_lw[:, :, 1] / 2),
#                      (tar_pos[:, :, 0] - tar_lw[:, :, 0] / 2),
#                      (tar_pos[:, :, 1] + tar_lw[:, :, 1] / 2)), dim=1)
#
# tar_pos_px = tar_pos * mm2px
# tar_lw_px = tar_lw * mm2px
#
#
# total_min = []
# for i in num_cls:
#     cls_index = tar_cls == i
#     cls_mask = cls_index & cls_index.transpose(1, 2)
#
#     tar_pos_x_low = (tar_pos[:, :, 0] - tar_lw[:, :, 0] / 2).unsqueeze(2)
#     tar_pos_x_high = (tar_pos[:, :, 0] + tar_lw[:, :, 0] / 2).unsqueeze(2)
#     tar_pos_y_low = (tar_pos[:, :, 1] - tar_lw[:, :, 1] / 2).unsqueeze(2)
#     tar_pos_y_high = (tar_pos[:, :, 1] + tar_lw[:, :, 1] / 2).unsqueeze(2)
#
#     x_distance = tar_pos_x_low - tar_pos_x_high.transpose(1, 2)
#     y_distance = tar_pos_y_low - tar_pos_y_high.transpose(1, 2)
#     x_mask = x_distance <= 0
#     y_mask = y_distance <= 0
#     x_distance.masked_fill_(x_mask, 100)
#     y_distance.masked_fill_(y_mask, 100)
#
#     # x_min = torch.min(x_distance, dim=2)[0]
#     # y_min = torch.min(y_distance, dim=2)[0]
#
#     x_temp = x_distance[cls_mask]
#     x_gap = x_temp[x_temp < 100]
#     y_temp = y_distance[cls_mask]
#     y_gap = y_temp[y_temp < 100]
#
#     total_min.append(torch.min(x_gap))
#     total_min.append(torch.min(y_gap))
#
# total_min = torch.mean(torch.tensor(total_min))
#
# print('here')
#
# # if losstype == 0:
# #
# #     tar_total_min = []
# #     pred_total_min = []
# #     for i in num_cls:
# #         cls_index = tar_cls == i
# #         cls_mask = cls_index & cls_index.transpose(1, 2)
# #
# #         tar_x_distance = tar_pos_x_low - tar_pos_x_high.transpose(1, 2)
# #         tar_y_distance = tar_pos_y_low - tar_pos_y_high.transpose(1, 2)
# #         tar_x_mask = tar_x_distance <= 0
# #         tar_y_mask = tar_y_distance <= 0
# #         tar_x_distance.masked_fill_(tar_x_mask, 100)
# #         tar_y_distance.masked_fill_(tar_y_mask, 100)
# #
# #         tar_x_temp = tar_x_distance[cls_mask]
# #         tar_x_gap = tar_x_temp[tar_x_temp < 100]
# #         tar_y_temp = tar_y_distance[cls_mask]
# #         tar_y_gap = tar_y_temp[tar_y_temp < 100]
# #         if len(tar_x_gap) == 0 and len(tar_y_gap) == 0:
# #             print('x none and y none')
# #             tar_x_gap = torch.tensor(0.0001)
# #             tar_y_gap = torch.tensor(0.0001)
# #         elif len(tar_x_gap) == 0 and len(tar_y_gap) != 0:
# #             print('x none')
# #             tar_x_gap = torch.clone(tar_y_gap)
# #         elif len(tar_x_gap) != 0 and len(tar_y_gap) == 0:
# #             print('y none')
# #             tar_y_gap = torch.clone(tar_x_gap)
# #
# #         # if len(tar_x_temp[tar_x_temp < 100]) == 0 or len(tar_y_temp[tar_y_temp < 100]) == 0:
# #         #     print('here')
# #         tar_total_min.append(torch.min(tar_x_gap))
# #         tar_total_min.append(torch.min(tar_y_gap))
# #
# #         pred_x_distance = pred_pos_x_low - pred_pos_x_high.transpose(1, 2)
# #         pred_y_distance = pred_pos_y_low - pred_pos_y_high.transpose(1, 2)
# #         pred_x_mask = pred_x_distance <= 0
# #         pred_y_mask = pred_y_distance <= 0
# #         pred_x_distance.masked_fill_(pred_x_mask, 100)
# #         pred_y_distance.masked_fill_(pred_y_mask, 100)
# #
# #         # pred_x_temp = pred_x_distance[cls_mask]
# #         # if len(pred_x_temp[pred_x_temp < 100]) == 0:
# #         #     # pred_x_gap = torch.zeros(len(pred_x_temp))
# #         #     pred_x_gap = 0.0001
# #         #     print('x none')
# #         # else:
# #         #     pred_x_gap = pred_x_temp[pred_x_temp < 100]
# #         #
# #         # pred_y_temp = pred_y_distance[cls_mask]
# #         # if len(pred_y_temp[pred_y_temp < 100]) == 0:
# #         #     # pred_y_gap = torch.zeros(len(pred_y_temp))
# #         #     pred_y_gap = 0.0001
# #         #     print('y none')
# #         # else:
# #         #     pred_y_gap = pred_y_temp[pred_y_temp < 100]
# #
# #         pred_x_temp = pred_x_distance[cls_mask]
# #         pred_y_temp = pred_y_distance[cls_mask]
# #         pred_x_gap = pred_x_temp[pred_x_temp < 100]
# #         pred_y_gap = pred_y_temp[pred_y_temp < 100]
# #         if len(pred_x_gap) == 0 and len(pred_y_gap) == 0:
# #             print('x none and y none')
# #             pred_x_gap = torch.tensor(0.0001)
# #             pred_y_gap = torch.tensor(0.0001)
# #         elif len(pred_x_gap) == 0 and len(pred_y_gap) != 0:
# #             print('x none')
# #             pred_x_gap = torch.clone(pred_y_gap)
# #         elif len(pred_x_gap) != 0 and len(pred_y_gap) == 0:
# #             print('y none')
# #             pred_y_gap = torch.clone(pred_x_gap)
# #
# #         try:
# #             pred_total_min.append(torch.min(pred_x_gap))
# #             pred_total_min.append(torch.min(pred_y_gap))
# #         except:
# #             print('this is pred_x_gap', pred_x_gap)
# #             print('this is pred_y_gap', pred_y_gap)
# #             print('this is len x', len(pred_x_temp))
# #             print('this is len y', len(pred_y_temp))
# #
# #     total_pred_min = torch.mean(torch.tensor(pred_total_min))
#
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# print(torch.min(torch.tensor(0)))
#
# def custom_function(x):
#     if x > 0.015:
#         return 0 + (x * 200 / 3 - 1) * 0.2
#     else:
#         return np.abs(np.log(0)) + 0
#
# # Generate x values for the plot
# x_values = np.linspace(0.001, 0.05, 1000)  # Adjust the range and number of points as needed
# y_values = [custom_function(x) for x in x_values]
#
# # Plot the function
# plt.plot(x_values, y_values, label='f(x)')
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.title('Custom Function f(x) with Value 0 at x = 0.015')
# plt.grid(True)
# plt.legend()
# plt.show()
#
# print(torch.log(torch.tensor(-1)))

import numpy as np

# Given array
Arr = np.array([3, 4, 6, 2, 7, 9])

# Number of rows in the mask
num_rows = len(Arr)

# Number of columns in the mask (maximum value in Arr)
num_cols = np.max(Arr)

# Initialize the mask with zeros
mask = np.zeros((num_rows, num_cols), dtype=int)

# Set the first n elements in each row to 1
for i in range(num_rows):
    mask[i, :Arr[i]] = 1

print(mask)

import numpy as np

# Example 2D numpy array
array_2d = np.array([[1, 2, 0],
                     [4, 5, 6],
                     [0, 0, 0],
                     [7, 8, 9]])

# Count rows where all elements are non-zero
rows_with_all_non_zero = np.sum(np.all(array_2d != 0, axis=1))

print("Number of rows with all non-zero values:", rows_with_all_non_zero)




