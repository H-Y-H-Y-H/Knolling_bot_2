import numpy as np
import torch

data_root = "../../../knolling_dataset/learning_data_1019/"
data_raw = np.loadtxt(data_root + 'num_30_after_0.txt')[:8].reshape(8, 30, 6)

tar_lw = torch.tensor(data_raw[:, :, 2:4])
tar_pos = torch.tensor(data_raw[:, :, :2])
tar_cls = torch.tensor(data_raw[:, :, 5]).unsqueeze(2)
num_cls = torch.unique(tar_cls.flatten()).numpy().astype(np.int32)

total_cls_mask = []
for i in num_cls:
    cls_index = tar_cls == i
    total_cls_mask.append(cls_index & cls_index.transpose(1, 2))

# cls_0 = tar_cls == 0
# cls_1 = tar_cls == 1
# cls_0_mask = cls_0 & cls_0.transpose(1, 2)
# cls_1_mask = cls_1 & cls_1.transpose(1, 2)

tar_pos_x_low = (tar_pos[:, :, 0] - tar_lw[:, :, 0] / 2).unsqueeze(2)
tar_pos_x_high = (tar_pos[:, :, 0] + tar_lw[:, :, 0] / 2).unsqueeze(2)
tar_pos_y_low = (tar_pos[:, :, 1] - tar_lw[:, :, 1] / 2).unsqueeze(2)
tar_pos_y_high = (tar_pos[:, :, 1] + tar_lw[:, :, 1] / 2).unsqueeze(2)

tar_pos_x_low_demo = tar_pos_x_low.numpy()
tar_pos_x_high_demo = tar_pos_x_high.numpy()
tar_pos_y_low_demo = tar_pos_y_low.numpy()
tar_pos_y_high_demo = tar_pos_y_high.numpy()
tar_cls_demo = tar_cls.numpy()

x_distance = tar_pos_x_low - tar_pos_x_high.transpose(1, 2)
y_distance = tar_pos_y_low - tar_pos_y_high.transpose(1, 2)
x_mask = x_distance <= 0
y_mask = y_distance <= 0
x_distance.masked_fill_(x_mask, 100)
y_distance.masked_fill_(y_mask, 100)

x_min = torch.min(x_distance, dim=2)[0]
y_min = torch.min(y_distance, dim=2)[0]

x_temp = x_distance[total_cls_mask[0]]

import torch

# Example tensors A and B
A = torch.randn(8, 30, 20)  # (batch_size, num_objects, num_features)
B = torch.randint(0, 3, (8, 30, 1))  # (batch_size, num_objects, 1) with classes 0, 1, 2

# Create masks for each class
class_0_mask = B == 0
class_1_mask = B == 1
class_2_mask = B == 2

# Use masks to divide tensor A
class_0_data = A[class_0_mask.expand(-1, -1, 20)].view(8, -1, 20)
class_1_data = A[class_1_mask.expand(-1, -1, 20)].view(8, -1, 20)
class_2_data = A[class_2_mask.expand(-1, -1, 20)].view(8, -1, 20)

# class_0_data, class_1_data, and class_2_data now contain the data from tensor A
# corresponding to classes 0, 1, and 2 specified in tensor B, respectively.

