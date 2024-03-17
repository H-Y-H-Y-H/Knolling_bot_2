import numpy as np

num_obj = 10
info_per_obj = 7

gt_data = np.loadtxt('demo_999/test24(11).txt').reshape(num_obj, info_per_obj)
# pred_data = np.loadtxt('num_10_pred.txt')[905].reshape(num_obj, info_per_obj)
print('here')