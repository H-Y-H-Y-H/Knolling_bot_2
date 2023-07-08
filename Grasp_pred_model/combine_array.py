import numpy as np

num_array = 3
array_1 = np.random.rand(5, 11)
array_2 = np.random.rand(7, 11)
array_3 = np.random.rand(2, 11)
num_len = np.array([array_1.shape[0], array_2.shape[0], array_3.shape[0]])

total_array = np.concatenate((array_1, array_2, array_3), axis=0)
print(array_1)

start_index = 0
array_after = []
for i in range(num_array):
    array_after.append(total_array[start_index:num_len[i] + start_index, :])

print(array_after)