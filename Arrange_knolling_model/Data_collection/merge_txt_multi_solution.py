import numpy as np
from tqdm import tqdm

configuration = [[2, 1],
                 [1, 2],
                 [1, 1]]
num = 5

start_evaluations = 0
end_evaluations =   100
step_num = 10
solution_num = 4
save_point = np.linspace(int((end_evaluations - start_evaluations) / step_num + start_evaluations), end_evaluations, step_num)

# def merge():
#
#     target_path = '../../knolling_dataset/learning_data_1013/'
#
#     for cfg in range(len(configuration)):
#
#         save_path = target_path + 'cfg_%s/' % cfg
#         for m in range(solution_num):
#
#             before_path = target_path + 'cfg_%s/' % cfg + 'labels_before_0/'
#             after_path = target_path + 'cfg_%s/' % cfg + 'labels_after_%s/' % m
#
#             total_data = []
#             for s in save_point:
#                 data = np.loadtxt(after_path + 'num_%d_%d.txt' % (num, int(s)))
#                 total_data.append(data)
#             total_data = np.asarray(total_data).reshape(-1, num * 5)
#             np.savetxt(save_path + 'num_%d_after_%d.txt' % (num, m), total_data)
#
#             if m == 0:
#                 total_data = []
#                 for s in save_point:
#                     data = np.loadtxt(before_path + 'num_%d_%d.txt' % (num, int(s)))
#                     total_data.append(data)
#                 total_data = np.asarray(total_data).reshape(-1, num * 5)
#                 np.savetxt(save_path + 'num_%d_before_0.txt' % num, total_data)
#                 # total_data = np.asarray(total_data).reshape(-1, num * 5)
#                 # np.savetxt(before_path + 'num_%d.txt' % num, total_data)


def merge(): # after that, the structure of dataset is cfg0_0, cfg0_1, cfg0_2,
                                                     # cfg1_0, cfg1_1, cfg1_2,
                                                     # cfg2_0, cfg2_1, cfg2_2


    for m in range(solution_num):

        target_path = '../../../knolling_dataset/learning_data_1228/'
        after_path = target_path + 'labels_after_%s/' % m
        output_path = target_path + 'num_%d_after_%d.txt' % (num, m)

        total_data = []
        for s in save_point:
            data = np.loadtxt(after_path + 'num_%d_%d.txt' % (num, int(s)))
            total_data.append(data)
        total_data = np.asarray(total_data).reshape(-1, num * 11)
        np.savetxt(output_path, total_data)
        # if m == 0:
        #     total_data = []
        #     for s in save_point:
        #         data = np.loadtxt(before_path + 'num_%d_%d.txt' % (num, int(s)))
        #         total_data.append(data)
        #     total_data = np.asarray(total_data).reshape(-1, num * 5)
        #     np.savetxt(target_path + 'num_%d_before_%d.txt' % (num, m), total_data)

def add():
    base_path = '../../../knolling_dataset/learning_data_817/'
    add_path = '../../knolling_dataset/learning_data_817_add/'
    # for m in tqdm(range(solution_num)):
    #     data_base = np.loadtxt(base_path + 'labels_after_%s/num_%d.txt' % (m, num_range[0]))
    #     data_add = np.loadtxt(add_path + 'labels_after_%s/num_%d.txt' % (m, num_range[0]))
    #     data_new = np.concatenate((data_base, data_add), axis=0)
    #     np.savetxt(base_path + 'labels_after_%s/num_%d_new.txt' % (m, num_range[0]), data_new)
    for m in tqdm(range(1)):
        data_base = np.loadtxt(base_path + 'labels_before_%s/num_%d.txt' % (m, num_range[0]))
        data_add = np.loadtxt(add_path + 'labels_before_%s/num_%d.txt' % (m, num_range[0]))
        data_new = np.concatenate((data_base, data_add), axis=0)
        np.savetxt(base_path + 'labels_before_%s/num_%d_new.txt' % (m, num_range[0]), data_new)

def add_noise():

    for m in range(solution_num):

        target_path = '../../../knolling_dataset/learning_data_1013/'
        after_path = target_path + 'labels_after_%s/' % m

        raw_data = np.loadtxt(after_path + 'num_5.txt')
        noise_mask = np.random.rand(5, 2) * 0.005

        new_data = []
        for i in range(len(raw_data)):
            one_img_data = raw_data[i].reshape(5, -1)
            one_img_data[:, 2:4] += noise_mask
            new_data.append(one_img_data.reshape(-1, ))
        new_data = np.asarray(new_data)
        np.savetxt(after_path + 'num_5_new.txt', new_data, fmt='%.05f')
        pass

merge()
# merge_test()
# add()
# add_noise()