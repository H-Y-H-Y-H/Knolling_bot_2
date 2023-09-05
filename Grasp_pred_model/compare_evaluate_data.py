import numpy as np
import matplotlib.pyplot as plt

PATH = '../models/LSTM_829_1_heavy_dropout0/'

data_1_high_conf = np.loadtxt(PATH + 'model_data_labels1.txt')
data_4_rdm_pos = np.loadtxt(PATH + 'model_data_labels4.txt')
model_threshold = data_1_high_conf[:, 0]

plt.figure(figsize=(24, 12))
plt.subplot(1, 3, 1)
plt.plot(model_threshold, data_1_high_conf[:, 1], label='1_recall')
plt.plot(model_threshold, data_4_rdm_pos[:, 1], label='4_recall')
plt.xlabel('model_threshold')
plt.title('analysis of model recall')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(model_threshold, data_1_high_conf[:, 2], label='1_precision')
plt.plot(model_threshold, data_4_rdm_pos[:, 2], label='4_precision')
plt.xlabel('model_threshold')
plt.title('analysis of model precision')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(model_threshold, data_1_high_conf[:, 3], label='1_accuracy')
plt.plot(model_threshold, data_4_rdm_pos[:, 3], label='4_accuracy')
plt.xlabel('model_threshold')
plt.title('analysis of model accuracy')
plt.legend()
plt.suptitle('compare 1:high conf, 4:baseline')

plt.savefig(PATH + 'compare_1_4.png')
plt.show()

