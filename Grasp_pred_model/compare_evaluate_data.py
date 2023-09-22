import numpy as np
import matplotlib.pyplot as plt

PATH = '../models/LSTM_918_0/'

data_1_baseline = np.loadtxt(PATH + 'model_data_labels_1.txt')
data_2_conf097 = np.loadtxt(PATH + 'model_data_labels_2.txt')
data_3_conf095 = np.loadtxt(PATH + 'model_data_labels_3.txt')
data_4_conf100 = np.loadtxt(PATH + 'model_data_labels_4.txt')
model_threshold = data_1_baseline[:, 0]

plt.figure(figsize=(24, 12))
plt.subplot(1, 3, 1)
plt.plot(model_threshold, data_1_baseline[:, 1], label='1_recall')
plt.plot(model_threshold, data_2_conf097[:, 1], label='2_recall')
plt.plot(model_threshold, data_3_conf095[:, 1], label='3_recall')
plt.plot(model_threshold, data_4_conf100[:, 1], label='4_recall')
plt.xlabel('model_threshold')
plt.title('analysis of model recall')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(model_threshold, data_1_baseline[:, 2], label='1_precision')
plt.plot(model_threshold, data_2_conf097[:, 2], label='2_precision')
plt.plot(model_threshold, data_3_conf095[:, 2], label='3_precision')
plt.plot(model_threshold, data_4_conf100[:, 2], label='4_precision')
plt.xlabel('model_threshold')
plt.title('analysis of model precision')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(model_threshold, data_1_baseline[:, 3], label='1_accuracy')
plt.plot(model_threshold, data_2_conf097[:, 3], label='2_accuracy')
plt.plot(model_threshold, data_3_conf095[:, 3], label='3_accuracy')
plt.plot(model_threshold, data_4_conf100[:, 3], label='4_accuracy')
plt.xlabel('model_threshold')
plt.title('analysis of model accuracy')
plt.legend()
plt.suptitle('compare 1:baseline, 2:conf0.97, 3:conf0.95, 4:conf1.00')

plt.savefig(PATH + 'compare.png')
plt.show()

