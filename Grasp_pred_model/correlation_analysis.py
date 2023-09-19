import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

PATH = '../models/LSTM_918_0/'

data_1_baseline = np.loadtxt(PATH + 'model_data_labels_1.txt')
data_2_conf097 = np.loadtxt(PATH + 'model_data_labels_2.txt')
data_3_conf095 = np.loadtxt(PATH + 'model_data_labels_3.txt')
model_threshold = data_1_baseline[1:49, 0]

precision_1 = data_1_baseline[1:49, 2].reshape(-1, 1)
precision_2 = data_2_conf097[1:49, 2].reshape(-1, 1)
precision_3 = data_3_conf095[1:49, 2].reshape(-1, 1)


data_1 = pd.DataFrame(np.concatenate((precision_1, precision_2), axis=1), columns=['baseline', 'label_097'])
r,p = stats.pearsonr(data_1.baseline,data_1.label_097)  # 相关系数和P值
print('baseline和conf097的相关系数r为 = %6.3f，p值为 = %6.3f'%(r,p))

data_2 = pd.DataFrame(np.concatenate((precision_1, precision_3), axis=1), columns=['baseline', 'label_095'])
r,p = stats.pearsonr(data_2.baseline,data_2.label_095)  # 相关系数和P值
print('baseline和conf095的相关系数r为 = %6.3f，p值为 = %6.3f'%(r,p))

plt.figure(figsize=(24, 24))
plt.subplot(2, 2, 1)
plt.scatter(data_1.baseline, data_1.label_097,color="blue")  # 散点图绘制
plt.grid()  # 显示网格线

plt.subplot(2, 2, 2)
plt.plot(model_threshold, precision_1, label='baseline_precision')
plt.plot(model_threshold, precision_2, label='conf097_precision')
plt.xlabel('model_threshold')
plt.title('analysis of model precision')
plt.legend()

plt.subplot(2, 2, 3)
plt.scatter(data_2.baseline, data_2.label_095,color="blue")  # 散点图绘制
plt.grid()  # 显示网格线

plt.subplot(2, 2, 4)
plt.plot(model_threshold, precision_1, label='baseline_precision')
plt.plot(model_threshold, precision_3, label='conf095_precision')
plt.xlabel('model_threshold')
plt.title('analysis of model precision')
plt.legend()

plt.suptitle('compare 1:baseline, 2:conf0.97, 3:conf0.95')
plt.savefig(PATH + 'compare_precision.png')
plt.show()