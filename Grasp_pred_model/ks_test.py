import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

PATH = '../models/LSTM_918_0/'

data_1_baseline = np.loadtxt(PATH + 'model_data_labels_1.txt')
data_2_conf097 = np.loadtxt(PATH + 'model_data_labels_2.txt')
data_3_conf095 = np.loadtxt(PATH + 'model_data_labels_3.txt')
model_threshold = data_1_baseline[:, 0]

recall_1 = data_1_baseline[:, 1]
recall_2 = data_2_conf097[:, 1]
recall_3 = data_3_conf095[:, 1]

precision_1 = data_1_baseline[:, 2]
precision_2 = data_2_conf097[:, 2]
precision_3 = data_3_conf095[:, 2]

accuracy_1 = data_1_baseline[:, 3]
accuracy_2 = data_2_conf097[:, 3]
accuracy_3 = data_3_conf095[:, 3]

KS_recall_12 = stats.ks_2samp(recall_1, recall_2)
KS_recall_13 = stats.ks_2samp(recall_1, recall_3)
KS_precision_12 = stats.ks_2samp(precision_1, precision_2)
KS_precision_13 = stats.ks_2samp(precision_1, precision_3)
KS_accuracy_12 = stats.ks_2samp(accuracy_1, accuracy_2)
KS_accuracy_13 = stats.ks_2samp(accuracy_1, accuracy_3)

print(f'the statistic of KS_recall_12 is:{KS_recall_12.statistic}, the pvalue is:{KS_recall_12.pvalue}')
print(f'the statistic of KS_recall_13 is:{KS_recall_13.statistic}, the pvalue is:{KS_recall_13.pvalue}')
print(f'the statistic of KS_precision_12 is:{KS_precision_12.statistic}, the pvalue is:{KS_precision_12.pvalue}')
print(f'the statistic of KS_precision_13 is:{KS_precision_13.statistic}, the pvalue is:{KS_precision_13.pvalue}')
print(f'the statistic of KS_accuracy_12 is:{KS_accuracy_12.statistic}, the pvalue is:{KS_accuracy_12.pvalue}')
print(f'the statistic of KS_accuracy_13 is:{KS_accuracy_13.statistic}, the pvalue is:{KS_accuracy_13.pvalue}')