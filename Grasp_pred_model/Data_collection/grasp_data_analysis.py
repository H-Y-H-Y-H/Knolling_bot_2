import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

import seaborn as sns
import matplotlib.pyplot as plt


def heatmap(path):

    data = pd.read_csv(path)
    data.head()

    plt.figure(figsize=(12, 10))
    print(data.corr())
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=.5, fmt=".2f")
    plt.show()

def EDA(path):

    df = pd.read_csv(path)
    df.iloc[:, [1, 2, 3, 4, 5, 6, 7]].plot(kind='box', subplots=True, layout=(1, 7), whis=1.5)
    print('this is the description of the dataset\n', df.describe())
    plt.show()

    # x pos and y pos and z pos
    df_xy = df.iloc[:, 1:4]
    df_xy.plot(kind='hist', subplots=True, layout=(1, 3), bins=20)
    plt.show()

    # length and width
    df_lw = df.iloc[:, 4:6]
    df_lw.plot(kind='hist', subplots=True, layout=(1, 2), bins=20)
    plt.show()

def statistics(path):
    df = pd.read_csv(path)
    grasp_1_frame = df[df['grasp_flag'] == 1]
    grasp_0_frame = df[df['grasp_flag'] == 0]
    print('grasp_flag 1 summary')
    print(grasp_1_frame.iloc[:, [0, -1]].describe())
    print('grasp_flag 0 summary')
    print(grasp_0_frame.iloc[:, [0, -1]].describe())
    # print(df['grasp_flag'])


def r_square(path):

    pass

if __name__ == '__main__':

    data_root = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/'
    data_path = data_root + 'grasp_pile_707_laptop/grasp_data.csv'
    heatmap(data_path)
    EDA(data_path)
    statistics(data_path)
