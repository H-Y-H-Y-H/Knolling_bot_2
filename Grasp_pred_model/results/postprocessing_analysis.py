import matplotlib.pyplot as plt
import numpy as np

path = './LSTM_703/'
train_loss = np.loadtxt(path + 'train_loss_LSTM.txt')
valid_loss = np.loadtxt(path + 'valid_loss_LSTM.txt')
print('this is min of train loss:', np.min(train_loss))
print('this is min of valid loss:', np.min(valid_loss))
x = np.arange(len(valid_loss))
plt.plot(x, valid_loss, label='valid_loss')
plt.plot(x, train_loss, label='train_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()