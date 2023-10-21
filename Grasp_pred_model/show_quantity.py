import numpy as np

PATH = '../ASSET/models/LSTM_918_0/'

lstm_loss = np.loadtxt(PATH + 'valid_loss_LSTM.txt')
yolo_loss = np.loadtxt(PATH + 'yolo_loss.txt')

print(lstm_loss)
print(yolo_loss)
print('----------------------')
print('lstm')
print(np.mean(lstm_loss))
print(np.min(lstm_loss))
print(np.max(lstm_loss))
print(np.std(lstm_loss))
print('----------------------')
print('yolo')
print(np.mean(yolo_loss))
print(np.min(yolo_loss))
print(np.max(yolo_loss))
print(np.std(yolo_loss))