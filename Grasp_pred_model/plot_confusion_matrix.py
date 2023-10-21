import matplotlib.pyplot as plt
import numpy as np

# Replace these with your known TP, TN, FP, FN values
tp = 14805
tn = 42015
fp = 8183
fn = 8774

# tp = 21591
# tn = 48889
# fp = 1222
# fn = 1954

path = '../ASSET/models/LSTM_918_0/'
# Create the confusion matrix
confusion_matrix = np.array([[tn, fp], [fn, tp]])

def plot_confusion_matrix(confusion_matrix, title='VB Baseline', cmap=plt.cm.Blues):
    plt.figure(figsize=(6, 4))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1], ["Predicted Negative", "Predicted Positive"])
    plt.yticks([0, 1], ["Actual Negative", "Actual Positive"])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(confusion_matrix[i, j]), horizontalalignment='center', verticalalignment='center')

# Plot the confusion matrix
plot_confusion_matrix(confusion_matrix)
plt.savefig(path + 'yolo_confusion_matrix.png')
plt.show()
