import numpy as np

from keras.utils import np_utils


train_label = [0, 1, 2, 0, 4, 3, 6, 1, 4, 6]
print(np.asarray(train_label).shape)
train_label = np_utils.to_categorical(train_label, num_classes=7)
print(train_label)
print(np.asarray(train_label).shape)
