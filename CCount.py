import numpy as np
import matplotlib.pyplot as plt
import timeit
import gzip, json, urllib.request
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import *
from keras.utils import plot_model
from IPython.display import display, Image
from keras.callbacks import EarlyStopping
from keras import regularizers


# =============================================== Load CIFAR-10 Data-set =============================================

raw_data = np.genfromtxt('/Users/Ido/Documents/MATLAB/Valerann/TU/DataSet_17-12-2018_S10.csv', delimiter=',')

test_end = 5500
val_start = 5500
val_end = 6000

np.random.shuffle(raw_data)


train_var = np.amax(raw_data[0:test_end, 0:600]) - np.amin(raw_data[0:test_end, 0:600])

train_mean = np.mean(raw_data[0:test_end, 0:600].reshape(-1, 600, 1), axis=0)/train_var

X_ = raw_data[0:test_end, 0:600].reshape(-1, 600, 1)/train_var - train_mean
X_test = raw_data[val_start:val_end, 0:600].reshape(-1, 600, 1)/train_var - train_mean

y_ = raw_data[0:test_end, 600].astype(int)
y_test = raw_data[val_start:val_end, 600].astype(int)

classes = max(max(y_), max(y_test)) + 1

y_ = keras.utils.to_categorical(y_, num_classes=classes)
y_test = keras.utils.to_categorical(y_test, num_classes=classes)


# ================================================== Build model ==================================================


input_shape = [1, 600, 1]
num_classes = int(classes)
batch_size = 128
epochs = 500


model = Sequential()

model.add(Conv1D(32, kernel_size=10, activation='relu', padding='same', input_shape=(600, 1)))
model.add(Conv1D(40, kernel_size=5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling1D(pool_size=5))
model.add(Dropout(0.25))

model.add(Conv1D(128, kernel_size=5, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling1D(pool_size=5))
# model.add(Conv1D(128, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
# model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.05)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])

model.summary()

history = model.fit(
    X_, y_,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    shuffle=True,
    validation_data=(X_test, y_test))

model.save('countModel_S3.h5')



