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


# =============================================== Load CIFAR-10 Data-set =============================================

raw_data = np.genfromtxt('/Users/Ido/Documents/MATLAB/Valerann/TU/DataSet_17-12-2018_test.csv', delimiter=',')

train_end = 100
val_start = 0
val_end = 600

np.random.shuffle(raw_data)

# X_ = raw_data[0:train_end, 0:600].reshape(-1, 600, 1) - 0.5
X_test = raw_data[val_start:val_end, 0:600].reshape(-1, 600, 1) - 0.5

# y_ = raw_data[0:train_end, 600].astype(int)
y_test = raw_data[val_start:val_end, 600].astype(int)

# y_ = keras.utils.to_categorical(y_)
y_test = keras.utils.to_categorical(y_test)

# ================================================== Load model ==================================================


def print_results(y, y_bar):

    x_axis = np.linspace(1, len(y), len(y))

    ls = plt.figure(1)
    plt.plot(x_axis, y, 'r')
    plt.plot(x_axis, y_bar, 'b')
    plt.legend(["Y", "Y_bar"])
    plt.ylabel('count')
    plt.xlabel('sample')
    plt.title('True Count VS Predicted')
    plt.show(block=True)


model = load_model('countModel_all_2.h5')

model.summary()

y_fit = model.predict(X_test, batch_size=10, verbose=1)
m = y_fit.shape[0]

error = abs(np.argmax(y_test, axis=1) - np.argmax(y_fit, axis=1))
error_mean = np.sum(error)/m


guess = np.zeros((y_fit.shape[0], 1))
guess[np.argmax(y_fit, axis=1) == np.argmax(y_test, axis=1)] = 1
accu = 100*np.sum(guess) / m

var = max(error)
print('Mean Error: %f' % error_mean)
print('Accuracy: %f percent' % accu)
print('Max error: %f' % var)


print_results(np.argmax(y_test, axis=1), np.argmax(y_fit, axis=1))

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adam(lr=0.0001, decay=1e-6),
#               metrics=['accuracy'])

# scores = model.evaluate(x=X_test, y=y_test, batch_size=10, verbose=1, sample_weight=None, steps=None)
#
# print('Loss: %.3f' % scores[0])
# print('Accuracy: %.3f' % scores[1])



