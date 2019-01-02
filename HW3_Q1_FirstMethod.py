import numpy as np
from keras.layers import Dense, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Conv2D, Dropout, Concatenate, Input
from keras.layers import Flatten, InputLayer, BatchNormalization, Activation
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.initializers import Constant
from keras.datasets import cifar10
import keras.backend as K
from keras.layers.core import Lambda
from keras import regularizers
from keras import optimizers
import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

num_classes = 10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)


X_train = K.cast_to_floatx(X_train) / 255
X_test = K.cast_to_floatx(X_test) / 255


def normalize(X_train,X_test):
    """
    This function normalize inputs for zero mean and unit variance
    it is used when training a model.
    Input: training set and test set
    Output: normalized training set and test set according to the
    training set statistics.
    """
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train-mean)/(std+1e-7)
    X_test = (X_test-mean)/(std+1e-7)
    return X_train, X_test


X_train, X_test = normalize(X_train, X_test)

input_shape = [32, 32, 3]
num_classes = 10

batch_size = 128
epochs = 150

model = Sequential()

model.add(Conv2D(30, kernel_size=(3, 3), activation='relu', padding='valid', input_shape=(32, 32, 3)))
model.add(Conv2D(30, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(48, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(48, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])

# plot_model(model, to_file='model.png', show_shapes=True, rankdir='TB')
# display(Image(filename='model.png'))

model.summary()

history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(X_test, y_test)
    )


his = history.history
x = list(range(epochs))
y_1 = his['val_acc']
y_2 = his['acc']
plt.plot(x, y_1)
plt.plot(x, y_2)
plt.legend(['validation accuracy', 'training_accuracy'])
plt.show()