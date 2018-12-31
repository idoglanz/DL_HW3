import numpy as np
import matplotlib.pyplot as plt
import timeit
import gzip, json, urllib.request
import keras
from keras.models import Sequential
from keras.layers import *
from keras.utils import plot_model
from IPython.display import display, Image


# =========================================== Data loading utility functions  ==========================================


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_CIFAR(batches):
    X_train = np.empty((0, 3072))
    y_train = np.empty(0)
    for batch in batches:
        filename = './cifar-10-batches-py/data_batch_%d' % batch
        batch_temp = unpickle(filename)
        X_train = np.append(X_train, batch_temp[b'data'], axis=0)
        y_train = np.append(y_train, batch_temp[b'labels'], axis=0)

    print(X_train.shape, len(y_train))
    return X_train, y_train


def print_CIFAR(img_as_vector, label=None):

    R = img_as_vector[0:1024].reshape(32, 32) / 255.0
    G = img_as_vector[1024:2048].reshape(32, 32) / 255.0
    B = img_as_vector[2048:].reshape(32, 32) / 255.0

    img = np.dstack((R, G, B))
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)
    ax.imshow(img)
    ax.imshow(img, interpolation='bicubic')
    if label is not None:
        ax.set_title('Category = ' + str(label), fontsize=15)
    plt.show()


# =============================================== Load CIFAR-10 Data-set =============================================


label_name = unpickle('./cifar-10-batches-py/batches.meta')[b'label_names']

[X_train, y_train] = load_CIFAR([1, 2,3, 4, 5])

test_set = unpickle('./cifar-10-batches-py/test_batch')

print('Data loaded successfully')

img_rows, img_cols, img_layers = 32, 32, 3

X_test = test_set[b'data']
y_test = test_set[b'labels']

# pic = 3

# print_CIFAR(X_test[pic, :], label_name[y_test[pic]])

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_layers) / 255
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_layers) / 255

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

print(X_train.shape)


# ================================================== Build model ==================================================


input_shape = [32, 32, 3]
num_classes = 10

batch_size = 128
epochs = 10

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
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

