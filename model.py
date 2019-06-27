import keras
from keras.datasets import fashion_mnist
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import print_summary
from keras.optimizers import Adam
from keras.regularizers import l2
import os

# Ucitavanje FashionMNIST skupa podataka
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# print(y_train)

# Koristimo samo deo trening skupa (prvi od 10 fold-ova) radi efikasnosti treninga
skf = StratifiedKFold(n_splits=6, random_state=0, shuffle=False)
for train_index, test_index in skf.split(x_train, y_train):
    x_train, y_train = x_train[test_index], y_train[test_index]
    break
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#################################################################################
batch_size = 32  # You can try 64 or 128 if you'd like to
num_classes = 10
epochs = 5  # loss function value will be stabilized after 93rd epoch
# To save the model:
# save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_fashion_mnist_trained_model.h5'
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
input_shape = (28, 28, 1)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

#################################################################################

# Cuvanje istreniranog modela u fajl
model.save('model.h5')
