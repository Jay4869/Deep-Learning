import tensorflow as tf
import numpy as np
import time, os, datetime
from Model import Residual_Unit
from Model import Attention_Block
from Model import AttentionResNet56_mini
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping,TensorBoard


def training_cifar10(n, method):

    # Load the CIFAR10 data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train[:n, :, :, :]
    y_train = y_train[:n]
    x_test = x_test[:(n*0.2), :, :, :]
    y_test = y_test[:(n*0.2)]

    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)

    # define generators for training and validation data
    train_datagen = ImageDataGenerator(featurewise_center=True,
                                       featurewise_std_normalization=True,
                                       rotation_range=20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    val_datagen = ImageDataGenerator(featurewise_center=True,
                                     featurewise_std_normalization=True)

    # compute quantities required for feature normalization
    train_datagen.fit(x_train)
    val_datagen.fit(x_train)

    # build a model
    model = AttentionResNet56_mini(shape=(32, 32, 3), in_channel=32,
                                   kernel_size=5, skip=2, n_classes=10,
                                   dropout=0.3, regularization=0.01)
    if method == 'SGD':
        optimizer = SGD(learning_rate=0.0001, momentum=0.9, nesterov=True)
    elif method == 'Adam':
        optimizer = Adam(learning_rate=0.0001)
    else:
        print('Input method is not being implemented')

    # define loss, metrics, optimizer
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # training model
    batch_size = 64
    epc = 50
    start = time.time()

    train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
    step_size_train = train_generator.n // train_generator.batch_size
    test_generator = val_datagen.flow(x_test, y_test, batch_size=batch_size)
    step_size_test = test_generator.n // test_generator.batch_size

    log_dir = 'Logs/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit_generator(train_generator,
                        steps_per_epoch=step_size_train,
                        epochs=epc,
                        validation_data=test_generator,
                        validation_steps=step_size_test,
                        callbacks=[tensorboard_callback])

    end = time.time()
    print("Time taken by above cell is {}.".format((end - start) / 60))

    # evaluation
    scores = model.evaluate(test_generator, steps=step_size_test)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    # save model
    model.save(log_dir + '.h5')

    return model

if __name__ == '__main__':
    import sys

    # n = int(sys.argv)
    # method = sys.argv

    print('Input Training Size')
    n = int(input())
    print('Optimizer method (SGD, Adam)')
    method = input()

    model = training_cifar10(n, method)