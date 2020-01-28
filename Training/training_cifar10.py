import tensorflow as tf
import numpy as np
import time, os, datetime
from Model import AttentionResNet56
from Model import AttentionResNet56_mini
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping,TensorBoard

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.compat.v1.Session(config=config)

def training_cifar10(version, n1, n2, method, epc):

    # Load the CIFAR10 data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train[:n1, :, :, :]
    y_train = y_train[:n1]
    x_val = x_train[-5000:, :, :, :]
    y_val = y_train[-5000:]
    x_test = x_test[:n2, :, :, :]
    y_test = y_test[:n2]

    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
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

    test_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

    # compute quantities required for feature normalization
    train_datagen.fit(x_train)
    test_datagen.fit(x_val)
    test_datagen.fit(x_test)

    if version == 'mini':
        # build a model
        print('Training AttentionResNet56_mini')
        model = AttentionResNet56_mini(shape=(32, 32, 3), in_channel=32,
                                       kernel_size=5, n_classes=10,
                                       dropout=0.4, regularization=0.01)
    elif version == '56':
        print('Training AttentionResNet56')
        model = AttentionResNet56(shape=(32, 32, 3), in_channel=64,
                                  kernel_size=7, n_classes=10,
                                  dropout=0.4, regularization=0.01)
    else:
        print('Input model is not being implemented')

    if method == 'SGD':
        optimizer = SGD(learning_rate=0.0001, momentum=0.9, nesterov=True)
    elif method == 'Adam':
        optimizer = Adam(learning_rate=0.0001)
    else:
        print('Input method is not being implemented')

    # define loss, metrics, optimizer
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # training model
    batch_size = 256
    epc = epc
    start = time.time()

    train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
    step_size_train = train_generator.n // train_generator.batch_size

    val_generator = test_datagen.flow(x_val, y_val)
    test_generator = test_datagen.flow(x_test, y_test, batch_size=batch_size)

    log_dir = 'Logs/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit_generator(train_generator,
                        steps_per_epoch=step_size_train,
                        epochs=epc,
                        validation_data=val_generator,
                        callbacks=[tensorboard_callback])

    end = time.time()
    print("Time taken by above cell is {}.".format((end - start) / 60))

    # evaluation
    val_scores = model.evaluate_generator(val_generator, verbose=0)
    test_scores = model.evaluate_generator(test_generator, verbose=0)
    print('validation loss:', val_scores[0])
    print('validation accuracy:', val_scores[1])
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])

    return model

if __name__ == '__main__':
    import sys

    version = sys.argv[1]
    n1 = int(sys.argv[2])
    n2 = int(sys.argv[3])
    method = sys.argv[4]
    epc = int(sys.argv[5])

    model = training_cifar10(version, n1, n2, method, epc)
    model.save('Model56_paper.h5')  # 0.7418
