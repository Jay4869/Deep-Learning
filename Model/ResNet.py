from tensorflow.keras.layers import Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

from Module import Residual_Unit_pre

def ResNet(shape, filters, kernel_size, n_classes, dropout=None, regularization=None):

    """
    :param shape: The tuple of input data.
    :param in_channel: The 4-th dimension (channel number) of input weight matrix. For example, in_channel=3 means the input contains 3 channels.
    :param kernel_size: Integer. the shape of the kernel. For example, default kernel_size = 3 means you have a 3*3 kernel.
    :param n_classes: Integer. The number of target classes. For example, n_classes = 10 means you have 10 class labels.
    :param dropout: Float between 0 and 1. Fraction of the input units to drop.
    :param regularization: Float. Fraction of the input units to drop.
    """

    input_data = Input(shape=shape)  # 32x32x3
    x = Conv2D(filters, kernel_size=kernel_size, padding='same')(input_data)

    for _ in range(2):
        x = Residual_Unit_pre(x, 32)

    x = MaxPooling2D(pool_size=2)(x)
    for _ in range(2):
        x = Residual_Unit_pre(x, 64)

    x = MaxPooling2D(pool_size=2)(x)
    for _ in range(2):
        x = Residual_Unit_pre(x, 128)

    x = MaxPooling2D(pool_size=2)(x)
    for _ in range(2):
        x = Residual_Unit_pre(x, 256)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if dropout:
        x = Dropout(dropout)(x)
    x = AveragePooling2D(pool_size=4, strides=1)(x)
    x = Flatten()(x)
    output = Dense(n_classes, kernel_regularizer=l2(regularization), activation='softmax')(x)

    model = Model(input_data, output)

    return model

