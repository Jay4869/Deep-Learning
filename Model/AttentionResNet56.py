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
from .Residual_Unit import Residual_Unit
from .Attention_Block import Attention_Block


def AttentionResNet56(x, in_channel=64, kernel_size=7, n_classes=None, dropout=None, regularization=0.01):

    """
    :param in_channel: The 4-th dimension (channel number) of input weight matrix. For example, in_channel=3 means the input contains 3 channels.
    :param kernel_size: the shape of the kernel. For example, default kernel_size = 3 means you have a 3*3 kernel.
    :param n_classes: Integer. The number of target classes. For example, n_classes = 10 means you have 10 class labels.
    :param dropout: Float between 0 and 1. Fraction of the input units to drop.
    :param regularization: Float. Fraction of the input units to drop.
    """

    # x = Input()
    x = Conv2D(in_channel, kernel_size=kernel_size, strides=2, padding='same')(x)  # 112x112x64
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)  # 56x56x64

    x = Residual_Unit(x, 64, 256)  # 56x56x256
    x = Attention_Block(x)

    x = Residual_Unit(x, 128, 512, stride=2)  # 28x28x512
    x = Attention_Block(x)

    x = Residual_Unit(x, 256, 1024, stride=2)  # 14x14x1024
    x = Attention_Block(x)

    x = Residual_Unit(x, 512, 2048, stride=2)  # 7x7x2048
    x = Residual_Unit(x, 512, 2048)
    x = Residual_Unit(x, 512, 2048)

    x = AveragePooling2D(pool_size=7, strides=1)(x)  # 1x1x2048
    x = Flatten()(x)

    if dropout:
        x = Dropout(dropout)(x)

    output = Dense(n_classes, kernel_regularizer=l2(regularization), activation='softmax')(x)

    # model = Model(input_, output)
    return output

