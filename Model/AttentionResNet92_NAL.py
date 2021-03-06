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

from Module import Residual_Unit
from Module import Attention_Block_NAL

def AttentionResNet92_NAL(shape, in_channel, kernel_size, n_classes, dropout=None, regularization=0.01):

    """
    :param shape: The tuple of input data.
    :param in_channel: The 4-th dimension (channel number) of input weight matrix. For example, in_channel=3 means the input contains 3 channels.
    :param kernel_size: Integer. the shape of the kernel. For example, default kernel_size = 3 means you have a 3*3 kernel.
    :param n_classes: Integer. The number of target classes. For example, n_classes = 10 means you have 10 class labels.
    :param dropout: Float between 0 and 1. Fraction of the input units to drop.
    :param regularization: Float. Fraction of the input units to drop.
    """

    input_data = Input(shape=shape)  # 32x32x3
    x = Conv2D(in_channel, kernel_size=kernel_size, padding='same')(input_data)  # 32x32x32
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=2, padding='same')(x)  # 16x16x32

    out_channel = in_channel * 4  # 256
    x = Residual_Unit(x, in_channel, out_channel)  # 16x16x128
    x = Attention_Block_NAL(x, skip=2)

    in_channel = out_channel // 2  # 64
    out_channel = in_channel * 4  # 256
    x = Residual_Unit(x, in_channel, out_channel, stride=2)  # 8x8x256
    x = Attention_Block_NAL(x, skip=1)
    x = Attention_Block_NAL(x, skip=1)

    in_channel = out_channel // 2  # 128
    out_channel = in_channel * 4  # 512
    x = Residual_Unit(x, in_channel, out_channel, stride=2)  # 512
    x = Attention_Block_NAL(x, skip=1)
    x = Attention_Block_NAL(x, skip=1)
    x = Attention_Block_NAL(x, skip=1)

    in_channel = out_channel // 2  # 256
    out_channel = in_channel * 4  # 1024
    x = Residual_Unit(x, in_channel, out_channel, stride=1)  # 4x4x1024
    x = Residual_Unit(x, in_channel, out_channel)
    x = Residual_Unit(x, in_channel, out_channel)

    x = AveragePooling2D(pool_size=4, strides=1)(x)  # 1x1x1024
    x = Flatten()(x)

    if dropout:
        x = Dropout(dropout)(x)

    output = Dense(n_classes, kernel_regularizer=l2(regularization), activation='softmax')(x)
    model = Model(input_data, output)

    return model

