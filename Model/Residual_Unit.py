import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.compat.v1.Session(config=config)

"""
Residual Unit is implemented above proposed Residual Block
input >> BN >> ReLU >> Conv2DLayer(1,1)
      >> BN >> ReLU >> Conv2DLayer(3,3)
      >> BN >> ReLU >> Conv2DLayer(1,1) + identify
      >> output
"""

def Residual_Unit(input, in_channel, out_channel, stride=1):

    """
    :param input: The input of the Residual_Unit. Should be a 4D array like (batch_num, img_len, img_len, channel_num)
    :param in_channel: The 4-th dimension (channel number) of input matrix. For example, in_channel=3 means the input contains 3 channels.
    :param out_channel: The 4-th dimension (channel number) of output matrix. For example, out_channel=5 means the output contains 5 channels (feature maps).
    :param stride: Integer. The number of pixels to move between 2 neighboring receptive fields.
    """

    # initialize as the input (identity) data
    shortcut = input
    in_channel = int(in_channel)
    out_channel = int(out_channel)

    # RestNet module
    x = BatchNormalization()(input)
    x = Activation('relu')(x)
    x = Conv2D(in_channel, (1, 1))(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(in_channel, (3, 3), padding='same', strides=stride)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(out_channel, (1, 1), padding='same')(x)

    # reduce the identity size
    shortcut = Conv2D(out_channel, (1, 1), padding='same', strides=stride)(shortcut)

    x = Add()([x, shortcut])

    return x






