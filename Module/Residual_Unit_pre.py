from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add

"""
Residual Unit is implemented above proposed Residual Block
input >> BN >> ReLU >> Conv2DLayer(3,3)
      >> BN >> ReLU >> Conv2DLayer(3,3) + identify
      >> output
"""

def Residual_Unit_pre(input, out_channel, stride=1):

    """
    :param input: The input of the Residual_Unit. Should be a 4D array like (batch_num, img_len, img_len, channel_num)
    :param out_channel: The 4-th dimension (channel number) of output matrix. For example, out_channel=5 means the output contains 5 channels (feature maps).
    :param stride: Integer. The number of pixels to move between 2 neighboring receptive fields.
    """

    # initialize as the input (identity) data
    shortcut = input
    if shortcut.shape[-1] != out_channel:
        shortcut = Conv2D(out_channel, (1, 1), padding='same')(shortcut)

    # RestNet module
    x = BatchNormalization()(input)
    x = Activation('relu')(x)
    x = Conv2D(out_channel, (3, 3), padding='same')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(out_channel, (3, 3), padding='same')(x)

    x = Add()([x, shortcut])

    return x






