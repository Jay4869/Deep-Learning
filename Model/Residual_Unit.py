from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, Add

"""
Residual Unit is implemented above proposed Residual Block
input >> BN >> ReLU >> Conv2DLayer(1,1)
      >> BN >> ReLU >> Conv2DLayer(3,3)
      >> BN >> ReLU >> Conv2DLayer(1,1) + identify
      >> output
"""

def Residual_Unit(input, in_channel, out_channel, kernel_size=(3, 3), stride=1):

    """
    :param input: The input of the Residual_Unit. Should be a 4D array like (batch_num, img_len, img_len, channel_num)
    :param in_channel: The 4-th dimension (channel number) of input matrix. For example, in_channel=3 means the input contains 3 channels.
    :param out_channel: The 4-th dimension (channel number) of output matrix. For example, out_channel=5 means the output contains 5 channels (feature maps).
    :param kernel_size: the shape of the kernel. For example, default kernel_size = 3 means you have a 3*3 kernel.
    :param stride: Integer. The number of pixels to move between 2 neighboring receptive fields.
    """

    x = BatchNormalization()(input)
    x = Activation('relu')(x)
    x = Conv2D(in_channel, (1, 1))(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(in_channel, kernel_size, padding='same', strides=stride)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(out_channel, (1, 1), padding='same', strides=stride)(x)

    x = Add()([x, input])

    return x






